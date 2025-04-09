import argparse
import collections
import os

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Pool
from chess_game import ChessGame  

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--processes", default=6, type=int, help="Maximum number of threads for generation to use.")
parser.add_argument("--alpha", default=0.3, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=64, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=1, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="model.pt", type=str, help="Model path")
parser.add_argument("--num_simulations", default=500, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--sampling_moves", default=3, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=False, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=20, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=100, type=int, help="Update steps in every iteration.")
parser.add_argument("--window_length", default=100_000, type=int, help="Replay buffer max length.")
parser.add_argument("--final_learning_rate", default=0.0001, type=float, help="Final minimum learning rate.")
parser.add_argument("--total_decay_iterations", default=100, type=int, help="Total iterations over which the learning rate will decay linearly.")
parser.add_argument("--infer", default=False, type=bool, help="Inference mode ON or OFF.")

class ReplayBuffer:
    """Simple replay buffer with possibly limited capacity."""
    def __init__(self, max_length=None):
        self._max_length = max_length
        self._data = []
        self._offset = 0

    def __len__(self):
        return len(self._data)

    @property
    def max_length(self):
        return self._max_length

    def append(self, item):
        if self._max_length is not None and len(self._data) >= self._max_length:
            self._data[self._offset] = item
            self._offset = (self._offset + 1) % self._max_length
        else:
            self._data.append(item)

    def extend(self, items):
        if self._max_length is None:
            self._data.extend(items)
        else:
            for item in items:
                if len(self._data) >= self._max_length:
                    self._data[self._offset] = item
                    self._offset = (self._offset + 1) % self._max_length
                else:
                    self._data.append(item)

    def __getitem__(self, index):
        assert -len(self._data) <= index < len(self._data)
        return self._data[(self._offset + index) % len(self._data)]

    def sample(self, size, generator=np.random, replace=True):
        # By default, the same element can be sampled multiple times. Making sure the samples
        # are unique is costly, and we do not mind the duplicites much during training.
        if replace:
            return [self._data[index] for index in generator.randint(len(self._data), size=size)]
        else:
            return [self._data[index] for index in generator.choice(len(self._data), size=size, replace=False)]

def adjust_learning_rate(optimizer, iteration, args):
    lr = args.learning_rate - (args.learning_rate - args.final_learning_rate) * (iteration / args.total_decay_iterations)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Agent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, args):
        class TransformerModel(nn.Module):
            def __init__(self, args):
                super(TransformerModel, self).__init__()
                self.board_size = ChessGame.N  # 8
                self.initial_channels = 119
                self.dim_model = 512  # Similar to num_channels in original
                self.num_actions = ChessGame.ACTIONS
                self.num_layers = 6
                self.num_heads = 8
                
                # Input projection
                self.input_proj = nn.Conv2d(self.initial_channels, self.dim_model, kernel_size=1)
                
                # Positional encoding
                self.pos_encoding = self.create_positional_encoding()
                
                # Transformer layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.dim_model,
                    nhead=self.num_heads,
                    dim_feedforward=self.dim_model*4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers= self.num_layers  # Same number of layers as residual blocks
                )
                
                # Policy head
                self.policy_conv = nn.Conv2d(self.dim_model, 2, kernel_size=3, padding=1)
                self.policy_flatten = nn.Flatten()
                self.policy_dense = nn.Linear(2 * self.board_size * self.board_size, self.num_actions)
                
                # Value head
                self.value_conv = nn.Conv2d(self.dim_model, 1, kernel_size=3, padding=1)
                self.value_flatten = nn.Flatten()
                self.value_dense = nn.Linear(self.board_size * self.board_size, 1)
                
            def create_positional_encoding(self):
                position = torch.arange(self.board_size * self.board_size).unsqueeze(1).float()
                div_term = torch.exp(torch.arange(0, self.dim_model, 2).float() * (-math.log(10000.0) / self.dim_model))
                
                pe = torch.zeros(self.board_size * self.board_size, self.dim_model)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                return pe.unsqueeze(0)  # [1, board_size*board_size, dim_model]
            
            def forward(self, x):
                # Input shape: [batch_size, board_size, board_size, channels]
                batch_size = x.size(0)
                
                # Move channels to second dimension
                x = x.permute(0, 3, 1, 2)  # [batch_size, channels, board_size, board_size]
                
                # Project to model dimension
                x = self.input_proj(x)  # [batch_size, dim_model, board_size, board_size]
                
                # Reshape for transformer
                x = x.flatten(2).transpose(1, 2)  # [batch_size, board_size*board_size, dim_model]
                
                # Add positional encoding
                pos_encoding = self.pos_encoding.to(x.device)
                x = x + pos_encoding
                
                # Transformer encoding
                x = self.transformer(x)  # [batch_size, board_size*board_size, dim_model]
                
                # Reshape back to 2D for conv heads
                x = x.transpose(1, 2).reshape(batch_size, self.dim_model, self.board_size, self.board_size)
                
                # Policy head
                policy_x = self.policy_conv(x)
                policy_x = self.policy_flatten(policy_x)
                policy = F.softmax(self.policy_dense(policy_x), dim=-1)
                
                # Value head
                value_x = self.value_conv(x)
                value_x = self.value_flatten(value_x)
                value = torch.tanh(self.value_dense(value_x))
                
                return policy, value

        self._model = TransformerModel(args).to(self.device)
        self.optimizer = torch.optim.AdamW(self._model.parameters(), lr=args.learning_rate)

    @classmethod
    def load(cls, path: str, args) -> "Agent":
        agent = Agent(args)
        agent._model.load_state_dict(torch.load(path, map_location=agent.device))
        return agent

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    def train(self, boards: torch.Tensor, target_policies: torch.Tensor, target_values: torch.Tensor) -> None:
        boards = boards.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)
        
        policy, value = self._model(boards)
        value = value.squeeze(-1)
        
        loss = (F.cross_entropy(policy, target_policies) + 
                F.mse_loss(value, target_values))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        boards = boards.to(self.device)
        with torch.no_grad():
            policy, value = self._model(boards)
        return policy.detach().cpu().numpy(), value.detach().cpu().numpy()

    def board(self, game) -> torch.Tensor:
        #if game.to_play != 0:
        #    game = game.clone(swap_players=True)
        return game.board


########
# MCTS #
########
class MCTNode:
    def __init__(self, prior: float | None):
        self.prior = prior  # Prior probability from the agent.
        self.game = None    # If the node is evaluated, the corresponding game instance.
        self.children = {}  # If the node is evaluated, mapping of valid actions to the child `MCTNode`s.
        self.visit_count = 0
        self.total_value = 0

    def value(self) -> float:
        # TODO: Return the value of the current node, handling the
        # case when `self.visit_count` is 0.
        if self.visit_count == 0:
            return 0.
        return self.total_value / self.visit_count

    def is_evaluated(self) -> bool:
        # A node is evaluated if it has non-zero `self.visit_count`.
        # In such case `self.game` is not None.
        return self.visit_count > 0

    def evaluate(self, gm: ChessGame, agent: Agent) -> None:
        # Each node can be evaluated at most once
        #assert self.game is None
        self.game = gm
        #self.game = gm.clone() 
        #self.game = game

        # TODO: Compute the value of the current game.
        # - If the game has ended, compute the value directly
        # - Otherwise, use the given `agent` to evaluate the current
        #   game. Then, for all valid actions, populate `self.children` with
        #   new `MCTNodes` with the priors from the policy predicted
        #   by the network.

        if self.game.winner is not None:

            self.children = {}
            if self.game.winner == -1:
                value = 0
            elif self.game.winner == self.game.to_play:
                value = 1
            else:
                value = -1

        else:
            agent_board = torch.from_numpy(agent.board(self.game)[np.newaxis])

            policy, _ = agent.predict(agent_board)
            policy = policy[0]

            valid_actions = self.game.valid_actions()
            self.children = {action: MCTNode(policy[action]) for action in valid_actions}
            
            # NOTE: don't think I need this
            #total = sum(policy[action] for action in valid_actions)
            #for action in valid_actions:
            #    self.children[action].prior /= total

            value = self.value()

        self.visit_count, self.total_value = 1, value

    def add_exploration_noise(self, epsilon: float, alpha: float) -> None:
        # TODO: Update the children priors by exploration noise
        # Dirichlet(alpha), so that the resulting priors are
        #   epsilon * Dirichlet(alpha) + (1 - epsilon) * original_prior
        for _, child in self.children.items():
            child.prior = epsilon * np.random.dirichlet([alpha]) + (1 - epsilon) * child.prior
        

    def select_child(self) -> tuple[int, "MCTNode"]:
        def ucb_score(child: "MCTNode"):
            Q = - child.value()
            P = child.prior
            N = self.visit_count
            N_sa = child.visit_count

            C = np.log((1 + N + 1965.2) / 1965.2) + 1.25
            ucb_score = Q + C * P * np.sqrt(N) / (N_sa + 1)
            return ucb_score 

        best_action, best_child = None, None 

        for action, child in self.children.items():
            if best_action is None or ucb_score(child) > ucb_score(best_child):
                best_action, best_child = action, child
        return best_action, best_child


def mcts(game: ChessGame, agent: Agent, args: argparse.Namespace, explore: bool) -> np.ndarray:
    root = MCTNode(None)
    root.evaluate(gm=game, agent=agent)

    if explore:
        root.add_exploration_noise(args.epsilon, args.alpha)

    # Run MCTS for `args.num_simulations` iterations.
    path = []
    for _ in range(args.num_simulations):
        node = root 
        action = None

        while node.children:
            game = node.game
            if node.is_evaluated():
                action, node = node.select_child()

                path.append((node, action))
            else:
                break

        # If the node has not been evaluated, evaluate it.
        if not node.is_evaluated():

            if action is None or len(game.valid_actions()) == 0:
                game = game.clone()
            else:
                game = game.clone()
                game.move(action)

                node.evaluate(game, agent)

        else:
            node.evaluate(game, agent)

        # Get the value of the node.
        value = node.value()

        for node, action in reversed(path):
            node.visit_count += 1
            node.total_value += value
            # Invert the value for the opponent's perspective
            value = -value  
        path = []

    policy = np.zeros(game.ACTIONS, dtype=np.float32)
    total_visits = sum(child.visit_count for child in root.children.values())

    for action, child in root.children.items():
        if total_visits > 0:
            policy[action] = child.visit_count / total_visits
    return policy


# TRAINING
ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])

def sim_game(agent: Agent, args: argparse.Namespace) -> list[ReplayBufferEntry]:
    # Simulate a game, return a list of `ReplayBufferEntry`s.
    game = ChessGame()
    game_states = []  
    moves = 0

    while game.winner is None:
        print(moves)
        policy = mcts(game, agent, args, explore=True)

        mask = np.zeros(game.ACTIONS, dtype=bool)
        mask[game.valid_actions()] = True
        policy[~mask] = 0
        if moves >= args.sampling_moves:
            action = np.argmax(policy)

        else:
            action = np.random.choice(np.arange(game.ACTIONS), p=policy)

        game.move(action)
        moves += 1

        game_states.append((game.board, policy))
    
    game_winnner = game.winner
    entries = [ReplayBufferEntry(board, policy, game_winnner) for board, policy in game_states]
    return entries
def simulate_single_game(args):
    agent = Agent(args)  
    return sim_game(agent, args)

def train(args: argparse.Namespace) -> Agent:
    agent = Agent(args)
    # TODO implement ReplayBuffer
    replay_buffer = ReplayBuffer(max_length=args.window_length)

    iteration = 0
    training = True

    score_deque = collections.deque(maxlen=5)
    
    while training:
        iteration += 1

        print(f"Iteration {iteration}:")
        # Generate simulated games
        with Pool(processes=args.processes) as pool:
            games = pool.map(simulate_single_game, [args] * args.sim_games)

            for game in games:
                replay_buffer.extend(game)



        adjust_learning_rate(agent.optimizer, iteration, args)
        for _ in range(args.train_for):
            # Perform training by sampling an `args.batch_size` of positions
            # from the `replay_buffer` and running `agent.train` on them.
            samples = replay_buffer.sample(args.batch_size)
            boards, policies, outcome = map(np.array, zip(*samples))

            agent.train(torch.tensor(boards, dtype=torch.float32),
                        torch.tensor(policies, dtype=torch.float32),
                        torch.tensor(outcome, dtype=torch.float32)) 

        if iteration % args.evaluate_each == 0:

            if np.mean(np.array(score_deque)) > 0.9:
                training = False

    agent.save(args.model_path)

    return agent


# Evaluation Player 
class Player:
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: ChessGame) -> int:
        # Predict a best possible action.
        if self.args.num_simulations == 0:
            # If no simulations should be performed, use directly the policy predicted by the agent on the current game board.
            agent_board = torch.from_numpy(self.agent.board(game)[np.newaxis])
            policy, _ = self.agent.predict(agent_board)
            policy = policy[0]
        else:
            policy = mcts(game, self.agent, self.args, explore=False) 
            mask = np.zeros(game.ACTIONS, dtype=bool)
            mask[game.valid_actions()] = True
            policy[~mask] = 0

        # Select the action with the highest probability
        return max(game.valid_actions(), key=lambda action: policy[action])


def main(args: argparse.Namespace) -> Player:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    if args.infer:
        args.num_simulations = 100 
        agent = Agent.load(args.model_path, args)
    else:
        agent = train(args)

    return Player(agent, args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    player = main(args)
