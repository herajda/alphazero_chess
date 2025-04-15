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
parser.add_argument("--processes", default=1, type=int, help="Maximum number of threads for generation to use.")
parser.add_argument("--alpha", default=0.3, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=1, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=1, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="model.pt", type=str, help="Model path")
parser.add_argument("--num_simulations", default=100, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--sampling_moves", default=3, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=False, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=1, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=1, type=int, help="Update steps in every iteration.")
parser.add_argument("--window_length", default=100_000, type=int, help="Replay buffer max length.")
parser.add_argument("--final_learning_rate", default=0.0001, type=float, help="Final minimum learning rate.")
parser.add_argument("--weight_decay", default=0.001, type=float, help="Weight decay for AdamW.")
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
    lr = max(args.learning_rate - (args.learning_rate - args.final_learning_rate) * (iteration / args.total_decay_iterations), args.final_learning_rate)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# Add the initializer function
def init_worker():
    seed = os.getpid()
    np.random.seed(seed)
    torch.manual_seed(seed)

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
                # Initialize positional encoding for 8x8 board
                pe = torch.zeros(self.board_size, self.board_size, self.dim_model)
                
                # Generate row and column frequencies
                position_row = torch.arange(self.board_size).float().unsqueeze(1)
                position_col = torch.arange(self.board_size).float().unsqueeze(1)
                
                # Frequency terms (same for rows/columns but scaled by dimension)
                div_term = torch.exp(
                    torch.arange(0, self.dim_model, 2).float() *
                    (-math.log(10000.0) / self.dim_model)
                )
                
                # Compute row and column encodings
                pe_row = torch.zeros(self.board_size, self.dim_model)
                pe_col = torch.zeros(self.board_size, self.dim_model)
                
                pe_row[:, 0::2] = torch.sin(position_row * div_term)
                pe_row[:, 1::2] = torch.cos(position_row * div_term)
                pe_col[:, 0::2] = torch.sin(position_col * div_term)
                pe_col[:, 1::2] = torch.cos(position_col * div_term)
                
                # Combine row and column encodings for each (i,j) position
                for i in range(self.board_size):
                    for j in range(self.board_size):
                        pe[i, j] = pe_row[i] + pe_col[j]
                
                # Flatten to [64, 512] and add batch dimension [1, 64, 512]
                pe = pe.view(-1, self.dim_model).unsqueeze(0)
                return pe
            
            def forward(self, x):
                # Input shape: [batch_size, 8, 8, 119] (chess board tensor)
                batch_size = x.size(0)
                
                # --- Input Projection ---
                # Move channels to dimension 1 (for Conv2d)
                x = x.permute(0, 3, 1, 2)  # [batch_size, 119, 8, 8]
                
                # Project input to model dimension
                x = self.input_proj(x)  # [batch_size, 512, 8, 8]
                
                # --- Prepare for Transformer ---
                # Flatten spatial dimensions (8x8 -> 64)
                x = x.flatten(2)  # [batch_size, 512, 64]
                x = x.transpose(1, 2)  # [batch_size, 64, 512]
                
                # --- Add 2D Positional Encoding ---
                x = x + self.pos_encoding.to(x.device)  # [batch_size, 64, 512]
                
                # --- Transformer Layers ---
                x = self.transformer(x)  # [batch_size, 64, 512]
                
                # --- Reshape for Policy/Value Heads ---
                # Convert back to 2D grid
                x = x.transpose(1, 2)  # [batch_size, 512, 64]
                x = x.reshape(batch_size, self.dim_model, 8, 8)  # [batch_size, 512, 8, 8]
                
                # --- Policy Head ---
                # 1. Convolution to reduce channels
                policy_x = self.policy_conv(x)  # [batch_size, 2, 8, 8]
                # 2. Flatten spatial dimensions
                policy_x = self.policy_flatten(policy_x)  # [batch_size, 2*8*8 = 128]
                # 3. Dense layer to action space
                policy = F.softmax(self.policy_dense(policy_x), dim=-1)  # [batch_size, 4672]
                
                # --- Value Head ---
                # 1. Convolution to single channel
                value_x = self.value_conv(x)  # [batch_size, 1, 8, 8]
                # 2. Flatten spatial dimensions
                value_x = self.value_flatten(value_x)  # [batch_size, 8*8 = 64]
                # 3. Dense layer to scalar value
                value = torch.tanh(self.value_dense(value_x))  # [batch_size, 1]
                
                return policy, value


        self._model = TransformerModel(args).to(self.device)
        self.optimizer = torch.optim.AdamW(self._model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

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
        loss_policy = -torch.sum(target_policies * torch.log(policy + 1e-8), dim=1).mean()
        loss_value = F.mse_loss(value, target_values)
        loss = loss_policy + loss_value
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        boards = boards.to(self.device)
        self._model.eval()
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

            policy, predicted_value_tensor = agent.predict(agent_board) 
            policy = policy[0]

            valid_actions = self.game.valid_actions()
            mask = np.zeros_like(policy)
            mask[valid_actions] = 1
            
            # Apply mask and renormalize
            policy *= mask  # Zero out invalid actions
            policy_sum = policy.sum()
            
            if policy_sum < 1e-8:  # Handle division by zero
                # Uniform distribution over valid moves if sum is near zero
                policy[valid_actions] = 1 / len(valid_actions)
            else:
                policy /= policy_sum  # Normalize valid actions
            self.children = {action: MCTNode(policy[action]) for action in valid_actions}
            value = predicted_value_tensor[0, 0]

        self.visit_count, self.total_value = 1, value

    def add_exploration_noise(self, epsilon: float, alpha: float) -> None:
       num_children = len(self.children)
       if num_children == 0:
           return
       
       child_items = list(self.children.items())
       noise = np.random.dirichlet([alpha] * num_children)
       
       # Apply noise and calculate new priors
       new_priors = []
       for i, (action, child) in enumerate(child_items):
           new_prior = epsilon * noise[i] + (1 - epsilon) * child.prior
           new_priors.append(new_prior)
       
       # Explicit normalization to handle numerical stability
       total = sum(new_priors)
       if total <= 1e-8:  # Fallback to uniform distribution if sum is near zero
           new_priors = [1/num_children] * num_children
       else:
           new_priors = [p / total for p in new_priors]

       # Update children with normalized priors
       for i, (action, child) in enumerate(child_items):
           child.prior = new_priors[i]       

    def select_child(self) -> tuple[int, "MCTNode"]:
        def ucb_score(child: "MCTNode"):
            Q = child.value()
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
            # this should not happen
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
    game = ChessGame(gui_enabled=True)
    game_states = []  
    moves = 0

    while game.winner is None:
        current_board_tensor = agent.board(game)
        policy = mcts(game, agent, args, explore=True)

        mask = np.zeros(game.ACTIONS, dtype=bool)
        mask[game.valid_actions()] = True
        policy[~mask] = 0
        policy /= np.sum(policy) 
        if moves >= args.sampling_moves:
            action = np.argmax(policy)

        else:
            action = np.random.choice(np.arange(game.ACTIONS), p=policy)

        game_states.append((current_board_tensor, policy, game.to_play))
        game.move(action)
        moves += 1

    game_winner = game.winner
    # Compute outcome z from the perspective of the player to move
    entries = []
    for board, policy, to_play in game_states:
        if game_winner == -1:  # Draw
            z = 0
        elif game_winner == to_play:  # Player to move wins
            z = 1
        else:  # Player to move loses
            z = -1
        entries.append(ReplayBufferEntry(board, policy, z))
    return entries
def simulate_single_game(packed_args_and_state):
    args, state_dict = packed_args_and_state # Unpack the arguments

    # Create a new agent instance IN THE WORKER PROCESS
    # This agent might be on CPU or GPU depending on args and availability
    worker_agent = Agent(args)

    # Load the state_dict received from the main process
    worker_agent._model.load_state_dict(state_dict)

    return sim_game(worker_agent, args)


def train(args: argparse.Namespace) -> Agent:
    agent = Agent(args)
    replay_buffer = ReplayBuffer(max_length=args.window_length)

    iteration = 0
    training = True
    score_deque = collections.deque(maxlen=5) # Still needs implementation for evaluation

    # Ensure init_worker is defined or imported if needed for seeding
    # def init_worker():
    #     seed = os.getpid() + iteration # Add iteration for potentially more unique seeds
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)

    while training:
        iteration += 1
        print(f"Iteration {iteration}:")

        # --- Prepare state dict for workers ---
        agent._model.eval() # Good practice before getting state_dict if dropout/batchnorm are used
                            # Although workers call eval() again, doesn't hurt.

        # Get the current state dictionary from the agent's model
        current_state_dict = agent._model.state_dict()

        # IMPORTANT: Move the state_dict to CPU before sending to workers
        # This ensures it can be pickled and sent regardless of worker device (CPU/GPU)
        cpu_state_dict = {k: v.cpu() for k, v in current_state_dict.items()}
        # ------------------------------------

        # Generate simulated games using the POOL
        # Pass the init_worker for proper seeding in each process
        with Pool(processes=args.processes, initializer=init_worker) as pool:
            # Prepare arguments for each worker: a tuple of (args, cpu_state_dict)
            worker_args = [(args, cpu_state_dict)] * args.sim_games

            # Map the simulate_single_game function over the arguments
            games_data = pool.map(simulate_single_game, worker_args)

            # games_data is now a list of lists of ReplayBufferEntry
            for game_entries in games_data:
                replay_buffer.extend(game_entries)

        # --- Training Phase ---
        agent._model.train() # Set model back to training mode
        adjust_learning_rate(agent.optimizer, iteration, args)

        # Check if buffer has enough samples for a batch
        if len(replay_buffer) >= args.batch_size:
            for _ in range(args.train_for):
                samples = replay_buffer.sample(args.batch_size)
                # Check if sampling returned enough items (can happen if buffer < batch_size)
                if not samples:
                    print("Warning: Replay buffer smaller than batch size, skipping training step.")
                    break
                boards, policies, outcome = map(np.array, zip(*samples))

                # Ensure outcomes are properly shaped for MSE loss (e.g., [batch_size])
                # The original outcome might be single values, ensure they are float tensors
                outcomes_tensor = torch.tensor(outcome, dtype=torch.float32)
                # If value is shape [batch_size, 1], ensure target is too, or squeeze value
                # Current network outputs [batch_size, 1], so target should be [batch_size, 1] or value squeezed
                # Let's make target [batch_size] to match squeezed value head output
                # value = value.squeeze(-1) in agent.train suggests target should be [batch_size]

                agent.train(torch.tensor(boards, dtype=torch.float32),
                            torch.tensor(policies, dtype=torch.float32),
                            outcomes_tensor) # Pass the correctly typed tensor
        else:
             print(f"Replay buffer size {len(replay_buffer)} < batch size {args.batch_size}, skipping training.")

        # --- Evaluation / Stopping Condition ---
        if iteration % args.evaluate_each == 0:
            # TODO: Implement actual evaluation (e.g., play vs baseline/previous version)
            # and update score_deque based on evaluation results.
            print(f"Evaluation step needed at iteration {iteration}")
            # Example placeholder: if np.mean(np.array(score_deque)) > 0.9:
            #     training = False
            pass # Replace pass with evaluation logic

        # Optional: Save checkpoints periodically
        if iteration % 50 == 0: # Save every 50 iterations, adjust as needed
             print(f"Saving checkpoint at iteration {iteration}")
             agent.save(f"model_checkpoint_{iteration}.pt")


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
