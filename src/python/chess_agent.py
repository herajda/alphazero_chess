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
parser.add_argument("--network", default="resnet", choices=["resnet", "transformer"], help="Neural network architecture.")
parser.add_argument("--residual_channels", default=192, type=int, help="Channels in the ResNet trunk.")
parser.add_argument("--residual_blocks", default=12, type=int, help="Residual blocks in the ResNet trunk.")
parser.add_argument("--transformer_dim_model", default=512, type=int, help="Transformer model width.")
parser.add_argument("--transformer_layers", default=6, type=int, help="Transformer encoder layers.")
parser.add_argument("--transformer_heads", default=8, type=int, help="Transformer attention heads.")
parser.add_argument("--transformer_ff_multiplier", default=2, type=int, help="Transformer feed-forward width multiplier.")

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

def _arg(args, name, default):
    if not hasattr(args, name):
        setattr(args, name, default)
    return getattr(args, name)


def ensure_model_args(args):
    _arg(args, "network", "resnet")
    _arg(args, "residual_channels", 192)
    _arg(args, "residual_blocks", 12)
    _arg(args, "transformer_dim_model", 512)
    _arg(args, "transformer_layers", 6)
    _arg(args, "transformer_heads", 8)
    _arg(args, "transformer_ff_multiplier", 2)
    _arg(args, "learning_rate", 0.001)
    _arg(args, "weight_decay", 0.001)
    return args


def model_config_from_args(args) -> dict:
    ensure_model_args(args)
    return {
        "network": args.network,
        "residual_channels": int(args.residual_channels),
        "residual_blocks": int(args.residual_blocks),
        "transformer_dim_model": int(args.transformer_dim_model),
        "transformer_layers": int(args.transformer_layers),
        "transformer_heads": int(args.transformer_heads),
        "transformer_ff_multiplier": int(args.transformer_ff_multiplier),
    }


def apply_model_config(args, config: dict):
    for key, value in config.items():
        setattr(args, key, value)
    ensure_model_args(args)


def infer_network_from_state_dict(state_dict: dict) -> str:
    keys = state_dict.keys()
    if any(k.startswith("stem.") or k.startswith("blocks.") for k in keys):
        return "resnet"
    return "transformer"


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ResNetModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        channels = int(args.residual_channels)
        blocks = int(args.residual_blocks)
        self.board_size = ChessGame.N
        self.num_actions = ChessGame.ACTIONS

        self.stem = nn.Sequential(
            nn.Conv2d(119, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # Action ids are file * 8 * 73 + rank * 73 + move_type.
        self.policy_conv = nn.Conv2d(channels, 73, kernel_size=1)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_size * self.board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: [B, 8, 8, 119]
        x = x.permute(0, 3, 1, 2)
        x = self.blocks(self.stem(x))

        policy = self.policy_conv(x)
        policy_logits = policy.permute(0, 3, 2, 1).contiguous().view(x.size(0), self.num_actions)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.flatten(1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy_logits, value


class TransformerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.board_size = ChessGame.N
        self.initial_channels = 119
        self.dim_model = int(args.transformer_dim_model)
        self.num_actions = ChessGame.ACTIONS
        self.num_layers = int(args.transformer_layers)
        self.num_heads = int(args.transformer_heads)
        self.ff_multiplier = int(args.transformer_ff_multiplier)

        self.input_proj = nn.Conv2d(self.initial_channels, self.dim_model, kernel_size=1)
        self.register_buffer("pos_encoding", self.create_positional_encoding())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_model * self.ff_multiplier,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.policy_conv = nn.Conv2d(self.dim_model, 2, kernel_size=3, padding=1)
        self.policy_flatten = nn.Flatten()
        self.policy_dense = nn.Linear(2 * self.board_size * self.board_size, self.num_actions)

        self.value_conv = nn.Conv2d(self.dim_model, 1, kernel_size=3, padding=1)
        self.value_flatten = nn.Flatten()
        self.value_dense = nn.Linear(self.board_size * self.board_size, 1)

    def create_positional_encoding(self):
        pe = torch.zeros(self.board_size, self.board_size, self.dim_model)
        pos_row = torch.arange(self.board_size).float().unsqueeze(1)
        pos_col = torch.arange(self.board_size).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2).float() * (-math.log(10000.0) / self.dim_model)
        )

        pe_row = torch.zeros(self.board_size, self.dim_model)
        pe_col = torch.zeros(self.board_size, self.dim_model)
        pe_row[:, 0::2] = torch.sin(pos_row * div_term)
        pe_row[:, 1::2] = torch.cos(pos_row * div_term)
        pe_col[:, 0::2] = torch.sin(pos_col * div_term)
        pe_col[:, 1::2] = torch.cos(pos_col * div_term)

        for i in range(self.board_size):
            for j in range(self.board_size):
                pe[i, j] = pe_row[i] + pe_col[j]
        return pe.view(-1, self.dim_model).unsqueeze(0)

    def forward(self, x):
        bsz = x.size(0)
        x = x.permute(0, 3, 1, 2)
        x = self.input_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.transpose(1, 2).view(bsz, self.dim_model, self.board_size, self.board_size)

        px = self.policy_conv(x)
        px = self.policy_flatten(px)
        policy_logits = self.policy_dense(px)

        vx = self.value_conv(x)
        vx = self.value_flatten(vx)
        value = torch.tanh(self.value_dense(vx))
        return policy_logits, value


def build_model(args):
    ensure_model_args(args)
    if args.network == "resnet":
        return ResNetModel(args)
    if args.network == "transformer":
        return TransformerModel(args)
    raise ValueError(f"Unknown network architecture: {args.network}")


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


class Agent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, args):
        ensure_model_args(args)
        self.args = args
        self.model_config = model_config_from_args(args)
        self._model = build_model(args).to(self.device)
        print(f"Network: {args.network}")
        print(f"Model parameters: {sum(p.numel() for p in self._model.parameters())}")
        self.optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    @classmethod
    def load(cls, path: str, args) -> "Agent":
        ensure_model_args(args)
        checkpoint = torch.load(path, map_location="cpu")
        optimizer_state = None
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            optimizer_state = checkpoint.get("optimizer_state_dict")
            apply_model_config(args, checkpoint.get("model_config", {}))
        else:
            state_dict = checkpoint
            if isinstance(state_dict, dict):
                args.network = infer_network_from_state_dict(state_dict)
            ensure_model_args(args)

        agent = Agent(args)
        agent._model.load_state_dict(state_dict)
        if optimizer_state is not None:
            agent.optimizer.load_state_dict(optimizer_state)
            move_optimizer_state_to_device(agent.optimizer, agent.device)
        return agent

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self._model.state_dict(),
            "model_config": self.model_config,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def train(self, boards: torch.Tensor, target_policies: torch.Tensor, target_values: torch.Tensor) -> dict[str, float]:
        self._model.train()
        boards = boards.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device).view(-1)

        policy_logits, value = self._model(boards)
        value = value.squeeze(-1)
        log_policy = F.log_softmax(policy_logits, dim=1)
        policy = log_policy.exp()
        loss_policy = -torch.sum(target_policies * log_policy, dim=1).mean()
        loss_value = F.mse_loss(value, target_values)
        loss = loss_policy + loss_value

        metrics = {
            "loss": float(loss.detach().item()),
            "policy_loss": float(loss_policy.detach().item()),
            "value_loss": float(loss_value.detach().item()),
            "policy_entropy": float((-policy * log_policy).sum(dim=1).mean().detach().item()),
            "target_policy_entropy": float((-target_policies * torch.log(target_policies + 1e-8)).sum(dim=1).mean().detach().item()),
            "value_mean": float(value.detach().mean().item()),
            "target_value_mean": float(target_values.detach().mean().item()),
            "target_value_abs_mean": float(target_values.detach().abs().mean().item()),
            "target_value_nonzero_fraction": float((target_values.detach() != 0).float().mean().item()),
        }

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_sq = 0.0
        for parameter in self._model.parameters():
            if parameter.grad is not None:
                grad_norm_sq += parameter.grad.detach().data.norm(2).item() ** 2
        metrics["grad_norm"] = grad_norm_sq ** 0.5
        self.optimizer.step()
        return metrics

    def predict(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if type(boards) is not torch.Tensor:
            boards = torch.from_numpy(boards).float()
        boards = boards.to(self.device)
        self._model.eval()
        with torch.no_grad():
            policy_logits, value = self._model(boards)
            policy = F.softmax(policy_logits, dim=1)
        return policy.detach().cpu().numpy(), value.detach().cpu().numpy()

    def board(self, game) -> torch.Tensor:
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
            Q = -child.value()
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
        
        if path:
            for node, action in reversed(path[:-1]): # exclude leaf
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
