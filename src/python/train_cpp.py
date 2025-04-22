#!/usr/bin/env python3
"""
train_cpp.py

Self-play training script using the C++ `chess_engine.simulate_games` backend.
"""
import chess_engine
import argparse
import collections
import numpy as np
import torch
import os, struct


from chess_agent import Agent, ReplayBuffer, adjust_learning_rate


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero training with C++ self-play backend")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--threads", type=int, default=1, help="Number of C++ self-play threads (and inference batch).")
    parser.add_argument("--sim_games", type=int, default=1, help="Number of self-play games per iteration.")
    parser.add_argument("--num_simulations", type=int, default=100, help="MCTS simulations per move.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet alpha for root noise.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Exploration epsilon for root noise.")
    parser.add_argument("--sampling_moves", type=int, default=3,
                        help="Number of moves to sample before switching to greedy.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--train_for", type=int, default=4, help="Training steps per iteration.")
    parser.add_argument("--window_length", type=int, default=100_000,
                        help="Replay buffer max length.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--final_learning_rate", type=float, default=0.0001,
                        help="Final learning rate after decay.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="AdamW weight decay.")
    parser.add_argument("--total_decay_iterations", type=int, default=100,
                        help="Iterations over which to linearly decay the learning rate.")
    parser.add_argument("--evaluate_each", type=int, default=1,
                        help="Perform evaluation every N iterations.")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                        help="Save model checkpoint every N iterations.")
    parser.add_argument("--max_iterations", type=int, default=1000,
                        help="Maximum number of training iterations.")
    parser.add_argument("--model_path", type=str, default="model.pt",
                        help="Path to save final model.")
    return parser.parse_args()

GAMES_FILE = "games.bin"
def load_games_from_file(path: str):
    """Read every full trajectory dumped to `path`.  If we hit a short or
    missing chunk at any point, stop and return what we've got."""
    replays = []
    if not os.path.exists(path):
        return replays

    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                # no more complete headers
                break
            steps, = struct.unpack("<i", hdr)

            # read each step; bail out if any part is incomplete
            failed = False
            for _ in range(steps):
                # 1) flat state
                state_bytes = f.read(4 * 8 * 8 * 119)
                if len(state_bytes) < 4 * 8 * 8 * 119:
                    failed = True
                    break
                state = np.frombuffer(state_bytes, dtype=np.float32).reshape(8, 8, 119)

                # 2) policy length
                plen_bytes = f.read(4)
                if len(plen_bytes) < 4:
                    failed = True
                    break
                p_len, = struct.unpack("<i", plen_bytes)

                policy_bytes = f.read(4 * p_len)
                if len(policy_bytes) < 4 * p_len:
                    failed = True
                    break
                policy = np.frombuffer(policy_bytes, dtype=np.float32)

                # 3) z
                z_bytes = f.read(4)
                if len(z_bytes) < 4:
                    failed = True
                    break
                z, = struct.unpack("<f", z_bytes)

                replays.append((state, policy, z))

            if failed:
                # We encountered a truncated record—stop parsing further
                break

    # now clear the file (so we won't re‑read old or partial data next time)
    with open(path, "wb"):
        pass

    return replays
def simulate_and_dump(model_ts: str, sim_args, games_file: str = GAMES_FILE):
    """Wrapper around the new C++ binding."""
    chess_engine.simulate_games_dump(
        model_ts,
        num_games=sim_args.sim_games,
        num_threads=sim_args.threads,
        num_simulations=sim_args.num_simulations,
        alpha=sim_args.alpha,
        epsilon=sim_args.epsilon,
        sampling_moves=sim_args.sampling_moves,
        output_file=games_file,
    )

def main():
    args = parse_args()

    # Set random seeds and threading
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Initialize agent and replay buffer
    agent = Agent(args)
    replay_buffer = ReplayBuffer(max_length=args.window_length)

    iteration = 0
    training = True
    initial = load_games_from_file(GAMES_FILE)
    if initial:
        replay_buffer.extend(initial)
        print(f"Loaded {len(initial)} positions from previous dump.")

    while training:
        iteration += 1
        print(f"--- Iteration {iteration} ---")

        # --- Self-play generation via C++ backend ---
        agent._model.eval()
        # Export current model to TorchScript
        ts_path = f"model_ts_{iteration}.pt"
        import subprocess
        subprocess.run(["python3", "export_torchscript.py", args.model_path, ts_path], check=True)
        # dump to disk, then load it immediately
        simulate_and_dump(ts_path, args, GAMES_FILE)
        new = load_games_from_file(GAMES_FILE)
        if not new:
            print("Warning: no games were dumped!")
        else:
            replay_buffer.extend(new)
            print(f"Appended {len(new)} new positions from disk.")
        

        # --- Training phase ---
        agent._model.train()
        adjust_learning_rate(agent.optimizer, iteration, args)

        if len(replay_buffer) >= args.batch_size:
            for _ in range(args.train_for):
                samples = replay_buffer.sample(args.batch_size)
                if not samples:
                    break
                boards, policies, outcomes = map(np.array, zip(*samples))
                # Convert to torch tensors
                boards_tensor   = torch.tensor(boards, dtype=torch.float32)
                boards_tensor = boards_tensor.view(-1, 8, 8, 119)
                policies_tensor = torch.tensor(policies, dtype=torch.float32)
                values_tensor   = torch.tensor(outcomes, dtype=torch.float32)

                agent.train(boards_tensor, policies_tensor, values_tensor)
            print("Training step completed")
        else:
            print("Not enough examples for training; skipping")

        # --- Evaluation / Checkpointing ---
        if iteration % args.evaluate_each == 0:
            print(f"[Eval] Placeholder for evaluation at iteration {iteration}")

        if iteration % args.checkpoint_interval == 0:
            ckpt_path = f"model_checkpoint_{iteration}.pt"
            agent.save(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        if iteration >= args.max_iterations:
            training = False

    # Save final model
    agent.save(args.model_path)
    print(f"Training complete; model saved to {args.model_path}")


if __name__ == "__main__":
    main()
