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
    parser.add_argument("--train_for", type=int, default=1, help="Training steps per iteration.")
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

    while training:
        iteration += 1
        print(f"--- Iteration {iteration} ---")

        # --- Self-play generation via C++ backend ---
        agent._model.eval()
        # Export current model to TorchScript
        ts_path = f"model_ts_{iteration}.pt"
        import subprocess
        subprocess.run(["python3", "export_torchscript.py", args.model_path, ts_path], check=True)
        games = chess_engine.simulate_games(
            ts_path,
            num_games=args.sim_games,
            num_threads=args.threads,
            num_simulations=args.num_simulations,
            alpha=args.alpha,
            epsilon=args.epsilon,
            sampling_moves=args.sampling_moves,
        )
        print(f"Generated {len(games)} games")

        # Collect into replay buffer
        for traj in games:
            replay_buffer.extend(traj)
        print(f"Replay buffer size: {len(replay_buffer)}")

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
