#!/usr/bin/env python3
"""
train_cpp.py

Self-play training script using the C++ `chess_engine.simulate_games_buffered` backend,
with on-disk circular replay buffer.
"""
import argparse
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"  
import struct
import random
import subprocess

import numpy as np
import torch

import chess_engine
from chess_agent import Agent, adjust_learning_rate

# Constants for record sizes
RECORD_STATE = 8 * 8 * 119         # number of floats in state
RECORD_POLICY = 4672               # number of floats in policy (8*8*73)
RECORD_Z = 1                       # one float for outcome
RECORD_FLOATS = RECORD_STATE + RECORD_POLICY + RECORD_Z
RECORD_BYTES = RECORD_FLOATS * 4    # bytes per record
HEADER_BYTES = 24                   # bytes for [capacity, size, head]


def sample_from_file(path: str, batch_size: int):
    """Randomly sample batch_size entries from the on-disk ring buffer."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        hdr = f.read(HEADER_BYTES)
        if len(hdr) < HEADER_BYTES:
            return []
        capacity, size, head = struct.unpack("<qqq", hdr)
        if size <= 0:
            return []

        # choose random indices among [0, size)
        idxs = random.sample(range(size), k=min(batch_size, size))
        batch = []
        for i in idxs:
            off = HEADER_BYTES + i * RECORD_BYTES
            f.seek(off)
            # read state
            b = f.read(RECORD_STATE * 4)
            state = np.frombuffer(b, dtype=np.float32).reshape(8, 8, 119)
            # read policy
            pb = f.read(RECORD_POLICY * 4)
            policy = np.frombuffer(pb, dtype=np.float32)
            # read z
            zb = f.read(4)
            z, = struct.unpack("<f", zb)
            batch.append((state, policy, z))
    return batch


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero training with C++ buffered self-play backend")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--threads", type=int, default=1, help="Number of C++ self-play threads (and inference batch).")
    parser.add_argument("--sim_games", type=int, default=1, help="Number of self-play games per iteration.")
    parser.add_argument("--num_simulations", type=int, default=100, help="MCTS simulations per move.")
    parser.add_argument("--num_simulations_eval", type=int, default=100, help="MCTS simulations per move in the evaluation mode.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet alpha for root noise.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Exploration epsilon for root noise.")
    parser.add_argument("--sampling_moves", type=int, default=3, help="Number of moves to sample before switching to greedy.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--train_for", type=int, default=1, help="Training steps per iteration.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--final_learning_rate", type=float, default=0.0001, help="Final learning rate after decay.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="AdamW weight decay.")
    parser.add_argument("--total_decay_iterations", type=int, default=100, help="Iterations over which to linearly decay the learning rate.")
    parser.add_argument("--evaluate_each", type=int, default=1, help="Perform evaluation every N iterations.")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="Save model checkpoint every N iterations.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of training iterations.")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to save final model.")
    parser.add_argument("--replay_buffer_capacity", type=int, default=100000, help="Max on-disk entries in ring buffer.")
    parser.add_argument("--resume_model", type=str, default=None, help="Optional path to pretrained model to resume training.")
    return parser.parse_args()

def evaluate_model(agent, ts_path, num_games, num_threads, num_simulations_eval, alpha, epsilon, sampling_moves):
    """Evaluate the model against a random player."""
    w_win, w_loss, w_draw, b_win, b_loss, b_draw = chess_engine.evaluate_vs_random(
        ts_path,
        num_games,
        num_threads,
        num_simulations_eval,
        alpha,
        epsilon,
        sampling_moves
    )
    print(f"As White: {w_win}W / {w_loss}L / {w_draw}D")
    print(f"As Black: {b_win}W / {b_loss}L / {b_draw}D")
    return w_win, w_loss, w_draw, b_win, b_loss, b_draw
def main():
    args = parse_args()

    # set seeds and threading
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # initialize agent (optionally resume)
    if args.resume_model:
        agent = Agent.load(args.resume_model, args)
        print(f"Resumed model from {args.resume_model}")
    else:
        agent = Agent(args)

    iteration = 0
    training = True

    # evaluate the model before training
    if args.resume_model:
        print("Evaluating model before training...")
        ts_path = f"model_ts_initial.pt"
        subprocess.run(["python3", "export_torchscript.py", args.model_path, ts_path], check=True)
        evaluate_model(
            agent,
            ts_path,
            num_games=10,
            num_threads=args.threads,
            num_simulations_eval=args.num_simulations_eval,
            alpha=args.alpha,
            epsilon=args.epsilon,
            sampling_moves=args.sampling_moves
        )

    while training and iteration < args.max_iterations:
        iteration += 1
        print(f"--- Iteration {iteration} ---")

        # export to TorchScript
        ts_path = f"model_ts_{iteration}.pt"
        subprocess.run(["python3", "export_torchscript.py", args.model_path, ts_path], check=True)

        # buffered self-play generation
        chess_engine.simulate_games_buffered(
            ts_path,
            num_games=args.sim_games,
            num_threads=args.threads,
            num_simulations=args.num_simulations,
            alpha=args.alpha,
            epsilon=args.epsilon,
            sampling_moves=args.sampling_moves,
            filename="games.bin",
            replay_buffer_capacity=args.replay_buffer_capacity
        )
        print("Generated self-play games into buffer.")

        # training phase
        agent._model.train()
        adjust_learning_rate(agent.optimizer, iteration, args)

        batch = sample_from_file("games.bin", args.batch_size)
        if not batch:
            print("No games to train on; skipping training")
        else:
            for _ in range(args.train_for):
                boards, policies, zs = map(np.array, zip(*batch))
                boards_tensor   = torch.tensor(boards,   dtype=torch.float32)
                policies_tensor = torch.tensor(policies, dtype=torch.float32)
                zs_tensor       = torch.tensor(zs,       dtype=torch.float32)
                agent.train(boards_tensor, policies_tensor, zs_tensor)
            print(f"Training step completed on batch of {len(batch)} entries")

        # save model after each training phase
        agent.save(args.model_path)
        print(f"Saved model to {args.model_path}")

        # periodic evaluation placeholder
        if iteration % args.evaluate_each == 0:
            evaluate_model(
                agent,
                ts_path,
                num_games=10,
                num_threads=args.threads,
                num_simulations_eval=args.num_simulations_eval,
                alpha=args.alpha,
                epsilon=args.epsilon,
                sampling_moves=args.sampling_moves
            )

        # periodic checkpoint
        if iteration % args.checkpoint_interval == 0:
            ckpt = f"model_checkpoint_{iteration}.pt"
            agent.save(ckpt)
            print(f"Saved checkpoint: {ckpt}")

        # stopping condition
        if iteration >= args.max_iterations:
            training = False

    print(f"Training complete; final model saved to {args.model_path}")


if __name__ == "__main__":
    main()
