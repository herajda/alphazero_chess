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

import threading
from torch.utils.tensorboard import SummaryWriter
import chess.engine
import chess
import math
import chess_moves
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
    parser.add_argument("--threads", type=int, default=30, help="Number of C++ self-play threads (and inference batch).")
    parser.add_argument("--sim_games", type=int, default=200, help="Number of self-play games per iteration.")
    parser.add_argument("--start_num_simulations", type=int, default=30, help="Initial number of MCTS simulations per move.")
    parser.add_argument("--end_num_simulations", type=int, default=200, help="Final number of MCTS simulations per move.")
    parser.add_argument("--num_simulations_steps", type=int, default=100, help="Number of steps over which to linearly decay num_simulations.")
    parser.add_argument("--num_simulations_eval", type=int, default=100, help="Eval num_simulations")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet alpha for root noise.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Exploration epsilon for root noise.")
    parser.add_argument("--sampling_moves", type=int, default=30, help="Number of moves to sample before switching to greedy.")
    parser.add_argument("--batch_size", type=int, default=400, help="Training batch size.")
    parser.add_argument("--train_for", type=int, default=120, help="Training steps per iteration.")
    parser.add_argument("--learning_rate", type=float, default=0.0015, help="Initial learning rate.")
    parser.add_argument("--final_learning_rate", type=float, default=0.0001, help="Final learning rate after decay.")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="AdamW weight decay.")
    parser.add_argument("--total_decay_iterations", type=int, default=500, help="Iterations over which to linearly decay the learning rate.")
    parser.add_argument("--evaluate_each", type=int, default=15, help="Perform evaluation every N iterations.")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="Save model checkpoint every N iterations.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of training iterations.")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to save final model.")
    parser.add_argument("--replay_buffer_capacity", type=int, default=1000000, help="Max on-disk entries in ring buffer.")
    parser.add_argument("--resume_model", type=str, default=None, help="Optional path to pretrained model to resume training.")
    parser.add_argument("--pretrain", type=bool, default=False, help="Pretrain the model before self-play.")
    parser.add_argument("--eval_games_per_color", type=int, default=10, help="Number of eval games per color vs Stockfish and vs Random.")
    parser.add_argument("--stockfish_path", type=str, default="/usr/games/stockfish",help="Path to Stockfish binary for ELO evaluation.")
    parser.add_argument("--stockfish_depth", type=int, default=12, help="Search depth for Stockfish during evaluation.")
    parser.add_argument("--stockfish_elo", type=int, default=1500, help="Simulated Elo for Stockfish (via UCI_LimitStrength/UCI_Elo)")
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
def evaluate_vs_stockfish(ts_path, num_games, args):
    """
    Play `num_games` per color between our model (using the C++ select_move binding)
    and Stockfish (via python-chess UCI). Return (w, l, d) from our model’s POV.
    """
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    engine.configure({"Threads": 4, "UCI_LimitStrength": True,
        "UCI_Elo": args.stockfish_elo})
    wins = losses = draws = 0

    for our_color in [chess.WHITE, chess.BLACK]:
        for _ in range(num_games):
            board = chess.Board()
            # alternate turns until game over
            while board.is_game_over(claim_draw=True) is False:
                if board.turn == our_color:
                    # ask C++ binding for our move
                    fen = board.fen()
                    action = chess_engine.select_move(
                        ts_path, fen,
                        400,
                        args.alpha, 0, args.sampling_moves
                    )
                    uci = chess_moves.action_to_uci(board, action)
                    if uci is None:
                        raise RuntimeError("Invalid action {action} returned by C++ engine")
                        
                    board.push_uci(uci)
                else:
                    # let Stockfish reply
                    result = engine.play(board, chess.engine.Limit(depth=args.stockfish_depth))
                    board.push(result.move)

            result = board.result(claim_draw=True)
            if result == "1-0":
                winner = chess.WHITE
            elif result == "0-1":
                winner = chess.BLACK
            else:
                winner = None

            if winner is None:
                draws += 1
            elif winner == our_color:
                wins += 1
            else:
                losses += 1

    engine.quit()
    return wins, losses, draws

def spawn_async_evaluation(iteration, ts_path, agent, args, writer):
    """
    Kick off both Random and Stockfish evals in a background thread,
    compute Elo vs Stockfish, and write all scalars to TensorBoard.
    """
    def _job():
        # 2) Stockfish eval
        sW, sL, sD = evaluate_vs_stockfish(ts_path, args.eval_games_per_color, args)
        score_sf = (sW + 0.5*sD) / (args.eval_games_per_color * 2)
        # 1) Random eval
        print("Evaluating model vs Random player...")
        scheduled_num_simulations = get_scheduled_num_simulations(iteration, args)
        wW, wL, wD, bW, bL, bD = evaluate_model(
            agent, ts_path,
            args.eval_games_per_color,
            args.threads,
            scheduled_num_simulations,  # Use scheduled num_simulations
            args.alpha, args.epsilon, args.sampling_moves
        )
        total = 2 * args.eval_games_per_color
        score_random = (wW + bW + 0.5*(wD + bD)) / total


        # 3) Elo estimate vs Stockfish 
        R_sf = args.stockfish_elo  
        # avoid scores=0 or 1
        eps = 1e-4
        p = max(eps, min(1 - eps, score_sf))
        dR = -400 * math.log10(1/p - 1)
        R_model = R_sf + dR

        # 4) log to TensorBoard
        writer.add_scalar("Eval/WinRate_vs_Random", score_random, iteration)
        writer.add_scalar("Eval/WinRate_vs_Stockfish", score_sf, iteration)
        writer.add_scalar("Eval/Elo_vs_Stockfish", R_model, iteration)
        writer.flush()

    th = threading.Thread(target=_job, daemon=True)
    th.start()
def get_scheduled_num_simulations(iteration, args):
    """Linearly interpolate num_simulations from start to end over num_simulations_steps."""
    if args.num_simulations_steps <= 1:
        return args.end_num_simulations
    if iteration >= args.num_simulations_steps:
        return args.end_num_simulations
    frac = iteration / (args.num_simulations_steps - 1)
    return int(round(args.start_num_simulations + frac * (args.end_num_simulations - args.start_num_simulations)))

def main():
    args = parse_args()
    writer = SummaryWriter(log_dir=getattr(args, "log_dir", None))


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
        spawn_async_evaluation(iteration, ts_path, agent, args, writer)

    if args.pretrain:
        # pretraining 
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
    torch.cuda.empty_cache()



    while training and iteration < args.max_iterations:
        torch.cuda.empty_cache()
        iteration += 1
        print(f"--- Iteration {iteration} ---")

        # schedule num_simulations
        scheduled_num_simulations = get_scheduled_num_simulations(iteration, args)
        print(f"Using num_simulations={scheduled_num_simulations}")

        # export to TorchScript
        ts_path = f"model_ts_{iteration}.pt"
        subprocess.run(["python3", "export_torchscript.py", args.model_path, ts_path], check=True)

        # buffered self-play generation
        chess_engine.simulate_games_buffered(
            ts_path,
            num_games=args.sim_games,
            num_threads=args.threads,
            num_simulations=scheduled_num_simulations,
            alpha=args.alpha,
            epsilon=args.epsilon,
            sampling_moves=args.sampling_moves,
            filename="games.bin",
            replay_buffer_capacity=args.replay_buffer_capacity
        )
        print("Generated self-play games into buffer.")
        torch.cuda.empty_cache()

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
        torch.cuda.empty_cache()

        # periodic evaluation placeholder
        if iteration % args.evaluate_each == 0:
            spawn_async_evaluation(iteration, ts_path, agent, args, writer)
            torch.cuda.empty_cache()

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
