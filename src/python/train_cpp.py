#!/usr/bin/env python3
"""
train_cpp.py

Self-play training script using the C++ `chess_engine.simulate_games_buffered` backend,
with on-disk circular replay buffer.
"""
import argparse
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"  
import struct, random, subprocess, json, threading, math

import numpy as np, torch, chess_engine
from torch.utils.tensorboard import SummaryWriter
from chess_agent import Agent, adjust_learning_rate

# Constants for record sizes
RECORD_STATE = 8 * 8 * 119         # number of floats in state
RECORD_POLICY = 4672               # number of floats in policy (8*8*73)
RECORD_Z = 1                       # one float for outcome
RECORD_FLOATS = RECORD_STATE + RECORD_POLICY + RECORD_Z
RECORD_BYTES = RECORD_FLOATS * 4    # bytes per record
HEADER_BYTES = 24                   # bytes for [capacity, size, head]


# ── sample_from_file -------------------------------------------------
def sample_from_file(path: str, batch_size: int):
    if not os.path.exists(path):
        return []

    with open(path, "rb") as f:
        hdr = f.read(HEADER_BYTES)
        if len(hdr) < HEADER_BYTES:
            return []

        capacity, size, head = struct.unpack("<qqq", hdr)
        #print(f"Buffer capacity={capacity}, size={size}, head={head}")
        if size <= 0:
            return []

        idxs = random.sample(range(size), k=min(batch_size, size))
        batch = []

        for i in idxs:
            phys = (head - size + i) % capacity      # <── NEW
            off  = HEADER_BYTES + phys * RECORD_BYTES
            f.seek(off)

            # read state
            sb = f.read(RECORD_STATE * 4)
            if len(sb) != RECORD_STATE * 4:          # corrupt/partial – skip
                continue
            state = np.frombuffer(sb, dtype=np.float32).reshape((8, 8, 119))  # reshape to (8, 8, 119)

            # read policy
            pb = f.read(RECORD_POLICY * 4)
            if len(pb) != RECORD_POLICY * 4:
                continue
            policy = np.frombuffer(pb, dtype=np.float32)

            # read z
            zb = f.read(4)
            if len(zb) != 4:
                continue
            z, = struct.unpack("<f", zb)

            batch.append((state, policy, z))

    return batch
# ──────────────────────────────────────────────────────────────────────
def load_all_records(path: str):
    """Return *all* (state, π, z) tuples from the on-disk ring buffer."""
    if not os.path.exists(path):
        return []

    records = []
    with open(path, "rb") as f:
        hdr = f.read(HEADER_BYTES)
        if len(hdr) < HEADER_BYTES:
            return []
        capacity, size, head = struct.unpack("<qqq", hdr)

        for i in range(size):                                    # logical order
            phys = (head - size + i) % capacity                  # physical slot
            off  = HEADER_BYTES + phys * RECORD_BYTES
            f.seek(off)

            # state 8·8·119
            sb = f.read(RECORD_STATE * 4)
            if len(sb) != RECORD_STATE * 4:
                break
            state = np.frombuffer(sb, dtype=np.float32).reshape((8, 8, 119))

            # policy 4672
            pb = f.read(RECORD_POLICY * 4)
            if len(pb) != RECORD_POLICY * 4:
                break
            policy = np.frombuffer(pb, dtype=np.float32)

            # z
            zb = f.read(4)
            if len(zb) != 4:
                break
            z, = struct.unpack("<f", zb)

            records.append((state, policy, z))
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero training with C++ buffered self-play backend")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--threads", type=int, default=90, help="Number of C++ self-play threads (and inference batch).")
    parser.add_argument("--sim_games", type=int, default=100, help="Number of self-play games per iteration.")
    parser.add_argument("--start_num_simulations", type=int, default=100, help="Initial number of MCTS simulations per move.")
    parser.add_argument("--end_num_simulations", type=int, default=600, help="Final number of MCTS simulations per move.")
    parser.add_argument("--num_simulations_steps", type=int, default=200, help="Number of steps over which to linearly decay num_simulations.")
    parser.add_argument("--num_simulations_eval", type=int, default=300, help="Eval num_simulations")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet alpha for root noise.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Exploration epsilon for root noise.")
    parser.add_argument("--sampling_moves", type=int, default=30, help="Number of moves to sample before switching to greedy.")
    parser.add_argument("--batch_size", type=int, default=400, help="Training batch size.")
    parser.add_argument("--train_for", type=int, default=120, help="Training steps per iteration.")
    parser.add_argument("--learning_rate", type=float, default=0.0015, help="Initial learning rate.")
    parser.add_argument("--final_learning_rate", type=float, default=0.0001, help="Final learning rate after decay.")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="AdamW weight decay.")
    parser.add_argument("--total_decay_iterations", type=int, default=500, help="Iterations over which to linearly decay the learning rate.")
    parser.add_argument("--evaluate_each", type=int, default=5, help="Perform evaluation every N iterations.")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Save model checkpoint every N iterations.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of training iterations.")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to save final model.")
    parser.add_argument("--replay_buffer_capacity", type=int, default=200000, help="Max on-disk entries in ring buffer.")
    parser.add_argument("--resume_model", type=str, default=None, help="Optional path to pretrained model to resume training.")
    parser.add_argument("--pretrain", type=bool, default=False, help="Pretrain the model before self-play.")
    # ───────────────── bootstrap / pure-MCTS pretraining ─────────────────
    parser.add_argument("--bootstrap_games", type=int, default=1000,
                        help="If >0 run this many self-play games with a "
                             "uniform dummy network *before* iteration 1.")
    parser.add_argument("--bootstrap_num_simulations", type=int, default=600)
    parser.add_argument("--bootstrap_threads", type=int, default=90)
    parser.add_argument("--bootstrap_train_for", type=int, default=60,
                        help="SGD steps (on the just generated buffer) "
                             "before entering the regular loop.")
    parser.add_argument("--bootstrap_batch_size", type=int, default=256)
    parser.add_argument("--bootstrap_replay_buffer_capacity", type=int, default=900000, help="Max on-disk entries in ring buffer.")

    parser.add_argument("--eval_games_per_color", type=int, default=25, help="Number of eval games per color vs Stockfish and vs Random.")
    parser.add_argument("--stockfish_path", type=str, default="/usr/games/stockfish",help="Path to Stockfish binary for ELO evaluation.")
    parser.add_argument("--stockfish_depth", type=int, default=12, help="Search depth for Stockfish during evaluation.")
    parser.add_argument("--stockfish_elo", type=int, default=1320, help="Simulated Elo for Stockfish (via UCI_LimitStrength/UCI_Elo)")
    return parser.parse_args()

# ---------------------------------------------------------------------------
#                         ###  BEGIN EVAL SECTION  ###
# ---------------------------------------------------------------------------

def spawn_async_evaluation(iteration: int, ts_path: str, args, writer):
    """
    Launch `evaluate_worker.py` in a *separate* process and stream its JSON
    result into TensorBoard without blocking training.
    """

    cmd = [
        "python3", "-u", "evaluate_worker.py",
        "--model-ts", ts_path,
        "--games",      str(args.eval_games_per_color),
        "--sims",       str(args.num_simulations_eval),
        "--alpha",      str(args.alpha),
        "--epsilon",    "0",
        "--sampling",   str(args.sampling_moves),
        "--sf-bin",     args.stockfish_path,
        "--sf-depth",   str(args.stockfish_depth),
        "--sf-elo",     str(args.stockfish_elo),
        "--threads",    str(20), 
    ]

    def _job():
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            report= json.loads(proc.stdout)        # {'random': {...}, 'stockfish': {...}}
            print(f"EVAL report= {report}")
        except Exception as e:
            print(f"[Eval] worker failed: {e}\ncmd: {' '.join(cmd)}")
            return

        # ---------- Random baseline ----------
# ---------- Random baseline (compute total from the actual JSON counts) ----------
        rnd = report["random"]
        white_games = rnd["wW"] + rnd["wL"] + rnd["wD"]
        black_games = rnd["bW"] + rnd["bL"] + rnd["bD"]
        total_random_games = white_games + black_games
        if total_random_games == 0:
            win_rate_random = 0.0
        else:
            win_rate_random = (
                rnd["wW"] + rnd["bW"] + 0.5 * (rnd["wD"] + rnd["bD"])
            ) / total_random_games

        # ---------- Stockfish baseline (again, compute total from JSON) ----------
        sf_list = report["stockfish"]  # [wW, wL, wD, bW, bL, bD]
        sf_white_games = sf_list[0] + sf_list[1] + sf_list[2]
        sf_black_games = sf_list[3] + sf_list[4] + sf_list[5]
        total_sf_games = sf_white_games + sf_black_games
        if total_sf_games == 0:
            win_rate_sf = 0.0
        else:
            wins  = sf_list[0] + sf_list[3]  # wW + bW
            draws = sf_list[2] + sf_list[5]  # wD + bD
            win_rate_sf = (wins + 0.5 * draws) / total_sf_games

        # ---------- Elo vs Stockfish (compute rating difference instead of absolute) ----------
        # If you want an absolute rating that never dips below zero, you can do:
        #    elo_diff = 400 * math.log10(win_rate_sf / (1 - win_rate_sf + 1e-12))
        #    agent_elo = args.stockfish_elo + elo_diff
        # But if you really only care about “how many Elo points below Stockfish”:
        eps = 1e-4
        p = max(eps, min(1 - eps, win_rate_sf))
        elo_diff = -400.0 * math.log10(1.0 / p - 1.0)
        # (This elo_diff is negative when win_rate_sf < 0.5, positive when > 0.5.)
        agent_elo = args.stockfish_elo + elo_diff

        # ---------- log ----------
        writer.add_scalar("Eval/WinRate_vs_Random",   win_rate_random, iteration)
        writer.add_scalar("Eval/WinRate_vs_Stockfish", win_rate_sf,     iteration)
        writer.add_scalar("Eval/Elo_vs_Stockfish",     agent_elo,            iteration)
        writer.flush()

        print(f"[Eval] iteration {iteration}: "
              f"Rnd={win_rate_random:.3f}, SF={win_rate_sf:.3f}, Elo={agent_elo:.0f}")

    threading.Thread(target=_job, daemon=True).start()

# ---------------------------------------------------------------------------
#                          ###  END EVAL SECTION  ###
# ---------------------------------------------------------------------------


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
        print(f"Resumed model from {args.resume_model}");
    else:
        agent = Agent(args)

    iteration = 0
    training = True

    # ───────────────────────── BOOTSTRAP stage ──────────────────────────
    if args.bootstrap_games > 0:

        # (a) prepare dummy TorchScript once
        if not os.path.exists("dummy_model.pt"):
            subprocess.run(["python3", "dummy_model.py",
                            "dummy_model.pt"], check=True)

        # (b) generate pure-MCTS games
        print(f"[bootstrap] generating {args.bootstrap_games} games "
              f"({args.bootstrap_num_simulations} sims, "
              f"{args.bootstrap_threads} threads, uniform priors)")

        chess_engine.simulate_games_buffered(
            "dummy_model.pt",
            num_games      = args.bootstrap_games,
            num_threads    = args.bootstrap_threads,
            num_simulations= args.bootstrap_num_simulations,
            alpha          = args.alpha,
            epsilon        = args.epsilon,
            sampling_moves = args.sampling_moves,
            filename       = "games.bin",
            replay_buffer_capacity = args.bootstrap_replay_buffer_capacity
        )
        # (c) load ALL records, shuffle once, train epoch-style
        print("[bootstrap] loading all records into memory …")
        records = load_all_records("games.bin")
        if not records:
            raise RuntimeError("games.bin is empty – bootstrap failed.")

        rng = np.random.default_rng(args.seed)
        rng.shuffle(records)
        print(f"[bootstrap] training on {len(records)} positions "
              f"in batches of {args.bootstrap_batch_size}")

        agent._model.train()
        for i in range(0, len(records), args.bootstrap_batch_size):
            batch = records[i : i + args.bootstrap_batch_size]
            boards, policies, zs = map(np.array, zip(*batch))
            agent.train(torch.tensor(boards,   dtype=torch.float32),
                        torch.tensor(policies, dtype=torch.float32),
                        torch.tensor(zs,       dtype=torch.float32))

        # (d) clean up – free disk space
        try:
            os.remove("games.bin")
            print("[bootstrap] games.bin removed")
        except FileNotFoundError:
            pass

        # (d) save & continue with normal loop
        agent.save(args.model_path)
        print("[bootstrap] done – switching to network-guided training")

    # evaluate the model before training
    if args.resume_model:
        print("Evaluating model before training...")
        ts_path = f"model_ts_initial.pt"
        subprocess.run(["python3", "export_torchscript.py", args.model_path, ts_path], check=True)
        spawn_async_evaluation(iteration, ts_path, args, writer)

    if args.pretrain:
        # pretraining 
        agent._model.train()
        adjust_learning_rate(agent.optimizer, iteration, args)

        batchsize = args.batch_size
        for step in range(args.train_for):
            batch = sample_from_file("games.bin", args.batch_size)
            if not batch:
                print("No games to train on; skipping training")
                continue
            batchsize = len(batch)
            boards, policies, zs = map(np.array, zip(*batch))
            boards_tensor   = torch.tensor(boards,   dtype=torch.float32)
            policies_tensor = torch.tensor(policies, dtype=torch.float32)
            zs_tensor       = torch.tensor(zs,       dtype=torch.float32)
            agent.train(boards_tensor, policies_tensor, zs_tensor)
        print(f"Training step completed on batch of {batchsize} entries")

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

        # training phase (apply pretrain logic here)
        agent._model.train()
        adjust_learning_rate(agent.optimizer, iteration, args)

        batchsize = args.batch_size
        for step in range(args.train_for):
            batch = sample_from_file("games.bin", args.batch_size)
            if not batch:
                print("No games to train on; skipping training")
                continue
            batchsize = len(batch)
            boards, policies, zs = map(np.array, zip(*batch))
            boards_tensor   = torch.tensor(boards,   dtype=torch.float32)
            policies_tensor = torch.tensor(policies, dtype=torch.float32)
            zs_tensor       = torch.tensor(zs,       dtype=torch.float32)
            agent.train(boards_tensor, policies_tensor, zs_tensor)
        print(f"Training step completed on batch of {batchsize} entries")

        # save model after each training phase
        agent.save(args.model_path)
        print(f"Saved model to {args.model_path}")
        torch.cuda.empty_cache()

        # periodic evaluation placeholder
        if iteration % args.evaluate_each == 0:
            new_ts = f"model_ts_{iteration}_eval.pt"
            subprocess.run(["python3", "export_torchscript.py", args.model_path, new_ts], check=True)
            spawn_async_evaluation(iteration, new_ts, args, writer)
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
