#!/usr/bin/env python3
"""
train_cpp.py

Self-play training script using the C++ `chess_engine.simulate_games_buffered` backend,
with on-disk circular replay buffer.
"""
import argparse
import sys
import types
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import struct, random, subprocess, json, threading, math, csv, time, shutil
from pathlib import Path

import numpy as np, torch, chess_engine
sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))
from torch.utils.tensorboard import SummaryWriter
from chess_agent import Agent, adjust_learning_rate

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Constants for record sizes
RECORD_STATE = 8 * 8 * 119         # number of floats in state
RECORD_POLICY = 4672               # number of floats in policy (8*8*73)
RECORD_Z = 1                       # one float for outcome
RECORD_FLOATS = RECORD_STATE + RECORD_POLICY + RECORD_Z
RECORD_BYTES = RECORD_FLOATS * 4    # bytes per record
HEADER_BYTES = 24                   # bytes for [capacity, size, head]


# ── sample_from_file -------------------------------------------------
def record_offset(capacity: int, size: int, head: int, logical_idx: int) -> int:
    phys = (head - size + logical_idx) % capacity
    return HEADER_BYTES + phys * RECORD_BYTES


def read_record_at(f, offset: int):
    f.seek(offset)

    sb = f.read(RECORD_STATE * 4)
    if len(sb) != RECORD_STATE * 4:
        return None
    state = np.frombuffer(sb, dtype=np.float32).reshape((8, 8, 119))

    pb = f.read(RECORD_POLICY * 4)
    if len(pb) != RECORD_POLICY * 4:
        return None
    policy = np.frombuffer(pb, dtype=np.float32)

    zb = f.read(4)
    if len(zb) != 4:
        return None
    z, = struct.unpack("<f", zb)

    return state, policy, z


def read_z_at(f, offset: int):
    f.seek(offset + (RECORD_STATE + RECORD_POLICY) * 4)
    zb = f.read(4)
    if len(zb) != 4:
        return None
    z, = struct.unpack("<f", zb)
    return z


def sample_from_file(
    path: str,
    batch_size: int,
    decisive_fraction: float = 0.0,
    max_decisive_attempts_per_item: int = 50,
):
    if not os.path.exists(path):
        return []

    with open(path, "rb") as f:
        hdr = f.read(HEADER_BYTES)
        if len(hdr) < HEADER_BYTES:
            return []

        capacity, size, head = struct.unpack("<qqq", hdr)
        if size <= 0:
            return []

        target_size = min(batch_size, size)
        decisive_fraction = max(0.0, min(1.0, decisive_fraction))
        target_decisive = int(round(target_size * decisive_fraction))
        batch = []

        if target_decisive > 0:
            max_attempts = max(0, max_decisive_attempts_per_item) * target_decisive
            attempts = 0
            while len(batch) < target_decisive and attempts < max_attempts:
                attempts += 1
                i = random.randrange(size)
                off = record_offset(capacity, size, head, i)
                z = read_z_at(f, off)
                if z is None or z == 0.0:
                    continue
                record = read_record_at(f, off)
                if record is not None:
                    batch.append(record)

        while len(batch) < target_size:
            i = random.randrange(size)
            off = record_offset(capacity, size, head, i)
            record = read_record_at(f, off)
            if record is not None:
                batch.append(record)

    random.shuffle(batch)
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
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                        help="Number of concurrent C++ self-play actors. May exceed CPU cores to keep GPU inference batches full.")
    parser.add_argument("--selfplay_batch_size", type=int, default=64,
                        help="Maximum neural-network inference batch size for C++ self-play.")
    parser.add_argument("--torch_threads", type=int, default=0,
                        help="PyTorch CPU worker threads. 0 means min(--threads, CPU cores).")
    parser.add_argument("--sim_games", type=int, default=100, help="Number of self-play games per iteration.")
    parser.add_argument("--start_num_simulations", type=int, default=100, help="Initial number of MCTS simulations per move.")
    parser.add_argument("--end_num_simulations", type=int, default=600, help="Final number of MCTS simulations per move.")
    parser.add_argument("--num_simulations_steps", type=int, default=200, help="Number of steps over which to linearly decay num_simulations.")
    parser.add_argument("--num_simulations_eval", type=int, default=300, help="Eval num_simulations")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet alpha for root noise.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Exploration epsilon for root noise.")
    parser.add_argument("--sampling_moves", type=int, default=30, help="Number of moves to sample before switching to greedy.")
    parser.add_argument("--opening_random_plies", type=int, default=12,
                        help="For self-play, sample 0..N uniformly random legal plies before recording positions. 0 starts every game from the initial position.")
    parser.add_argument("--network", type=str, default="resnet", choices=["resnet", "transformer"], help="Neural network architecture for new models.")
    parser.add_argument("--residual_channels", type=int, default=192, help="Channels in the ResNet trunk.")
    parser.add_argument("--residual_blocks", type=int, default=12, help="Residual blocks in the ResNet trunk.")
    parser.add_argument("--transformer_dim_model", type=int, default=512, help="Transformer model width when --network=transformer.")
    parser.add_argument("--transformer_layers", type=int, default=6, help="Transformer encoder layers when --network=transformer.")
    parser.add_argument("--transformer_heads", type=int, default=8, help="Transformer attention heads when --network=transformer.")
    parser.add_argument("--transformer_ff_multiplier", type=int, default=2, help="Transformer feed-forward width multiplier.")
    parser.add_argument("--batch_size", type=int, default=400, help="Training batch size.")
    parser.add_argument("--decisive_sample_fraction", type=float, default=0.5,
                        help="Target fraction of each SGD batch sampled from decisive replay records (z != 0).")
    parser.add_argument("--max_decisive_sample_attempts_per_item", type=int, default=50,
                        help="Rejection-sampling probes per desired decisive record before falling back to uniform sampling.")
    parser.add_argument("--train_for", type=int, default=120, help="Training steps per iteration.")
    parser.add_argument("--learning_rate", type=float, default=0.0015, help="Initial learning rate.")
    parser.add_argument("--final_learning_rate", type=float, default=0.0001, help="Final learning rate after decay.")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="AdamW weight decay.")
    parser.add_argument("--total_decay_iterations", type=int, default=500, help="Iterations over which to linearly decay the learning rate.")
    parser.add_argument("--evaluate_each", type=int, default=5, help="Perform evaluation every N iterations.")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Save model checkpoint every N iterations.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of training iterations.")
    parser.add_argument("--start_iteration", type=int, default=None,
                        help="Completed training iteration to resume from. Defaults to the latest Train row in --metrics_path when --resume_model is used.")
    parser.add_argument("--run_dir", type=str, default="runs/train", help="Directory for this training run's artifacts.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to save the current model. Defaults to <run_dir>/model.pt.")
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory. Defaults to <run_dir>/tensorboard.")
    parser.add_argument("--metrics_path", type=str, default=None, help="CSV file for per-iteration training metrics. Defaults to <run_dir>/training_metrics.csv.")
    parser.add_argument("--eval_metrics_path", type=str, default=None, help="CSV file for async evaluation results. Defaults to <run_dir>/eval_metrics.csv.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory for periodic checkpoints. Defaults to <run_dir>/checkpoints.")
    parser.add_argument("--ts_dir", type=str, default=None, help="Directory for TorchScript exports. Defaults to <run_dir>/torchscript.")
    parser.add_argument("--replay_buffer_path", type=str, default=None, help="Replay buffer path. Defaults to <run_dir>/games.bin.")
    parser.add_argument("--best_model_path", type=str, default=None, help="Accepted best model path. Defaults to <run_dir>/best_model.pt.")
    parser.add_argument("--promotion_metrics_path", type=str, default=None, help="CSV file for candidate-vs-best arena results. Defaults to <run_dir>/promotion_metrics.csv.")
    parser.add_argument("--replay_buffer_capacity", type=int, default=200000, help="Max on-disk entries in ring buffer.")
    parser.add_argument("--resume_model", type=str, default=None, help="Optional path to pretrained model to resume training.")
    parser.add_argument("--no_resume_eval", action="store_true",
                        help="When resuming, skip the immediate pre-loop evaluation.")
    parser.add_argument("--pretrain", action="store_true", help="Train once on an existing games.bin before self-play.")
    # ───────────────── bootstrap / pure-MCTS pretraining ─────────────────
    parser.add_argument("--bootstrap_games", type=int, default=0,
                        help="If >0 run this many self-play games with a "
                             "uniform dummy network *before* iteration 1.")
    parser.add_argument("--bootstrap_num_simulations", type=int, default=600)
    parser.add_argument("--bootstrap_threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--bootstrap_train_for", type=int, default=60,
                        help="SGD steps (on the just generated buffer) "
                             "before entering the regular loop.")
    parser.add_argument("--bootstrap_batch_size", type=int, default=256)
    parser.add_argument("--bootstrap_replay_buffer_capacity", type=int, default=900000, help="Max on-disk entries in ring buffer.")

    parser.add_argument("--eval_games_per_color", type=int, default=25, help="Number of eval games per color vs Stockfish and vs Random.")
    parser.add_argument("--stockfish_eval_each", type=int, default=None,
                        help="Run Stockfish eval every N iterations. 0 disables Stockfish eval. Default follows --evaluate_each.")
    parser.add_argument("--eval_threads", type=int, default=min(20, os.cpu_count() or 1), help="Threads used by the async evaluation worker.")
    parser.add_argument("--max_parallel_evals", type=int, default=1, help="Maximum async evaluations allowed at the same time.")
    parser.add_argument("--eval_timeout", type=int, default=0, help="Seconds before an async evaluation is killed. 0 disables timeout.")
    parser.add_argument("--eval_initial", action="store_true", help="Evaluate the initial model before iteration 1.")
    parser.add_argument("--promotion_interval", type=int, default=5, help="Run candidate-vs-best promotion arena every N iterations. 0 disables gating and self-play uses the latest candidate.")
    parser.add_argument("--promotion_games_per_color", type=int, default=10, help="Arena games as each color for candidate-vs-best promotion.")
    parser.add_argument("--promotion_threshold", type=float, default=0.55, help="Promote candidate when arena score is at least this fraction.")
    parser.add_argument("--arena_num_simulations", type=int, default=300, help="MCTS simulations per move in candidate-vs-best arena.")
    parser.add_argument("--arena_threads", type=int, default=min(8, os.cpu_count() or 1), help="Threads for candidate-vs-best arena games.")
    parser.add_argument("--arena_opening_random_plies", type=int, default=-1, help="Random legal opening plies for arena. -1 uses --opening_random_plies.")
    parser.add_argument("--stockfish_path", type=str, default="/usr/games/stockfish", help="Path to Stockfish binary for ELO evaluation.")
    parser.add_argument("--stockfish_depth", type=int, default=12, help="Search depth for Stockfish during evaluation.")
    parser.add_argument("--stockfish_elo", type=int, default=1320, help="Simulated Elo for Stockfish (via UCI_LimitStrength/UCI_Elo)")
    return parser.parse_args()

# ---------------------------------------------------------------------------
#                         ###  BEGIN EVAL SECTION  ###
# ---------------------------------------------------------------------------


def elo_from_score(score: float, anchor_elo: int):
    eps = 1e-4
    p = max(eps, min(1 - eps, score))
    elo_diff = -400.0 * math.log10(1.0 / p - 1.0)
    return anchor_elo + elo_diff


def append_eval_csv(path: str, row: dict):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fieldnames = [
        "iteration",
        "status",
        "seconds",
        "model_ts",
        "games_per_color",
        "num_simulations",
        "stockfish_elo",
        "stockfish_depth",
        "win_rate_random",
        "win_rate_stockfish",
        "elo_vs_stockfish",
        "random_wW",
        "random_wL",
        "random_wD",
        "random_bW",
        "random_bL",
        "random_bD",
        "stockfish_wW",
        "stockfish_wL",
        "stockfish_wD",
        "stockfish_bW",
        "stockfish_bL",
        "stockfish_bD",
        "error",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def active_eval_threads(eval_threads):
    return sum(thread.is_alive() for thread in eval_threads)


def cleanup_eval_threads(eval_threads):
    eval_threads[:] = [thread for thread in eval_threads if thread.is_alive()]


def spawn_async_evaluation(iteration: int, ts_path: str, args, writer, eval_threads):
    """Launch evaluation in a separate process and persist TensorBoard + CSV metrics."""
    cleanup_eval_threads(eval_threads)
    if active_eval_threads(eval_threads) >= args.max_parallel_evals:
        print(
            f"[Eval] skipping iteration {iteration}: "
            f"{active_eval_threads(eval_threads)} evaluation(s) already running"
        )
        writer.add_scalar("Eval/Skipped", 1, iteration)
        writer.flush()
        append_eval_csv(args.eval_metrics_path, {
            "iteration": iteration,
            "status": "skipped_busy",
            "model_ts": ts_path,
            "games_per_color": args.eval_games_per_color,
            "num_simulations": args.num_simulations_eval,
            "stockfish_elo": args.stockfish_elo,
            "stockfish_depth": args.stockfish_depth,
        })
        return

    run_stockfish = args.stockfish_eval_each > 0 and iteration % args.stockfish_eval_each == 0
    cmd = [
        sys.executable, "-u", str(SCRIPT_DIR / "evaluate_worker.py"),
        "--model-ts", ts_path,
        "--games", str(args.eval_games_per_color),
        "--sims", str(args.num_simulations_eval),
        "--alpha", str(args.alpha),
        "--epsilon", "0",
        "--sampling", str(args.sampling_moves),
        "--sf-bin", args.stockfish_path,
        "--sf-depth", str(args.stockfish_depth),
        "--sf-elo", str(args.stockfish_elo),
        "--threads", str(args.eval_threads),
    ]
    if not run_stockfish:
        cmd.append("--skip-stockfish")

    def _job():
        started = time.perf_counter()
        mode = "random+stockfish" if run_stockfish else "random-only"
        print(
            f"[Eval] iteration {iteration}: starting {args.eval_games_per_color} games/color, "
            f"sims={args.num_simulations_eval}, mode={mode}"
        )
        try:
            timeout = args.eval_timeout or None
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(SCRIPT_DIR),
                timeout=timeout,
            )
            report = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as e:
            seconds = time.perf_counter() - started
            error = str(e)
            print(f"[Eval] iteration {iteration} failed after {seconds:.1f}s: {error}")
            writer.add_scalar("Eval/Failed", 1, iteration)
            writer.flush()
            append_eval_csv(args.eval_metrics_path, {
                "iteration": iteration,
                "status": "failed",
                "seconds": seconds,
                "model_ts": ts_path,
                "games_per_color": args.eval_games_per_color,
                "num_simulations": args.num_simulations_eval,
                "stockfish_elo": args.stockfish_elo,
                "stockfish_depth": args.stockfish_depth,
                "error": error,
            })
            return

        seconds = time.perf_counter() - started

        rnd = report["random"]
        white_games = rnd["wW"] + rnd["wL"] + rnd["wD"]
        black_games = rnd["bW"] + rnd["bL"] + rnd["bD"]
        total_random_games = white_games + black_games
        win_rate_random = 0.0 if total_random_games == 0 else (
            rnd["wW"] + rnd["bW"] + 0.5 * (rnd["wD"] + rnd["bD"])
        ) / total_random_games

        sf_list = report.get("stockfish")
        win_rate_sf = None
        agent_elo = None
        if sf_list is not None:
            sf_white_games = sf_list[0] + sf_list[1] + sf_list[2]
            sf_black_games = sf_list[3] + sf_list[4] + sf_list[5]
            total_sf_games = sf_white_games + sf_black_games
            win_rate_sf = 0.0 if total_sf_games == 0 else (
                sf_list[0] + sf_list[3] + 0.5 * (sf_list[2] + sf_list[5])
            ) / total_sf_games
            agent_elo = elo_from_score(win_rate_sf, args.stockfish_elo)

        writer.add_scalar("Eval/WinRate_vs_Random", win_rate_random, iteration)
        if sf_list is not None:
            writer.add_scalar("Eval/WinRate_vs_Stockfish", win_rate_sf, iteration)
            writer.add_scalar("Eval/Elo_vs_Stockfish", agent_elo, iteration)
        writer.add_scalar("Eval/Seconds", seconds, iteration)
        writer.add_scalar("Eval/Failed", 0, iteration)
        writer.flush()

        row = {
            "iteration": iteration,
            "status": "ok" if sf_list is not None else "ok_random_only",
            "seconds": seconds,
            "model_ts": ts_path,
            "games_per_color": args.eval_games_per_color,
            "num_simulations": args.num_simulations_eval,
            "stockfish_elo": args.stockfish_elo,
            "stockfish_depth": args.stockfish_depth,
            "win_rate_random": win_rate_random,
            "random_wW": rnd["wW"],
            "random_wL": rnd["wL"],
            "random_wD": rnd["wD"],
            "random_bW": rnd["bW"],
            "random_bL": rnd["bL"],
            "random_bD": rnd["bD"],
        }
        if sf_list is not None:
            row.update({
                "win_rate_stockfish": win_rate_sf,
                "elo_vs_stockfish": agent_elo,
                "stockfish_wW": sf_list[0],
                "stockfish_wL": sf_list[1],
                "stockfish_wD": sf_list[2],
                "stockfish_bW": sf_list[3],
                "stockfish_bL": sf_list[4],
                "stockfish_bD": sf_list[5],
            })
        append_eval_csv(args.eval_metrics_path, row)

        if sf_list is None:
            print(
                f"[Eval] iteration {iteration}: random_wr={win_rate_random:.3f}, "
                f"stockfish=skipped, seconds={seconds:.1f}"
            )
        else:
            print(
                f"[Eval] iteration {iteration}: random_wr={win_rate_random:.3f}, "
                f"stockfish_wr={win_rate_sf:.3f}, elo={agent_elo:.0f}, seconds={seconds:.1f}"
            )

    thread = threading.Thread(target=_job, daemon=False, name=f"eval-{iteration}")
    thread.start()
    eval_threads.append(thread)


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

def move_agent_to_device(agent, device):
    agent._model.to(device)
    for state in agent.optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


TRAINING_METRIC_KEYS = [
    "loss",
    "policy_loss",
    "value_loss",
    "grad_norm",
    "policy_entropy",
    "target_policy_entropy",
    "value_mean",
    "target_value_mean",
    "target_value_abs_mean",
    "target_value_nonzero_fraction",
]


def current_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


def replay_buffer_stats(path: str):
    stats = {
        "capacity": 0,
        "size": 0,
        "head": 0,
        "fill_ratio": 0.0,
        "file_mb": 0.0,
    }
    if not os.path.exists(path):
        return stats

    stats["file_mb"] = os.path.getsize(path) / (1024 * 1024)
    with open(path, "rb") as f:
        hdr = f.read(HEADER_BYTES)
    if len(hdr) < HEADER_BYTES:
        return stats

    capacity, size, head = struct.unpack("<qqq", hdr)
    stats.update({
        "capacity": int(capacity),
        "size": int(size),
        "head": int(head),
        "fill_ratio": float(size / capacity) if capacity > 0 else 0.0,
    })
    return stats


def summarize_metrics(step_metrics):
    if not step_metrics:
        return {}
    return {
        key: float(np.mean([metrics[key] for metrics in step_metrics if key in metrics]))
        for key in TRAINING_METRIC_KEYS
    }


def append_metrics_csv(path: str, row: dict):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fieldnames = [
        "phase",
        "iteration",
        "update_step",
        "updates",
        "loss",
        "policy_loss",
        "value_loss",
        "grad_norm",
        "policy_entropy",
        "target_policy_entropy",
        "value_mean",
        "target_value_mean",
        "target_value_abs_mean",
        "target_value_nonzero_fraction",
        "learning_rate",
        "num_simulations",
        "opening_random_plies",
        "replay_size",
        "replay_capacity",
        "replay_fill_ratio",
        "replay_file_mb",
        "self_play_seconds",
        "train_seconds",
        "iteration_seconds",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_promotion_csv(path: str, row: dict):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fieldnames = [
        "iteration",
        "status",
        "seconds",
        "candidate_ts",
        "best_ts",
        "games_per_color",
        "num_simulations",
        "opening_random_plies",
        "threshold",
        "score",
        "wins",
        "losses",
        "draws",
        "candidate_wW",
        "candidate_wL",
        "candidate_wD",
        "candidate_bW",
        "candidate_bL",
        "candidate_bD",
        "promoted",
        "error",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def arena_score(counts: tuple[int, int, int, int, int, int]):
    cww, cwl, cwd, cbw, cbl, cbd = counts
    wins = cww + cbw
    losses = cwl + cbl
    draws = cwd + cbd
    total = wins + losses + draws
    score = (wins + 0.5 * draws) / total if total else 0.0
    return score, wins, losses, draws


def log_training_phase(
    *,
    phase: str,
    iteration: int,
    update_step: int,
    step_metrics: list[dict[str, float]],
    replay_stats: dict,
    learning_rate: float,
    writer,
    metrics_path: str,
    num_simulations: int | None = None,
    opening_random_plies: int | None = None,
    self_play_seconds: float | None = None,
    train_seconds: float | None = None,
    iteration_seconds: float | None = None,
):
    summary = summarize_metrics(step_metrics)
    if not summary:
        print(f"[metrics] {phase} iteration={iteration}: no optimizer updates recorded")
        return

    scalar_names = {
        "loss": "Loss",
        "policy_loss": "PolicyLoss",
        "value_loss": "ValueLoss",
        "grad_norm": "GradNorm",
        "policy_entropy": "PolicyEntropy",
        "target_policy_entropy": "TargetPolicyEntropy",
        "value_mean": "ValueMean",
        "target_value_mean": "TargetValueMean",
        "target_value_abs_mean": "TargetValueAbsMean",
        "target_value_nonzero_fraction": "TargetValueNonzeroFraction",
    }
    for key, scalar_name in scalar_names.items():
        writer.add_scalar(f"{phase}/{scalar_name}", summary[key], iteration)
    writer.add_scalar(f"{phase}/LearningRate", learning_rate, iteration)
    writer.add_scalar(f"{phase}/ReplayBufferSize", replay_stats["size"], iteration)
    writer.add_scalar(f"{phase}/ReplayBufferFillRatio", replay_stats["fill_ratio"], iteration)
    if num_simulations is not None:
        writer.add_scalar(f"{phase}/NumSimulations", num_simulations, iteration)
    if opening_random_plies is not None:
        writer.add_scalar(f"{phase}/OpeningRandomPlies", opening_random_plies, iteration)
    if self_play_seconds is not None:
        writer.add_scalar(f"{phase}/SelfPlaySeconds", self_play_seconds, iteration)
    if train_seconds is not None:
        writer.add_scalar(f"{phase}/TrainSeconds", train_seconds, iteration)
    if iteration_seconds is not None:
        writer.add_scalar(f"{phase}/IterationSeconds", iteration_seconds, iteration)
    writer.flush()

    row = {
        "phase": phase,
        "iteration": iteration,
        "update_step": update_step,
        "updates": len(step_metrics),
        "learning_rate": learning_rate,
        "num_simulations": num_simulations,
        "opening_random_plies": opening_random_plies,
        "replay_size": replay_stats["size"],
        "replay_capacity": replay_stats["capacity"],
        "replay_fill_ratio": replay_stats["fill_ratio"],
        "replay_file_mb": replay_stats["file_mb"],
        "self_play_seconds": self_play_seconds,
        "train_seconds": train_seconds,
        "iteration_seconds": iteration_seconds,
        **summary,
    }
    append_metrics_csv(metrics_path, row)

    print(
        f"[metrics] {phase} iter={iteration} updates={len(step_metrics)} "
        f"loss={summary['loss']:.4f} policy={summary['policy_loss']:.4f} "
        f"value={summary['value_loss']:.4f} grad={summary['grad_norm']:.3f} "
        f"target_nonzero={summary['target_value_nonzero_fraction']:.3f} "
        f"lr={learning_rate:.3g} buffer={replay_stats['size']}/{replay_stats['capacity']}"
    )


def resolve_path(path_value: str | None, default: Path | None = None) -> str:
    path = Path(path_value).expanduser() if path_value is not None else default
    if path is None:
        raise ValueError("path is required")
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def prepare_run_paths(args):
    args.run_dir = resolve_path(args.run_dir)
    run_dir = Path(args.run_dir)
    args.log_dir = resolve_path(args.log_dir, run_dir / "tensorboard")
    args.model_path = resolve_path(args.model_path, run_dir / "model.pt")
    args.metrics_path = resolve_path(args.metrics_path, run_dir / "training_metrics.csv")
    args.eval_metrics_path = resolve_path(args.eval_metrics_path, run_dir / "eval_metrics.csv")
    args.promotion_metrics_path = resolve_path(args.promotion_metrics_path, run_dir / "promotion_metrics.csv")
    args.best_model_path = resolve_path(args.best_model_path, run_dir / "best_model.pt")
    args.checkpoint_dir = resolve_path(args.checkpoint_dir, run_dir / "checkpoints")
    args.ts_dir = resolve_path(args.ts_dir, run_dir / "torchscript")
    args.replay_buffer_path = resolve_path(args.replay_buffer_path, run_dir / "games.bin")

    for directory in (
        args.run_dir,
        args.log_dir,
        args.checkpoint_dir,
        args.ts_dir,
        os.path.dirname(args.model_path),
        os.path.dirname(args.metrics_path),
        os.path.dirname(args.eval_metrics_path),
        os.path.dirname(args.promotion_metrics_path),
        os.path.dirname(args.best_model_path),
        os.path.dirname(args.replay_buffer_path),
    ):
        if directory:
            os.makedirs(directory, exist_ok=True)


def export_torchscript(model_path: str, ts_path: str):
    os.makedirs(os.path.dirname(ts_path), exist_ok=True)
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "export_torchscript.py"), model_path, ts_path],
        check=True,
        cwd=str(SCRIPT_DIR),
    )


def copy_model_file(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def infer_resume_counters(metrics_path: str) -> tuple[int, int]:
    last_iteration = 0
    last_update_step = 0
    if not os.path.exists(metrics_path):
        return last_iteration, last_update_step

    try:
        with open(metrics_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    if row.get("phase") == "Train":
                        last_iteration = max(last_iteration, int(row.get("iteration") or 0))
                    last_update_step = max(last_update_step, int(row.get("update_step") or 0))
                except (TypeError, ValueError):
                    continue
    except OSError as exc:
        print(f"[train_cpp] Warning: could not infer resume counters from {metrics_path}: {exc}")
    return last_iteration, last_update_step


def export_best_torchscript(args, iteration: int | None = None) -> str:
    if iteration is None:
        filename = "best_model_ts.pt"
    else:
        filename = f"best_model_ts_{iteration:06d}_accepted.pt"
    ts_path = os.path.join(args.ts_dir, filename)
    export_torchscript(args.best_model_path, ts_path)
    return ts_path


def run_promotion_arena(iteration: int, args, writer, best_ts_path: str) -> tuple[str, bool]:
    candidate_ts = os.path.join(args.ts_dir, f"candidate_ts_{iteration:06d}.pt")
    export_torchscript(args.model_path, candidate_ts)
    if not os.path.exists(best_ts_path):
        best_ts_path = export_best_torchscript(args)

    started = time.perf_counter()
    try:
        counts = chess_engine.evaluate_model_vs_model(
            candidate_ts,
            best_ts_path,
            games_per_color=args.promotion_games_per_color,
            num_threads=args.arena_threads,
            num_simulations=args.arena_num_simulations,
            alpha=args.alpha,
            opening_random_plies=args.arena_opening_random_plies,
        )
    except Exception as exc:
        seconds = time.perf_counter() - started
        append_promotion_csv(args.promotion_metrics_path, {
            "iteration": iteration,
            "status": "error",
            "seconds": seconds,
            "candidate_ts": candidate_ts,
            "best_ts": best_ts_path,
            "games_per_color": args.promotion_games_per_color,
            "num_simulations": args.arena_num_simulations,
            "opening_random_plies": args.arena_opening_random_plies,
            "threshold": args.promotion_threshold,
            "error": repr(exc),
        })
        raise

    seconds = time.perf_counter() - started
    score, wins, losses, draws = arena_score(tuple(int(x) for x in counts))
    promoted = score >= args.promotion_threshold
    cww, cwl, cwd, cbw, cbl, cbd = [int(x) for x in counts]

    append_promotion_csv(args.promotion_metrics_path, {
        "iteration": iteration,
        "status": "ok",
        "seconds": seconds,
        "candidate_ts": candidate_ts,
        "best_ts": best_ts_path,
        "games_per_color": args.promotion_games_per_color,
        "num_simulations": args.arena_num_simulations,
        "opening_random_plies": args.arena_opening_random_plies,
        "threshold": args.promotion_threshold,
        "score": score,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "candidate_wW": cww,
        "candidate_wL": cwl,
        "candidate_wD": cwd,
        "candidate_bW": cbw,
        "candidate_bL": cbl,
        "candidate_bD": cbd,
        "promoted": int(promoted),
    })

    writer.add_scalar("Arena/Score", score, iteration)
    writer.add_scalar("Arena/Wins", wins, iteration)
    writer.add_scalar("Arena/Losses", losses, iteration)
    writer.add_scalar("Arena/Draws", draws, iteration)
    writer.add_scalar("Arena/Promoted", int(promoted), iteration)
    writer.flush()

    print(
        f"[arena] iter={iteration} score={score:.3f} "
        f"W/L/D={wins}/{losses}/{draws} "
        f"threshold={args.promotion_threshold:.3f} promoted={promoted}"
    )

    if promoted:
        copy_model_file(args.model_path, args.best_model_path)
        best_ts_path = export_best_torchscript(args, iteration=iteration)
        print(f"[arena] promoted candidate to best model: {args.best_model_path}")

    return best_ts_path, promoted


def write_run_config(args):
    config_path = Path(args.run_dir) / "config.json"
    config = vars(args).copy()
    config["python"] = sys.version
    config["torch_version"] = torch.__version__
    config["cuda_available"] = torch.cuda.is_available()
    config["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    config["script_dir"] = str(SCRIPT_DIR)
    config["project_root"] = str(PROJECT_ROOT)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return str(config_path)


def main():
    args = parse_args()
    prepare_run_paths(args)
    writer = SummaryWriter(log_dir=args.log_dir)
    metrics_path = args.metrics_path

    cpu_count = os.cpu_count() or 1
    args.threads = max(1, args.threads)
    args.sim_games = max(1, args.sim_games)
    args.selfplay_batch_size = max(1, args.selfplay_batch_size)
    if args.torch_threads <= 0:
        args.torch_threads = min(args.threads, cpu_count)
    else:
        args.torch_threads = max(1, min(args.torch_threads, cpu_count))
    args.bootstrap_threads = min(args.bootstrap_threads, os.cpu_count() or args.bootstrap_threads)
    args.evaluate_each = max(1, args.evaluate_each)
    args.eval_threads = max(1, min(args.eval_threads, os.cpu_count() or args.eval_threads))
    args.max_parallel_evals = max(1, args.max_parallel_evals)
    if args.stockfish_eval_each is None:
        args.stockfish_eval_each = args.evaluate_each
    args.stockfish_eval_each = max(0, args.stockfish_eval_each)
    args.opening_random_plies = max(0, args.opening_random_plies)
    args.residual_channels = max(16, args.residual_channels)
    args.residual_blocks = max(1, args.residual_blocks)
    args.transformer_dim_model = max(32, args.transformer_dim_model)
    args.transformer_layers = max(1, args.transformer_layers)
    args.transformer_heads = max(1, args.transformer_heads)
    args.transformer_ff_multiplier = max(1, args.transformer_ff_multiplier)
    args.promotion_interval = max(0, args.promotion_interval)
    args.promotion_games_per_color = max(1, args.promotion_games_per_color)
    args.promotion_threshold = max(0.0, min(1.0, args.promotion_threshold))
    args.arena_num_simulations = args.num_simulations_eval if args.arena_num_simulations <= 0 else args.arena_num_simulations
    args.arena_threads = max(1, min(args.arena_threads, os.cpu_count() or args.arena_threads))
    if args.arena_opening_random_plies < 0:
        args.arena_opening_random_plies = args.opening_random_plies
    else:
        args.arena_opening_random_plies = max(0, args.arena_opening_random_plies)

    config_path = write_run_config(args)
    print(f"Run directory: {args.run_dir}")
    print(f"TensorBoard logs: {args.log_dir}")
    print(f"Training metrics CSV: {metrics_path}")
    print(f"Evaluation metrics CSV: {args.eval_metrics_path}")
    print(f"Promotion metrics CSV: {args.promotion_metrics_path}")
    print(f"Best model path: {args.best_model_path}")
    print(f"Network: {args.network}")
    print(f"Self-play actors: {args.threads}")
    print(f"Self-play inference batch size: {args.selfplay_batch_size}")
    print(f"PyTorch CPU threads: {args.torch_threads}")
    if args.threads < args.selfplay_batch_size:
        print(
            f"[train_cpp] Warning: --threads ({args.threads}) is below --selfplay_batch_size "
            f"({args.selfplay_batch_size}); inference batches cannot fill."
        )
    if args.sim_games < min(args.threads, args.selfplay_batch_size):
        print(
            f"[train_cpp] Warning: --sim_games ({args.sim_games}) is below available self-play concurrency; "
            f"late-iteration batches will underfill."
        )
    print(f"Run config: {config_path}")
    if not os.path.exists(args.stockfish_path):
        print(f"[train_cpp] Warning: Stockfish binary not found at {args.stockfish_path}; scheduled evals will fail.")

    # set seeds and threading
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_threads)

    # initialize agent (optionally resume)
    if args.resume_model:
        agent = Agent.load(args.resume_model, args)
        print(f"Resumed model from {args.resume_model}");
        if args.model_path != args.resume_model:
            agent.save(args.model_path)
            print(f"Saved resumed weights to {args.model_path} for ongoing training.")
    else:
        agent = Agent(args)
        agent.save(args.model_path)
        print(f"Saved initial random model to {args.model_path}")

    if args.resume_model and os.path.exists(args.best_model_path):
        print(f"Using existing accepted best model at {args.best_model_path}")
    else:
        copy_model_file(args.model_path, args.best_model_path)
        print(f"Initialized accepted best model at {args.best_model_path}")

    best_ts_path = ""

    move_agent_to_device(agent, torch.device("cpu"))
    torch.cuda.empty_cache()

    inferred_iteration, inferred_update_steps = infer_resume_counters(args.metrics_path)
    if args.start_iteration is None:
        args.start_iteration = inferred_iteration if args.resume_model else 0
    else:
        args.start_iteration = max(0, args.start_iteration)

    iteration = args.start_iteration
    total_train_steps = inferred_update_steps if args.resume_model else 0
    if iteration > 0:
        print(
            f"[train_cpp] Resuming after completed iteration {iteration}; "
            f"next iteration is {iteration + 1}."
        )
    eval_threads = []
    training = True

    # ───────────────────────── BOOTSTRAP stage ──────────────────────────
    if args.bootstrap_games > 0:

        # (a) prepare dummy TorchScript once
        dummy_model_path = os.path.join(args.ts_dir, "dummy_model.pt")
        if not os.path.exists(dummy_model_path):
            subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "dummy_model.py"), dummy_model_path],
                check=True,
                cwd=str(SCRIPT_DIR),
            )

        # (b) generate pure-MCTS games
        print(f"[bootstrap] generating {args.bootstrap_games} games "
              f"({args.bootstrap_num_simulations} sims, "
              f"{args.bootstrap_threads} threads, uniform priors)")

        chess_engine.simulate_games_buffered(
            dummy_model_path,
            num_games      = args.bootstrap_games,
            num_threads    = args.bootstrap_threads,
            num_simulations= args.bootstrap_num_simulations,
            alpha          = args.alpha,
            epsilon        = args.epsilon,
            sampling_moves = args.sampling_moves,
            filename       = args.replay_buffer_path,
            replay_buffer_capacity = args.bootstrap_replay_buffer_capacity,
            opening_random_plies = args.opening_random_plies
        )
        # (c) load ALL records, shuffle once, train epoch-style
        print("[bootstrap] loading all records into memory …")
        records = load_all_records(args.replay_buffer_path)
        if not records:
            raise RuntimeError(f"{args.replay_buffer_path} is empty - bootstrap failed.")

        rng = np.random.default_rng(args.seed)
        rng.shuffle(records)
        print(f"[bootstrap] training on {len(records)} positions "
              f"in batches of {args.bootstrap_batch_size}")

        move_agent_to_device(agent, Agent.device)
        agent._model.train()
        bootstrap_metrics = []
        bootstrap_train_started = time.perf_counter()
        for i in range(0, len(records), args.bootstrap_batch_size):
            batch = records[i : i + args.bootstrap_batch_size]
            boards, policies, zs = map(np.array, zip(*batch))
            metrics = agent.train(torch.tensor(boards,   dtype=torch.float32),
                                  torch.tensor(policies, dtype=torch.float32),
                                  torch.tensor(zs,       dtype=torch.float32))
            bootstrap_metrics.append(metrics)
            total_train_steps += 1
        bootstrap_train_seconds = time.perf_counter() - bootstrap_train_started
        log_training_phase(
            phase="Bootstrap",
            iteration=0,
            update_step=total_train_steps,
            step_metrics=bootstrap_metrics,
            replay_stats=replay_buffer_stats(args.replay_buffer_path),
            learning_rate=current_learning_rate(agent.optimizer),
            writer=writer,
            metrics_path=metrics_path,
            num_simulations=args.bootstrap_num_simulations,
            opening_random_plies=args.opening_random_plies,
            train_seconds=bootstrap_train_seconds,
        )

        # (d) clean up – free disk space
        try:
            os.remove(args.replay_buffer_path)
            print(f"[bootstrap] removed {args.replay_buffer_path}")
        except FileNotFoundError:
            pass

        # (d) save & continue with normal loop
        agent.save(args.model_path)
        copy_model_file(args.model_path, args.best_model_path)
        best_ts_path = export_best_torchscript(args)
        print("[bootstrap] done - accepted bootstrap-trained model as best")
        move_agent_to_device(agent, torch.device("cpu"))
        torch.cuda.empty_cache()

    # evaluate the model before training
    if args.eval_initial or (args.resume_model and not args.no_resume_eval):
        print("Evaluating model before training...")
        ts_path = os.path.join(args.ts_dir, "best_model_ts_initial.pt")
        export_torchscript(args.best_model_path, ts_path)
        spawn_async_evaluation(iteration, ts_path, args, writer, eval_threads)

    if args.pretrain:
        # pretraining
        move_agent_to_device(agent, Agent.device)
        # pretraining
        agent._model.train()
        adjust_learning_rate(agent.optimizer, iteration, args)

        batchsize = args.batch_size
        pretrain_metrics = []
        pretrain_started = time.perf_counter()
        for step in range(args.train_for):
            batch = sample_from_file(
                args.replay_buffer_path,
                args.batch_size,
                decisive_fraction=args.decisive_sample_fraction,
                max_decisive_attempts_per_item=args.max_decisive_sample_attempts_per_item,
            )
            if not batch:
                print("No games to train on; skipping training")
                continue
            batchsize = len(batch)
            boards, policies, zs = map(np.array, zip(*batch))
            boards_tensor   = torch.tensor(boards,   dtype=torch.float32)
            policies_tensor = torch.tensor(policies, dtype=torch.float32)
            zs_tensor       = torch.tensor(zs,       dtype=torch.float32)
            metrics = agent.train(boards_tensor, policies_tensor, zs_tensor)
            pretrain_metrics.append(metrics)
            total_train_steps += 1
        pretrain_seconds = time.perf_counter() - pretrain_started
        print(f"Training step completed on batch of {batchsize} entries")
        log_training_phase(
            phase="Pretrain",
            iteration=0,
            update_step=total_train_steps,
            step_metrics=pretrain_metrics,
            replay_stats=replay_buffer_stats(args.replay_buffer_path),
            learning_rate=current_learning_rate(agent.optimizer),
            writer=writer,
            metrics_path=metrics_path,
            train_seconds=pretrain_seconds,
        )

        # save model after each training phase
        agent.save(args.model_path)
        copy_model_file(args.model_path, args.best_model_path)
        best_ts_path = export_best_torchscript(args)
        print(f"Saved pretrain model to {args.model_path} and accepted it as best")
        move_agent_to_device(agent, torch.device("cpu"))
        torch.cuda.empty_cache()

    if not best_ts_path:
        best_ts_path = export_best_torchscript(args)

    while training and iteration < args.max_iterations:
        torch.cuda.empty_cache()
        iteration += 1
        iteration_started = time.perf_counter()
        print(f"--- Iteration {iteration} ---")

        # schedule num_simulations
        scheduled_num_simulations = get_scheduled_num_simulations(iteration, args)
        print(f"Using num_simulations={scheduled_num_simulations}")

        # buffered self-play generation uses the accepted best model. If gating is
        # disabled, fall back to the latest candidate model for legacy behavior.
        if args.promotion_interval > 0:
            ts_path = best_ts_path
            print(f"Using accepted best model for self-play: {args.best_model_path}")
        else:
            ts_path = os.path.join(args.ts_dir, f"model_ts_{iteration:06d}.pt")
            export_torchscript(args.model_path, ts_path)
            print("Promotion gating disabled; using latest candidate for self-play")

        self_play_started = time.perf_counter()
        chess_engine.simulate_games_buffered(
            ts_path,
            num_games=args.sim_games,
            num_threads=args.threads,
            num_simulations=scheduled_num_simulations,
            alpha=args.alpha,
            epsilon=args.epsilon,
            sampling_moves=args.sampling_moves,
            filename=args.replay_buffer_path,
            replay_buffer_capacity=args.replay_buffer_capacity,
            opening_random_plies=args.opening_random_plies,
            inference_batch_size=args.selfplay_batch_size
        )
        self_play_seconds = time.perf_counter() - self_play_started
        print("Generated self-play games into buffer.")
        torch.cuda.empty_cache()

        # training phase (apply pretrain logic here)
        move_agent_to_device(agent, Agent.device)
        agent._model.train()
        adjust_learning_rate(agent.optimizer, iteration, args)

        batchsize = args.batch_size
        train_metrics = []
        train_started = time.perf_counter()
        for step in range(args.train_for):
            batch = sample_from_file(
                args.replay_buffer_path,
                args.batch_size,
                decisive_fraction=args.decisive_sample_fraction,
                max_decisive_attempts_per_item=args.max_decisive_sample_attempts_per_item,
            )
            if not batch:
                print("No games to train on; skipping training")
                continue
            batchsize = len(batch)
            boards, policies, zs = map(np.array, zip(*batch))
            boards_tensor   = torch.tensor(boards,   dtype=torch.float32)
            policies_tensor = torch.tensor(policies, dtype=torch.float32)
            zs_tensor       = torch.tensor(zs,       dtype=torch.float32)
            metrics = agent.train(boards_tensor, policies_tensor, zs_tensor)
            train_metrics.append(metrics)
            total_train_steps += 1
        train_seconds = time.perf_counter() - train_started
        iteration_seconds = time.perf_counter() - iteration_started
        print(f"Training step completed on batch of {batchsize} entries")
        log_training_phase(
            phase="Train",
            iteration=iteration,
            update_step=total_train_steps,
            step_metrics=train_metrics,
            replay_stats=replay_buffer_stats(args.replay_buffer_path),
            learning_rate=current_learning_rate(agent.optimizer),
            writer=writer,
            metrics_path=metrics_path,
            num_simulations=scheduled_num_simulations,
            opening_random_plies=args.opening_random_plies,
            self_play_seconds=self_play_seconds,
            train_seconds=train_seconds,
            iteration_seconds=iteration_seconds,
        )

        # save model after each training phase
        agent.save(args.model_path)
        print(f"Saved candidate model to {args.model_path}")
        torch.cuda.empty_cache()
        move_agent_to_device(agent, torch.device("cpu"))
        torch.cuda.empty_cache()

        # Candidate-vs-best arena. Self-play remains pinned to best unless promoted.
        if args.promotion_interval > 0 and iteration % args.promotion_interval == 0:
            best_ts_path, _ = run_promotion_arena(iteration, args, writer, best_ts_path)
            torch.cuda.empty_cache()

        # periodic evaluation of the accepted best model
        if iteration % args.evaluate_each == 0:
            new_ts = os.path.join(args.ts_dir, f"best_model_ts_{iteration:06d}_eval.pt")
            export_torchscript(args.best_model_path, new_ts)
            spawn_async_evaluation(iteration, new_ts, args, writer, eval_threads)
            torch.cuda.empty_cache()

        # periodic checkpoint
        if iteration % args.checkpoint_interval == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"model_checkpoint_{iteration:06d}.pt")
            best_ckpt = os.path.join(args.checkpoint_dir, f"best_model_checkpoint_{iteration:06d}.pt")
            agent.save(ckpt)
            copy_model_file(args.best_model_path, best_ckpt)
            print(f"Saved candidate checkpoint: {ckpt}")
            print(f"Saved best checkpoint: {best_ckpt}")

        # stopping condition
        if iteration >= args.max_iterations:
            training = False

    cleanup_eval_threads(eval_threads)
    if eval_threads:
        print(f"Waiting for {len(eval_threads)} async evaluation(s) to finish...")
        for thread in eval_threads:
            thread.join()
    writer.close()
    print(f"Training complete; final candidate model saved to {args.model_path}")
    print(f"Accepted best model: {args.best_model_path}")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}")
    print(f"Training metrics: {metrics_path}")
    print(f"Evaluation metrics: {args.eval_metrics_path}")
    print(f"Promotion metrics: {args.promotion_metrics_path}")


if __name__ == "__main__":
    main()
