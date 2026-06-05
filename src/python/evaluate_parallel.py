#!/usr/bin/env python3
"""
Batch‑evaluate sequential *model_ts_*.pt* checkpoints and log results to
TensorBoard **with fully parallel game generation for _both_ baselines**
(Random _and_ Stockfish).

Changes vs the previous draft
-----------------------------
* **Stockfish evaluation now parallel** – a configurable pool of worker
  processes (each with its own Stockfish instance) split the workload across
  CPU cores.  This avoids the earlier serial bottleneck while still keeping
  evaluation of _different_ checkpoints sequential.
* Metrics and CLI remain identical apart from two new flags:
  * `--stockfish-processes` – # worker processes (defaults to half the CPU
    cores, min 1).
  * `--stockfish-threads`   – threads _inside_ each Stockfish instance.

Example
-------
```bash
python3 evaluate_all_models.py \
        --models-dir checkpoints \
        --games-per-color 50 --num-threads 32 \
        --stockfish-depth 12 --stockfish-processes 8 --stockfish-threads 1
```
"""
from __future__ import annotations

import argparse, glob, math, os, random, re, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import torch
import chess
import chess.engine as uci
import chess_moves  # local helper for <action> → UCI
import chess_engine  # C++ extension
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
#                               helpers
# ---------------------------------------------------------------------------

def _natural_key(p: str) -> int:
    m = re.search(r"(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else -1


# ---------------------------  RANDOM baseline  -----------------------------

def evaluate_vs_random(ts: str, *, games: int, threads: int, sims: int,
                        alpha: float, epsilon: float, sampling: int) -> Tuple[float, Dict]:
    wW, wL, wD, bW, bL, bD = chess_engine.evaluate_vs_random(
        ts, games, threads, sims, alpha, epsilon, sampling
    )
    total = 2 * games
    win_rate = (wW + bW + 0.5 * (wD + bD)) / total
    return win_rate, dict(wW=wW, wL=wL, wD=wD, bW=bW, bL=bL, bD=bD)


# ---------------------------  STOCKFISH baseline  --------------------------

def _sf_worker(task):
    """Play *n_games* with our model on one colour vs Stockfish and return counts."""
    (ts, n_games, sims, depth, sf_bin, elo, alpha, epsilon, sampling,
     sf_threads, colour, seed) = task

    rng = random.Random(seed)

    engine = uci.SimpleEngine.popen_uci(sf_bin)
    engine.configure({
        "Threads": sf_threads,
        "UCI_LimitStrength": True,
        "UCI_Elo": elo,
    })

    W = L = D = 0
    for _ in range(n_games):
        board = chess.Board()
        while not board.is_game_over(claim_draw=True):
            if board.turn == colour:
                a   = chess_engine.select_move(ts, board.fen(), sims,
                                               alpha, epsilon, sampling)
                uci_move = chess_moves.action_to_uci(board, a)
                if uci_move is None:
                    raise RuntimeError(f"Invalid action {a} from C++ engine")
                board.push_uci(uci_move)
            else:
                board.push(engine.play(board, uci.Limit(depth=depth)).move)

        res = board.result()
        winner = {"1-0": chess.WHITE, "0-1": chess.BLACK}.get(res)  # None → draw
        if winner is None:
            D += 1
        elif winner == colour:
            W += 1
        else:
            L += 1

    engine.quit()
    return dict(W=W, L=L, D=D, colour=colour)


def evaluate_vs_stockfish_parallel(ts: str, *, games: int, sims: int, depth: int,
                                   sf_bin: str, elo: int, alpha: float, epsilon: float,
                                   sampling: int, processes: int, sf_threads: int) -> Tuple[float, float, Dict]:
    """Run Stockfish evaluation with *processes* workers in parallel."""
    tasks = []
    # Evenly split games for each colour across workers
    for colour in (chess.WHITE, chess.BLACK):
        base = games // processes
        extra = games % processes
        for i in range(processes):
            n = base + (1 if i < extra else 0)
            if n == 0:
                continue
            seed = random.randint(0, 2**32 - 1)
            tasks.append((ts, n, sims, depth, sf_bin, elo, alpha, epsilon,
                          sampling, sf_threads, colour, seed))

    agg = {"W": 0, "L": 0, "D": 0}
    with ProcessPoolExecutor(max_workers=processes) as ex:
        futs = [ex.submit(_sf_worker, t) for t in tasks]
        for fut in as_completed(futs):
            res = fut.result()
            agg["W"] += res["W"]
            agg["L"] += res["L"]
            agg["D"] += res["D"]

    total = 2 * games
    win_rate = (agg["W"] + 0.5 * agg["D"]) / total

    # Elo estimate vs limited‑strength Stockfish
    eps = 1e-4
    p   = max(eps, min(1 - eps, win_rate))
    elo_est = elo - 400 * math.log10(1 / p - 1)

    return win_rate, elo_est, agg


# ---------------------------------------------------------------------------
#                               main driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("AlphaZero checkpoint evaluator")
    ap.add_argument("--models-dir",   type=str, default=".")
    ap.add_argument("--pattern",      type=str, default="model_ts_*.pt")
    ap.add_argument("--log-dir",      type=str, default="runs/eval")

    # Random baseline parameters
    ap.add_argument("--games-per-color", type=int, default=50)
    ap.add_argument("--num-threads",     type=int, default=os.cpu_count() or 1)
    ap.add_argument("--num-simulations", type=int, default=100)
    ap.add_argument("--alpha",           type=float, default=0.3)
    ap.add_argument("--epsilon",         type=float, default=0.0)
    ap.add_argument("--sampling-moves",  type=int,   default=3)

    # Stockfish parameters
    ap.add_argument("--stockfish-bin",   type=str,   default="/usr/games/stockfish")
    ap.add_argument("--stockfish-depth", type=int,   default=12)
    ap.add_argument("--stockfish-elo",   type=int,   default=1320)
    ap.add_argument("--stockfish-processes", type=int, default=max(1, (os.cpu_count() or 2)//2),
                    help="Parallel processes for Stockfish games")
    ap.add_argument("--stockfish-threads",   type=int, default=1,
                    help="Threads *inside* each Stockfish instance")

    args = ap.parse_args()

    writer = SummaryWriter(args.log_dir)

    models = sorted(
        glob.glob(str(Path(args.models_dir) / args.pattern)),
        key=_natural_key
    )
    if not models:
        print("[eval] No checkpoints found – aborting.")
        return

    print(f"[eval] {len(models)} checkpoints detected. Starting evaluation…\n")

    for step, ts in enumerate(models, 1):
        base = os.path.basename(ts)
        print(f"=== {base} ({step}/{len(models)}) ===")

        # ------------------ Random baseline ------------------
        wr_rnd, rnd_counts = evaluate_vs_random(
            ts,
            games=args.games_per_color,
            threads=args.num_threads,
            sims=args.num_simulations,
            alpha=args.alpha,
            epsilon=args.epsilon,
            sampling=args.sampling_moves,
        )
        writer.add_scalar("Random/WinRate", wr_rnd, step)
        for k, v in rnd_counts.items():
            writer.add_scalar(f"Random/{k}", v, step)

        # ------------------ Stockfish baseline ----------------
        wr_sf, elo, sf_counts = evaluate_vs_stockfish_parallel(
            ts,
            games=args.games_per_color,
            sims=args.num_simulations,
            depth=args.stockfish_depth,
            sf_bin=args.stockfish_bin,
            elo=args.stockfish_elo,
            alpha=args.alpha,
            epsilon=args.epsilon,
            sampling=args.sampling_moves,
            processes=args.stockfish_processes,
            sf_threads=args.stockfish_threads,
        )
        writer.add_scalar("Stockfish/WinRate",      wr_sf, step)
        writer.add_scalar("Stockfish/EloEstimate",  elo,   step)
        for k, v in sf_counts.items():
            writer.add_scalar(f"Stockfish/{k}", v, step)

        writer.flush()
        print(f"  → Random WR: {wr_rnd:.3f} | Stockfish WR: {wr_sf:.3f} | Elo: {elo:.0f}\n")

    writer.close()
    print("All checkpoints evaluated. TensorBoard logs at:", args.log_dir)


if __name__ == "__main__":
    main()
