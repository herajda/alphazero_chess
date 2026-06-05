# file: src/python/evaluate_worker.py
#!/usr/bin/env python3
"""
Run both Random-baseline and Stockfish evaluations in *this* process and
return a one-line JSON report on stdout.

Designed to be launched from the training script via subprocess.
"""
from __future__ import annotations
import os
import argparse, json, random, sys, pathlib, torch, chess, chess_moves
import chess_engine
import chess.engine as uci


# ----------------------------------------------------------------------
# helpers ------------------------------------------------------------------
# evaluate_worker.py  (only this function changed)

# evaluate_worker.py
from chess_engine import evaluate_vs_random, evaluate_vs_stockfish

def eval_vs_random(ts_path, games, sims, alpha, eps, sampling, threads):
    wW,wL,wD,bW,bL,bD = evaluate_vs_random(
        ts_path,
        games_per_color = games,
        num_threads     = threads,     # e.g. 16 or 32
        num_simulations = sims,
        alpha           = alpha,
        epsilon         = eps,
        sampling_moves  = sampling
    )
    return dict(wW=wW, wL=wL, wD=wD, bW=bW, bL=bL, bD=bD)


def eval_vs_stockfish(ts, games, depth, alpha, epsilon, sampling, elo, bin_path):
    eng = uci.SimpleEngine.popen_uci(bin_path)
    eng.configure({"Threads": 8, "UCI_LimitStrength": True, "UCI_Elo": elo})
    W=L=D=0
    for us in (chess.WHITE, chess.BLACK):
        for _ in range(games):
            board = chess.Board()
            while not board.is_game_over(claim_draw=True):
                if board.turn == us:                                   # our model
                    a = chess_engine.select_move(ts, board.fen(),
                                                  400, alpha, 0, sampling)
                    u = chess_moves.action_to_uci(board, a)
                    board.push_uci(u)
                else:                                                  # Stockfish
                    board.push(eng.play(board, uci.Limit(depth=depth)).move)
            res = board.result()
            win = {"1-0": chess.WHITE, "0-1": chess.BLACK}.get(res, None)
            if   win is None: D += 1
            elif win == us:  W += 1
            else:            L += 1
    eng.quit()
    return dict(W=W, L=L, D=D)


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ts",   required=True)
    p.add_argument("--games",      type=int, default=100)
    p.add_argument("--sims",       type=int, default=300)
    p.add_argument("--alpha",      type=float, default=0.3)
    p.add_argument("--epsilon",    type=float, default=0.0)
    p.add_argument("--sampling",   type=int,   default=3)
    p.add_argument("--seed",       type=int,   default=None)
    p.add_argument("--sf-bin",     type=str,   default="/usr/games/stockfish")
    p.add_argument("--sf-depth",   type=int,   default=12)
    p.add_argument("--sf-elo",     type=int,   default=1320)
    p.add_argument("--threads",     type=int,   default=8)
    p.add_argument("--skip-stockfish", action="store_true",
                   help="Only run the random baseline and return stockfish=null.")
    args = p.parse_args()

    rnd = eval_vs_random(args.model_ts, args.games, args.sims,
                         args.alpha, args.epsilon, args.sampling, args.threads)

    sf = None
    if not args.skip_stockfish:
        sf = evaluate_vs_stockfish(
            args.model_ts,           # TorchScript
            args.sf_bin,
            args.games,        # games / colour
            args.threads,
            args.sf_depth,
            args.sf_elo,
            args.sims,         # our sims / move
            args.alpha, args.epsilon, args.sampling)
    # single-line JSON makes parsing trivial
    print(json.dumps({"random": rnd, "stockfish": sf}), flush=True)


if __name__ == "__main__":
    main()
