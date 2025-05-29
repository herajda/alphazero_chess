# file: src/python/evaluate_worker.py
#!/usr/bin/env python3
"""
Run both Random-baseline and Stockfish evaluations in *this* process and
return a one-line JSON report on stdout.

Designed to be launched from the training script via subprocess.
"""
from __future__ import annotations
import argparse, json, random, sys, pathlib, chess, chess_engine, chess_moves
import chess.engine as uci


# ----------------------------------------------------------------------
# helpers ------------------------------------------------------------------
# evaluate_worker.py  (only this function changed)

def eval_vs_random(ts, games, sims, alpha, epsilon, sampling, seed):
    rng = random.Random(seed)
    wW=wL=wD=bW=bL=bD = 0

    for us in (chess.WHITE, chess.BLACK):
        for _ in range(games):
            board = chess.Board()
            while not board.is_game_over(claim_draw=True):
                if board.turn == us:  # our model
                    a   = chess_engine.select_move(ts, board.fen(),
                                                   sims, alpha, epsilon, sampling)
                    uci = chess_moves.action_to_uci(board, a)
                    if uci is None:
                        raise RuntimeError(f"Bad action {a}")
                    board.push_uci(uci)
                else:                 # random baseline
                    board.push(rng.choice(list(board.legal_moves)))

            res = board.result()
            winner = {"1-0": chess.WHITE, "0-1": chess.BLACK}.get(res)

            if winner is None:           # draw
                if us == chess.WHITE: wD += 1
                else:                  bD += 1
            elif winner == us:           # we won
                if us == chess.WHITE: wW += 1
                else:                  bW += 1
            else:                       # we lost
                if us == chess.WHITE: wL += 1
                else:                  bL += 1

    return dict(wW=wW, wL=wL, wD=wD, bW=bW, bL=bL, bD=bD)


def eval_vs_stockfish(ts, games, depth, alpha, epsilon, sampling, elo, bin_path):
    eng = uci.SimpleEngine.popen_uci(bin_path)
    eng.configure({"Threads": 4, "UCI_LimitStrength": True, "UCI_Elo": elo})
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
    args = p.parse_args()

    rnd = eval_vs_random(args.model_ts, args.games, args.sims,
                         args.alpha, args.epsilon, args.sampling, args.seed)

    sf  = eval_vs_stockfish(args.model_ts, args.games,
                            args.sf_depth, args.alpha, args.epsilon,
                            args.sampling, args.sf_elo, args.sf_bin)

    # single-line JSON makes parsing trivial
    print(json.dumps({"random": rnd, "stockfish": sf}), flush=True)


if __name__ == "__main__":
    main()
