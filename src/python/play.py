#!/usr/bin/env python3
"""
Interactive script to play against your AlphaZero model via the C++ backend.
"""

import argparse
import chess
import chess_engine
import chess_moves


def main():
    parser = argparse.ArgumentParser(
        description="Play interactively against the AlphaZero C++ engine"
    )
    parser.add_argument(
        "--model-path", default="model_ts.pt",
        help="Path to your TorchScript model (used by the C++ BatchManager)"
    )
    parser.add_argument(
        "--num-simulations", type=int, default=100,
        help="Number of MCTS simulations per move"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.3,
        help="Dirichlet‐alpha for root noise (only matters if epsilon > 0)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.0,
        help="Exploration epsilon at root (0 for greedy play)"
    )
    parser.add_argument(
        "--sampling-moves", type=int, default=0,
        help="Number of opening moves to sample before switching to greedy"
    )
    parser.add_argument(
        "--human-color", choices=["white", "black"], default="white",
        help="Which side you play"
    )
    args = parser.parse_args()

    # Who moves on board.turn == True
    human_is_white = (args.human_color == "white")

    board = chess.Board()
    print(board, "\n")

    while not board.is_game_over():
        if board.turn == human_is_white:
            # **** HUMAN MOVE ****
            move_uci = input("Your move (UCI): ").strip()
            try:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    print("Illegal move, try again.")
                    continue
                board.push(move)
            except Exception:
                print("Invalid UCI. Format like e2e4 or g1f3.")
                continue
        else:
            # **** AI MOVE via C++ select_move ****
            fen = board.fen()
            action = chess_engine.select_move(
                args.model_path,
                fen,
                args.num_simulations,
                args.alpha,
                args.epsilon,
                args.sampling_moves
            )
            uci = chess_moves.action_to_uci(board, action)
            if uci is None:
                raise RuntimeError(f"C++ engine returned invalid action {action}")
            print(f"AI plays {uci}")
            board.push_uci(uci)

        # show updated board
        print(board, "\n")

    print("Game over:", board.result())


if __name__ == "__main__":
    main()
