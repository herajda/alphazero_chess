#!/usr/bin/env python3
"""
Play a human vs the supervised-trained chess transformer.

Loads the saved supervised weights and lets a human play interactively in the
terminal (optionally with the Tk GUI that `ChessGame` exposes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import chess_agent
from chess_agent import Agent
from chess_game import ChessGame
import chess_moves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play against the supervised-trained chess agent."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("supervised_model.pt"),
        help="Path to the trained supervised model weights.",
    )
    parser.add_argument(
        "--human-color",
        choices=("white", "black"),
        default="white",
        help="Choose which side the human controls.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the agent policy (0 for greedy play).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed controlling agent move sampling when temperature > 0.",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=0,
        help="Optional move cap to avoid infinite games (0 disables the cap).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable Tkinter board GUI alongside the terminal view.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
        help="Computation precision to load the model with (fp16/bf16 require compatible CUDA).",
    )
    return parser.parse_args()


def load_agent(model_path: Path, precision: str) -> tuple[Agent, argparse.Namespace]:
    """Instantiate an Agent in inference mode and load weights from disk."""
    agent_args = chess_agent.parser.parse_args([])
    agent_args.infer = True
    agent_args.model_path = str(model_path)
    agent_args.num_simulations = 0  # rely on direct policy predictions
    agent_args.precision = precision
    agent = Agent.load(str(model_path), agent_args)
    return agent, agent_args


def select_agent_action(
    agent: Agent,
    game: ChessGame,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """Sample or pick the agent move from the policy head."""
    # Agent expects a tensor with shape [1, 8, 8, 119]
    if agent.dtype == torch.float16:
        np_dtype = np.float16
    elif agent.dtype == torch.bfloat16 and hasattr(np, "bfloat16"):
        np_dtype = np.bfloat16  # type: ignore[attr-defined]
    else:
        np_dtype = np.float32
    board_tensor = agent.board(game).astype(np_dtype, copy=False)[np.newaxis]
    policy, _ = agent.predict(board_tensor)
    policy = policy[0]

    valid_actions = [action for action in game.valid_actions() if action is not None]
    if not valid_actions:
        raise RuntimeError("No valid moves available for the agent.")

    probs = policy[valid_actions]
    probs = np.clip(probs, 1e-12, None)

    if temperature <= 0:
        best_index = int(np.argmax(probs))
        return int(valid_actions[best_index])

    scaled = np.power(probs, 1.0 / max(temperature, 1e-6))
    total = scaled.sum()
    if total <= 0:
        return int(rng.choice(valid_actions))
    scaled /= total
    choice = rng.choice(len(valid_actions), p=scaled)
    return int(valid_actions[int(choice)])


def print_board(game: ChessGame) -> None:
    board = game.get_board_state()
    print(board)
    print()


def prompt_human_move(game: ChessGame) -> bool:
    """Prompt the human for a legal move. Returns False if the player quits."""
    while True:
        try:
            user_input = input("Your move (UCI, or 'quit'): ").strip().lower()
        except EOFError:
            print()
            return False

        if user_input in ("quit", "exit", "resign"):
            return False
        if user_input == "":
            continue

        try:
            current_board = game.get_board_state()
            action = chess_moves.uci_to_action(current_board, user_input)
        except ValueError:
            action = None

        if action is None:
            print("Illegal or unsupported move, please try again.")
            continue

        try:
            game.move(action)
        except ValueError as exc:
            print(f"{exc}")
            continue
        return True


def describe_result(result: int | None) -> str:
    if result == 1:
        return "White wins"
    if result == 0:
        return "Black wins"
    if result == -1:
        return "Draw"
    return "Game ongoing"


def main() -> int:
    args = parse_args()

    model_path = args.model_path.expanduser()
    if not model_path.exists():
        print(f"Model path '{model_path}' does not exist.", file=sys.stderr)
        return 1

    agent, _ = load_agent(model_path, args.precision)
    rng = np.random.default_rng(args.seed)

    game = ChessGame(gui_enabled=args.gui)
    human_is_white = args.human_color == "white"

    print("Supervised chess play started. Enter moves in UCI format (e.g., e2e4).")
    print_board(game)

    move_count = 0
    while game.winner is None:
        human_turn = (game.to_play == 1 and human_is_white) or (
            game.to_play == 0 and not human_is_white
        )

        if human_turn:
            if not prompt_human_move(game):
                print("Human resigned.")
                return 0
        else:
            board_before = game.get_board_state()
            action = select_agent_action(agent, game, args.temperature, rng)
            uci = chess_moves.action_to_uci(board_before, action)
            game.move(action)
            if uci is None:
                uci = "<unknown>"
            print(f"Agent plays {uci}")

        move_count += 1
        print_board(game)

        if args.max_moves > 0 and move_count >= args.max_moves:
            print("Reached move cap, declaring draw.")
            return 0

    print("Game over:", describe_result(game.winner))
    return 0


if __name__ == "__main__":
    sys.exit(main())
