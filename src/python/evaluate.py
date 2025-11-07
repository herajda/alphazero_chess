#!/usr/bin/env python3
"""
Evaluate the trained AlphaZero agent against a uniformly random player.
Plays a specified number of games with the agent as White and as Black.
"""
import argparse
import random
import chess_agent
from chess_agent import Agent, Player
from chess_game import ChessGame


def evaluate(player: Player, num_games_per_color: int = 10, seed: int = None):
    rng = random.Random(seed)
    results = {
        'White': {'win': 0, 'loss': 0, 'draw': 0},
        'Black': {'win': 0, 'loss': 0, 'draw': 0},
    }

    for color_name, color in [('White', 1), ('Black', 0)]:
        print(f"Evaluating {num_games_per_color} games as {color_name}")
        for i in range(1, num_games_per_color + 1):
            game = ChessGame(gui_enabled=False)
            while game.winner is None:
                if game.to_play == color:
                    # Agent plays
                    action = player.play(game)
                else:
                    # Random player
                    actions = game.valid_actions()
                    action = rng.choice(actions)
                game.move(action)

            outcome = game.winner  # 1 for White win, 0 for Black win, -1 for draw
            if outcome == -1:
                results[color_name]['draw'] += 1
                result_str = 'draw'
            elif outcome == color:
                results[color_name]['win'] += 1
                result_str = 'win'
            else:
                results[color_name]['loss'] += 1
                result_str = 'loss'

            print(f"  Game {i}/{num_games_per_color} as {color_name}: {result_str}")
        print()

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate AlphaZero agent vs a random baseline'
    )
    parser.add_argument(
        '--model-path', type=str, default='model.pt',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--games-per-color', type=int, default=10,
        help='Number of games to play as each color'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for the random player'
    )
    parser.add_argument(
        '--precision',
        choices=('fp32', 'fp16', 'bf16'),
        default='bf16',
        help='Computation precision for loading the model (fp16/bf16 require compatible CUDA).'
    )
    args = parser.parse_args()

    # Prepare agent arguments for inference
    agent_args = chess_agent.parser.parse_args([])
    agent_args.infer = True
    agent_args.model_path = args.model_path
    agent_args.precision = args.precision

    # Load the agent and wrap in a Player
    agent = Agent.load(args.model_path, agent_args)
    player = Player(agent, agent_args)

    # Run evaluation
    summary = evaluate(player, num_games_per_color=args.games_per_color, seed=args.seed)

    # Print summary results
    print("\nEvaluation Summary:")
    for color_name in ['White', 'Black']:
        stats = summary[color_name]
        print(
            f"As {color_name}: {stats['win']} wins, {stats['loss']} losses, {stats['draw']} draws"
        )
