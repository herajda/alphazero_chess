#!/usr/bin/env python3
import argparse
import glob
import itertools
import logging
import os
from pathlib import Path
import numpy as np
import torch
from chess_game import ChessGame
from chess_agent import Agent, mcts

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Arena:
    def __init__(self, models_dir, games_per_match=10, mcts_sims=50):
        self.models_dir = Path(models_dir)
        self.games_per_match = games_per_match
        self.mcts_sims = mcts_sims
        self.agents = {}
        self.load_models()

    def load_models(self):
        model_files = list(self.models_dir.glob("*.pt"))
        if not model_files:
            logging.warning(f"No models found in {self.models_dir}")
            return

        logging.info(f"Found {len(model_files)} models: {[f.name for f in model_files]}")
        
        # Dummy args for loading if needed, though load() should handle it
        dummy_args = argparse.Namespace(
            seed=42, threads=1, processes=1, alpha=0.3, epsilon=0.25, 
            num_simulations=self.mcts_sims, sampling_moves=0, 
            infer=True, precision="fp32", compile=False
        )

        for model_path in model_files:
            try:
                agent = Agent.load(str(model_path), args=dummy_args)
                # Ensure agent uses the arena's MCTS settings
                agent.args.num_simulations = self.mcts_sims
                self.agents[model_path.name] = agent
                logging.info(f"Loaded {model_path.name}")
            except Exception as e:
                logging.error(f"Failed to load {model_path.name}: {e}")

    def play_game(self, agent_white, agent_black):
        game = ChessGame()
        moves = 0
        while game.winner is None:
            # Determine current player
            current_agent = agent_white if game.to_play == 1 else agent_black
            
            # Get policy from MCTS
            # We need to pass args to mcts. Agent.load attaches args to agent.
            # But mcts function takes explicit args.
            # Let's use the agent's stored args which we updated in load_models
            policy = mcts(game, current_agent, current_agent.args, explore=True) # Explore=True for variety? Or False for strength?
            # Usually for eval we want explore=False (deterministic best move) or very low temp.
            # But mcts implementation with explore=False returns policy based on visit counts directly.
            
            # Mask invalid moves
            mask = np.zeros(game.ACTIONS, dtype=bool)
            mask[game.valid_actions()] = True
            policy[~mask] = 0
            
            if policy.sum() > 0:
                policy /= policy.sum()
                action = np.argmax(policy)
            else:
                logging.warning("No valid moves from policy, choosing random valid move.")
                action = np.random.choice(game.valid_actions())

            game.move(action)
            moves += 1
            
            if moves > 200: # Draw by length
                return 0

        return game.winner # 1 for White, -1 for Black, 0 for Draw

    def run_tournament(self):
        model_names = list(self.agents.keys())
        n_models = len(model_names)
        scores = {name: 0.0 for name in model_names}
        results = {name: {"wins": 0, "losses": 0, "draws": 0} for name in model_names}

        # Round Robin
        for i in range(n_models):
            for j in range(i + 1, n_models):
                name_a = model_names[i]
                name_b = model_names[j]
                agent_a = self.agents[name_a]
                agent_b = self.agents[name_b]

                logging.info(f"Match: {name_a} vs {name_b}")
                
                # Play games
                for g in range(self.games_per_match):
                    # Swap colors every game
                    if g % 2 == 0:
                        white, black = agent_a, agent_b
                        p1, p2 = name_a, name_b
                        p1_color = 1
                    else:
                        white, black = agent_b, agent_a
                        p1, p2 = name_b, name_a
                        p1_color = -1 # p1 is black

                    winner = self.play_game(white, black)
                    
                    if winner == 0:
                        scores[name_a] += 0.5
                        scores[name_b] += 0.5
                        results[name_a]["draws"] += 1
                        results[name_b]["draws"] += 1
                    elif winner == 1: # White wins
                        if p1_color == 1: # p1 was white
                            scores[p1] += 1
                            results[p1]["wins"] += 1
                            results[p2]["losses"] += 1
                        else: # p2 was white
                            scores[p2] += 1
                            results[p2]["wins"] += 1
                            results[p1]["losses"] += 1
                    else: # Black wins (winner == -1)
                        if p1_color == -1: # p1 was black
                            scores[p1] += 1
                            results[p1]["wins"] += 1
                            results[p2]["losses"] += 1
                        else: # p2 was black
                            scores[p2] += 1
                            results[p2]["wins"] += 1
                            results[p1]["losses"] += 1

        print("\n--- Tournament Results ---")
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"{'Model':<40} | {'Score':<6} | {'W':<4} | {'L':<4} | {'D':<4}")
        print("-" * 70)
        for name, score in sorted_scores:
            r = results[name]
            print(f"{name:<40} | {score:<6} | {r['wins']:<4} | {r['losses']:<4} | {r['draws']:<4}")

        if sorted_scores:
            best_model = sorted_scores[0][0]
            print(f"\nBest Model: {best_model}")

def main():
    parser = argparse.ArgumentParser(description="Chess Agent Arena")
    parser.add_argument("--models-dir", type=str, default="experiments", help="Directory containing .pt models")
    parser.add_argument("--games", type=int, default=2, help="Games per match (even number recommended)")
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulations per move")
    args = parser.parse_args()

    arena = Arena(args.models_dir, games_per_match=args.games, mcts_sims=args.sims)
    if not arena.agents:
        print("No agents loaded. Run grid_search.py first.")
        return
    arena.run_tournament()

if __name__ == "__main__":
    main()
