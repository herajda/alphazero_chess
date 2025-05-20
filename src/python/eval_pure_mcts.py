# scripts/eval_pure_mcts.py
import argparse
import chess_engine

def main():
    p = argparse.ArgumentParser(description="Pure-MCTS vs Random (no NN)")
    p.add_argument("--model-path", type=str, default="dummy_model.pt",
                   help="TorchScript model that returns uniform/zero")
    p.add_argument("--games-per-color", type=int, default=20)
    p.add_argument("--num-threads",    type=int, default=1)
    p.add_argument("--num-simulations",type=int, default=200)
    p.add_argument("--alpha",          type=float, default=0.3)
    p.add_argument("--epsilon",        type=float, default=0.25)
    p.add_argument("--sampling-moves", type=int, default=3)
    args = p.parse_args()

    w_win, w_loss, w_draw, b_win, b_loss, b_draw = chess_engine.evaluate_vs_random(
        args.model_path,
        args.games_per_color,
        args.num_threads,
        args.num_simulations,
        args.alpha,
        args.epsilon,
        args.sampling_moves
    )

    print("\n=== Pure-MCTS (uniform priors, leaf-only true values) vs Random ===")
    print(f"As White: {w_win} wins / {w_loss} losses / {w_draw} draws")
    print(f"As Black: {b_win} wins / {b_loss} losses / {b_draw} draws")

if __name__ == "__main__":
    main()
