#!/usr/bin/env python3
import argparse
import itertools
import subprocess
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Grid search for chess agent hyperparameters.")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"), help="Directory to save experiments.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    args.experiments_dir.mkdir(parents=True, exist_ok=True)

    # Define the grid
    grid = {
        "model_type": ["cnn", "resnet", "transformer"],
        "learning_rate": [1e-3, 1e-4],
    }
    
    # Specific params for each model type can be handled by conditionally adding them
    # For simplicity, we will just iterate over the main grid and set defaults for others, 
    # or we can make the grid more complex.
    
    # Let's do a list of explicit configurations to try
    configurations = [
        # CNN Baselines
        {"model_type": "cnn", "num_filters": 128, "num_layers": 5, "learning_rate": 1e-3},
        {"model_type": "cnn", "num_filters": 256, "num_layers": 10, "learning_rate": 1e-4},
        
        # ResNet Baselines
        {"model_type": "resnet", "num_filters": 128, "num_layers": 5, "learning_rate": 1e-3},
        {"model_type": "resnet", "num_filters": 256, "num_layers": 10, "learning_rate": 1e-4},
        
        # Transformer Baselines
        {"model_type": "transformer", "dim_model": 256, "num_layers": 4, "num_heads": 4, "learning_rate": 1e-3},
        {"model_type": "transformer", "dim_model": 512, "num_layers": 6, "num_heads": 8, "learning_rate": 1e-4},
    ]

    base_cmd = [
        "python3", "src/python/train_supervised.py",
        "--epochs", "1", # Short training for demo/grid search speed
        "--max-games", "50000", # Smaller dataset for speed
        "--batch-size", "512",
    ]

    for i, config in enumerate(configurations):
        exp_name = f"exp_{i}_{config['model_type']}"
        model_path = args.experiments_dir / f"{exp_name}.pt"
        log_dir = args.experiments_dir / "logs" / exp_name
        
        print(f"--- Running Experiment {i+1}/{len(configurations)}: {exp_name} ---")
        
        cmd = base_cmd + [
            "--model-path", str(model_path),
            "--logdir", str(log_dir),
        ]
        
        for key, value in config.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
            
        print("Command:", " ".join(cmd))
        
        if not args.dry_run:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Experiment {exp_name} failed: {e}")

if __name__ == "__main__":
    main()
