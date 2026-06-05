#!/usr/bin/env python3
"""Overfit a fixed replay batch to sanity-check policy/value learnability."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chess_agent import Agent
from train_cpp import sample_from_file


def parse_args():
    parser = argparse.ArgumentParser(description="Overfit one fixed AlphaZero replay batch.")
    parser.add_argument("--replay_buffer", required=True, help="Path to games.bin replay buffer.")
    parser.add_argument("--model_path", default=None, help="Optional checkpoint to start from.")
    parser.add_argument("--save_model", default=None, help="Optional path to save the overfit checkpoint.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--report_each", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.0015)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--decisive_sample_fraction", type=float, default=0.5)
    parser.add_argument("--max_decisive_sample_attempts_per_item", type=int, default=50)
    parser.add_argument("--network", choices=["resnet", "transformer"], default="resnet")
    parser.add_argument("--residual_channels", type=int, default=192)
    parser.add_argument("--residual_blocks", type=int, default=12)
    parser.add_argument("--transformer_dim_model", type=int, default=512)
    parser.add_argument("--transformer_layers", type=int, default=6)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_ff_multiplier", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    batch = sample_from_file(
        args.replay_buffer,
        args.batch_size,
        decisive_fraction=args.decisive_sample_fraction,
        max_decisive_attempts_per_item=args.max_decisive_sample_attempts_per_item,
    )
    if not batch:
        raise RuntimeError(f"No replay records found in {args.replay_buffer}")

    boards, policies, values = map(np.array, zip(*batch))
    boards_t = torch.tensor(boards, dtype=torch.float32)
    policies_t = torch.tensor(policies, dtype=torch.float32)
    values_t = torch.tensor(values, dtype=torch.float32)

    if args.model_path:
        agent = Agent.load(args.model_path, args)
    else:
        agent = Agent(args)

    first = None
    last = None
    for step in range(1, args.steps + 1):
        metrics = agent.train(boards_t, policies_t, values_t)
        if first is None:
            first = metrics
        last = metrics
        if step == 1 or step % args.report_each == 0 or step == args.steps:
            print(
                f"step={step} loss={metrics['loss']:.4f} "
                f"policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f} "
                f"entropy={metrics['policy_entropy']:.4f} "
                f"target_entropy={metrics['target_policy_entropy']:.4f} "
                f"nonzero={metrics['target_value_nonzero_fraction']:.3f}",
                flush=True,
            )

    if args.save_model:
        agent.save(args.save_model)
        print(f"saved={args.save_model}")

    print(
        "summary "
        f"policy_loss_start={first['policy_loss']:.4f} "
        f"policy_loss_end={last['policy_loss']:.4f} "
        f"entropy_start={first['policy_entropy']:.4f} "
        f"entropy_end={last['policy_entropy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
