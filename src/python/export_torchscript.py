#!/usr/bin/env python3
"""
Export a trained PyTorch model to TorchScript for C++ inference.
"""
import argparse
import os

import torch

from chess_agent import Agent, ReplayBuffer, adjust_learning_rate


def infer_precision(state_dict) -> str:
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float16:
                return "fp16"
            if value.dtype == torch.bfloat16:
                return "bf16"
            return "fp32"
    return "fp32"


def main(model_in: str, model_out: str):
    # Determine precision from checkpoint (default to bf16 to align with training pipeline).
    state = None
    precision = "bf16"
    if os.path.isfile(model_in):
        state = torch.load(model_in, map_location="cpu")
        precision = infer_precision(state)

    args = argparse.Namespace(model_path=model_in, learning_rate=0.001, weight_decay=0.001, precision=precision)

    if state is not None:
        print(f"Loading weights from '{model_in}' ({precision})…")
        agent = Agent(args)
        agent._model.load_state_dict(state)
        agent._model = agent._model.to(agent.device)
    else:
        print(f"Warning: '{model_in}' not found; initializing a new random model.")
        agent = Agent(args)
    agent._model.eval()
    scripted = torch.jit.script(agent._model)
    scripted.save(model_out)
    print(f"Saved TorchScript model to {model_out}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: export_torchscript.py <in.pt> <out_ts.pt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
