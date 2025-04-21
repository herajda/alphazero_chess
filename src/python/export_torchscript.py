#!/usr/bin/env python3
"""
Export a trained PyTorch model to TorchScript for C++ inference.
"""
import torch
from chess_agent import Agent, ReplayBuffer, adjust_learning_rate
import argparse
import os

def main(model_in: str, model_out: str):
    # Load your Python Agent (with same args used for training)
    args = argparse.Namespace(model_path=model_in, learning_rate=0.001, weight_decay=0.001)
    # if the input file exists, load it; otherwise start from scratch
    if os.path.isfile(model_in):
        print(f"Loading weights from '{model_in}'…")
        agent = Agent.load(model_in, args)
    else:
        print(f"Warning: '{model_in}' not found; initializing a new random model.")
        agent = Agent(args)
    agent._model.eval()
    # Script & save
    scripted = torch.jit.script(agent._model)
    scripted.save(model_out)
    print(f"Saved TorchScript model to {model_out}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: export_torchscript.py <in.pt> <out_ts.pt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
