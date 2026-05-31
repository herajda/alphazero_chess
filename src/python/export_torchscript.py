#!/usr/bin/env python3
"""
Export a trained PyTorch model to TorchScript for C++ inference.
"""
import argparse
import os
import torch

from chess_agent import Agent


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def infer_precision(state_dict) -> str:
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float16:
                return "fp16"
            if value.dtype == torch.bfloat16:
                return "bf16"
            return "fp32"
    return "fp32"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained model to TorchScript.")
    parser.add_argument("model_in", help="Path to the PyTorch checkpoint to export.")
    parser.add_argument("model_out", help="Output TorchScript path.")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=None,
                        help="Override precision detection (defaults to checkpoint dtype or bf16).")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate used only when instantiating the Agent wrapper.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay used only when instantiating the Agent wrapper.")
    parser.add_argument("--model_arch", choices=("transformer", "cnn", "resnet", "convnext"), default="transformer",
                        help="Backbone architecture to instantiate for export.")
    parser.add_argument("--transformer_dim", type=int, default=512, help="Transformer embedding dimension.")
    parser.add_argument("--transformer_heads", type=int, default=8, help="Transformer attention heads.")
    parser.add_argument("--transformer_layers", type=int, default=6, help="Transformer encoder layers.")
    parser.add_argument("--transformer_ff_multiplier", type=int, default=2, help="Transformer FFN expansion ratio.")
    parser.add_argument("--cnn_channels", type=int, default=256, help="CNN base channels.")
    parser.add_argument("--cnn_depth", type=int, default=8, help="CNN depth (number of conv blocks).")
    parser.add_argument("--cnn_kernel_size", type=int, default=3, help="CNN kernel size.")
    parser.add_argument("--resnet_channels", type=int, default=256, help="ResNet channel width.")
    parser.add_argument("--resnet_blocks", type=int, default=6, help="Number of residual blocks.")
    parser.add_argument("--resnet_bottleneck", action="store_true", help="Use bottleneck-style residual blocks.")
    parser.add_argument("--convnext_dims", nargs="*", type=int, default=None, help="ConvNeXt-V2 dims per stage (len=4).")
    parser.add_argument("--convnext_depths", nargs="*", type=int, default=None, help="ConvNeXt-V2 depths per stage (len=4).")
    parser.add_argument("--convnext_drop_path", type=float, default=0.1, help="ConvNeXt-V2 stochastic depth rate.")
    parser.add_argument("--convnext_ffn_multiplier", type=float, default=2.0, help="ConvNeXt-V2 FFN multiplier.")
    parser.add_argument("--convnext_layer_scale_init", type=float, default=1e-6, help="ConvNeXt-V2 layer scale init.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    # Determine precision from checkpoint (default to bf16 to align with training pipeline).
    state_dict = None
    precision = args.precision or "bf16"
    if os.path.isfile(args.model_in):
        checkpoint = torch.load(args.model_in, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)
        if args.precision is None:
            precision = infer_precision(state_dict)

    agent_args = argparse.Namespace(
        model_path=args.model_in,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        precision=precision,
        model_arch=args.model_arch,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff_multiplier=args.transformer_ff_multiplier,
        cnn_channels=args.cnn_channels,
        cnn_depth=args.cnn_depth,
        cnn_kernel_size=args.cnn_kernel_size,
        resnet_channels=args.resnet_channels,
        resnet_blocks=args.resnet_blocks,
        resnet_bottleneck=args.resnet_bottleneck,
        convnext_dims=args.convnext_dims,
        convnext_depths=args.convnext_depths,
        convnext_drop_path=args.convnext_drop_path,
        convnext_ffn_multiplier=args.convnext_ffn_multiplier,
        convnext_layer_scale_init=args.convnext_layer_scale_init,
    )

    if state_dict is not None:
        print(f"Loading weights from '{args.model_in}' ({precision})…")
        agent = Agent.load(args.model_in, agent_args)
    else:
        print(f"Warning: '{args.model_in}' not found; initializing a new random model.")
        agent = Agent(agent_args)
    agent._model.eval()
    scripted = torch.jit.script(agent._model)
    scripted.save(args.model_out)
    print(f"Saved TorchScript model to {args.model_out}")


if __name__ == '__main__':
    main()
