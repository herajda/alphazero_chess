#!/usr/bin/env python3
"""
Supervised training pipeline for the transformer-based chess agent.

This script parses a PGN dataset with the C++ chess_engine helpers (chess-library
PGN parser), writes the encounters into a compact binary buffer, and then trains
the transformer policy/value network against the human move targets.
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import subprocess
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import chess_engine
from chess_agent import Agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised chess training using PGN data.")
    default_pgn = Path("data") / "lichess_db_standard_rated_2017-09.pgn.zst"
    parser.add_argument("--pgn", type=Path, default=default_pgn,
                        help="Path to the source PGN (optionally .zst compressed).")
    parser.add_argument("--buffer", type=Path, default=Path("data") / "supervised.bin",
                        help="Output path for the binary supervised buffer.")
    parser.add_argument("--max-games", type=int, default=200_000,
                        help="Limit the number of PGN games to ingest (-1 for all).")
    parser.add_argument("--batch-size", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=0,
                        help="Training steps per epoch (0 → computed from dataset size).")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping value (<=0 disables clipping).")
    parser.add_argument("--value-loss-weight", type=float, default=1.0,
                        help="Weight for the value head MSE component.")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Steps between logging updates.")
    parser.add_argument("--logdir", type=Path, default=None,
                        help="Optional TensorBoard log directory.")
    parser.add_argument("--model-path", type=Path, default=Path("supervised_model.pt"),
                        help="Where to save the trained model weights.")
    parser.add_argument("--rebuild-buffer", action="store_true",
                        help="Force rebuilding the supervised buffer even if it exists.")
    parser.add_argument("--force-decompress", action="store_true",
                        help="Force regenerating the decompressed PGN when source is .zst.")
    parser.add_argument("--prefetch-workers", type=int, default=8,
                        help="Number of parallel workers fetching training batches.")
    parser.add_argument("--shuffle-buffer", type=int, default=1_000_000,
                        help="Number of samples kept in the sequential shuffle buffer.")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Optional RNG seed for the sequential shuffle buffer (defaults to --seed).")
    parser.add_argument("--seed", type=int )
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing model checkpoint if available.")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16",
                        help="Computation precision for training (fp16/bf16 require compatible CUDA).")
    parser.add_argument("--compile", dest="compile", action="store_true",
                        help="Enable torch.compile for the model (default).")
    parser.add_argument("--no-compile", dest="compile", action="store_false",
                        help="Disable torch.compile even if available.")
    parser.add_argument("--compile-backend", type=str, default=None,
                        help="Optional backend to pass to torch.compile (e.g., 'inductor').")
    parser.add_argument("--compile-mode", type=str, default="default",
                        help="torch.compile mode to use (e.g., 'default', 'reduce-overhead', 'max-autotune').")
    parser.add_argument("--compile-fullgraph", action="store_true",
                        help="Request fullgraph=True when compiling (experimental).")
    
    # Model Architecture Arguments
    parser.add_argument("--model_type", default="transformer", choices=["transformer", "cnn", "resnet"], help="Model architecture type.")
    parser.add_argument("--num_layers", default=6, type=int, help="Number of layers (transformer/resnet).")
    parser.add_argument("--num_heads", default=8, type=int, help="Number of heads (transformer).")
    parser.add_argument("--dim_model", default=512, type=int, help="Model dimension (transformer).")
    parser.add_argument("--num_filters", default=256, type=int, help="Number of filters (cnn/resnet).")

    parser.set_defaults(compile=True)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    if seed is None:
        seed = random.randint(0, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_decompressed_pgn(path: Path, force: bool) -> Path:
    if path.suffix.lower() != ".zst":
        return path

    target = path.with_suffix("")
    if target.exists() and not force:
        logging.info("Using cached decompressed PGN: %s", target)
        return target

    if shutil.which("zstd") is None:
        raise RuntimeError(
            "zstd binary not found. Install zstd or decompress the PGN manually."
        )

    logging.info("Decompressing %s → %s", path, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out_file:
        subprocess.run(
            ["zstd", "--decompress", "--stdout", str(path)],
            check=True,
            stdout=out_file,
        )
    return target


def build_supervised_buffer(pgn_path: Path, buffer_path: Path, max_games: int, rebuild: bool) -> int:
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild or not buffer_path.exists():
        logging.info("Converting PGN to supervised buffer...")
        chess_engine.convert_pgn_to_supervised_buffer(
            str(pgn_path),
            str(buffer_path),
            max_games if max_games is not None else -1,
        )
    else:
        logging.info("Reusing existing supervised buffer: %s", buffer_path)

    total_records = chess_engine.supervised_buffer_size(str(buffer_path))
    if total_records <= 0:
        raise RuntimeError(f"No training samples found in buffer {buffer_path}.")
    logging.info("Buffer ready with %d positions.", total_records)
    return total_records


class BatchPrefetcher:
    """Asynchronously fetch batches from the C++ sampler."""

    def __init__(self,
                 buffer_path: Path,
                 batch_size: int,
                 workers: int,
                 shuffle_buffer: int,
                 shuffle_seed: int) -> None:
        self.buffer_path = str(buffer_path)
        self.batch_size = batch_size
        self.workers = max(1, workers)
        if shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be positive.")
        self.shuffle_buffer = max(self.batch_size, shuffle_buffer)
        self.shuffle_seed = shuffle_seed if shuffle_seed is not None else -1
        self.shuffle_seed = int(self.shuffle_seed)
        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="prefetch")
        self._alive = True
        self._queue: deque[Future] = deque()
        for _ in range(self.workers):
            self._submit()

    def _submit(self) -> None:
        if not self._alive:
            return
        future = self.executor.submit(
            chess_engine.sample_supervised_batch_v2,
            self.buffer_path,
            self.batch_size,
            self.shuffle_buffer,
            self.shuffle_seed,
        )
        self._queue.append(future)

    def next_batch(self):
        if not self._queue:
            self._submit()
        future = self._queue.popleft()
        batch = future.result()
        if self._alive:
            self._submit()
        return batch

    def shutdown(self) -> None:
        self._alive = False
        while self._queue:
            future = self._queue.popleft()
            future.cancel()
        self.executor.shutdown(wait=True)


def batch_to_tensors(batch, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype == torch.float16:
        np_dtype = np.float16
    elif dtype == torch.bfloat16 and hasattr(np, "bfloat16"):
        np_dtype = np.bfloat16  # type: ignore[attr-defined]
    else:
        np_dtype = np.float32
    states = np.stack([np.asarray(sample[0], dtype=np_dtype) for sample in batch], axis=0)
    actions = np.fromiter((int(sample[1]) for sample in batch), dtype=np.int64)
    values = np.fromiter((float(sample[2]) for sample in batch), dtype=np_dtype)

    state_tensor = torch.from_numpy(states).to(device=device, dtype=dtype)
    action_tensor = torch.from_numpy(actions).to(device=device, dtype=torch.long)
    value_tensor = torch.from_numpy(values).to(device=device, dtype=dtype)
    return state_tensor, action_tensor, value_tensor


def maybe_load_checkpoint(agent: Agent, model_path: Path, resume: bool) -> None:
    if resume and model_path.exists():
        logging.info("Loading checkpoint from %s", model_path)
        state = torch.load(model_path, map_location=agent.device)
        agent._model.load_state_dict(state)
        agent._model = agent._model.to(agent.device)


def train_supervised(args: argparse.Namespace) -> None:
    setup_logging()
    set_seed(args.seed)

    logging.info("Preparing dataset...")
    pgn_path = ensure_decompressed_pgn(args.pgn, args.force_decompress)
    total_records = build_supervised_buffer(pgn_path, args.buffer, args.max_games, args.rebuild_buffer)

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch <= 0:
        steps_per_epoch = max(1, total_records // args.batch_size)
        logging.info("Auto steps-per-epoch: %d", steps_per_epoch)

    agent = Agent(args)
    maybe_load_checkpoint(agent, args.model_path, args.resume)

    writer: SummaryWriter | None = SummaryWriter(str(args.logdir)) if args.logdir else None
    shuffle_seed = args.shuffle_seed if args.shuffle_seed is not None else args.seed
    effective_shuffle_seed = shuffle_seed if shuffle_seed is not None else -1
    prefetcher = BatchPrefetcher(
        args.buffer,
        args.batch_size,
        args.prefetch_workers,
        args.shuffle_buffer,
        effective_shuffle_seed,
    )

    device = agent.device
    scaler: GradScaler = agent.scaler
    use_scaler = scaler is not None and scaler.is_enabled()
    autocast_enabled = agent.autocast_dtype is not None
    global_step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            agent._model.train()

            running_loss = running_policy = running_value  = 0.0
            running_actions = 0
            running_time = 0.0
            running_fetch = 0.0
            running_compute = 0.0

            for step in range(1, steps_per_epoch + 1):
                step_start = time.perf_counter()
                fetch_start = time.perf_counter()
                batch = prefetcher.next_batch()
                if not batch:
                    logging.warning("Received empty batch from sampler; skipping step.")
                    continue
                fetch_duration = time.perf_counter() - fetch_start

                states, actions, targets = batch_to_tensors(batch, device, agent.dtype)

                # Build policy targets as one-hot vectors matching the action space
                num_actions = agent._model.num_actions if hasattr(agent._model, "num_actions") else None
                if num_actions is None:
                    # Fallback: infer from model output by a quick forward pass
                    with torch.no_grad():
                        tmp_logits, _ = agent._model(states[:1])
                        num_actions = tmp_logits.shape[-1]
                policy_targets = F.one_hot(actions, num_classes=num_actions).to(dtype=states.dtype)

                compute_start = time.perf_counter()
                # Perform training step via Agent to trigger internal DEBUG prints

                policy_loss, value_loss, loss = agent.train(states, policy_targets, targets)
                compute_duration = time.perf_counter() - compute_start


                running_loss += float(loss)
                running_policy += float(policy_loss)
                running_value += float(value_loss)
                running_actions += actions.numel()
                running_time += time.perf_counter() - step_start
                running_fetch += fetch_duration
                running_compute += compute_duration
                global_step += 1

                if step % args.log_interval == 0:
                    scale = 1.0 / args.log_interval
                    log_loss = running_loss * scale
                    log_policy = running_policy * scale
                    log_value = running_value * scale
                    actions_per_sec = running_actions / running_time if running_time > 0 else 0.0
                    avg_fetch = running_fetch * scale
                    avg_compute = running_compute * scale
                    logging.info(
                        "epoch %d/%d | step %d/%d | loss=%.4f | policy=%.4f | value=%.4f | actions/s=%.1f | fetch=%.3fs | compute=%.3fs",
                        epoch, args.epochs, step, steps_per_epoch, log_loss, log_policy, log_value, actions_per_sec, avg_fetch, avg_compute,
                    )
                    if writer:
                        writer.add_scalar("train/loss", log_loss, global_step)
                        writer.add_scalar("train/policy_loss", log_policy, global_step)
                        writer.add_scalar("train/value_loss", log_value, global_step)
                        writer.add_scalar("train/actions_per_second", actions_per_sec, global_step)
                        writer.add_scalar("train/fetch_time_sec", avg_fetch, global_step)
                        writer.add_scalar("train/compute_time_sec", avg_compute, global_step)
                    running_loss = running_policy = running_value = 0.0
                    running_actions = 0
                    running_time = 0.0
                    running_fetch = 0.0
                    running_compute = 0.0

            agent.save(str(args.model_path))
            logging.info("Epoch %d complete. Model saved to %s", epoch, args.model_path)
    finally:
        prefetcher.shutdown()
        if writer:
            writer.close()


if __name__ == "__main__":
    cli_args = parse_args()
    train_supervised(cli_args)
