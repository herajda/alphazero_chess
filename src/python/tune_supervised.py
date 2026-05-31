#!/usr/bin/env python3
"""
Hyper-parameter and architecture search driver for the supervised chess agent.

The script trains a collection of model variants (CNN, ResNet, Transformer, ...)
and then runs an all-vs-all arena to report the strongest checkpoint.

Custom grids can be provided through a JSON file with the following structure:
[
  {
    "name": "custom_transformer",
    "overrides": {
      "model_arch": "transformer",
      "transformer_dim": 768,
      "transformer_layers": 8,
      "epochs": 6,
      "learning_rate": 0.00025
    }
  }
]
"""

from __future__ import annotations

import argparse
import json
import logging
import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

import train_supervised
from chess_agent import Agent, Player
from chess_game import ChessGame


@dataclass
class ExperimentSpec:
    name: str
    overrides: dict[str, Any]


class LossCurveAggregator:
    """Collect per-log-interval losses for each run and render them in a single plot."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._interval_curves: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def record(self, run_name: str, metrics: dict[str, Any]) -> None:
        if metrics.get("scope") != "interval":
            return
        loss = metrics.get("avg_loss")
        step = metrics.get("global_step")
        if loss is None or step is None:
            return
        self._interval_curves[run_name].append((int(step), float(loss)))

    def save_plot(self) -> None:
        if not self._interval_curves:
            logging.info("No aggregated loss data recorded; skipping combined plot.")
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: PERF203
            logging.warning("Unable to import matplotlib; skipping combined loss plot: %s", exc)
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        for run_name, points in sorted(self._interval_curves.items()):
            if not points:
                continue
            points.sort(key=lambda item: item[0])
            steps, losses = zip(*points)
            ax.plot(steps, losses, label=run_name)

        ax.set_xlabel("Global step (log interval)")
        ax.set_ylabel("Average loss")
        ax.set_title("Supervised training loss (per log interval)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_path)
        plt.close(fig)
        logging.info(
            "Saved combined loss plot for %d models to %s",
            len([points for points in self._interval_curves.values() if points]),
            self.output_path,
        )


def default_search_space() -> list[ExperimentSpec]:
    """Return a diverse, curated grid of model/hparam combinations."""
    specs: list[ExperimentSpec] = [
        ExperimentSpec(
            name="resnet_224_deep12",
            overrides={
                "model_arch": "resnet",
                "resnet_channels": 224,
                "resnet_blocks": 12,
                "learning_rate": 3.2e-4,
                "weight_decay": 1.2e-4,
                "epochs": 4,
                "batch_size": 1152,
            },
        ),
        ExperimentSpec(
            name="resnet_256_bottleneck12",
            overrides={
                "model_arch": "resnet",
                "resnet_channels": 256,
                "resnet_blocks": 12,
                "resnet_bottleneck": True,
                "learning_rate": 3e-4,
                "weight_decay": 1.1e-4,
                "epochs": 4,
                "batch_size": 1024,
            },
        ),
        ExperimentSpec(
            name="resnet_288_bottleneck14",
            overrides={
                "model_arch": "resnet",
                "resnet_channels": 288,
                "resnet_blocks": 14,
                "resnet_bottleneck": True,
                "learning_rate": 2.7e-4,
                "weight_decay": 1e-4,
                "epochs": 4,
                "batch_size": 896,
            },
        ),
        ExperimentSpec(
            name="resnet_320_bottleneck16",
            overrides={
                "model_arch": "resnet",
                "resnet_channels": 320,
                "resnet_blocks": 16,
                "resnet_bottleneck": True,
                "learning_rate": 2.5e-4,
                "weight_decay": 9e-5,
                "epochs": 4,
                "batch_size": 832,
            },
        ),
        ExperimentSpec(
            name="resnet_352_bottleneck12",
            overrides={
                "model_arch": "resnet",
                "resnet_channels": 352,
                "resnet_blocks": 12,
                "resnet_bottleneck": True,
                "learning_rate": 2.4e-4,
                "weight_decay": 9e-5,
                "epochs": 5,
                "batch_size": 768,
            },
        ),
        ExperimentSpec(
            name="resnet_256_standard10",
            overrides={
                "model_arch": "resnet",
                "resnet_channels": 256,
                "resnet_blocks": 10,
                "learning_rate": 3.6e-4,
                "weight_decay": 1.3e-4,
                "epochs": 4,
                "batch_size": 1152,
            },
        ),
        ExperimentSpec(
            name="resnet_288_standard12",
            overrides={
                "model_arch": "resnet",
                "resnet_channels": 288,
                "resnet_blocks": 12,
                "learning_rate": 3.1e-4,
                "weight_decay": 1.1e-4,
                "epochs": 4,
                "batch_size": 992,
            },
        ),
        ExperimentSpec(
            name="resnet_224_light8",
            overrides={
                "model_arch": "resnet",
                "resnet_channels": 224,
                "resnet_blocks": 8,
                "learning_rate": 4e-4,
                "weight_decay": 1.4e-4,
                "epochs": 3,
                "batch_size": 1408,
            },
        ),
        ExperimentSpec(
            name="resnet_192_fast10",
            overrides={
                "model_arch": "resnet",
                "resnet_channels": 192,
                "resnet_blocks": 10,
                "learning_rate": 4.2e-4,
                "weight_decay": 1.5e-4,
                "epochs": 3,
                "batch_size": 1536,
            },
        ),
        ExperimentSpec(
            name="resnet_256_bottleneck8",
            overrides={
                "model_arch": "resnet",
                "resnet_channels": 256,
                "resnet_blocks": 8,
                "resnet_bottleneck": True,
                "learning_rate": 3.8e-4,
                "weight_decay": 1.3e-4,
                "epochs": 4,
                "batch_size": 1280,
            },
        ),
        ExperimentSpec(
            name="resnet_320_standard14",
            overrides={
                "model_arch": "resnet",
                "resnet_channels": 320,
                "resnet_blocks": 14,
                "learning_rate": 2.8e-4,
                "weight_decay": 1e-4,
                "epochs": 4,
                "batch_size": 896,
            },
        ),
        ExperimentSpec(
            name="resnet_288_bottleneck10",
            overrides={
                "model_arch": "resnet",
                "resnet_channels": 288,
                "resnet_blocks": 10,
                "resnet_bottleneck": True,
                "learning_rate": 3e-4,
                "weight_decay": 1.05e-4,
                "epochs": 4,
                "batch_size": 1056,
            },
        ),
        ExperimentSpec(
            name="convnext_t_base",
            overrides={
                "model_arch": "convnext",
                "compile": False,
                "convnext_dims": [96, 192, 384, 576],
                "convnext_depths": [2, 2, 6, 2],
                "convnext_drop_path": 0.1,
                "convnext_ffn_multiplier": 2.0,
                "epochs": 4,
                "batch_size": 1152,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
            },
        ),
        ExperimentSpec(
            name="convnext_t_deep",
            overrides={
                "model_arch": "convnext",
                "compile": False,
                "convnext_dims": [96, 192, 384, 768],
                "convnext_depths": [3, 3, 9, 3],
                "convnext_drop_path": 0.15,
                "convnext_ffn_multiplier": 2.0,
                "epochs": 4,
                "batch_size": 1024,
                "learning_rate": 2.7e-4,
                "weight_decay": 9e-5,
            },
        ),
        ExperimentSpec(
            name="convnext_t_wide",
            overrides={
                "model_arch": "convnext",
                "compile": False,
                "convnext_dims": [128, 256, 512, 768],
                "convnext_depths": [2, 2, 6, 2],
                "convnext_drop_path": 0.12,
                "convnext_ffn_multiplier": 2.0,
                "epochs": 4,
                "batch_size": 960,
                "learning_rate": 2.8e-4,
                "weight_decay": 1e-4,
            },
        ),
        ExperimentSpec(
            name="convnext_t_compact",
            overrides={
                "model_arch": "convnext",
                "compile": False,
                "convnext_dims": [80, 160, 320, 480],
                "convnext_depths": [3, 3, 6, 2],
                "convnext_drop_path": 0.08,
                "convnext_ffn_multiplier": 2.0,
                "epochs": 4,
                "batch_size": 1280,
                "learning_rate": 3.2e-4,
                "weight_decay": 1.1e-4,
            },
        ),
        ExperimentSpec(
            name="convnext_t_scalable",
            overrides={
                "model_arch": "convnext",
                "compile": False,
                "convnext_dims": [112, 224, 448, 672],
                "convnext_depths": [2, 2, 6, 2],
                "convnext_drop_path": 0.12,
                "convnext_ffn_multiplier": 2.25,
                "epochs": 4,
                "batch_size": 960,
                "learning_rate": 2.9e-4,
                "weight_decay": 9e-5,
            },
        ),
        ExperimentSpec(
            name="convnext_t_regularized",
            overrides={
                "model_arch": "convnext",
                "compile": False,
                "convnext_dims": [96, 192, 384, 640],
                "convnext_depths": [3, 3, 8, 3],
                "convnext_drop_path": 0.2,
                "convnext_ffn_multiplier": 2.0,
                "epochs": 4,
                "batch_size": 896,
                "learning_rate": 2.5e-4,
                "weight_decay": 9e-5,
            },
        ),
        ExperimentSpec(
            name="convnext_t_fast",
            overrides={
                "model_arch": "convnext",
                "compile": False,
                "convnext_dims": [96, 192, 320, 480],
                "convnext_depths": [2, 2, 6, 2],
                "convnext_drop_path": 0.05,
                "convnext_ffn_multiplier": 1.8,
                "epochs": 3,
                "batch_size": 1344,
                "learning_rate": 3.4e-4,
                "weight_decay": 1.2e-4,
            },
        ),
        ExperimentSpec(
            name="convnext_t_bold",
            overrides={
                "model_arch": "convnext",
                "compile": False,
                "convnext_dims": [128, 256, 384, 512],
                "convnext_depths": [3, 3, 9, 3],
                "convnext_drop_path": 0.18,
                "convnext_ffn_multiplier": 2.5,
                "epochs": 4,
                "batch_size": 832,
                "learning_rate": 2.6e-4,
                "weight_decay": 1e-4,
            },
        ),
    ]
    return specs


def parse_cli() -> argparse.Namespace:
    default_pgn = Path("data") / "lichess_db_standard_rated_2017-09.pgn.zst"
    parser = argparse.ArgumentParser(
        description="Run a supervised model/architecture grid search with arena evaluation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments") / "supervised_search",
        help="Where to store checkpoints, logs, and arena summaries.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Optional JSON file describing the grid to run.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional subset of configuration names to run.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only run the arena on previously saved models.",
    )
    parser.add_argument(
        "--games-per-match",
        type=int,
        default=2,
        help="Number of games per pairing (per color).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=320,
        help="Move cap for arena games (0 disables).",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
        help="Precision to load checkpoints for arena evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base RNG seed used for configs (each config increments this).",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="Optional hard limit on how many configs to run (0 means all).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile during search runs.",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile during search runs.",
    )
    parser.set_defaults(compile=True)
    parser.add_argument(
        "--pgn",
        type=Path,
        default=default_pgn,
        help="PGN source used for training.",
    )
    parser.add_argument(
        "--buffer",
        type=Path,
        default=Path("data") / "supervised.bin",
        help="Binary supervised buffer file.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=200_000,
        help="Maximum PGN games to parse.",
    )
    parser.add_argument(
        "--force-decompress",
        action="store_true",
        help="Force PGN decompression even if cached.",
    )
    parser.add_argument(
        "--rebuild-buffer",
        action="store_true",
        help="Force rebuilding the supervised buffer.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Logging interval passed to training runs.",
    )
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=1.0,
        help="Override for value head loss weight if configs do not specify one.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training if checkpoints already exist.",
    )
    parser.add_argument(
        "--trial-steps",
        type=int,
        default=2,
        help="Number of dummy minibatches to run per config before launching full training.",
    )
    parser.add_argument(
        "--tune-logdir",
        type=Path,
        default=None,
        help="Optional TensorBoard directory aggregating per-model training curves.",
    )
    return parser.parse_args()


def load_specs_from_file(path: Path) -> list[ExperimentSpec]:
    data = json.loads(path.read_text())
    specs: list[ExperimentSpec] = []
    for entry in data:
        name = entry.get("name")
        overrides = entry.get("overrides", {})
        if not name or not isinstance(overrides, dict):
            raise ValueError(f"Invalid entry in {path}: {entry}")
        specs.append(ExperimentSpec(name=name, overrides=overrides))
    return specs


def filtered_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    specs = load_specs_from_file(args.config_file) if args.config_file else default_search_space()
    if args.include:
        wanted = set(args.include)
        specs = [spec for spec in specs if spec.name in wanted]
    max_configs = args.max_configs or len(specs)
    return specs[:max_configs]


def build_training_args(
    template_args: argparse.Namespace,
    cli_args: argparse.Namespace,
    spec: ExperimentSpec,
    idx: int,
    exp_dir: Path,
) -> argparse.Namespace:
    train_args = copy.deepcopy(template_args)
    train_args.pgn = cli_args.pgn
    train_args.buffer = cli_args.buffer
    train_args.max_games = cli_args.max_games
    train_args.force_decompress = cli_args.force_decompress
    train_args.rebuild_buffer = cli_args.rebuild_buffer
    train_args.log_interval = cli_args.log_interval
    train_args.precision = cli_args.precision
    train_args.compile = cli_args.compile
    train_args.value_loss_weight = spec.overrides.get("value_loss_weight", cli_args.value_loss_weight)
    train_args.model_arch = spec.overrides.get("model_arch", getattr(train_args, "model_arch", "transformer"))
    train_args.resume = cli_args.resume
    train_args.trial_steps = max(1, int(getattr(cli_args, "trial_steps", 1)))

    for key, value in spec.overrides.items():
        setattr(train_args, key, value)

    exp_dir.mkdir(parents=True, exist_ok=True)
    if not getattr(train_args, "logdir", None):
        train_args.logdir = exp_dir / "runs"
    else:
        train_args.logdir = exp_dir / "runs"
    train_args.model_path = exp_dir / "model.pt"
    train_args.seed = (cli_args.seed or 1) + idx
    train_args.shuffle_seed = getattr(train_args, "seed", None)
    return train_args


def build_progress_callback(
    writer: SummaryWriter | None,
    run_name: str,
    aggregator: LossCurveAggregator | None = None,
):
    if writer is None and aggregator is None:
        return None

    tag_prefix = run_name

    def _callback(metrics: dict[str, Any]):
        epoch = metrics.get("epoch", 0)
        loss = metrics.get("avg_loss")
        actions_per_sec = metrics.get("actions_per_sec")
        duration = metrics.get("duration_sec")
        value_loss = metrics.get("avg_value_loss")
        policy_loss = metrics.get("avg_policy_loss")

        if writer is not None:
            if loss is not None:
                writer.add_scalar(f"{tag_prefix}/loss", loss, epoch)
            if policy_loss is not None:
                writer.add_scalar(f"{tag_prefix}/policy_loss", policy_loss, epoch)
            if value_loss is not None:
                writer.add_scalar(f"{tag_prefix}/value_loss", value_loss, epoch)
            if actions_per_sec is not None:
                writer.add_scalar(f"{tag_prefix}/actions_per_sec", actions_per_sec, epoch)
            if duration is not None:
                writer.add_scalar(f"{tag_prefix}/epoch_duration_sec", duration, epoch)
            writer.flush()
        if aggregator is not None:
            aggregator.record(run_name, metrics)

    return _callback


def _dummy_training_step(agent: Agent, batch_size: int) -> None:
    board_channels = getattr(agent._model, "initial_channels", 119)
    num_actions = getattr(agent._model, "num_actions", ChessGame.ACTIONS)
    states = torch.randn(batch_size, ChessGame.N, ChessGame.N, board_channels, dtype=torch.float32)
    policy_targets = torch.zeros(batch_size, num_actions, dtype=torch.float32)
    action_indices = torch.randint(0, num_actions, (batch_size, 1))
    policy_targets.scatter_(1, action_indices, 1.0)
    value_targets = torch.empty(batch_size, dtype=torch.float32).uniform_(-1.0, 1.0)
    agent.train(states, policy_targets, value_targets)
    del states, policy_targets, value_targets


def _run_trial_training(agent: Agent, batch_size: int, trial_steps: int) -> None:
    for step in range(trial_steps):
        logging.debug(
            "Trial training step %d/%d (batch_size=%d)",
            step + 1,
            trial_steps,
            batch_size,
        )
        _dummy_training_step(agent, batch_size)


def check_model_fits_vram(train_args: argparse.Namespace) -> tuple[bool, str | None]:
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
    agent: Agent | None = None
    try:
        agent = Agent(train_args)
        batch_size = int(getattr(train_args, "batch_size", 32))
        trial_steps = max(1, int(getattr(train_args, "trial_steps", 1)))
        _run_trial_training(agent, batch_size, trial_steps)
        return True, None
    except RuntimeError as exc:  # noqa: PERF203
        if cuda_available and "out of memory" in str(exc).lower():
            return False, str(exc)
        raise
    finally:
        if agent is not None:
            del agent
        if cuda_available:
            torch.cuda.empty_cache()


def verify_vram_capacity(experiments: list[tuple[ExperimentSpec, argparse.Namespace]]) -> bool:
    if not experiments:
        return True
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        logging.warning(
            "CUDA not available; running CPU trial steps instead of VRAM warm-up."
        )
    failures: list[tuple[str, str]] = []
    for spec, train_args in experiments:
        trial_steps = max(1, int(getattr(train_args, "trial_steps", 1)))
        logging.info(
            "Performing VRAM warm-up/trial (%d step%s) for %s...",
            trial_steps,
            "" if trial_steps == 1 else "s",
            spec.name,
        )
        fits, message = check_model_fits_vram(train_args)
        if not fits:
            summary = (
                f"arch={getattr(train_args, 'model_arch', 'transformer')} | "
                f"batch_size={getattr(train_args, 'batch_size', 'n/a')} | "
                f"overrides={spec.overrides}"
            )
            failures.append((summary, message or "CUDA OOM"))
            logging.error("✗ %s does not fit on the GPU: %s", spec.name, message)
        else:
            suffix = (
                "fits in GPU memory"
                if cuda_available
                else "completed CPU trial run"
            )
            logging.info(
                "✓ %s %s (%d step%s).",
                spec.name,
                suffix,
                trial_steps,
                "" if trial_steps == 1 else "s",
            )
    if failures:
        logging.error("Aborting search because some configs do not fit in VRAM:")
        for summary, message in failures:
            logging.error(" - %s :: %s", summary, message)
        return False
    return True


def ensure_agents(
    experiments: list[tuple[ExperimentSpec, argparse.Namespace]],
    precision: str,
) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for spec, train_args in experiments:
        model_path = Path(train_args.model_path)
        if not model_path.exists():
            logging.warning("Skipping %s because checkpoint %s is missing.", spec.name, model_path)
            continue
        eval_args = copy.deepcopy(train_args)
        eval_args.infer = True
        eval_args.num_simulations = 0
        eval_args.model_path = str(model_path)
        eval_args.precision = precision
        agent = Agent.load(str(model_path), eval_args)
        match_args = argparse.Namespace(num_simulations=0, epsilon=0.25, alpha=0.3)
        player = Player(agent, match_args)
        loaded.append(
            {
                "name": spec.name,
                "agent": agent,
                "player": player,
                "model_path": model_path,
                "train_args": train_args,
            }
        )
    return loaded


def play_single_game(player_white: Player, player_black: Player, max_moves: int) -> tuple[int | None, int]:
    game = ChessGame(gui_enabled=False)
    move_count = 0
    while game.winner is None:
        if max_moves > 0 and move_count >= max_moves:
            return -1, move_count
        current_player = player_white if game.to_play == 1 else player_black
        action = current_player.play(game)
        game.move(action)
        move_count += 1
    return game.winner, move_count


def run_arena(
    competitors: list[dict[str, Any]],
    games_per_match: int,
    max_moves: int,
) -> dict[str, Any]:
    scoreboard = {
        entry["name"]: {"score": 0.0, "wins": 0, "losses": 0, "draws": 0}
        for entry in competitors
    }
    history: list[dict[str, Any]] = []

    def record_result(white: str, black: str, outcome: int | None):
        if outcome == 1:
            scoreboard[white]["wins"] += 1
            scoreboard[white]["score"] += 1.0
            scoreboard[black]["losses"] += 1
        elif outcome == 0:
            scoreboard[black]["wins"] += 1
            scoreboard[black]["score"] += 1.0
            scoreboard[white]["losses"] += 1
        else:
            scoreboard[white]["draws"] += 1
            scoreboard[black]["draws"] += 1
            scoreboard[white]["score"] += 0.5
            scoreboard[black]["score"] += 0.5

    for i in range(len(competitors)):
        for j in range(i + 1, len(competitors)):
            a = competitors[i]
            b = competitors[j]
            for _ in range(games_per_match):
                outcome, moves = play_single_game(a["player"], b["player"], max_moves)
                history.append(
                    {"white": a["name"], "black": b["name"], "result": outcome, "moves": moves}
                )
                record_result(a["name"], b["name"], outcome)

                outcome, moves = play_single_game(b["player"], a["player"], max_moves)
                history.append(
                    {"white": b["name"], "black": a["name"], "result": outcome, "moves": moves}
                )
                record_result(b["name"], a["name"], outcome)

    ranking = sorted(
        scoreboard.items(),
        key=lambda item: (item[1]["score"], item[1]["wins"]),
        reverse=True,
    )
    return {"scoreboard": scoreboard, "history": history, "ranking": ranking}


def main() -> int:
    args = parse_cli()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    template_args = train_supervised.parse_args([])
    specs = filtered_specs(args)
    if not specs:
        logging.error("No experiment configurations resolved. Nothing to do.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tune_writer: SummaryWriter | None = None
    curve_aggregator: LossCurveAggregator | None = None
    if not args.skip_training:
        tune_logdir = args.tune_logdir or (args.output_dir / "tune_tensorboard")
        tune_logdir.parent.mkdir(parents=True, exist_ok=True)
        tune_writer = SummaryWriter(str(tune_logdir))
        curve_aggregator = LossCurveAggregator(args.output_dir / "combined_loss_curves.png")

    try:
        experiments: list[tuple[ExperimentSpec, argparse.Namespace]] = []
        for idx, spec in enumerate(specs):
            exp_dir = args.output_dir / spec.name
            train_args = build_training_args(template_args, args, spec, idx, exp_dir)
            progress_cb = build_progress_callback(tune_writer, spec.name, curve_aggregator)
            if progress_cb is not None:
                setattr(train_args, "_progress_callback", progress_cb)
            experiments.append((spec, train_args))

        pending_experiments: list[tuple[ExperimentSpec, argparse.Namespace]] = []
        for spec, train_args in experiments:
            model_path = Path(train_args.model_path)
            checkpoint_ready = model_path.is_file() and model_path.stat().st_size > 0
            if checkpoint_ready and not args.resume:
                logging.info(
                    "Checkpoint already exists for %s at %s; skipping training.",
                    spec.name,
                    model_path,
                )
            else:
                pending_experiments.append((spec, train_args))

        if not args.skip_training:
            if not pending_experiments:
                logging.info("No experiments require training; reusing existing checkpoints.")
            else:
                if not verify_vram_capacity(pending_experiments):
                    return 1
                for spec, train_args in pending_experiments:
                    logging.info("Training %s with overrides: %s", spec.name, spec.overrides)
                    train_supervised.train_supervised(train_args)

        competitors = ensure_agents(experiments, args.precision)
        if len(competitors) < 2:
            logging.error("Need at least two trained agents for an arena (found %d).", len(competitors))
            return 1

        logging.info("Running arena with %d agents (%d games per pairing).", len(competitors), args.games_per_match)
        arena_summary = run_arena(competitors, args.games_per_match, args.max_moves)

        summary_path = args.output_dir / "arena_summary.json"
        summary_path.write_text(json.dumps(arena_summary, indent=2))
        best_name, best_stats = arena_summary["ranking"][0]
        logging.info("Best model: %s (score %.2f, %dW/%dL/%dD)",
                     best_name,
                     best_stats["score"],
                     best_stats["wins"],
                     best_stats["losses"],
                     best_stats["draws"])
        print("\nArena ranking:")
        for rank, (name, stats) in enumerate(arena_summary["ranking"], 1):
            print(
                f"{rank:2d}. {name:20s} | score={stats['score']:.2f} | "
                f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}"
            )
        print(f"\nFull arena log stored at: {summary_path}")
        return 0
    finally:
        if tune_writer is not None:
            tune_writer.close()
        if curve_aggregator is not None:
            curve_aggregator.save_plot()


if __name__ == "__main__":
    raise SystemExit(main())
