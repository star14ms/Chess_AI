import argparse
import hashlib
import json
import math
import os
import time
import random

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from omegaconf import OmegaConf
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
import matplotlib.pyplot as plt

from chess_gym.chess_custom import LegacyChessBoard, FullyTrackedBoard
from MCTS.training_modules.chess import create_chess_network
from utils.profile_model import profile_model
from utils.training_utils import (
    count_dataset_entries,
    iter_csv_rows,
    iter_json_array,
    load_checkpoint,
    save_checkpoint,
    select_device,
    fast_count_json_lines,
    fast_count_jsonl_lines,
)
from utils.dataset_labels import format_dataset_label, truncate_label


class NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        return False

    def add_task(self, *args, **kwargs):
        return 0

    def reset(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return None


class MateInOneDataset(Dataset):
    def __init__(self, cfg, rows):
        self.cfg = cfg
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        obs, action_index = self.rows[idx]
        return torch.from_numpy(obs).float(), torch.tensor(action_index, dtype=torch.long)


class MateInOneIterableDataset(IterableDataset):
    def __init__(
        self,
        cfg,
        data_paths: list[str],
        split: str,
        val_split: float,
        seed: int,
        max_rows: int | None = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'.")
        self.cfg = cfg
        self.data_paths = data_paths
        self.split = split
        self.val_split = val_split
        self.seed = seed
        self.max_rows = max_rows
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        global_worker_id = worker_id + self.rank * num_workers
        global_num_workers = num_workers * self.world_size
        max_rows = self.max_rows
        if max_rows is not None and global_num_workers > 1:
            max_rows = math.ceil(max_rows / global_num_workers)

        sources = []
        for data_path in self.data_paths:
            lower_path = data_path.lower()
            if lower_path.endswith(".jsonl"):
                entry_iter = _iter_json_lines(data_path)
                source_type = "jsonl"
            elif lower_path.endswith(".json"):
                entry_iter = iter_json_array(data_path)
                source_type = "json"
            else:
                entry_iter = iter_csv_rows(data_path)
                source_type = "csv"
            source_is_endgame = "endgame" in os.path.basename(lower_path)
            sources.append((entry_iter, source_type, source_is_endgame))

        action_space_size = self.cfg.network.action_space_size
        input_channels = self.cfg.network.input_channels
        history_steps = self.cfg.env.history_steps

        raw_index = 0
        yielded = 0
        rng = random.Random(self.seed + global_worker_id)
        active = list(range(len(sources)))
        while active:
            active_idx = rng.randrange(len(active))
            source_id = active[active_idx]
            iterator, source_type, source_is_endgame = sources[source_id]
            try:
                entry = next(iterator)
            except StopIteration:
                active.pop(active_idx)
                continue
            if raw_index % global_num_workers != global_worker_id:
                raw_index += 1
                continue
            raw_index += 1

            if not self._entry_matches_split(entry):
                continue

            if _is_processed_entry(entry, source_type):
                processed = _extract_processed_entry(entry, source_type)
                if processed is None:
                    continue
                fen, value_target, policy_mask, policy_uci, themes, source_override, moves_to_mate = processed
                board = create_board_from_fen(fen, action_space_size)
                action_index = 0
                if policy_mask:
                    try:
                        move = chess.Move.from_uci(policy_uci)
                    except ValueError:
                        continue
                    if not board.is_legal(move):
                        continue
                    action_id = board.move_to_action_id(move)
                    if action_id is None or action_id < 1 or action_id > action_space_size:
                        continue
                    action_index = action_id - 1
                obs = board.get_board_vector(history_steps=history_steps)
                if obs.shape[0] != input_channels:
                    raise ValueError(
                        f"Observation channels {obs.shape[0]} != expected {input_channels}. "
                        "Check action_space_size vs input_channels and board type."
                    )
                resolved_source_id = _source_id_from_moves_to_mate(
                    moves_to_mate, is_endgame=source_is_endgame
                )
                yield (
                    torch.from_numpy(obs.astype(np.float32, copy=False)).float(),
                    torch.tensor(action_index, dtype=torch.long),
                    torch.tensor(value_target, dtype=torch.float32),
                    torch.tensor(policy_mask, dtype=torch.bool),
                    resolved_source_id,
                    themes,
                )
                yielded += 1
                if max_rows is not None and yielded >= max_rows:
                    break
                continue

            fen, moves = _extract_fen_and_moves(entry, source_type)
            if not fen or not moves:
                continue
            themes = _extract_themes(entry, source_type)

            board = create_board_from_fen(fen, action_space_size)
            parsed_moves: list[chess.Move] = []
            temp_board = board.copy(stack=False)
            valid = True
            for uci in moves:
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    valid = False
                    break
                if not temp_board.is_legal(move):
                    valid = False
                    break
                parsed_moves.append(move)
                temp_board.push(move)
            if not valid:
                continue

            total_plies = len(parsed_moves)
            for ply_index, move in enumerate(parsed_moves):
                policy_mask = True
                action_index = 0
                if policy_mask:
                    action_id = board.move_to_action_id(move)
                    if action_id is None or action_id < 1 or action_id > action_space_size:
                        valid = False
                        break
                    action_index = action_id - 1
                obs = board.get_board_vector(history_steps=history_steps)
                if obs.shape[0] != input_channels:
                    raise ValueError(
                        f"Observation channels {obs.shape[0]} != expected {input_channels}. "
                        "Check action_space_size vs input_channels and board type."
                    )
                value_target = 1.0 if ply_index % 2 == 0 else -1.0
                moves_to_mate = total_plies - ply_index
                yield (
                    torch.from_numpy(obs.astype(np.float32, copy=False)).float(),
                    torch.tensor(action_index, dtype=torch.long),
                    torch.tensor(value_target, dtype=torch.float32),
                    torch.tensor(policy_mask, dtype=torch.bool),
                    _source_id_from_moves_to_mate(
                        moves_to_mate, is_endgame=source_is_endgame
                    ),
                    themes,
                )
                yielded += 1
                if max_rows is not None and yielded >= max_rows:
                    break
                board.push(move)
            if max_rows is not None and yielded >= max_rows:
                break

    def _entry_matches_split(self, entry: dict) -> bool:
        if self.val_split <= 0:
            return self.split == "train"
        if self.val_split >= 1:
            return self.split == "val"
        key = entry.get("PuzzleId") or entry.get("puzzle_id") or entry.get("FEN") or entry.get("fen")
        if not key:
            key = json.dumps(entry, sort_keys=True, ensure_ascii=True)
        digest = hashlib.sha256(f"{self.seed}:{key}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        is_val = bucket < self.val_split
        return is_val if self.split == "val" else not is_val


def create_board_from_fen(fen: str, action_space_size: int):
    if action_space_size == 4672:
        board = LegacyChessBoard(fen)
    else:
        board = FullyTrackedBoard(fen)
    return board


def _extract_moves_list(moves_field):
    if not moves_field:
        return []
    if isinstance(moves_field, str):
        tokens = moves_field.split()
    elif isinstance(moves_field, list):
        tokens = [str(t).strip() for t in moves_field if str(t).strip()]
    else:
        return []
    return [token for token in tokens if token]


def _iter_json_lines(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_themes(themes_field) -> list[str]:
    if not themes_field:
        return []
    if isinstance(themes_field, list):
        return [str(t).strip() for t in themes_field if str(t).strip()]
    if isinstance(themes_field, str):
        return [t for t in themes_field.split() if t.strip()]
    return []


def _is_processed_entry(entry: dict, source_type: str) -> bool:
    if source_type == "jsonl":
        return True
    if not isinstance(entry, dict):
        return False
    return "value_target" in entry or "value" in entry


def _extract_processed_entry(entry: dict, source_type: str):
    fen = entry.get("fen") or entry.get("FEN")
    if not fen:
        return None
    value_target = entry.get("value_target")
    if value_target is None:
        value_target = entry.get("value")
    if value_target is None:
        return None
    if "policy_mask" in entry:
        policy_mask = bool(entry.get("policy_mask", False))
    else:
        policy_mask = True
    policy_uci = entry.get("policy_uci") or entry.get("policy_move") or entry.get("best") or entry.get("move")
    if policy_mask and not policy_uci:
        return None
    if not policy_mask:
        policy_uci = ""
    themes_field = entry.get("themes") or entry.get("Themes")
    themes = _normalize_themes(themes_field)
    moves_to_mate = entry.get("moves_to_mate")
    if moves_to_mate is not None:
        try:
            moves_to_mate = int(moves_to_mate)
        except (TypeError, ValueError):
            moves_to_mate = None
    source_override = entry.get("source_id")
    if source_override is None:
        source_override = entry.get("source_label")
    return fen, float(value_target), policy_mask, policy_uci, themes, source_override, moves_to_mate


def _extract_themes(entry: dict, source_type: str) -> list[str]:
    if source_type == "csv":
        return []
    themes_field = entry.get("Themes") or entry.get("themes")
    return _normalize_themes(themes_field)


def _extract_fen_and_moves(entry: dict, source_type: str):
    if source_type == "csv":
        moves_field = entry.get("moves") or entry.get("best")
        return entry.get("fen"), _extract_moves_list(moves_field)
    fen = entry.get("FEN") or entry.get("fen")
    moves_field = entry.get("Moves") or entry.get("moves")
    moves = _extract_moves_list(moves_field)
    return fen, moves


def _source_id_from_moves_to_mate(
    moves_to_mate: int | None, is_endgame: bool = False
) -> torch.Tensor:
    if is_endgame:
        return torch.tensor(6, dtype=torch.long)
    if moves_to_mate is None:
        return torch.tensor(6, dtype=torch.long)
    try:
        moves_to_mate = int(moves_to_mate)
    except (TypeError, ValueError):
        return torch.tensor(6, dtype=torch.long)
    if moves_to_mate <= 0:
        return torch.tensor(6, dtype=torch.long)
    mate_in = (moves_to_mate + 1) // 2
    if mate_in <= 1:
        return torch.tensor(0, dtype=torch.long)
    if mate_in == 2:
        return torch.tensor(1, dtype=torch.long)
    if mate_in == 3:
        return torch.tensor(2, dtype=torch.long)
    if mate_in == 4:
        return torch.tensor(3, dtype=torch.long)
    if mate_in == 5:
        return torch.tensor(4, dtype=torch.long)
    if mate_in <= 5:
        return torch.tensor(mate_in - 1, dtype=torch.long)
    return torch.tensor(5, dtype=torch.long)


def save_learning_curve(
    train_losses: list[float],
    train_accs: list[float],
    val_losses: list[float],
    val_accs: list[float],
    checkpoint_dir: str,
    theme_metrics_path: str | None = None,
    ignored_themes: set[str] | None = None,
    theme_plot_include_missing: bool = False,
    per_source_train_loss: dict[str, list[float]] | None = None,
    per_source_train_acc: dict[str, list[float]] | None = None,
    per_source_val_loss: dict[str, list[float]] | None = None,
    per_source_val_acc: dict[str, list[float]] | None = None,
) -> None:
    if not train_losses or not val_losses:
        return
    epochs = list(range(1, len(train_losses) + 1))
    theme_curves = None
    if theme_metrics_path and os.path.exists(theme_metrics_path):
        theme_curves = _load_theme_curves(
            theme_metrics_path,
            epochs,
            ignored_themes=ignored_themes,
            include_missing=theme_plot_include_missing,
        )

    if theme_curves:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = list(axes)

    if per_source_train_loss and per_source_train_acc:
        labels = list(per_source_train_loss.keys())
        for idx, label in enumerate(labels):
            color = f"C{idx % 10}"
            axes[0].plot(
                epochs,
                per_source_train_loss.get(label, []),
                label=f"{label} (train)",
                color=color,
                linestyle="-",
            )
            if per_source_val_loss and per_source_val_loss.get(label):
                axes[0].plot(
                    epochs,
                    per_source_val_loss.get(label, []),
                    label=f"{label} (val)",
                    color=color,
                    linestyle="--",
                )
            axes[1].plot(
                epochs,
                per_source_train_acc.get(label, []),
                label=f"{label} (train)",
                color=color,
                linestyle="-",
            )
            if per_source_val_acc and per_source_val_acc.get(label):
                axes[1].plot(
                    epochs,
                    per_source_val_acc.get(label, []),
                    label=f"{label} (val)",
                    color=color,
                    linestyle="--",
                )

        axes[0].plot(
            epochs,
            train_losses,
            label="train (overall)",
            color="black",
            linestyle="-",
            linewidth=2,
        )
        axes[0].plot(
            epochs,
            val_losses,
            label="val (overall)",
            color="black",
            linestyle="--",
            linewidth=2,
        )
        axes[1].plot(
            epochs,
            [a * 100 for a in train_accs],
            label="train (overall)",
            color="black",
            linestyle="-",
            linewidth=2,
        )
        axes[1].plot(
            epochs,
            [a * 100 for a in val_accs],
            label="val (overall)",
            color="black",
            linestyle="--",
            linewidth=2,
        )
        axes[0].set_title("Loss by dataset")
        axes[1].set_title("Accuracy by dataset")
    else:
        axes[0].plot(epochs, train_losses, label="train", color="C0", linestyle="-")
        axes[0].plot(epochs, val_losses, label="val", color="C0", linestyle="--")
        axes[0].set_title("Loss")
        axes[1].plot(
            epochs,
            [a * 100 for a in train_accs],
            label="train",
            color="C1",
            linestyle="-",
        )
        axes[1].plot(
            epochs,
            [a * 100 for a in val_accs],
            label="val",
            color="C1",
            linestyle="--",
        )
        axes[1].set_title("Accuracy")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_ylim(0, 3)
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(50, 100)
    axes[1].legend()

    if theme_curves:
        theme_epochs, top1_5, top6_10 = theme_curves
        _plot_theme_group(
            axes[2],
            theme_epochs,
            top1_5,
            "Theme accuracy (top 1-5 by frequency)",
        )
        _plot_theme_group(
            axes[3],
            theme_epochs,
            top6_10,
            "Theme accuracy (top 6-10 by frequency)",
        )

    fig.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "learning_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # per-source curves are rendered in the main learning_curve plot


def _load_theme_curves(
    theme_metrics_path: str,
    epochs: list[int],
    ignored_themes: set[str] | None = None,
    include_missing: bool = False,
) -> tuple[list[int], list[tuple[str, list[float], list[float]]], list[tuple[str, list[float], list[float]]]] | None:
    theme_totals: dict[str, int] = {}
    theme_by_split: dict[str, dict[int, dict[str, float]]] = {"train": {}, "val": {}}
    with open(theme_metrics_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            split = payload.get("split")
            if split not in theme_by_split:
                continue
            epoch = int(payload.get("epoch", 0))
            themes = payload.get("themes") or []
            epoch_map: dict[str, float] = {}
            for entry in themes:
                theme = entry.get("theme")
                total = int(entry.get("total", 0))
                acc = float(entry.get("acc", 0.0))
                if not theme:
                    continue
                if ignored_themes and theme in ignored_themes:
                    continue
                epoch_map[theme] = acc
                theme_totals[theme] = max(theme_totals.get(theme, 0), total)
            theme_by_split[split][epoch] = epoch_map

    if not theme_totals:
        return None

    sorted_themes = sorted(theme_totals.items(), key=lambda item: (-item[1], item[0]))
    top_themes = [name for name, _ in sorted_themes[:10]]
    all_epochs = set()
    for split_map in theme_by_split.values():
        all_epochs.update(split_map.keys())
    theme_epochs = epochs if include_missing else sorted(all_epochs)
    theme_series: list[tuple[str, list[float], list[float]]] = []
    for theme in top_themes:
        train_series = []
        val_series = []
        for epoch in theme_epochs:
            train_series.append(
                theme_by_split["train"].get(epoch, {}).get(theme, float("nan"))
            )
            val_series.append(
                theme_by_split["val"].get(epoch, {}).get(theme, float("nan"))
            )
        theme_series.append((theme, train_series, val_series))

    return theme_epochs, theme_series[:5], theme_series[5:10]


def _plot_theme_group(
    axis: plt.Axes,
    epochs: list[int],
    theme_series: list[tuple[str, list[float], list[float]]],
    title: str,
) -> None:
    for idx, (theme, train_series, val_series) in enumerate(theme_series):
        color = f"C{idx % 10}"
        axis.plot(
            epochs,
            train_series,
            label=f"{theme} (train)",
            color=color,
            linestyle="-",
        )
        axis.plot(
            epochs,
            val_series,
            label=f"{theme} (val)",
            color=color,
            linestyle="--",
        )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(None, 100)
    if theme_series:
        axis.legend(fontsize=8)


def _update_theme_stats(
    theme_stats: dict, themes_batch: list[list[str]], correct_batch: list[bool]
) -> None:
    for themes, is_correct in zip(themes_batch, correct_batch):
        if not themes:
            continue
        for theme in set(themes):
            stats = theme_stats.setdefault(theme, {"total": 0, "correct": 0})
            stats["total"] += 1
            if is_correct:
                stats["correct"] += 1


def _format_theme_stats(theme_stats: dict) -> list[tuple[str, int, int, float]]:
    rows = []
    for theme, stats in theme_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = (correct / total * 100) if total else 0.0
        rows.append((theme, total, correct, acc))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows


def _collate_with_themes(batch):
    obs, target, value_target, policy_mask, source_id, themes = zip(*batch)
    collated = default_collate(list(zip(obs, target, value_target, policy_mask, source_id)))
    return collated[0], collated[1], collated[2], collated[3], collated[4], list(themes)


def _setup_ddp(rank: int, world_size: int) -> None:
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _ddp_all_reduce(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def _ddp_all_reduce_list(values: list[int], device: torch.device) -> list[int]:
    tensor = torch.tensor(values, device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def _ddp_all_reduce_float_list(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def _train_worker(rank: int, world_size: int, args) -> None:
    ddp_enabled = world_size > 1
    is_main = rank == 0
    cfg = OmegaConf.load(args.config)
    if ddp_enabled:
        torch.cuda.set_device(rank)
        _setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = select_device(args.device or cfg.training.get("device", "auto"))

    source_labels = ["m1", "m2", "m3", "m4", "m5", "m6plus", "endgame"]
    num_sources = len(source_labels)
    skip_value_sources = {s for s in args.skip_value_sources if s}
    skip_value_source_ids = {
        idx
        for idx, label in enumerate(source_labels)
        if label in skip_value_sources
    }
    skip_value_ids_tensor = None
    if skip_value_source_ids:
        skip_value_ids_tensor = torch.tensor(
            sorted(skip_value_source_ids), device=device, dtype=torch.long
        )
    per_source_train_loss_history = {label: [] for label in source_labels}
    per_source_train_acc_history = {label: [] for label in source_labels}
    per_source_val_loss_history = {label: [] for label in source_labels}
    per_source_val_acc_history = {label: [] for label in source_labels}

    total_rows = None
    json_only = True
    if is_main:
        total_rows = 0
        for data_path in args.data_paths:
            lower_path = data_path.lower()
            if lower_path.endswith(".json"):
                data_lines = fast_count_json_lines(data_path)
                print(
                    f"Total lines: {data_lines + 2} | Data lines: {data_lines} | Invalid: 0 (fast count)"
                )
                total_rows += data_lines
            elif lower_path.endswith(".jsonl"):
                data_lines = fast_count_jsonl_lines(data_path)
                print(
                    f"Total lines: {data_lines} | Data lines: {data_lines} | Invalid: 0 (fast count)"
                )
                total_rows += data_lines
            else:
                json_only = False
        if not json_only:
            total_rows = None
    if ddp_enabled:
        total_rows_tensor = torch.tensor(
            -1 if total_rows is None else total_rows, device=device, dtype=torch.long
        )
        dist.broadcast(total_rows_tensor, src=0)
        total_rows = None if total_rows_tensor.item() < 0 else int(total_rows_tensor.item())

    batch_size = args.batch_size or cfg.training.batch_size
    learning_rate = args.learning_rate or cfg.optimizer.learning_rate
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg.optimizer.weight_decay
    if args.policy_dropout is not None:
        cfg.network.policy_dropout = float(args.policy_dropout)
    if args.value_dropout is not None:
        cfg.network.value_dropout = float(args.value_dropout)
    train_dataset = MateInOneIterableDataset(
        cfg,
        args.data_paths,
        split="train",
        val_split=args.val_split,
        seed=args.seed,
        max_rows=args.max_rows,
        rank=rank,
        world_size=world_size,
    )
    val_dataset = MateInOneIterableDataset(
        cfg,
        args.data_paths,
        split="val",
        val_split=args.val_split,
        seed=args.seed,
        max_rows=args.max_rows,
        rank=rank,
        world_size=world_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_with_themes,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_with_themes,
    )

    model = create_chess_network(cfg, device)
    if ddp_enabled:
        model = DDP(model, device_ids=[rank])
    model.train()
    model_ref = model.module if ddp_enabled else model

    if is_main:
        try:
            dummy_input = torch.zeros(
                1,
                cfg.network.input_channels,
                cfg.network.board_size,
                cfg.network.board_size,
                device=device,
            )
            profile_model(model_ref, inputs=(dummy_input,))
        except Exception as exc:
            print(f"Warning: Model profiling failed: {exc}")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_patience, factor=args.lr_factor
    )
    criterion = nn.CrossEntropyLoss()

    run_id = time.strftime("%Y-%m-%d/%H-%M-%S")
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_id)
    if is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
    theme_metrics_path = os.path.join(checkpoint_dir, "theme_metrics.jsonl")
    ignored_themes = {t.strip() for t in args.theme_ignore if t and t.strip()}

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    train_accs: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []
    if args.resume:
        resume_data = load_checkpoint(args.resume, model_ref, optimizer, scheduler, device)
        start_epoch = resume_data.get("epoch", 0) + 1
        best_val_loss = resume_data.get("best_val_loss", best_val_loss)
        patience_counter = resume_data.get("patience_counter", patience_counter)
        train_losses = list(resume_data.get("train_losses", train_losses))
        train_accs = list(resume_data.get("train_accs", train_accs))
        val_losses = list(resume_data.get("val_losses", val_losses))
        val_accs = list(resume_data.get("val_accs", val_accs))
        if is_main:
            print(f"Resumed from {args.resume} (next epoch={start_epoch}).")
        if start_epoch > args.epochs:
            if is_main:
                print(
                    f"Checkpoint epoch={start_epoch - 1} exceeds requested epochs={args.epochs}. "
                    "Nothing to train."
                )
            if ddp_enabled:
                _cleanup_ddp()
            return

    if is_main:
        max_rows_label = args.max_rows if args.max_rows is not None else "all"
        joined_paths = ", ".join(args.data_paths)
        print(f"Streaming data from {joined_paths} (max_rows={max_rows_label}).")
        print(f"Split: val={args.val_split:.2f} | seed={args.seed}")
        cpu_cores = os.cpu_count() or 0
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        total_workers = args.num_workers * world_size if ddp_enabled else args.num_workers
        print(
            "Resources | "
            f"cpu_cores={cpu_cores} gpus={gpus} "
            f"num_workers={args.num_workers} total_workers={total_workers}"
        )
        print(f"Training on {device} | batch_size={batch_size} | epochs={args.epochs}")
        if ddp_enabled:
            print(f"DDP enabled | world_size={world_size}")
        if args.amp:
            amp_status = (
                "enabled" if device.type in {"cuda", "mps"} else "disabled (non-CUDA)"
            )
            print(f"AMP: {amp_status}")

    amp_enabled = bool(args.amp and device.type in {"cuda", "mps"})
    amp_device = "cuda" if device.type == "cuda" else "mps"
    scaler = GradScaler(amp_device, enabled=args.amp and device.type == "cuda")

    train_total = None
    val_total = None
    if total_rows is not None:
        effective_rows = total_rows
        if args.max_rows is not None:
            effective_rows = min(effective_rows, args.max_rows)
        if args.val_split <= 0:
            val_count = 0
        elif args.val_split >= 1:
            val_count = effective_rows
        else:
            val_count = int(effective_rows * args.val_split)
        train_count = max(effective_rows - val_count, 0)
    else:
        train_count = count_dataset_entries(train_dataset)
        val_count = count_dataset_entries(val_dataset)
    train_total = math.ceil(train_count / batch_size) if train_count else 0
    val_total = math.ceil(val_count / batch_size) if val_count else 0
    if is_main:
        print(f"Total data entries | train={train_count} | val={val_count}")
        print(f"Total batches      | train={train_total} | val={val_total}")

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        TextColumn(
            "loss={task.fields[loss]:.4f} "
            "pol={task.fields[pol]:.4f} val={task.fields[val]:.4f} "
            "acc={task.fields[acc]:.2f}%"
        ),
        TextColumn("src={task.fields[src]}"),
    )

    progress_manager = (
        NullProgress() if args.no_progress or not is_main else Progress(*progress_columns, transient=False)
    )
    with progress_manager as progress:
        epoch_task = progress.add_task(
            "Epochs",
            total=args.epochs,
            completed=start_epoch - 1,
            loss=float("nan"),
            pol=float("nan"),
            val=float("nan"),
            acc=float("nan"),
            src="-",
        )
        train_task = progress.add_task(
            "Train",
            total=train_total,
            loss=float("nan"),
            pol=float("nan"),
            val=float("nan"),
            acc=float("nan"),
            src="-",
        )
        val_task = progress.add_task(
            "Val",
            total=val_total,
            loss=float("nan"),
            pol=float("nan"),
            val=float("nan"),
            acc=float("nan"),
            src="-",
        )
        last_val_loss = None
        last_val_acc = None

        collect_theme_stats = is_main and not ddp_enabled

        for epoch in range(start_epoch, args.epochs + 1):
            total_loss = 0.0
            total_samples = 0
            policy_loss_sum = 0.0
            policy_samples = 0
            value_loss_sum = 0.0
            value_samples = 0
            correct = 0
            total = 0
            train_theme_stats: dict[str, dict[str, int]] = {}
            per_source_correct = [0 for _ in range(num_sources)]
            per_source_total = [0 for _ in range(num_sources)]
            per_source_loss_sum = [0.0 for _ in range(num_sources)]
            progress.reset(
                train_task,
                total=train_total,
                completed=0,
                loss=float("nan"),
                pol=float("nan"),
                val=float("nan"),
                acc=float("nan"),
                src="-",
            )
            progress.update(train_task, description=f"Train (epoch {epoch})")
            progress.reset(
                val_task,
                total=val_total,
                completed=0,
                loss=float("nan"),
                pol=float("nan"),
                val=float("nan"),
                acc=float("nan"),
                src="-",
            )
            progress.update(val_task, description="Val")
            batch_idx = 0
            for obs, target, value_target, policy_mask, source_ids, themes in train_loader:
                batch_idx += 1
                obs = obs.to(device)
                target = target.to(device)
                value_target = value_target.to(device)
                policy_mask = policy_mask.to(device)
                source_ids = source_ids.to(device)
                optimizer.zero_grad()
                with autocast(amp_device, enabled=amp_enabled):
                    logits, value = model(obs)
                    value = value.view(-1)
                    if skip_value_ids_tensor is not None:
                        value_mask = ~torch.isin(source_ids, skip_value_ids_tensor)
                        if value_mask.any():
                            value_loss = F.mse_loss(value[value_mask], value_target[value_mask])
                        else:
                            value_loss = torch.tensor(0.0, device=device)
                    else:
                        value_mask = None
                        value_loss = F.mse_loss(value, value_target)
                    if policy_mask.any():
                        policy_logits = logits[policy_mask]
                        policy_target = target[policy_mask]
                        policy_loss = criterion(policy_logits, policy_target)
                        per_sample_loss = F.cross_entropy(
                            policy_logits, policy_target, reduction="none"
                        )
                    else:
                        policy_loss = torch.tensor(0.0, device=device)
                        per_sample_loss = None
                    loss = policy_loss + value_loss
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                batch_size = obs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                if value_mask is None:
                    value_loss_sum += value_loss.item() * batch_size
                    value_samples += batch_size
                else:
                    masked_count = value_mask.sum().item()
                    value_loss_sum += value_loss.item() * masked_count
                    value_samples += masked_count
                preds = torch.argmax(logits, dim=1)
                if policy_mask.any():
                    policy_preds = preds[policy_mask]
                    policy_target = target[policy_mask]
                    policy_source_ids = source_ids[policy_mask]
                    correct += (policy_preds == policy_target).sum().item()
                    total += policy_target.numel()
                    policy_loss_sum += policy_loss.item() * policy_target.numel()
                    policy_samples += policy_target.numel()
                    for source_idx in range(num_sources):
                        mask = policy_source_ids == source_idx
                        count = mask.sum().item()
                        if count:
                            per_source_total[source_idx] += count
                            per_source_correct[source_idx] += (
                                policy_preds[mask] == policy_target[mask]
                            ).sum().item()
                            if per_sample_loss is not None:
                                per_source_loss_sum[source_idx] += per_sample_loss[
                                    mask
                                ].sum().item()
                if collect_theme_stats and policy_mask.any():
                    policy_mask_cpu = policy_mask.detach().cpu().tolist()
                    policy_themes = [t for t, keep in zip(themes, policy_mask_cpu) if keep]
                    correct_batch = (preds[policy_mask] == target[policy_mask]).detach().cpu().tolist()
                    _update_theme_stats(train_theme_stats, policy_themes, correct_batch)
                avg_loss = total_loss / max(total_samples, 1)
                avg_policy_loss = policy_loss_sum / max(policy_samples, 1)
                avg_value_loss = value_loss_sum / max(value_samples, 1)
                acc = correct / max(total, 1)
                per_source_train = [
                    f"{label}={per_source_correct[i] / per_source_total[i] * 100:.1f}%"
                    for i, label in enumerate(source_labels)
                    if per_source_total[i] > 0
                ]
                per_source_train_str = " | ".join(per_source_train) if per_source_train else "-"
                progress.update(
                    train_task,
                    advance=1,
                    loss=avg_loss,
                    pol=avg_policy_loss,
                    val=avg_value_loss,
                    acc=acc * 100,
                    src=per_source_train_str,
                )

            epoch_loss_sum = total_loss
            epoch_loss_samples = total_samples
            epoch_policy_loss_sum = policy_loss_sum
            epoch_policy_samples = policy_samples
            epoch_value_loss_sum = value_loss_sum
            epoch_value_samples = value_samples
            epoch_correct = correct
            epoch_total = total
            if ddp_enabled:
                epoch_loss_sum = _ddp_all_reduce(epoch_loss_sum, device)
                epoch_loss_samples = _ddp_all_reduce(epoch_loss_samples, device)
                epoch_policy_loss_sum = _ddp_all_reduce(epoch_policy_loss_sum, device)
                epoch_policy_samples = _ddp_all_reduce(epoch_policy_samples, device)
                epoch_value_loss_sum = _ddp_all_reduce(epoch_value_loss_sum, device)
                epoch_value_samples = _ddp_all_reduce(epoch_value_samples, device)
                epoch_correct = _ddp_all_reduce(epoch_correct, device)
                epoch_total = _ddp_all_reduce(epoch_total, device)
                per_source_correct = _ddp_all_reduce_list(per_source_correct, device)
                per_source_total = _ddp_all_reduce_list(per_source_total, device)
                per_source_loss_sum = _ddp_all_reduce_float_list(
                    per_source_loss_sum, device
                )

            avg_loss = epoch_loss_sum / max(epoch_loss_samples, 1)
            avg_policy_loss = epoch_policy_loss_sum / max(epoch_policy_samples, 1)
            avg_value_loss = epoch_value_loss_sum / max(epoch_value_samples, 1)
            acc = epoch_correct / max(epoch_total, 1)
            progress.update(
                epoch_task,
                advance=1,
                loss=avg_loss,
                pol=avg_policy_loss,
                val=avg_value_loss,
                acc=acc * 100,
            )

            model.eval()
            val_loss = 0.0
            val_loss_samples = 0
            val_policy_loss_sum = 0.0
            val_policy_samples = 0
            val_value_loss_sum = 0.0
            val_value_samples = 0
            val_correct = 0
            val_samples = 0
            val_theme_stats: dict[str, dict[str, int]] = {}
            val_per_source_correct = [0 for _ in range(num_sources)]
            val_per_source_total = [0 for _ in range(num_sources)]
            val_per_source_loss_sum = [0.0 for _ in range(num_sources)]
            with torch.no_grad():
                for obs, target, value_target, policy_mask, source_ids, themes in val_loader:
                    obs = obs.to(device)
                    target = target.to(device)
                    value_target = value_target.to(device)
                    policy_mask = policy_mask.to(device)
                    source_ids = source_ids.to(device)
                    with autocast(amp_device, enabled=amp_enabled):
                        logits, value = model(obs)
                        value = value.view(-1)
                        if skip_value_ids_tensor is not None:
                            value_mask = ~torch.isin(source_ids, skip_value_ids_tensor)
                            if value_mask.any():
                                value_loss = F.mse_loss(
                                    value[value_mask], value_target[value_mask]
                                )
                            else:
                                value_loss = torch.tensor(0.0, device=device)
                        else:
                            value_mask = None
                            value_loss = F.mse_loss(value, value_target)
                        if policy_mask.any():
                            policy_logits = logits[policy_mask]
                            policy_target = target[policy_mask]
                            policy_loss = criterion(policy_logits, policy_target)
                            per_sample_loss = F.cross_entropy(
                                policy_logits, policy_target, reduction="none"
                            )
                        else:
                            policy_loss = torch.tensor(0.0, device=device)
                            per_sample_loss = None
                        loss = policy_loss + value_loss
                    batch_size = obs.size(0)
                    val_loss += loss.item() * batch_size
                    val_loss_samples += batch_size
                    if value_mask is None:
                        val_value_loss_sum += value_loss.item() * batch_size
                        val_value_samples += batch_size
                    else:
                        masked_count = value_mask.sum().item()
                        val_value_loss_sum += value_loss.item() * masked_count
                        val_value_samples += masked_count
                    preds = torch.argmax(logits, dim=1)
                    if policy_mask.any():
                        policy_preds = preds[policy_mask]
                        policy_target = target[policy_mask]
                        policy_source_ids = source_ids[policy_mask]
                        val_correct += (policy_preds == policy_target).sum().item()
                        val_samples += policy_target.numel()
                        val_policy_loss_sum += policy_loss.item() * policy_target.numel()
                        val_policy_samples += policy_target.numel()
                        for source_idx in range(num_sources):
                            mask = policy_source_ids == source_idx
                            count = mask.sum().item()
                            if count:
                                val_per_source_total[source_idx] += count
                                val_per_source_correct[source_idx] += (
                                    policy_preds[mask] == policy_target[mask]
                                ).sum().item()
                                if per_sample_loss is not None:
                                    val_per_source_loss_sum[source_idx] += per_sample_loss[
                                        mask
                                    ].sum().item()
                    if collect_theme_stats and policy_mask.any():
                        policy_mask_cpu = policy_mask.detach().cpu().tolist()
                        policy_themes = [t for t, keep in zip(themes, policy_mask_cpu) if keep]
                        correct_batch = (preds[policy_mask] == target[policy_mask]).detach().cpu().tolist()
                        _update_theme_stats(val_theme_stats, policy_themes, correct_batch)
                    val_avg_loss = val_loss / max(val_loss_samples, 1)
                    val_avg_policy_loss = val_policy_loss_sum / max(val_policy_samples, 1)
                    val_avg_value_loss = val_value_loss_sum / max(val_value_samples, 1)
                    val_acc = val_correct / max(val_samples, 1)
                    per_source_val = [
                        f"{label}={val_per_source_correct[i] / val_per_source_total[i] * 100:.1f}%"
                        for i, label in enumerate(source_labels)
                        if val_per_source_total[i] > 0
                    ]
                    per_source_val_str = " | ".join(per_source_val) if per_source_val else "-"
                    progress.update(
                        val_task,
                        advance=1,
                        loss=val_avg_loss,
                        pol=val_avg_policy_loss,
                        val=val_avg_value_loss,
                        acc=val_acc * 100,
                        src=per_source_val_str,
                    )
            model.train()

            train_losses.append(avg_loss)
            train_accs.append(acc)

            has_val = val_samples > 0
            if ddp_enabled:
                val_loss = _ddp_all_reduce(val_loss, device)
                val_loss_samples = _ddp_all_reduce(val_loss_samples, device)
                val_policy_loss_sum = _ddp_all_reduce(val_policy_loss_sum, device)
                val_policy_samples = _ddp_all_reduce(val_policy_samples, device)
                val_value_loss_sum = _ddp_all_reduce(val_value_loss_sum, device)
                val_value_samples = _ddp_all_reduce(val_value_samples, device)
                val_correct = _ddp_all_reduce(val_correct, device)
                val_samples = _ddp_all_reduce(val_samples, device)
                val_per_source_correct = _ddp_all_reduce_list(val_per_source_correct, device)
                val_per_source_total = _ddp_all_reduce_list(val_per_source_total, device)
                val_per_source_loss_sum = _ddp_all_reduce_float_list(
                    val_per_source_loss_sum, device
                )
            if has_val:
                val_avg_loss = val_loss / max(val_loss_samples, 1)
                val_avg_policy_loss = val_policy_loss_sum / max(val_policy_samples, 1)
                val_avg_value_loss = val_value_loss_sum / max(val_value_samples, 1)
                val_acc = val_correct / max(val_samples, 1)
                val_losses.append(val_avg_loss)
                val_accs.append(val_acc)
                if is_main:
                    per_source_train = [
                        f"{label}={per_source_correct[i] / per_source_total[i] * 100:.2f}%"
                        for i, label in enumerate(source_labels)
                        if per_source_total[i] > 0
                    ]
                    per_source_val = [
                        f"{label}={val_per_source_correct[i] / val_per_source_total[i] * 100:.2f}%"
                        for i, label in enumerate(source_labels)
                        if val_per_source_total[i] > 0
                    ]
                    per_source_train_str = " | ".join(per_source_train) if per_source_train else "N/A"
                    per_source_val_str = " | ".join(per_source_val) if per_source_val else "N/A"
                    print(
                        f"Epoch {epoch}/{args.epochs} | "
                        f"train loss={avg_loss:.4f} pol={avg_policy_loss:.4f} val={avg_value_loss:.4f} "
                        f"acc={acc*100:.2f}% | "
                        f"val loss={val_avg_loss:.4f} pol={val_avg_policy_loss:.4f} "
                        f"val={val_avg_value_loss:.4f} acc={val_acc*100:.2f}% | "
                        f"train acc by source: {per_source_train_str} | "
                        f"val acc by source: {per_source_val_str}"
                    )
                scheduler.step(val_avg_loss)

                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    patience_counter = 0
                    if is_main:
                        best_path = os.path.join(checkpoint_dir, "model_best.pth")
                        save_checkpoint(
                            best_path,
                            model_ref,
                            optimizer,
                            epoch,
                            cfg,
                            scheduler=scheduler,
                            best_val_loss=best_val_loss,
                            patience_counter=patience_counter,
                            train_losses=train_losses,
                            train_accs=train_accs,
                            val_losses=val_losses,
                            val_accs=val_accs,
                        )
                else:
                    patience_counter += 1
                    if patience_counter >= args.early_stop_patience:
                        if is_main:
                            print(f"Early stopping at epoch {epoch}.")
                        break
            else:
                if is_main:
                    per_source_train = [
                        f"{label}={per_source_correct[i] / per_source_total[i] * 100:.2f}%"
                        for i, label in enumerate(source_labels)
                        if per_source_total[i] > 0
                    ]
                    per_source_train_str = " | ".join(per_source_train) if per_source_train else "N/A"
                    print(
                        f"Epoch {epoch}/{args.epochs} | "
                        f"train loss={avg_loss:.4f} acc={acc*100:.2f}% | "
                        f"val loss=N/A acc=N/A (no validation batches) | "
                        f"train acc by source: {per_source_train_str}"
                    )

            for i, label in enumerate(source_labels):
                if per_source_total[i] > 0:
                    train_loss_value = per_source_loss_sum[i] / per_source_total[i]
                    train_acc_value = per_source_correct[i] / per_source_total[i] * 100
                else:
                    train_loss_value = float("nan")
                    train_acc_value = float("nan")
                per_source_train_loss_history[label].append(train_loss_value)
                per_source_train_acc_history[label].append(train_acc_value)

                if has_val and val_per_source_total[i] > 0:
                    val_loss_value = val_per_source_loss_sum[i] / val_per_source_total[i]
                    val_acc_value = val_per_source_correct[i] / val_per_source_total[i] * 100
                else:
                    val_loss_value = float("nan")
                    val_acc_value = float("nan")
                per_source_val_loss_history[label].append(val_loss_value)
                per_source_val_acc_history[label].append(val_acc_value)

            last_val_loss = val_avg_loss if has_val else None
            last_val_acc = val_acc if has_val else None
            if is_main and collect_theme_stats:
                with open(theme_metrics_path, "a", encoding="utf-8") as theme_out:
                    train_rows = _format_theme_stats(train_theme_stats)
                    train_payload = {
                        "epoch": epoch,
                        "split": "train",
                        "themes": [
                            {"theme": t, "total": total, "correct": correct, "acc": acc}
                            for t, total, correct, acc in train_rows
                        ],
                    }
                    theme_out.write(json.dumps(train_payload, ensure_ascii=True) + "\n")
                    if has_val:
                        val_rows = _format_theme_stats(val_theme_stats)
                        val_payload = {
                            "epoch": epoch,
                            "split": "val",
                            "themes": [
                                {"theme": t, "total": total, "correct": correct, "acc": acc}
                                for t, total, correct, acc in val_rows
                            ],
                        }
                        theme_out.write(json.dumps(val_payload, ensure_ascii=True) + "\n")

                min_theme_total = max(0, int(args.theme_log_min_total))
                if train_theme_stats:
                    print(f"Theme accuracy (train, total>={min_theme_total}):")
                    for theme, total, correct, acc in _format_theme_stats(train_theme_stats):
                        if theme in ignored_themes:
                            continue
                        if total < min_theme_total:
                            break
                        print(f"- {theme}: {correct}/{total} ({acc:.2f}%)")
                if has_val and val_theme_stats:
                    print(f"Theme accuracy (val, total>={min_theme_total}):")
                    for theme, total, correct, acc in _format_theme_stats(val_theme_stats):
                        if theme in ignored_themes:
                            continue
                        if total < min_theme_total:
                            break
                        print(f"- {theme}: {correct}/{total} ({acc:.2f}%)")
            elif is_main and ddp_enabled and epoch == start_epoch:
                print("DDP enabled: theme stats/plots are disabled to avoid partial metrics.")

            if is_main:
                save_learning_curve(
                    train_losses,
                    train_accs,
                    val_losses,
                    val_accs,
                    checkpoint_dir,
                    theme_metrics_path=theme_metrics_path if collect_theme_stats else None,
                    ignored_themes=ignored_themes,
                    theme_plot_include_missing=args.theme_plot_include_missing,
                    per_source_train_loss=per_source_train_loss_history,
                    per_source_train_acc=per_source_train_acc_history,
                    per_source_val_loss=per_source_val_loss_history if has_val else None,
                    per_source_val_acc=per_source_val_acc_history if has_val else None,
                )

            if args.save_every > 0 and epoch % args.save_every == 0 and is_main:
                if has_val:
                    ckpt_filename = (
                        f"model_epoch_{epoch}_val_{val_avg_loss:.4f}_acc_{val_acc*100:.2f}.pth"
                    )
                else:
                    ckpt_filename = f"model_epoch_{epoch}_noval.pth"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
                save_checkpoint(
                    ckpt_path,
                    model_ref,
                    optimizer,
                    epoch,
                    cfg,
                    scheduler=scheduler,
                    best_val_loss=best_val_loss,
                    patience_counter=patience_counter,
                    train_losses=train_losses,
                    train_accs=train_accs,
                    val_losses=val_losses,
                    val_accs=val_accs,
                )

    if is_main:
        if last_val_loss is not None and last_val_acc is not None:
            final_filename = (
                f"model_final_epoch_{epoch}_val_{last_val_loss:.4f}_acc_{last_val_acc*100:.2f}.pth"
            )
        else:
            final_filename = f"model_final_epoch_{epoch}_noval.pth"
        final_path = os.path.join(checkpoint_dir, final_filename)
        save_checkpoint(
            final_path,
            model_ref,
            optimizer,
            args.epochs,
            cfg,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            train_losses=train_losses,
            train_accs=train_accs,
            val_losses=val_losses,
            val_accs=val_accs,
        )
        print(f"Saved final checkpoint to {final_path}")
        if collect_theme_stats:
            save_learning_curve(
                train_losses,
                train_accs,
                val_losses,
                val_accs,
                checkpoint_dir,
                theme_metrics_path=theme_metrics_path,
                ignored_themes=ignored_themes,
                theme_plot_include_missing=args.theme_plot_include_missing,
                per_source_train_loss=per_source_train_loss_history,
                per_source_train_acc=per_source_train_acc_history,
                per_source_val_loss=per_source_val_loss_history if has_val else None,
                per_source_val_acc=per_source_val_acc_history if has_val else None,
            )

    if ddp_enabled:
        _cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Supervised training on puzzle positions.")
    parser.add_argument(
        "--data",
        "--csv",
        dest="data_paths",
        nargs="+",
        default=["data/mate_in_1_flipped_expanded.jsonl", "data/mate_in_2_flipped_expanded.jsonl", "data/mate_in_3_flipped_expanded.jsonl", "data/mate_in_4_flipped_expanded.jsonl", "data/mate_in_5_flipped_expanded.jsonl", "data/endgame_without_mate_flipped.json"],
        help="Paths to puzzle data (CSV or JSON). Pass multiple paths to mix datasets.",
    )
    parser.add_argument("--config", default="config/train_mcts.yaml", help="Config YAML for network settings.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--policy-dropout",
        type=float,
        default=0.25,
        help="Override cfg.network.policy_dropout when set.",
    )
    parser.add_argument(
        "--value-dropout",
        type=float,
        default=0.25,
        help="Override cfg.network.value_dropout when set.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--checkpoint-dir", default="outputs/supervised_train")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--num-workers",
        type=str,
        default="auto",
        help=(
            "DataLoader workers per GPU process (DDP). Use an int or 'auto'. "
            "Auto caps at 2: total workers = num_workers * num_gpus; keep total near CPU cores."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument(
        "--skip-value-sources",
        nargs="*",
        default=["endgame_without_mate_flipped"],
        help="Source labels to exclude from value loss (e.g., endgame_without_mate_flipped).",
    )
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA AMP (mixed precision) training.",
    )
    parser.add_argument(
        "--theme-log-min-total",
        type=int,
        default=100,
        help="Minimum theme count to log to console (default: 100).",
    )
    parser.add_argument(
        "--theme-ignore",
        nargs="*",
        default=["mate", "veryLong", "long", "short", "opening", "middlegame", "endgame"],
        help="Themes to exclude from theme logging/plots (default: mate, veryLong, long, short, opening, middlegame, endgame).",
    )
    parser.add_argument(
        "--theme-plot-include-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, keep missing epochs as gaps in theme plots (default: False).",
    )
    args = parser.parse_args()
    if args.num_workers == "auto":
        cpu_cores = os.cpu_count() or 1
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        args.num_workers = min(2, max(1, cpu_cores // max(1, num_gpus)))
    else:
        try:
            args.num_workers = int(args.num_workers)
        except ValueError as exc:
            raise ValueError("--num-workers must be an int or 'auto'.") from exc
    device_str = args.device or "auto"
    use_ddp = (
        device_str in {"auto", "cuda"}
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
        and args.num_workers > 1
    )
    if use_ddp:
        world_size = torch.cuda.device_count()
        mp.spawn(_train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        _train_worker(0, 1, args)


if __name__ == "__main__":
    main()
