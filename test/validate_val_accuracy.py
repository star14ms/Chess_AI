"""
Validate policy accuracy on the same validation split used in train_supervised.py,
using the same seed and config. Reports accuracy by data type: m1, m2, m3, m4, m5,
m6plus, and endgame by ply (endgame_ply_0, endgame_ply_1, ... for all plies present in data).

Usage (same config/seed as train_supervised; optional checkpoint):
  python validate_val_accuracy.py
  python validate_val_accuracy.py --checkpoint path/to/best.pth
  python validate_val_accuracy.py --checkpoint path/to/best.pth --device cuda:0
"""
import hashlib
import json
import math
import os
import random
from collections import defaultdict

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data._utils.collate import default_collate
import hydra
from omegaconf import DictConfig, OmegaConf

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import sys
import os
# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from chess_gym.chess_custom import LegacyChessBoard, FullyTrackedBoard
from MCTS.training_modules.chess import create_chess_network
from utils.training_utils import (
    count_dataset_entries,
    fast_count_json_lines,
    fast_count_jsonl_lines,
    iter_csv_rows,
    iter_json_array,
    select_device,
)


def create_board_from_fen(fen: str, action_space_size: int):
    if action_space_size == 4672:
        return LegacyChessBoard(fen)
    return FullyTrackedBoard(fen)


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


def _normalize_themes(themes_field) -> list:
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


def _source_id_from_moves_to_mate(moves_to_mate, is_endgame: bool = False) -> int:
    if is_endgame:
        return 6
    if moves_to_mate is None:
        return 6
    try:
        moves_to_mate = int(moves_to_mate)
    except (TypeError, ValueError):
        return 6
    if moves_to_mate <= 0:
        return 6
    mate_in = (moves_to_mate + 1) // 2
    if mate_in <= 1:
        return 0
    if mate_in == 2:
        return 1
    if mate_in == 3:
        return 2
    if mate_in == 4:
        return 3
    if mate_in == 5:
        return 4
    return 5


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
    policy_uci = (
        entry.get("policy_uci")
        or entry.get("policy_move")
        or entry.get("best")
        or entry.get("move")
    )
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
    return fen, float(value_target), policy_mask, policy_uci, themes, moves_to_mate


def _extract_themes(entry: dict, source_type: str) -> list:
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


def _entry_matches_split(entry: dict, split: str, val_split: float, seed: int) -> bool:
    if val_split <= 0:
        return split == "train"
    if val_split >= 1:
        return split == "val"
    key = (
        entry.get("PuzzleId")
        or entry.get("puzzle_id")
        or entry.get("FEN")
        or entry.get("fen")
    )
    if not key:
        key = json.dumps(entry, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    is_val = bucket < val_split
    return is_val if split == "val" else not is_val


class ValWithEndgamePlyDataset(IterableDataset):
    """Validation dataset that yields (obs, action_index, value_target, policy_mask, source_id, endgame_ply, themes).
    endgame_ply is 0/1/2 for endgame first/second/third move, else -1.
    Uses the same seed and val_split as train_supervised for identical validation split.
    """

    def __init__(
        self,
        cfg: DictConfig,
        data_paths: list,
        val_split: float,
        seed: int,
        max_rows: int | None = None,
    ):
        self.cfg = cfg
        self.data_paths = data_paths
        self.val_split = val_split
        self.seed = seed
        self.max_rows = max_rows

    def _entry_matches_split(self, entry: dict) -> bool:
        return _entry_matches_split(entry, "val", self.val_split, self.seed)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        global_worker_id = worker_id
        global_num_workers = num_workers
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
        rng = random.Random(self.seed)
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
            if not self._entry_matches_split(entry):
                continue
            if raw_index % global_num_workers != global_worker_id:
                raw_index += 1
                continue
            raw_index += 1

            if _is_processed_entry(entry, source_type):
                processed = _extract_processed_entry(entry, source_type)
                if processed is None:
                    continue
                (
                    fen,
                    value_target,
                    policy_mask,
                    policy_uci,
                    themes,
                    moves_to_mate,
                ) = processed
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
                    if (
                        action_id is None
                        or action_id < 1
                        or action_id > action_space_size
                    ):
                        continue
                    action_index = action_id - 1
                obs = board.get_board_vector(history_steps=history_steps)
                if obs.shape[0] != input_channels:
                    raise ValueError(
                        f"Observation channels {obs.shape[0]} != expected {input_channels}."
                    )
                resolved_source_id = _source_id_from_moves_to_mate(
                    moves_to_mate, is_endgame=source_is_endgame
                )
                if source_is_endgame:
                    try:
                        endgame_ply = max(0, int(entry.get("endgame_ply", 0)))
                    except (TypeError, ValueError):
                        endgame_ply = 0
                else:
                    endgame_ply = -1
                yield (
                    torch.from_numpy(obs.astype(np.float32, copy=False)).float(),
                    torch.tensor(action_index, dtype=torch.long),
                    torch.tensor(value_target, dtype=torch.float32),
                    torch.tensor(policy_mask, dtype=torch.bool),
                    torch.tensor(resolved_source_id, dtype=torch.long),
                    torch.tensor(endgame_ply, dtype=torch.long),
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
            parsed_moves = []
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
                action_id = board.move_to_action_id(move)
                if (
                    action_id is None
                    or action_id < 1
                    or action_id > action_space_size
                ):
                    valid = False
                    break
                action_index = action_id - 1
                obs = board.get_board_vector(history_steps=history_steps)
                if obs.shape[0] != input_channels:
                    raise ValueError(
                        f"Observation channels {obs.shape[0]} != expected {input_channels}."
                    )
                moves_to_mate = total_plies - ply_index
                resolved_source_id = _source_id_from_moves_to_mate(
                    moves_to_mate, is_endgame=source_is_endgame
                )
                if source_is_endgame and ply_index <= 2:
                    endgame_ply = ply_index
                else:
                    endgame_ply = -1
                yield (
                    torch.from_numpy(obs.astype(np.float32, copy=False)).float(),
                    torch.tensor(action_index, dtype=torch.long),
                    torch.tensor(
                        1.0 if ply_index % 2 == 0 else -1.0, dtype=torch.float32
                    ),
                    torch.tensor(policy_mask, dtype=torch.bool),
                    torch.tensor(resolved_source_id, dtype=torch.long),
                    torch.tensor(endgame_ply, dtype=torch.long),
                    themes,
                )
                yielded += 1
                if max_rows is not None and yielded >= max_rows:
                    break
                board.push(move)
            if max_rows is not None and yielded >= max_rows:
                break


def collate_val_with_endgame_ply(batch):
    obs, target, value_target, policy_mask, source_id, endgame_ply, themes = zip(
        *batch
    )
    collated = default_collate(
        list(zip(obs, target, value_target, policy_mask, source_id, endgame_ply))
    )
    return (
        collated[0],
        collated[1],
        collated[2],
        collated[3],
        collated[4],
        collated[5],
        list(themes),
    )


# source_id: 0=m1, 1=m2, 2=m3, 3=m4, 4=m5, 5=m6plus, 6=endgame
SOURCE_LABELS = ["m1", "m2", "m3", "m4", "m5", "m6plus", "endgame"]


def _endgame_ply_label(ply: int) -> str:
    """Label for endgame-by-move reporting (e.g. end_move1, end_move2) to distinguish from mate m1, m2."""
    return f"end_move{ply + 1}"


def _count_val_batches_and_samples(
    cfg: DictConfig,
    batch_size: int,
    max_rows: int | None,
    val_split: float,
    data_paths: list,
) -> tuple[int, int]:
    """Same fast count as train_supervised: line counts for .json/.jsonl, else one dataset pass."""
    total_rows = None
    json_only = True
    total_rows = 0
    for data_path in data_paths:
        lower_path = data_path.lower()
        if lower_path.endswith(".json"):
            total_rows += fast_count_json_lines(data_path)
        elif lower_path.endswith(".jsonl"):
            total_rows += fast_count_jsonl_lines(data_path)
        else:
            json_only = False
    if not json_only:
        total_rows = None

    if total_rows is not None:
        effective_rows = total_rows
        if max_rows is not None:
            effective_rows = min(effective_rows, max_rows)
        if val_split <= 0:
            val_count = 0
        elif val_split >= 1:
            val_count = effective_rows
        else:
            val_count = int(effective_rows * val_split)
        total_data = val_count
        total_batches = math.ceil(val_count / batch_size) if val_count else 0
        return total_batches, total_data

    val_dataset = ValWithEndgamePlyDataset(
        cfg,
        data_paths,
        val_split=val_split,
        seed=cfg.supervised.seed,
        max_rows=max_rows,
    )
    val_count = count_dataset_entries(val_dataset)
    total_batches = math.ceil(val_count / batch_size) if val_count else 0
    return total_batches, val_count


def run_validation(
    cfg: DictConfig,
    checkpoint_path: str | None,
    device: torch.device,
    num_workers_override: int | None = None,
    batch_size_override: int | None = None,
):
    supervised_cfg = cfg.supervised
    default_batch = (
        supervised_cfg.val_batch_size
        if getattr(supervised_cfg, "val_batch_size", None) is not None
        else supervised_cfg.batch_size
    )
    batch_size = batch_size_override if batch_size_override is not None else default_batch
    max_rows = supervised_cfg.max_rows

    if num_workers_override is not None:
        num_workers = num_workers_override
    elif getattr(supervised_cfg, "num_workers", None) == "auto":
        cpu_cores = os.cpu_count() or 1
        num_workers = max(1, cpu_cores - 1)
    else:
        try:
            num_workers = int(getattr(supervised_cfg, "num_workers", 0) or 0)
        except (TypeError, ValueError):
            num_workers = 0

    print("Counting validation data...")
    total_batches, total_data = _count_val_batches_and_samples(
        cfg,
        batch_size,
        max_rows,
        supervised_cfg.val_split,
        supervised_cfg.data_paths,
    )
    val_dataset = ValWithEndgamePlyDataset(
        cfg,
        supervised_cfg.data_paths,
        val_split=supervised_cfg.val_split,
        seed=supervised_cfg.seed,
        max_rows=max_rows,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_val_with_endgame_ply,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = create_chess_network(cfg, device)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint loaded; reporting accuracy with randomly initialized model.")

    # Use all available CUDA devices if possible
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        n_accel = torch.cuda.device_count()
        compute_str = f"{n_accel} CUDA" + (f" + {num_workers} CPU workers" if num_workers else "")
    elif device.type == "cuda":
        compute_str = "1 CUDA" + (f" + {num_workers} CPU workers" if num_workers else "")
    elif device.type == "mps":
        compute_str = "1 MPS" + (f" + {num_workers} CPU workers" if num_workers else "")
    else:
        compute_str = f"{num_workers} CPU workers" if num_workers else "1 CPU"

    print(
        f"Validation: {total_data:,} samples, {total_batches} batches "
        f"(batch_size={batch_size}) | compute: {compute_str}"
    )

    model.eval()
    criterion = nn.CrossEntropyLoss()
    skip_value_sources = {s for s in supervised_cfg.skip_value_sources if s}
    skip_value_source_ids = {
        idx
        for idx, label in enumerate(SOURCE_LABELS)
        if label in skip_value_sources
    }
    skip_value_ids_tensor = None
    if skip_value_source_ids:
        skip_value_ids_tensor = torch.tensor(
            sorted(skip_value_source_ids), device=device, dtype=torch.long
        )

    # Per source_id (m1..m5, m6plus, endgame)
    correct_by_source = {i: 0 for i in range(7)}
    total_by_source = {i: 0 for i in range(7)}
    # Endgame by ply (dynamic: any ply >= 0 that appears in data)
    correct_by_endgame_move = defaultdict(int)
    total_by_endgame_move = defaultdict(int)

    total_correct = 0
    total_samples = 0
    policy_loss_sum = 0.0
    policy_samples = 0
    value_loss_sum = 0.0
    value_samples = 0
    batch_idx = 0

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn(
            "• pol={task.fields[pol]:.4f} val={task.fields[val]:.4f} "
            "acc={task.fields[acc]:.1f}%"
        ),
        TextColumn("• {task.fields[mate_acc]}"),
        TextColumn("• {task.fields[endgame_acc]}"),
    )
    with Progress(*progress_columns) as progress:
        val_task = progress.add_task(
            "Validating",
            total=total_batches,
            completed=0,
            pol=0.0,
            val=0.0,
            acc=0.0,
            mate_acc="-",
            endgame_acc="-",
        )
        with torch.no_grad():
            for (
                obs,
                target,
                value_target,
                policy_mask,
                source_ids,
                endgame_ply,
                themes,
            ) in val_loader:
                obs = obs.to(device)
                target = target.to(device)
                value_target = value_target.to(device)
                source_ids = source_ids.to(device)
                endgame_ply = endgame_ply.to(device)
                batch_size = obs.size(0)

                logits, value = model(obs)
                value = value.view(-1)
                preds = torch.argmax(logits, dim=1)

                if skip_value_ids_tensor is not None:
                    value_mask = ~torch.isin(source_ids, skip_value_ids_tensor)
                    value_samples_this = value_mask.sum().item()
                    if value_mask.any():
                        value_loss = F.mse_loss(
                            value[value_mask], value_target[value_mask]
                        )
                    else:
                        value_loss = torch.tensor(0.0, device=device)
                else:
                    value_loss = F.mse_loss(value, value_target)
                    value_samples_this = batch_size
                value_loss_sum += value_loss.item() * value_samples_this
                value_samples += value_samples_this

                if policy_mask.any():
                    policy_logits = logits[policy_mask]
                    policy_target = target[policy_mask]
                    policy_loss = criterion(policy_logits, policy_target)
                    policy_loss_sum += policy_loss.item() * policy_target.numel()
                    policy_samples += policy_target.numel()
                else:
                    policy_loss = torch.tensor(0.0, device=device)

                if policy_mask.any():
                    mask = policy_mask.to(device)
                    policy_preds = preds[mask]
                    policy_target_masked = target[mask]
                    policy_source_ids = source_ids[mask]
                    policy_endgame_ply = endgame_ply[mask]

                    total_correct += (
                        (policy_preds == policy_target_masked).sum().item()
                    )
                    total_samples += policy_target_masked.numel()

                    for i in range(7):
                        src_mask = policy_source_ids == i
                        count = src_mask.sum().item()
                        if count:
                            total_by_source[i] += count
                            correct_by_source[i] += (
                                (
                                    policy_preds[src_mask]
                                    == policy_target_masked[src_mask]
                                )
                                .sum()
                                .item()
                            )

                    endgame_mask = policy_source_ids == 6
                    for ply_val in policy_endgame_ply[endgame_mask].unique().cpu().tolist():
                        ply = int(ply_val)
                        if ply < 0:
                            continue
                        move_mask = endgame_mask & (policy_endgame_ply == ply)
                        count = move_mask.sum().item()
                        if count:
                            total_by_endgame_move[ply] += count
                            correct_by_endgame_move[ply] += (
                                (
                                    policy_preds[move_mask]
                                    == policy_target_masked[move_mask]
                                )
                                .sum()
                                .item()
                            )

                batch_idx += 1
                avg_pol = (
                    policy_loss_sum / policy_samples if policy_samples else 0.0
                )
                avg_val = value_loss_sum / value_samples if value_samples else 0.0
                acc_so_far = (
                    total_correct / total_samples * 100 if total_samples else 0.0
                )
                mate_acc_parts = []
                for i in range(5):
                    t = total_by_source[i]
                    c = correct_by_source[i]
                    label = SOURCE_LABELS[i]
                    if t:
                        mate_acc_parts.append(
                            f"{label}={c / t * 100:.1f}%"
                        )
                    else:
                        mate_acc_parts.append(f"{label}=-")
                mate_acc_str = " ".join(mate_acc_parts) if mate_acc_parts else "-"
                endgame_acc_parts = []
                for ply in sorted(total_by_endgame_move.keys()):
                    t = total_by_endgame_move[ply]
                    c = correct_by_endgame_move[ply]
                    label = _endgame_ply_label(ply)
                    if t:
                        endgame_acc_parts.append(
                            f"{label}={c / t * 100:.1f}%"
                        )
                    else:
                        endgame_acc_parts.append(f"{label}=-")
                endgame_acc_str = " ".join(endgame_acc_parts) if endgame_acc_parts else "-"
                progress.update(
                    val_task,
                    advance=1,
                    description=f"Validating (batch {batch_idx}/{total_batches})",
                    pol=avg_pol,
                    val=avg_val,
                    acc=acc_so_far,
                    mate_acc=mate_acc_str,
                    endgame_acc=endgame_acc_str,
                )

    # Report
    avg_policy_loss = policy_loss_sum / max(policy_samples, 1)
    avg_value_loss = value_loss_sum / max(value_samples, 1)
    overall_acc = total_correct / max(total_samples, 1) * 100
    print("\n" + "=" * 60)
    print("Validation (same seed & val_split as train_supervised)")
    print(f"  seed={supervised_cfg.seed}  val_split={supervised_cfg.val_split}")
    print("=" * 60)
    print(
        f"\nPolicy loss: {avg_policy_loss:.4f}  |  "
        f"Value loss: {avg_value_loss:.4f}  |  "
        f"Accuracy: {total_correct}/{total_samples} = {overall_acc:.2f}%\n"
    )

    print("By data type (m1..m5, m6plus, endgame):")
    print("-" * 50)
    for i, label in enumerate(SOURCE_LABELS):
        t = total_by_source[i]
        c = correct_by_source[i]
        acc = (c / t * 100) if t else float("nan")
        print(f"  {label:12}  {c:6}/{t:<6}  =  {acc:.2f}%")
    print()

    print("Endgame by move (end_move1, end_move2, ...):")
    print("-" * 50)
    by_endgame_move = {}
    for ply in sorted(total_by_endgame_move.keys()):
        label = _endgame_ply_label(ply)
        t = total_by_endgame_move[ply]
        c = correct_by_endgame_move[ply]
        acc = (c / t * 100) if t else float("nan")
        print(f"  {label:25}  {c:6}/{t:<6}  =  {acc:.2f}%")
        by_endgame_move[label] = (c, t)
    if not by_endgame_move:
        print("  (no endgame samples with endgame_ply >= 0)")
    print("=" * 60)

    return {
        "overall_acc": overall_acc,
        "overall_correct": total_correct,
        "overall_total": total_samples,
        "by_source": {
            label: (correct_by_source[i], total_by_source[i])
            for i, label in enumerate(SOURCE_LABELS)
        },
        "by_endgame_move": by_endgame_move,
    }


def _parse_validate_args():
    """Parse --checkpoint and --device so Hydra doesn't consume them."""
    import argparse

    p = argparse.ArgumentParser(description="Validate accuracy by data type")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pth (optional)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: from config or 'auto')",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        metavar="N",
        help="DataLoader workers (default: from config or 'auto')",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Batch size (default: from config)",
    )
    args, unknown = p.parse_known_args()
    return args, unknown


@hydra.main(
    config_path=os.path.join(PROJECT_ROOT, "config"),
    config_name="train_supervised",
    version_base=None,
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    supervised_cfg = cfg.supervised
    checkpoint_path = os.environ.get("VALIDATE_CHECKPOINT") or None
    if checkpoint_path == "":
        checkpoint_path = None
    device_override = os.environ.get("VALIDATE_DEVICE") or None
    if device_override == "":
        device_override = None
    num_workers_override = os.environ.get("VALIDATE_NUM_WORKERS")
    if num_workers_override == "":
        num_workers_override = None
    elif num_workers_override is not None:
        try:
            num_workers_override = int(num_workers_override)
        except ValueError:
            num_workers_override = None
    batch_size_override = os.environ.get("VALIDATE_BATCH_SIZE")
    if batch_size_override == "":
        batch_size_override = None
    elif batch_size_override is not None:
        try:
            batch_size_override = int(batch_size_override)
        except ValueError:
            batch_size_override = None
    device = select_device(
        device_override or supervised_cfg.device or "auto"
    )
    run_validation(cfg, checkpoint_path, device, num_workers_override, batch_size_override)


if __name__ == "__main__":
    import sys

    args, unknown = _parse_validate_args()
    sys.argv = [sys.argv[0]] + unknown
    os.environ["VALIDATE_CHECKPOINT"] = args.checkpoint or ""
    os.environ["VALIDATE_DEVICE"] = args.device or ""
    os.environ["VALIDATE_NUM_WORKERS"] = (
        str(args.num_workers) if args.num_workers is not None else ""
    )
    os.environ["VALIDATE_BATCH_SIZE"] = (
        str(args.batch_size) if args.batch_size is not None else ""
    )
    main()
