import argparse
import hashlib
import json
import math
import os
import time
import random
import re

import chess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from omegaconf import OmegaConf
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
import matplotlib.pyplot as plt

from chess_gym.chess_custom import LegacyChessBoard, FullyTrackedBoard
from MCTS.training_modules.chess import create_chess_network
from utils.profile_model import profile_model
from utils.training_utils import (
    count_dataset_entries,
    freeze_value_head,
    iter_csv_rows,
    iter_json_array,
    load_checkpoint,
    save_checkpoint,
    select_device,
    validate_json_lines,
)


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
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'.")
        self.cfg = cfg
        self.data_paths = data_paths
        self.split = split
        self.val_split = val_split
        self.seed = seed
        self.max_rows = max_rows

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        max_rows = self.max_rows
        if max_rows is not None and num_workers > 1:
            max_rows = math.ceil(max_rows / num_workers)

        sources = []
        for data_path in self.data_paths:
            if data_path.lower().endswith(".json"):
                entry_iter = iter_json_array(data_path)
                source_type = "json"
            else:
                entry_iter = iter_csv_rows(data_path)
                source_type = "csv"
            sources.append((entry_iter, source_type))

        action_space_size = self.cfg.network.action_space_size
        input_channels = self.cfg.network.input_channels
        history_steps = self.cfg.env.history_steps

        raw_index = 0
        yielded = 0
        rng = random.Random(self.seed + worker_id)
        active = list(range(len(sources)))
        while active:
            active_idx = rng.randrange(len(active))
            source_id = active[active_idx]
            iterator, source_type = sources[source_id]
            try:
                entry = next(iterator)
            except StopIteration:
                active.pop(active_idx)
                continue
            if raw_index % num_workers != worker_id:
                raw_index += 1
                continue
            raw_index += 1

            if not self._entry_matches_split(entry):
                continue

            fen, best_uci = _extract_fen_and_move(entry, source_type)
            if not fen or not best_uci:
                continue

            board = create_board_from_fen(fen, action_space_size)
            try:
                move = chess.Move.from_uci(best_uci)
            except ValueError:
                continue
            if not board.is_legal(move):
                continue
            action_id = board.move_to_action_id(move)
            if action_id is None or action_id < 1 or action_id > action_space_size:
                continue
            obs = board.get_board_vector(history_steps=history_steps)
            if obs.shape[0] != input_channels:
                raise ValueError(
                    f"Observation channels {obs.shape[0]} != expected {input_channels}. "
                    "Check action_space_size vs input_channels and board type."
                )
            action_index = action_id - 1
            yield (
                torch.from_numpy(obs.astype(np.float32, copy=False)).float(),
                torch.tensor(action_index, dtype=torch.long),
                torch.tensor(source_id, dtype=torch.long),
            )
            yielded += 1
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


def _extract_best_move(moves_field):
    if not moves_field:
        return None
    if isinstance(moves_field, str):
        tokens = moves_field.split()
    elif isinstance(moves_field, list):
        tokens = [str(t).strip() for t in moves_field if str(t).strip()]
    else:
        return None
    return tokens[0] if tokens else None


def _extract_fen_and_move(entry: dict, source_type: str):
    if source_type == "csv":
        return entry.get("fen"), entry.get("best")
    fen = entry.get("FEN") or entry.get("fen")
    moves_field = entry.get("Moves") or entry.get("moves")
    best_uci = _extract_best_move(moves_field)
    return fen, best_uci


def save_learning_curve(
    train_losses: list[float],
    train_accs: list[float],
    val_losses: list[float],
    val_accs: list[float],
    checkpoint_dir: str,
) -> None:
    if not train_losses or not val_losses:
        return
    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_losses, label="train")
    axes[0].plot(epochs, val_losses, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [a * 100 for a in train_accs], label="train")
    axes[1].plot(epochs, [a * 100 for a in val_accs], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    fig.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "learning_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Supervised training on puzzle positions.")
    parser.add_argument(
        "--data",
        "--csv",
        dest="data_paths",
        nargs="+",
        default=["data/mate_in_1_flipped.json", "data/mate_in_2_flipped.json", "data/mate_in_3_flipped.json", "data/mate_in_4_flipped.json", "data/mate_in_5_flipped.json"],
        help="Paths to puzzle data (CSV or JSON). Pass multiple paths to mix datasets.",
    )
    parser.add_argument("--config", default="config/train_mcts.yaml", help="Config YAML for network settings.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--checkpoint-dir", default="outputs/supervised_train")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cuda, mps, or cpu")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--conv-dropout", type=float, default=0.1)
    parser.add_argument("--policy-dropout", type=float, default=0.25)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.network.policy_dropout = args.policy_dropout
    device = select_device(args.device or cfg.training.get("device", "auto"))

    raw_labels = [os.path.basename(p) or p for p in args.data_paths]
    shortened_labels: list[str] = []
    for label in raw_labels:
        base = os.path.splitext(label)[0]
        for suffix in ("_flipped", "_aug"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        lower = base.lower()
        mate_match = re.search(r"mate[_-]?in[_-]?(\d+)", lower)
        if mate_match:
            shortened = f"m{mate_match.group(1)}"
        else:
            shortened = base or "data"
            if len(shortened) > 12:
                shortened = shortened[:12]
        shortened_labels.append(shortened)
    label_counts: dict[str, int] = {}
    source_labels: list[str] = []
    for idx, label in enumerate(shortened_labels):
        count = label_counts.get(label, 0)
        label_counts[label] = count + 1
        if count > 0:
            source_labels.append(f"{label}#{count + 1}")
        else:
            source_labels.append(label)
    num_sources = len(source_labels)

    total_rows = 0
    json_only = True
    for data_path in args.data_paths:
        if data_path.lower().endswith(".json"):
            total_rows += validate_json_lines(data_path)
        else:
            json_only = False
    if not json_only:
        total_rows = None

    batch_size = args.batch_size or cfg.training.batch_size
    learning_rate = args.learning_rate or cfg.optimizer.learning_rate
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg.optimizer.weight_decay
    train_dataset = MateInOneIterableDataset(
        cfg,
        args.data_paths,
        split="train",
        val_split=args.val_split,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    val_dataset = MateInOneIterableDataset(
        cfg,
        args.data_paths,
        split="val",
        val_split=args.val_split,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    model = create_chess_network(cfg, device)
    freeze_value_head(model)
    model.train()
    try:
        dummy_input = torch.zeros(
            1,
            cfg.network.input_channels,
            cfg.network.board_size,
            cfg.network.board_size,
            device=device,
        )
        profile_model(model, inputs=(dummy_input,))
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

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    train_accs: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []
    if args.resume:
        resume_data = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = resume_data.get("epoch", 0) + 1
        best_val_loss = resume_data.get("best_val_loss", best_val_loss)
        patience_counter = resume_data.get("patience_counter", patience_counter)
        train_losses = list(resume_data.get("train_losses", train_losses))
        train_accs = list(resume_data.get("train_accs", train_accs))
        val_losses = list(resume_data.get("val_losses", val_losses))
        val_accs = list(resume_data.get("val_accs", val_accs))
        print(f"Resumed from {args.resume} (next epoch={start_epoch}).")
        if start_epoch > args.epochs:
            print(
                f"Checkpoint epoch={start_epoch - 1} exceeds requested epochs={args.epochs}. "
                "Nothing to train."
            )
            return

    max_rows_label = args.max_rows if args.max_rows is not None else "all"
    joined_paths = ", ".join(args.data_paths)
    print(f"Streaming data from {joined_paths} (max_rows={max_rows_label}).")
    print(f"Split: val={args.val_split:.2f} | seed={args.seed}")
    print(f"Training on {device} | batch_size={batch_size} | epochs={args.epochs}")

    train_total = None
    val_total = None
    print("Counting dataset for progress totals...")
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
        print("Using line-count totals (skips/filters not applied).")
    else:
        train_count = count_dataset_entries(train_dataset)
        val_count = count_dataset_entries(val_dataset)
    train_total = math.ceil(train_count / batch_size) if train_count else 0
    val_total = math.ceil(val_count / batch_size) if val_count else 0
    print(f"Total data entries | train={train_count} | val={val_count}")
    print(f"Total batches      | train={train_total} | val={val_total}")
    if args.num_workers > 0:
        print("Note: totals are computed without worker partitioning.")

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        TextColumn("loss={task.fields[loss]:.4f} acc={task.fields[acc]:.2f}%"),
        TextColumn("src={task.fields[src]}"),
    )

    progress_manager = (
        NullProgress() if args.no_progress else Progress(*progress_columns, transient=False)
    )
    with progress_manager as progress:
        epoch_task = progress.add_task(
            "Epochs",
            total=args.epochs,
            completed=start_epoch - 1,
            loss=float("nan"),
            acc=float("nan"),
            src="-",
        )
        train_task = progress.add_task(
            "Train", total=train_total, loss=float("nan"), acc=float("nan"), src="-"
        )
        val_task = progress.add_task(
            "Val", total=val_total, loss=float("nan"), acc=float("nan"), src="-"
        )
        last_val_loss = None
        last_val_acc = None

        for epoch in range(start_epoch, args.epochs + 1):
            total_loss = 0.0
            correct = 0
            total = 0
            per_source_correct = [0 for _ in range(num_sources)]
            per_source_total = [0 for _ in range(num_sources)]
            progress.reset(
                train_task,
                total=train_total,
                completed=0,
                loss=float("nan"),
                acc=float("nan"),
                src="-",
            )
            progress.update(train_task, description=f"Train (epoch {epoch})")
            progress.reset(
                val_task,
                total=val_total,
                completed=0,
                loss=float("nan"),
                acc=float("nan"),
                src="-",
            )
            progress.update(val_task, description="Val")

            for obs, target, source_ids in train_loader:
                obs = obs.to(device)
                target = target.to(device)
                source_ids = source_ids.to(device)
                optimizer.zero_grad()
                logits, _ = model(obs, policy_only=True)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * obs.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == target).sum().item()
                total += obs.size(0)
                for source_idx in range(num_sources):
                    mask = source_ids == source_idx
                    count = mask.sum().item()
                    if count:
                        per_source_total[source_idx] += count
                        per_source_correct[source_idx] += (preds[mask] == target[mask]).sum().item()

                avg_loss = total_loss / max(total, 1)
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
                    acc=acc * 100,
                    src=per_source_train_str,
                )

            avg_loss = total_loss / max(total, 1)
            acc = correct / max(total, 1)
            progress.update(epoch_task, advance=1, loss=avg_loss, acc=acc * 100)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_samples = 0
            val_per_source_correct = [0 for _ in range(num_sources)]
            val_per_source_total = [0 for _ in range(num_sources)]
            with torch.no_grad():
                for obs, target, source_ids in val_loader:
                    obs = obs.to(device)
                    target = target.to(device)
                    source_ids = source_ids.to(device)
                    logits, _ = model(obs, policy_only=True)
                    loss = criterion(logits, target)
                    val_loss += loss.item() * obs.size(0)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == target).sum().item()
                    val_samples += obs.size(0)
                    for source_idx in range(num_sources):
                        mask = source_ids == source_idx
                        count = mask.sum().item()
                        if count:
                            val_per_source_total[source_idx] += count
                            val_per_source_correct[source_idx] += (preds[mask] == target[mask]).sum().item()
                    val_avg_loss = val_loss / max(val_samples, 1)
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
                        acc=val_acc * 100,
                        src=per_source_val_str,
                    )
            model.train()

            train_losses.append(avg_loss)
            train_accs.append(acc)

            has_val = val_samples > 0
            if has_val:
                val_avg_loss = val_loss / max(val_samples, 1)
                val_acc = val_correct / max(val_samples, 1)
                val_losses.append(val_avg_loss)
                val_accs.append(val_acc)
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
                    f"train loss={avg_loss:.4f} acc={acc*100:.2f}% | "
                    f"val loss={val_avg_loss:.4f} acc={val_acc*100:.2f}% | "
                    f"train acc by source: {per_source_train_str} | "
                    f"val acc by source: {per_source_val_str}"
                )
                scheduler.step(val_avg_loss)

                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    patience_counter = 0
                    best_path = os.path.join(checkpoint_dir, "model_best.pth")
                    save_checkpoint(
                        best_path,
                        model,
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
                        print(f"Early stopping at epoch {epoch}.")
                        break
            else:
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

            last_val_loss = val_avg_loss if has_val else None
            last_val_acc = val_acc if has_val else None
            save_learning_curve(train_losses, train_accs, val_losses, val_accs, checkpoint_dir)

            if args.save_every > 0 and epoch % args.save_every == 0:
                if has_val:
                    ckpt_filename = (
                        f"model_epoch_{epoch}_val_{val_avg_loss:.4f}_acc_{val_acc*100:.2f}.pth"
                    )
                else:
                    ckpt_filename = f"model_epoch_{epoch}_noval.pth"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
                save_checkpoint(
                    ckpt_path,
                    model,
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

    if last_val_loss is not None and last_val_acc is not None:
        final_filename = (
            f"model_final_epoch_{epoch}_val_{last_val_loss:.4f}_acc_{last_val_acc*100:.2f}.pth"
        )
    else:
        final_filename = f"model_final_epoch_{epoch}_noval.pth"
    final_path = os.path.join(checkpoint_dir, final_filename)
    save_checkpoint(
        final_path,
        model,
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
    save_learning_curve(train_losses, train_accs, val_losses, val_accs, checkpoint_dir)


if __name__ == "__main__":
    main()
