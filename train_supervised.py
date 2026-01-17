import argparse
import csv
import os
import time

import chess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
import matplotlib.pyplot as plt

from chess_gym.chess_custom import LegacyChessBoard, FullyTrackedBoard
from MCTS.training_modules.chess import create_chess_network


class MateInOneDataset(Dataset):
    def __init__(self, cfg, rows):
        self.cfg = cfg
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        obs, action_index = self.rows[idx]
        return torch.from_numpy(obs).float(), torch.tensor(action_index, dtype=torch.long)


def select_device(device_str: str | None) -> torch.device:
    if device_str and device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_board_from_fen(fen: str, action_space_size: int):
    if action_space_size == 4672:
        board = LegacyChessBoard(fen)
    else:
        board = FullyTrackedBoard(fen)
    return board


def load_mate_in_one_rows(cfg, csv_path: str, max_rows: int | None = None):
    rows = []
    skipped = 0
    action_space_size = cfg.network.action_space_size
    input_channels = cfg.network.input_channels
    history_steps = cfg.env.history_steps

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if max_rows is not None and len(rows) >= max_rows:
                break
            fen = row.get("fen")
            best_uci = row.get("best")
            if not fen or not best_uci:
                skipped += 1
                continue
            board = create_board_from_fen(fen, action_space_size)
            try:
                move = chess.Move.from_uci(best_uci)
            except ValueError:
                skipped += 1
                continue
            if not board.is_legal(move):
                skipped += 1
                continue
            action_id = board.move_to_action_id(move)
            if action_id is None or action_id < 1 or action_id > action_space_size:
                skipped += 1
                continue
            obs = board.get_board_vector(history_steps=history_steps)
            if obs.shape[0] != input_channels:
                raise ValueError(
                    f"Observation channels {obs.shape[0]} != expected {input_channels}. "
                    "Check action_space_size vs input_channels and board type."
                )
            action_index = action_id - 1
            rows.append((obs.astype(np.float32, copy=False), action_index))

    return rows, skipped


def freeze_value_head(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("value_head"):
            param.requires_grad = False


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg_payload = OmegaConf.to_container(cfg, resolve=False)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg_payload,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Supervised training on mate-in-one positions.")
    parser.add_argument("--csv", default="data/mate_in_one.csv", help="Path to mate-in-one CSV file.")
    parser.add_argument("--config", default="config/train_mcts.yaml", help="Config YAML for network settings.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--checkpoint-dir", default="outputs/supervised_train")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cuda, mps, or cpu")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--policy-dropout", type=float, default=0.3)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.network.policy_dropout = args.policy_dropout
    device = select_device(args.device or cfg.training.get("device", "auto"))

    rows, skipped = load_mate_in_one_rows(cfg, args.csv, max_rows=args.max_rows)
    if not rows:
        raise RuntimeError("No usable rows loaded from mate_in_one.csv.")

    batch_size = args.batch_size or cfg.training.batch_size
    learning_rate = args.learning_rate or cfg.optimizer.learning_rate
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg.optimizer.weight_decay

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(rows))
    val_size = int(len(rows) * args.val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_rows = [rows[i] for i in train_indices]
    val_rows = [rows[i] for i in val_indices]

    train_dataset = MateInOneDataset(cfg, train_rows)
    val_dataset = MateInOneDataset(cfg, val_rows)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    model = create_chess_network(cfg, device)
    freeze_value_head(model)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_patience, factor=args.lr_factor
    )
    criterion = nn.CrossEntropyLoss()

    run_id = time.strftime("%Y-%m-%d/%H-%M-%S")
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_id)

    print(f"Loaded {len(rows)} rows (skipped {skipped}).")
    print(f"Train rows: {len(train_rows)} | Val rows: {len(val_rows)}")
    print(f"Training on {device} | batch_size={batch_size} | epochs={args.epochs}")

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        TextColumn("[{task.fields[details]}]"),
    )

    with Progress(*progress_columns, transient=False) as progress:
        epoch_task = progress.add_task("Epochs", total=args.epochs, details="starting")
        train_task = progress.add_task("Train", total=len(train_loader), details="")
        val_task = progress.add_task("Val", total=len(val_loader), details="")

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            total_loss = 0.0
            correct = 0
            total = 0
            progress.reset(train_task, total=len(train_loader), completed=0, details=f"epoch {epoch}")
            progress.reset(val_task, total=len(val_loader), completed=0, details="")

            for obs, target in train_loader:
                obs = obs.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                logits, _ = model(obs)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * obs.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == target).sum().item()
                total += obs.size(0)

                avg_loss = total_loss / max(total, 1)
                acc = correct / max(total, 1)
                progress.update(train_task, advance=1, details=f"loss={avg_loss:.4f} acc={acc*100:.2f}%")

            avg_loss = total_loss / max(total, 1)
            acc = correct / max(total, 1)
            progress.update(epoch_task, advance=1, details=f"loss={avg_loss:.4f} acc={acc*100:.2f}%")

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for obs, target in val_loader:
                    obs = obs.to(device)
                    target = target.to(device)
                    logits, _ = model(obs)
                    loss = criterion(logits, target)
                    val_loss += loss.item() * obs.size(0)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == target).sum().item()
                    val_total += obs.size(0)
                    val_avg_loss = val_loss / max(val_total, 1)
                    val_acc = val_correct / max(val_total, 1)
                    progress.update(val_task, advance=1, details=f"loss={val_avg_loss:.4f} acc={val_acc*100:.2f}%")
            model.train()

            val_avg_loss = val_loss / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)
            train_losses.append(avg_loss)
            train_accs.append(acc)
            val_losses.append(val_avg_loss)
            val_accs.append(val_acc)
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train loss={avg_loss:.4f} acc={acc*100:.2f}% | "
                f"val loss={val_avg_loss:.4f} acc={val_acc*100:.2f}%"
            )

            scheduler.step(val_avg_loss)

            if args.save_every > 0 and epoch % args.save_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"model.pth")
                save_checkpoint(ckpt_path, model, optimizer, epoch, cfg)

            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                patience_counter = 0
                best_path = os.path.join(checkpoint_dir, "model_best.pth")
                save_checkpoint(best_path, model, optimizer, epoch, cfg)
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    final_path = os.path.join(checkpoint_dir, "model.pth")
    save_checkpoint(final_path, model, optimizer, args.epochs, cfg)
    print(f"Saved final checkpoint to {final_path}")

    if train_losses and val_losses:
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
        print(f"Saved learning curve to {plot_path}")


if __name__ == "__main__":
    main()
