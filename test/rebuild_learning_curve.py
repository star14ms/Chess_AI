import argparse
import os
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from train_supervised import save_learning_curve


def _default_theme_ignore() -> list[str]:
    return ["mate", "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5", "veryLong", "long", "short", "oneMove", "opening", "middlegame", "endgame"]


def _resolve_theme_metrics_path(checkpoint_path: str, theme_metrics_path: str | None) -> str | None:
    if theme_metrics_path:
        return theme_metrics_path
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    candidate = os.path.join(checkpoint_dir, "theme_metrics.jsonl")
    return candidate if os.path.exists(candidate) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild learning_curve.png from a checkpoint and theme_metrics.jsonl."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a checkpoint that contains train/val history lists.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write learning_curve.png (defaults to checkpoint dir).",
    )
    parser.add_argument(
        "--theme-metrics",
        default=None,
        help="Path to theme_metrics.jsonl (defaults to checkpoint dir if present).",
    )
    parser.add_argument(
        "--theme-ignore",
        nargs="*",
        default=_default_theme_ignore(),
        help="Themes to exclude from plotting (default matches train_supervised.py).",
    )
    parser.add_argument(
        "--theme-plot-include-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, keep missing epochs as gaps in theme plots (default: False).",
    )
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else checkpoint_dir
    theme_metrics_path = _resolve_theme_metrics_path(checkpoint_path, args.theme_metrics)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_losses = list(checkpoint.get("train_losses", []))
    train_accs = list(checkpoint.get("train_accs", []))
    val_losses = list(checkpoint.get("val_losses", []))
    val_accs = list(checkpoint.get("val_accs", []))
    per_source_train_loss = checkpoint.get("per_source_train_loss")
    per_source_train_acc = checkpoint.get("per_source_train_acc")
    per_source_val_loss = checkpoint.get("per_source_val_loss")
    per_source_val_acc = checkpoint.get("per_source_val_acc")

    if not train_losses or not val_losses:
        raise RuntimeError(
            "Checkpoint is missing train/val history. "
            "Cannot rebuild learning_curve.png without these lists."
        )

    ignored_themes = {t.strip() for t in args.theme_ignore if t and t.strip()}
    save_learning_curve(
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        output_dir,
        theme_metrics_path=theme_metrics_path,
        ignored_themes=ignored_themes,
        theme_plot_include_missing=args.theme_plot_include_missing,
        per_source_train_loss=per_source_train_loss if isinstance(per_source_train_loss, dict) else None,
        per_source_train_acc=per_source_train_acc if isinstance(per_source_train_acc, dict) else None,
        per_source_val_loss=per_source_val_loss if isinstance(per_source_val_loss, dict) else None,
        per_source_val_acc=per_source_val_acc if isinstance(per_source_val_acc, dict) else None,
    )
    print(f"Saved learning_curve.png to {output_dir}")



if __name__ == "__main__":
    main()
