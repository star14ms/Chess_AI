from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count Lichess puzzle positions with a given theme."
    )
    parser.add_argument(
        "theme",
        help="Theme to count (exact match against Themes list, e.g. 'mateIn2').",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with: python -m pip install datasets"
        ) from exc

    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TaskProgressColumn,
            TimeElapsedColumn,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: rich. Install with: python -m pip install rich"
        ) from exc

    args = parse_args()
    theme = args.theme

    ds = load_dataset("Lichess/chess-puzzles")
    split_names = list(ds.keys())
    total_rows = sum(getattr(ds[split], "num_rows", 0) for split in split_names)

    total = 0
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning puzzles", total=total_rows)
        for split in split_names:
            for row in ds[split]:
                themes = row.get("Themes") or []
                if theme in themes:
                    total += 1
                progress.advance(task)

    print("splits:", split_names)
    print(f"theme: {theme}")
    print("count:", total)


if __name__ == "__main__":
    main()
