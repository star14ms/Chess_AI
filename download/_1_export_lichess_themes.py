"""Export Lichess puzzles by themes into JSON files."""

import argparse
import json
from pathlib import Path


def iter_requested_themes(themes, requested):
    if not themes:
        return []
    return [t for t in themes if t in requested]


def theme_to_filename(theme: str) -> str:
    normalized = []
    for idx, char in enumerate(theme):
        if idx > 0 and (char.isupper() or char.isdigit()):
            normalized.append("_")
        normalized.append(char.lower())
    return "".join(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Lichess puzzles by theme.")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write mate_in_#.json files (default: ./data)",
    )
    parser.add_argument(
        "themes",
        nargs="+",
        help="Themes to export (exact match, e.g. mateIn1 fork endgame).",
    )
    args = parser.parse_args()

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Lichess/chess-puzzles", split="train")
    total_rows = getattr(ds, "num_rows", None)

    requested_themes = set(args.themes)
    file_handles = {}
    item_counts = {}

    try:
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Processing puzzles", total=total_rows)
            for row in ds:
                themes = row.get("Themes") or []
                matched_themes = iter_requested_themes(themes, requested_themes)
                if not matched_themes:
                    progress.advance(task)
                    continue

                for theme in matched_themes:
                    if theme not in file_handles:
                        file_path = output_dir / f"{theme_to_filename(theme)}.json"
                        handle = file_path.open("w", encoding="utf-8")
                        handle.write("[\n")
                        file_handles[theme] = handle
                        item_counts[theme] = 0

                    handle = file_handles[theme]
                    if item_counts[theme] > 0:
                        handle.write(",\n")
                    handle.write(json.dumps(row, ensure_ascii=True))
                    item_counts[theme] += 1
                progress.advance(task)
    finally:
        for theme, handle in file_handles.items():
            if item_counts.get(theme, 0) > 0:
                handle.write("\n")
            handle.write("]\n")
            handle.close()

    for theme in args.themes:
        print(f"{theme}: {item_counts.get(theme, 0)}")


if __name__ == "__main__":
    main()
