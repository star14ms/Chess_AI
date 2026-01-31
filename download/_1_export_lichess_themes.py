"""Export Lichess puzzles by themes into JSON files."""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeQuery:
    raw: str
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    filename: str


def iter_matching_queries(themes, queries):
    if not themes:
        return []
    theme_set = set(themes)
    matched = []
    for query in queries:
        if query.includes and not set(query.includes).issubset(theme_set):
            continue
        if query.excludes and set(query.excludes).intersection(theme_set):
            continue
        matched.append(query)
    return matched


def theme_to_filename(theme: str) -> str:
    normalized = []
    for idx, char in enumerate(theme):
        if idx > 0 and (char.isupper() or char.isdigit()):
            normalized.append("_")
        normalized.append(char.lower())
    return "".join(normalized)


def query_to_filename(includes: tuple[str, ...], excludes: tuple[str, ...]) -> str:
    if not includes and not excludes:
        return "all"
    parts = []
    if includes:
        parts.append("-".join(theme_to_filename(t) for t in includes))
    else:
        parts.append("all")
    if excludes:
        parts.append("without")
        parts.append("-".join(theme_to_filename(t) for t in excludes))
    return "_".join(parts)


def parse_theme_query(raw: str) -> ThemeQuery:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    includes = []
    excludes = []
    for token in tokens:
        if token.startswith(("-", "!")):
            excludes.append(token[1:])
        else:
            includes.append(token)
    includes_sorted = tuple(sorted(set(includes)))
    excludes_sorted = tuple(sorted(set(excludes)))
    filename = query_to_filename(includes_sorted, excludes_sorted)
    return ThemeQuery(raw=raw, includes=includes_sorted, excludes=excludes_sorted, filename=filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Lichess puzzles by theme.")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write theme JSON files (default: ./data)",
    )
    parser.add_argument(
        "themes",
        nargs="+",
        help=(
            "Theme queries to export. Use comma-separated tokens per query. "
            "Prefix tokens with '-' or '!' to exclude. "
            "Examples: mateIn1, endgame,-mate, rookEndgame,!mate"
        ),
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

    queries = [parse_theme_query(raw) for raw in args.themes]
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
                matched_queries = iter_matching_queries(themes, queries)
                if not matched_queries:
                    progress.advance(task)
                    continue

                for query in matched_queries:
                    if query.filename not in file_handles:
                        file_path = output_dir / f"{query.filename}.json"
                        handle = file_path.open("w", encoding="utf-8")
                        handle.write("[\n")
                        file_handles[query.filename] = handle
                        item_counts[query.filename] = 0

                    handle = file_handles[query.filename]
                    if item_counts[query.filename] > 0:
                        handle.write(",\n")
                    handle.write(json.dumps(row, ensure_ascii=True))
                    item_counts[query.filename] += 1
                progress.advance(task)
    finally:
        for key, handle in file_handles.items():
            if item_counts.get(key, 0) > 0:
                handle.write("\n")
            handle.write("]\n")
            handle.close()

    for query in queries:
        print(f"{query.raw} -> {query.filename}.json: {item_counts.get(query.filename, 0)}")


if __name__ == "__main__":
    main()
