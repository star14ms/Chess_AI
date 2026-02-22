"""
Clean up trajectory JSON files: truncate sequences longer than max_moves by removing leading moves.

When max_saved_moves was even (e.g. 30), the losing color could start first, causing reward inversion.
Limiting to an odd max (e.g. 29) fixes this. For sequences with more than max_moves, we drop
leading moves and update FEN to the position after those moves.
"""

import argparse
import json
import sys
from pathlib import Path

import chess

# Default data dir: project's data/ folder, relative to this script
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _truncate_to_max_moves(record: dict, max_moves: int) -> dict:
    """
    If record has more than max_moves, return a new record with leading moves dropped.
    Otherwise return the record unchanged.
    """
    moves_str = record.get("Moves", "")
    moves_list = moves_str.split()
    n = len(moves_list)
    if n <= max_moves:
        return record

    fen = record.get("FEN", "")
    try:
        board = chess.Board(fen)
        drop_count = n - max_moves
        for i in range(drop_count):
            move = chess.Move.from_uci(moves_list[i])
            if move not in board.legal_moves:
                return record  # Invalid; leave as-is
            board.push(move)
        new_fen = board.fen()
    except Exception:
        return record  # Leave as-is on error

    new_moves = " ".join(moves_list[drop_count:])
    return {
        **record,
        "FEN": new_fen,
        "Moves": new_moves,
    }


def process_file(
    path: Path,
    max_moves: int,
    dry_run: bool = False,
    in_place: bool = True,
) -> tuple[int, int]:
    """
    Process a trajectory JSON file. Truncate records with more than max_moves moves.
    Returns (total_records, truncated_count).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"Skipping {path}: not a JSON array", file=sys.stderr)
        return 0, 0

    truncated = 0
    new_data = []
    for record in data:
        if not isinstance(record, dict) or "Moves" not in record:
            new_data.append(record)
            continue
        new_record = _truncate_to_max_moves(record, max_moves)
        if new_record is not record:
            truncated += 1
        new_data.append(new_record)

    if truncated > 0 and not dry_run:
        out_path = path if in_place else path.parent / f"{path.stem}_max{max_moves}{path.suffix}"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            for i, rec in enumerate(new_data):
                if i > 0:
                    f.write(",\n")
                f.write(json.dumps(rec))
            f.write("\n]\n")
        if not in_place:
            print(f"  -> wrote {out_path}")

    return len(data), truncated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Truncate trajectory sequences to at most max_moves by dropping leading moves"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Trajectory JSON files to process (default: all minimal_endgames_trajectories_*.json in --data-dir)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=29,
        metavar="N",
        help="Max moves per sequence; longer sequences have leading moves dropped (default: 29)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help=f"Data directory when using default file discovery (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be changed, don't write",
    )
    parser.add_argument(
        "--no-in-place",
        action="store_true",
        help="Write to *_maxN.json instead of overwriting (ignored if --dry-run)",
    )
    args = parser.parse_args()
    if args.max_moves < 1:
        parser.error("--max-moves must be >= 1")

    data_dir = Path(args.data_dir)
    if args.paths:
        paths = [Path(p) for p in args.paths if Path(p).exists()]
    else:
        paths = list(data_dir.glob("minimal_endgames_trajectories_*.json"))

    if not paths:
        print("No trajectory files found.", file=sys.stderr)
        sys.exit(1)

    total_records = 0
    total_truncated = 0
    for path in sorted(paths):
        n, truncated = process_file(
            path,
            max_moves=args.max_moves,
            dry_run=args.dry_run,
            in_place=not args.no_in_place,
        )
        total_records += n
        total_truncated += truncated
        if truncated > 0:
            mode = "(dry-run, no writes)" if args.dry_run else ""
            print(f"{path.name}: {truncated}/{n} records truncated {mode}")

    print(f"\nTotal: {total_truncated}/{total_records} records truncated (max_moves={args.max_moves}) across {len(paths)} files.")
    if args.dry_run and total_truncated > 0:
        print("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
