"""
Split trajectory JSON files into separate mate-in-1 through mate-in-5 datasets.

For each trajectory with N moves, extracts mate-in-k positions (when N >= 2k-1):
- mate_in_1: 1 move to mate
- mate_in_2: 3 moves (win-def-win)
- mate_in_3: 5 moves
- mate_in_4: 7 moves
- mate_in_5: 9 moves
"""

import argparse
import json
import sys
from pathlib import Path

import chess

# Default data dir: project's data/ folder, relative to this script
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Default trajectory files (KQ, KR, KBB, KBN)
_DEFAULT_TRAJECTORY_FILES = [
    "minimal_endgames_trajectories_K_vs_KQ.json",
    "minimal_endgames_trajectories_K_vs_KR.json",
    "minimal_endgames_trajectories_K_vs_KBB.json",
    "minimal_endgames_trajectories_K_vs_KBN.json",
]


def _num_moves_for_mate_in(k: int) -> int:
    """Mate-in-k has 2k-1 plies (e.g. mate_in_2 = 3 moves, mate_in_3 = 5 moves)."""
    return 2 * k - 1


def _extract_mate_in_positions(
    record: dict,
    moves_list: list[str],
) -> dict[int, dict]:
    """
    From a trajectory record, extract mate-in-1 through mate-in-5 puzzle positions.
    mate_in_k has (2k-1) moves: 1, 3, 5, 7, 9 for k=1..5.
    Returns {k: {puzzle_dict}} for each k where sequence has >= 2k-1 moves.
    """
    n = len(moves_list)
    if n < 1:
        return {}

    base = {
        "PuzzleId": record.get("PuzzleId", ""),
        "GameId": record.get("GameId"),
        "Rating": record.get("Rating"),
        "RatingDeviation": record.get("RatingDeviation"),
        "Popularity": record.get("Popularity"),
        "NbPlays": record.get("NbPlays"),
        "Themes": record.get("Themes", []),
        "OpeningTags": record.get("OpeningTags"),
    }

    result = {}
    fen = record.get("FEN", "")
    try:
        board = chess.Board(fen)
        for k in range(1, 6):  # mate_in_1 .. mate_in_5
            num_moves = _num_moves_for_mate_in(k)
            if n < num_moves:
                break
            # Position num_moves before mate: after playing first (n - num_moves) moves
            drop_count = n - num_moves
            b = board.copy()
            for i in range(drop_count):
                move = chess.Move.from_uci(moves_list[i])
                if move not in b.legal_moves:
                    break
                b.push(move)
            else:
                # All moves applied successfully
                mate_moves = " ".join(moves_list[drop_count:])
                themes = list(base.get("Themes") or [])
                if f"mateIn{k}" not in themes:
                    themes.append(f"mateIn{k}")
                result[k] = {
                    **base,
                    "FEN": b.fen(),
                    "Moves": mate_moves,
                    "Themes": themes,
                }
    except Exception:
        pass
    return result


def process_files(
    paths: list[Path],
    output_dir: Path,
    output_prefix: str,
    dry_run: bool,
) -> dict[int, list[dict]]:
    """
    Process trajectory files and collect mate-in-1..5 positions.
    Returns {k: [records]} for each k.
    Deduplicates by FEN+Moves to avoid duplicate initial positions.
    """
    collected: dict[int, list[dict]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    seen: dict[int, set[str]] = {k: set() for k in range(1, 6)}

    for path in sorted(paths):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for record in data:
            if not isinstance(record, dict) or "Moves" not in record:
                continue
            moves_list = record.get("Moves", "").split()
            for k, puzzle in _extract_mate_in_positions(record, moves_list).items():
                key = puzzle["FEN"] + "|" + puzzle["Moves"]
                if key not in seen[k]:
                    seen[k].add(key)
                    collected[k].append(puzzle)

    if not dry_run and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for k in range(1, 6):
            if collected[k]:
                out_path = output_dir / f"{output_prefix}mate_in_{k}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("[\n")
                    for i, rec in enumerate(collected[k]):
                        if i > 0:
                            f.write(",\n")
                        f.write(json.dumps(rec))
                    f.write("\n]\n")
                print(f"  {out_path.name}: {len(collected[k])} positions")

    return collected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split trajectory files into mate-in-1 through mate-in-5 datasets"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Trajectory JSON files (default: KQ, KR, KBB, KBN in --data-dir)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help=f"Data directory (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: same as --data-dir)",
    )
    parser.add_argument(
        "--output-prefix",
        default="minimal_endgames_",
        help="Prefix for output files, e.g. minimal_endgames_mate_in_1.json (default: minimal_endgames_)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report counts, don't write files",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.paths:
        paths = [Path(p) for p in args.paths if Path(p).exists()]
    else:
        paths = [data_dir / f for f in _DEFAULT_TRAJECTORY_FILES if (data_dir / f).exists()]

    if not paths:
        print("No trajectory files found.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    collected = process_files(
        paths,
        output_dir=output_dir,
        output_prefix=args.output_prefix,
        dry_run=args.dry_run,
    )

    print(f"\nExtracted from {len(paths)} trajectory files:")
    for k in range(1, 6):
        print(f"  mate_in_{k}: {len(collected[k])} positions")
    total = sum(len(collected[k]) for k in range(1, 6))
    print(f"  Total: {total}")
    if args.dry_run and total > 0:
        print("\nRun without --dry-run to write files.")


if __name__ == "__main__":
    main()
