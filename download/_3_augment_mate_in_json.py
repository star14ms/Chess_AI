"""Augment mate-in datasets by flipping board and colors."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Callable

import chess


def iter_json_array(path: Path, progress_cb: Callable[[int], None] | None = None):
    decoder = json.JSONDecoder()
    buffer = ""
    in_array = False
    with path.open("r", encoding="utf-8") as handle:
        item_count = 0
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            if progress_cb is not None:
                progress_cb(len(chunk))
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not in_array:
                    if buffer.startswith("["):
                        buffer = buffer[1:]
                        in_array = True
                    else:
                        idx = buffer.find("[")
                        if idx == -1:
                            buffer = ""
                            break
                        buffer = buffer[idx + 1 :]
                        in_array = True
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    buffer = buffer.lstrip()
                if buffer.startswith("]"):
                    return
                try:
                    obj, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError as exc:
                    break
                yield obj
                item_count += 1
                buffer = buffer[idx:]
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]

        buffer = buffer.strip()
        if in_array and buffer:
            if buffer.startswith("]"):
                return
            try:
                obj, _ = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                return
            yield obj
            item_count += 1


def transform_square_180(square: int) -> int:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    return chess.square(7 - file_idx, 7 - rank_idx)


def transform_uci_180(uci: str) -> str | None:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None
    new_move = chess.Move(
        transform_square_180(move.from_square),
        transform_square_180(move.to_square),
        promotion=move.promotion,
    )
    return new_move.uci()


def flip_board_and_color(fen: str) -> str:
    board = chess.Board(fen)
    board = board.transform(chess.flip_horizontal)
    board = board.mirror()
    return board.fen()


def reset_move_counters(fen: str) -> str:
    parts = fen.split()
    if len(parts) < 6:
        return fen
    parts[4] = "0"
    parts[5] = "0"
    return " ".join(parts)


def normalize_fen_for_dup(fen: str) -> str:
    parts = fen.split()
    if len(parts) < 4:
        return fen
    return " ".join(parts[:4])


def augment_file(
    path: Path,
    suffix: str,
    puzzle_id_suffix: str,
    progress_cb: Callable[[int], None] | None = None,
) -> dict:
    output_path = path.with_name(f"{path.stem}{suffix}.json")

    seen_fens: set[str] = set()
    stats = {
        "input_rows": 0,
        "written_rows": 0,
        "missing_fen": 0,
        "missing_moves": 0,
        "invalid_moves": 0,
        "duplicate_source": 0,
        "duplicate_flipped": 0,
        "duplicate_source_fens": [],
        "duplicate_flipped_fens": [],
    }

    with output_path.open("w", encoding="utf-8") as out:
        out.write("[\n")
        first = True
        for entry in iter_json_array(path, progress_cb=progress_cb):
            stats["input_rows"] += 1
            if not isinstance(entry, dict):
                continue
            fen = entry.get("FEN") or entry.get("fen")
            moves = entry.get("Moves") or entry.get("moves")
            if not fen:
                stats["missing_fen"] += 1
                continue
            normalized_fen = normalize_fen_for_dup(fen)
            if normalized_fen in seen_fens:
                stats["duplicate_source"] += 1
                stats["duplicate_source_fens"].append(normalized_fen)
                continue
            if not moves:
                stats["missing_moves"] += 1
                continue

            if isinstance(moves, str):
                move_list = moves.split()
            elif isinstance(moves, list):
                move_list = [str(m).strip() for m in moves if str(m).strip()]
            else:
                stats["invalid_moves"] += 1
                continue

            entry = dict(entry)
            entry["FEN"] = reset_move_counters(fen)
            seen_fens.add(normalized_fen)
            if not first:
                out.write(",\n")
            out.write(json.dumps(entry, ensure_ascii=True))
            first = False
            stats["written_rows"] += 1

            flipped_fen = flip_board_and_color(fen)
            flipped_fen = reset_move_counters(flipped_fen)
            normalized_flipped_fen = normalize_fen_for_dup(flipped_fen)
            if normalized_flipped_fen in seen_fens:
                stats["duplicate_flipped"] += 1
                stats["duplicate_flipped_fens"].append(normalized_flipped_fen)
                continue

            flipped_moves = []
            invalid_flip = False
            for uci in move_list:
                mapped = transform_uci_180(uci)
                if mapped is None:
                    invalid_flip = True
                    break
                flipped_moves.append(mapped)
            if invalid_flip:
                stats["invalid_moves"] += 1
                continue

            flipped_entry = dict(entry)
            flipped_entry["FEN"] = flipped_fen
            flipped_entry["Moves"] = " ".join(flipped_moves)
            if "PuzzleId" in flipped_entry and flipped_entry["PuzzleId"]:
                flipped_entry["PuzzleId"] = f"{flipped_entry['PuzzleId']}{puzzle_id_suffix}"
            seen_fens.add(normalized_flipped_fen)
            out.write(",\n")
            out.write(json.dumps(flipped_entry, ensure_ascii=True))
            stats["written_rows"] += 1

        out.write("\n]\n")

    stats["output_path"] = str(output_path)
    return stats


def augment_file_worker(path_str: str, suffix: str, puzzle_id_suffix: str, progress_map) -> dict:
    path = Path(path_str)
    bytes_acc = 0

    def on_progress(byte_count: int) -> None:
        nonlocal bytes_acc
        bytes_acc += byte_count
        progress_map[path_str] = bytes_acc

    return augment_file(path, suffix, puzzle_id_suffix, progress_cb=on_progress)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment mate_in_#.json files by flipping board and color."
    )
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory with mate_in_#.json files.",
    )
    parser.add_argument(
        "--suffix",
        default="_flipped",
        help="Suffix to append to output file name.",
    )
    parser.add_argument(
        "--puzzle-id-suffix",
        default="_aug",
        help="Suffix to append to PuzzleId for augmented rows.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    json_files = sorted(input_dir.glob("mate_in_[0-9].json"))
    if not json_files:
        raise SystemExit(f"No mate_in_*.json files found in {input_dir}")

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

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        manager = concurrent.futures.process.multiprocessing.Manager()
        progress_map = manager.dict()
        tasks: dict[str, int] = {}
        totals: dict[str, int] = {}
        for path in json_files:
            key = str(path)
            total_bytes = os.path.getsize(path)
            totals[key] = total_bytes
            progress_map[key] = 0
            tasks[key] = progress.add_task(f"Processing {path.name}", total=total_bytes)

        max_workers = min(len(json_files), os.cpu_count() or 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    augment_file_worker, str(path), args.suffix, args.puzzle_id_suffix, progress_map
                ): str(path)
                for path in json_files
            }
            pending = set(future_map.keys())
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for key, task_id in tasks.items():
                    progress.update(task_id, completed=progress_map.get(key, 0))
                for future in done:
                    stats = future.result()
                    key = future_map[future]
                    progress.update(tasks[key], completed=totals[key])
                    print(f"{Path(stats['output_path']).name} -> {stats['output_path']}")
                    print(
                        "rows: "
                        f"in={stats['input_rows']} out={stats['written_rows']} "
                        f"dup_src={stats['duplicate_source']} dup_flip={stats['duplicate_flipped']} "
                        f"missing_fen={stats['missing_fen']} missing_moves={stats['missing_moves']} "
                        f"invalid_moves={stats['invalid_moves']}"
                    )
                    # if stats["duplicate_source_fens"]:
                    #     print("duplicate_source_fens:")
                    #     for fen in stats["duplicate_source_fens"]:
                    #         print(f"  {fen}")
                    # if stats["duplicate_flipped_fens"]:
                    #     print("duplicate_flipped_fens:")
                    #     for fen in stats["duplicate_flipped_fens"]:
                    #         print(f"  {fen}")


if __name__ == "__main__":
    main()
