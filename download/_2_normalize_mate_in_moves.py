"""Normalize mate-in datasets by fixing move alignment and filtering multi-mates."""

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
                except json.JSONDecodeError:
                    break
                yield obj
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
            except json.JSONDecodeError:
                return
            yield obj


def parse_moves(moves):
    if isinstance(moves, str):
        move_list = moves.split()
        return move_list, "str"
    if isinstance(moves, list):
        move_list = [str(m).strip() for m in moves if str(m).strip()]
        return move_list, "list"
    return None, None


def format_moves(move_list, source_type):
    if source_type == "list":
        return move_list
    return " ".join(move_list)


def count_checkmating_moves(board: chess.Board) -> int:
    mate_count = 0
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            mate_count += 1
        board.pop()
        if mate_count > 1:
            break
    return mate_count


def normalize_entry(entry: dict, stats: dict) -> dict | None:
    fen = entry.get("FEN") or entry.get("fen")
    moves = entry.get("Moves") or entry.get("moves")
    if not fen:
        stats["missing_fen"] += 1
        return None
    if not moves:
        stats["missing_moves"] += 1
        return None

    move_list, source_type = parse_moves(moves)
    if move_list is None:
        stats["invalid_moves"] += 1
        return None

    if len(move_list) % 2 == 0 and move_list:
        board = chess.Board(fen)
        first_move = move_list[0]
        try:
            move = chess.Move.from_uci(first_move)
        except ValueError:
            stats["invalid_moves"] += 1
            return None
        if move not in board.legal_moves:
            stats["invalid_moves"] += 1
            return None
        board.push(move)
        entry = dict(entry)
        entry["FEN"] = board.fen()
        entry["Moves"] = format_moves(move_list[1:], source_type)
        stats["adjusted_rows"] += 1
        move_list = move_list[1:]
        fen = entry["FEN"]

    if len(move_list) == 1:
        board = chess.Board(fen)
        mate_count = count_checkmating_moves(board)
        if mate_count > 1:
            stats["multiple_mates"] += 1
            return None

    return entry


def normalize_file(
    path: Path, output_dir: Path | None, progress_cb: Callable[[int], None] | None = None
) -> dict:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / path.name
        temp_path = output_path
    else:
        output_path = path
        temp_path = path.with_suffix(path.suffix + ".tmp")

    stats = {
        "input_rows": 0,
        "written_rows": 0,
        "adjusted_rows": 0,
        "multiple_mates": 0,
        "missing_fen": 0,
        "missing_moves": 0,
        "invalid_moves": 0,
    }

    with temp_path.open("w", encoding="utf-8") as out:
        out.write("[\n")
        first = True
        for entry in iter_json_array(path, progress_cb=progress_cb):
            stats["input_rows"] += 1
            if not isinstance(entry, dict):
                continue
            normalized = normalize_entry(entry, stats)
            if normalized is None:
                continue
            if not first:
                out.write(",\n")
            out.write(json.dumps(normalized, ensure_ascii=True))
            first = False
            stats["written_rows"] += 1
        out.write("\n]\n")

    if output_dir is None:
        temp_path.replace(output_path)

    stats["output_path"] = str(output_path)
    return stats


def normalize_file_worker(path_str: str, output_dir_str: str | None, progress_map) -> dict:
    path = Path(path_str)
    output_dir = Path(output_dir_str) if output_dir_str else None
    bytes_acc = 0

    def on_progress(byte_count: int) -> None:
        nonlocal bytes_acc
        bytes_acc += byte_count
        progress_map[path_str] = bytes_acc

    return normalize_file(path, output_dir, progress_cb=on_progress)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize mate_in_#.json files so even-length move lists "
            "apply the opponent's first move and drop it from Moves."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory with mate_in_#.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory (default: overwrite input files).",
    )
    args = parser.parse_args()

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

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    json_files = sorted(input_dir.glob("mate_in_[0-9].json"))
    if not json_files:
        raise SystemExit(f"No mate_in_*.json files found in {input_dir}")

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
                    normalize_file_worker, str(path), str(output_dir) if output_dir else None, progress_map
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
                        f"adjusted={stats['adjusted_rows']} multiple_mates={stats['multiple_mates']} "
                        f"missing_fen={stats['missing_fen']} missing_moves={stats['missing_moves']} "
                        f"invalid_moves={stats['invalid_moves']}"
                    )


if __name__ == "__main__":
    main()
