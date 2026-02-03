"""Normalize puzzle datasets by fixing move alignment and filtering multi-mates."""

import argparse
import json
import os
import multiprocessing as mp
import concurrent.futures
import math
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

    if len(move_list) % 2 == 1:
        puzzle_id = entry.get("PuzzleId") or entry.get("puzzle_id") or "unknown"
        print(
            f"Skipping odd-length moves: puzzle_id={puzzle_id} "
            f"len={len(move_list)}"
        )
        stats["odd_moves"] += 1
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
        "odd_moves": 0,
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


def _count_json_array_lines(path: Path) -> int:
    total_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            total_lines += 1
    return max(total_lines - 2, 0)


def _iter_json_array_chunk(path: Path, start_idx: int, end_idx: int):
    data_idx = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped == "[" or stripped == "]":
                continue
            if stripped.endswith(","):
                stripped = stripped[:-1].rstrip()
            if data_idx < start_idx:
                data_idx += 1
                continue
            if data_idx >= end_idx:
                break
            data_idx += 1
            yield json.loads(stripped)


def normalize_chunk(
    path: Path,
    temp_path: Path,
    start_idx: int,
    end_idx: int,
    progress_queue,
    progress_batch: int,
) -> dict:
    stats = {
        "input_rows": 0,
        "written_rows": 0,
        "adjusted_rows": 0,
        "odd_moves": 0,
        "multiple_mates": 0,
        "missing_fen": 0,
        "missing_moves": 0,
        "invalid_moves": 0,
    }
    processed = 0
    with temp_path.open("w", encoding="utf-8") as out:
        for entry in _iter_json_array_chunk(path, start_idx, end_idx):
            processed += 1
            if processed % progress_batch == 0:
                progress_queue.put(("progress", str(path), progress_batch))
            stats["input_rows"] += 1
            if not isinstance(entry, dict):
                continue
            normalized = normalize_entry(entry, stats)
            if normalized is None:
                continue
            out.write(json.dumps(normalized, ensure_ascii=True) + "\n")
            stats["written_rows"] += 1
    remainder = processed % progress_batch
    if remainder:
        progress_queue.put(("progress", str(path), remainder))
    stats["temp_path"] = str(temp_path)
    stats["start_idx"] = start_idx
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize puzzle JSON files so even-length move lists "
            "apply the opponent's first move and drop it from Moves."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory with JSON files (default: mate_in_*.json).",
    )
    parser.add_argument(
        "--input-glob",
        default="mate_in_[0-9].json",
        help="Glob for input files when --files is not set (default: mate_in_[0-9].json).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory (default: overwrite input files).",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional JSON files to normalize (overrides --input-dir glob).",
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

    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.files:
        json_files = [Path(path) for path in args.files]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise SystemExit(f"Input directory not found: {input_dir}")
        json_files = sorted(input_dir.glob(args.input_glob))
        if not json_files:
            raise SystemExit(
                f"No files found in {input_dir} matching {args.input_glob}"
            )

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        progress_queue = mp.Manager().Queue()
        tasks: dict[str, int] = {}
        totals: dict[str, int] = {}
        chunk_tasks = []
        temp_dir = Path(output_dir) if output_dir else Path(args.input_dir)
        temp_dir = temp_dir / ".tmp_normalize_chunks"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for path in json_files:
            key = str(path)
            total_rows = _count_json_array_lines(path)
            totals[key] = total_rows
            tasks[key] = progress.add_task(f"Processing {path.name}", total=total_rows)
            chunk_size = max(1, math.ceil(total_rows / max(1, os.cpu_count() or 1)))
            num_chunks = math.ceil(total_rows / chunk_size)
            for chunk_index in range(num_chunks):
                start_idx = chunk_index * chunk_size
                end_idx = min(start_idx + chunk_size, total_rows)
                temp_path = temp_dir / f"{path.stem}_part_{chunk_index:03d}.jsonl"
                chunk_tasks.append((path, temp_path, start_idx, end_idx))

        max_workers = min(len(chunk_tasks), os.cpu_count() or 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    normalize_chunk,
                    path,
                    temp_path,
                    start_idx,
                    end_idx,
                    progress_queue,
                    1000,
                ): str(path)
                for path, temp_path, start_idx, end_idx in chunk_tasks
            }
            pending = set(future_map.keys())
            completed_chunks = 0
            per_file_stats: dict[str, dict] = {}
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED
                )
                while not progress_queue.empty():
                    msg_type, key, delta = progress_queue.get()
                    if msg_type == "progress":
                        progress.update(tasks[key], advance=delta)
                for future in done:
                    stats = future.result()
                    key = future_map[future]
                    completed_chunks += 1
                    file_stats = per_file_stats.setdefault(
                        key,
                        {
                            "input_rows": 0,
                            "written_rows": 0,
                            "adjusted_rows": 0,
                            "odd_moves": 0,
                            "multiple_mates": 0,
                            "missing_fen": 0,
                            "missing_moves": 0,
                            "invalid_moves": 0,
                            "chunks": [],
                        },
                    )
                    file_stats["input_rows"] += stats["input_rows"]
                    file_stats["written_rows"] += stats["written_rows"]
                    file_stats["adjusted_rows"] += stats["adjusted_rows"]
                    file_stats["odd_moves"] += stats["odd_moves"]
                    file_stats["multiple_mates"] += stats["multiple_mates"]
                    file_stats["missing_fen"] += stats["missing_fen"]
                    file_stats["missing_moves"] += stats["missing_moves"]
                    file_stats["invalid_moves"] += stats["invalid_moves"]
                    file_stats["chunks"].append((stats["start_idx"], stats["temp_path"]))

            for key, task_id in tasks.items():
                progress.update(task_id, completed=totals[key])

        for path in json_files:
            key = str(path)
            stats = per_file_stats.get(key)
            if stats is None:
                continue
            chunks = sorted(stats["chunks"], key=lambda item: item[0])
            output_path = (Path(output_dir) if output_dir else path).with_name(path.name)
            temp_path = output_path if output_dir else path.with_suffix(path.suffix + ".tmp")
            with temp_path.open("w", encoding="utf-8") as out:
                out.write("[\n")
                first = True
                for _, chunk_path in chunks:
                    with Path(chunk_path).open("r", encoding="utf-8") as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            if not first:
                                out.write(",\n")
                            out.write(line.strip())
                            first = False
                out.write("\n]\n")
            if output_dir is None:
                temp_path.replace(output_path)
            print(f"{Path(output_path).name} -> {output_path}")
            print(
                "rows: "
                f"in={stats['input_rows']} out={stats['written_rows']} "
                f"adjusted={stats['adjusted_rows']} odd_moves={stats['odd_moves']} "
                f"multiple_mates={stats['multiple_mates']} missing_fen={stats['missing_fen']} "
                f"missing_moves={stats['missing_moves']} invalid_moves={stats['invalid_moves']}"
            )

        for _, _, _, _ in chunk_tasks:
            pass
        for chunk_path in temp_dir.glob("*.jsonl"):
            chunk_path.unlink()
        try:
            temp_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
