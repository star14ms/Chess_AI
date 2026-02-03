import argparse
import json
import os
import sys
from pathlib import Path

import chess
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import multiprocessing as mp

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.training_utils import iter_json_array


def _extract_moves_list(moves_field):
    if not moves_field:
        return []
    if isinstance(moves_field, str):
        tokens = moves_field.split()
    elif isinstance(moves_field, list):
        tokens = [str(t).strip() for t in moves_field if str(t).strip()]
    else:
        return []
    return [token for token in tokens if token]


def _normalize_themes(themes_field):
    if not themes_field:
        return []
    if isinstance(themes_field, list):
        return [str(t).strip() for t in themes_field if str(t).strip()]
    if isinstance(themes_field, str):
        return [t for t in themes_field.split() if t.strip()]
    return []


def _extract_entry(entry):
    fen = entry.get("FEN") or entry.get("fen")
    moves_field = entry.get("Moves") or entry.get("moves")
    moves = _extract_moves_list(moves_field)
    puzzle_id = entry.get("PuzzleId") or entry.get("puzzle_id")
    themes = _normalize_themes(entry.get("Themes") or entry.get("themes"))
    return fen, moves, puzzle_id, themes


def build_value_dataset(
    inputs,
    output_path,
    max_rows=None,
    include_themes=True,
    workers=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_labels = [Path(path).stem for path in inputs]
    if workers is None:
        workers = min(os.cpu_count() or 1, len(inputs))
    if len(inputs) > 1 and workers > 1:
        written = _build_value_dataset_mp(
            inputs,
            output_path,
            max_rows=max_rows,
            include_themes=include_themes,
            workers=workers,
        )
        _write_source_labels(output_path, source_labels)
        return written
    return _build_value_dataset_single(
        inputs,
        output_path,
        max_rows=max_rows,
        include_themes=include_themes,
    )


def _build_value_dataset_single(inputs, output_path, max_rows=None, include_themes=True):
    written = 0
    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )

    with output_path.open("w", encoding="utf-8") as out_handle, Progress(
        *progress_columns, transient=False
    ) as progress:
        for source_id, input_path in enumerate(inputs):
            total = _count_json_array_lines(input_path)
            if max_rows is not None:
                total = min(total, max_rows)
            task_id = progress.add_task(f"Processing {Path(input_path).name}", total=total)
            for idx, entry in enumerate(iter_json_array(input_path), start=1):
                if max_rows is not None and idx > max_rows:
                    break
                fen, moves, puzzle_id, themes = _extract_entry(entry)
                if not fen or not moves:
                    progress.update(task_id, advance=1)
                    continue
                board = chess.Board(fen)
                parsed_moves = []
                valid = True
                for uci in moves:
                    try:
                        move = chess.Move.from_uci(uci)
                    except ValueError:
                        valid = False
                        break
                    if not board.is_legal(move):
                        valid = False
                        break
                    parsed_moves.append(move)
                    board.push(move)
                if not valid:
                    progress.update(task_id, advance=1)
                    continue

                board = chess.Board(fen)
                total_plies = len(parsed_moves)
                for ply_index, move in enumerate(parsed_moves):
                    policy_mask = ply_index == 0
                    value_target = 1.0 if ply_index % 2 == 0 else -1.0
                    moves_to_mate = total_plies - ply_index
                    payload = {
                        "fen": board.fen(),
                        "value_target": value_target,
                        "policy_mask": policy_mask,
                        "policy_uci": move.uci() if policy_mask else None,
                        "moves_to_mate": moves_to_mate,
                        "source_id": source_id,
                        "source_label": Path(input_path).stem,
                    }
                    if puzzle_id:
                        payload["puzzle_id"] = puzzle_id
                    if include_themes and themes:
                        payload["themes"] = themes
                    out_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                    written += 1
                    board.push(move)
                progress.update(task_id, advance=1)

    _write_source_labels(output_path, [Path(path).stem for path in inputs])
    return written


def _build_value_dataset_mp(inputs, output_path, max_rows=None, include_themes=True, workers=1):
    per_file_total = {}
    for input_path in inputs:
        count = _count_json_array_lines(input_path)
        if max_rows is not None:
            count = min(count, max_rows)
        per_file_total[input_path] = count

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )

    temp_dir = output_path.parent / ".tmp_value_build"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_paths = {
        input_path: temp_dir / f"{Path(input_path).stem}_value.jsonl"
        for input_path in inputs
    }

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = []
    for source_id, input_path in enumerate(inputs):
        temp_path = temp_paths[input_path]
        proc = ctx.Process(
            target=_process_input_file,
            args=(
                input_path,
                temp_path,
                max_rows,
                include_themes,
                progress_queue,
                1000,
                source_id,
                Path(input_path).stem,
            ),
        )
        proc.start()
        processes.append(proc)

    written_total = 0
    completed = 0
    with Progress(*progress_columns, transient=False) as progress:
        task_ids = {}
        for input_path in inputs:
            label = Path(input_path).name
            task_ids[input_path] = progress.add_task(
                f"Processing {label}",
                total=per_file_total.get(input_path, 0),
            )
        while completed < len(inputs):
            message = progress_queue.get()
            if not message:
                continue
            msg_type = message[0]
            if msg_type == "progress":
                _, input_path, delta = message
                task_id = task_ids.get(input_path)
                if task_id is not None:
                    progress.update(task_id, advance=delta)
            elif msg_type == "done":
                _, input_path, written = message
                written_total += written
                completed += 1

    for proc in processes:
        proc.join()

    with output_path.open("w", encoding="utf-8") as out_handle:
        for input_path in inputs:
            temp_path = temp_paths[input_path]
            if not temp_path.exists():
                continue
            with temp_path.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    out_handle.write(line)

    for temp_path in temp_paths.values():
        if temp_path.exists():
            temp_path.unlink()
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    return written_total


def _process_input_file(
    input_path,
    temp_path,
    max_rows,
    include_themes,
    progress_queue,
    progress_batch,
    source_id,
    source_label,
):
    written = 0
    processed = 0
    with Path(temp_path).open("w", encoding="utf-8") as out_handle:
        for idx, entry in enumerate(iter_json_array(input_path), start=1):
            if max_rows is not None and idx > max_rows:
                break
            processed += 1
            if processed % progress_batch == 0:
                progress_queue.put(("progress", input_path, progress_batch))
            fen, moves, puzzle_id, themes = _extract_entry(entry)
            if not fen or not moves:
                continue
            board = chess.Board(fen)
            parsed_moves = []
            valid = True
            for uci in moves:
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    valid = False
                    break
                if not board.is_legal(move):
                    valid = False
                    break
                parsed_moves.append(move)
                board.push(move)
            if not valid:
                continue

            board = chess.Board(fen)
            total_plies = len(parsed_moves)
            for ply_index, move in enumerate(parsed_moves):
                policy_mask = ply_index == 0
                value_target = 1.0 if ply_index % 2 == 0 else -1.0
                moves_to_mate = total_plies - ply_index
                payload = {
                    "fen": board.fen(),
                    "value_target": value_target,
                    "policy_mask": policy_mask,
                    "policy_uci": move.uci() if policy_mask else None,
                    "moves_to_mate": moves_to_mate,
                    "source_id": source_id,
                    "source_label": source_label,
                }
                if puzzle_id:
                    payload["puzzle_id"] = puzzle_id
                if include_themes and themes:
                    payload["themes"] = themes
                out_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                written += 1
                board.push(move)
    remainder = processed % progress_batch
    if remainder:
        progress_queue.put(("progress", input_path, remainder))
    progress_queue.put(("done", input_path, written))


def _count_json_array_lines(path: str | Path) -> int:
    total_lines = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for _ in handle:
            total_lines += 1
    return max(total_lines - 2, 0)


def _write_source_labels(output_path: Path, source_labels: list[str]) -> None:
    sidecar_path = Path(str(output_path) + ".sources.json")
    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(source_labels, handle, ensure_ascii=True)


def main():
    parser = argparse.ArgumentParser(description="Build value training dataset from mate puzzles.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/mate_in_1_flipped.json",
            "data/mate_in_2_flipped.json",
            "data/mate_in_3_flipped.json",
            "data/mate_in_4_flipped.json",
            "data/mate_in_5_flipped.json",
        ],
        help="Input mate JSON array files (e.g., data/mate_in_1_flipped.json).",
    )
    parser.add_argument(
        "--output",
        default="data/value_mate_processed.jsonl",
        help="Output JSONL path (e.g., data/value_mate_processed.jsonl).",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--include-themes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include themes field in output (default: True).",
    )
    args = parser.parse_args()

    written = build_value_dataset(
        inputs=args.inputs,
        output_path=args.output,
        max_rows=args.max_rows,
        include_themes=args.include_themes,
        workers=args.workers,
    )
    print(f"Wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
