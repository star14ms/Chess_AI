"""
Generate labeled mating sequences for minimal endgame positions using Stockfish.
Uses position generators from _5_generate_minimal_endgames, then plays out optimal
sequences with Stockfish to produce: FEN, best_move, and value (distance-to-mate based).

Output: JSON files with sequences suitable for training (each step has FEN, move, value).
"""

import json
import multiprocessing as mp
import os
import queue
import random
import shutil
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


import chess
import chess.engine

try:
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ImportError:
    Progress = None

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from download._5_generate_minimal_endgames import (
    TYPE_CONFIG,
    generate_kbb_vs_k,
    generate_kbn_vs_k,
    generate_kq_vs_k,
    generate_kr_vs_k,
    swap_colors_only,
    typed_puzzle_id,
)


def generate_labeled_sequence(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    max_moves: int = 50,
    depth: int = 20,
    timeout: float | None = None,
) -> list[dict] | None:
    """
    Play out optimal moves with Stockfish from the current position.
    Returns list of {"fen": str, "move": str, "value": float} or None.
    Only returns a sequence when the game ends in CHECKMATE.
    Draws (stalemate, 50-move, insufficient material, repetition) return None.
    timeout: max seconds per move; positions taking longer are skipped.
    """
    moves = []
    current = board.copy()
    limit_kw = {"depth": depth}
    if timeout is not None and timeout > 0:
        limit_kw["time"] = timeout

    while not current.is_game_over() and len(moves) < max_moves:
        result = engine.play(current, chess.engine.Limit(**limit_kw))
        if result.move is None:
            break

        # Value: positions closer to mate are more valuable (higher)
        # mate_in_X: value = 1.0 - (X / max_moves) so mate_in_1 ~ 0.98, mate_in_25 ~ 0.5
        remaining = max_moves - len(moves)
        value = 1.0 - (remaining / (max_moves + 10)) * 0.5  # scale 0.5–1.0

        moves.append({
            "fen": current.fen(),
            "move": result.move.uci(),
            "value": round(value, 4),
        })
        current.push(result.move)

    if current.is_checkmate():
        return moves
    # Draw or max_moves exceeded: omit from dataset
    return None


def _sequence_ends_in_checkmate(moves_list: list[dict], start_fen: str) -> bool:
    """Verify playing all moves from start_fen ends in checkmate."""
    if not moves_list:
        return False
    try:
        b = chess.Board(start_fen)
        for m in moves_list:
            move = chess.Move.from_uci(m["move"])
            b.push(move)
        return b.is_checkmate()
    except Exception:
        return False


def _process_positions_chunk(args: tuple) -> int | list[dict]:
    """
    Worker: process a chunk of positions with Stockfish. Used for multiprocessing.
    args: (positions, engine_path, max_moves, depth, timeout, id_prefix, start_index,
           progress_queue?, result_queue?)
    When result_queue is set: puts each sequence to queue, returns count.
    When result_queue is None: returns list of sequences (for memory when not writing).
    """
    parts = list(args)
    result_queue = parts.pop(-1) if len(parts) >= 9 else None
    progress_queue = parts.pop(-1) if len(parts) >= 8 else None
    (
        positions,
        engine_path,
        max_moves,
        depth,
        timeout,
        id_prefix,
        start_index,
    ) = parts[:7]
    engine_path = _resolve_engine_path(engine_path)
    sequences: list[dict] = []
    seq_count = 0
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        for pos in positions:
            board = chess.Board(pos["FEN"])
            if board.is_game_over():
                continue
            seq = generate_labeled_sequence(
                board,
                engine,
                max_moves=max_moves,
                depth=depth,
                timeout=timeout,
            )
            if seq and _sequence_ends_in_checkmate(seq, seq[0]["fen"]):
                if progress_queue:
                    progress_queue.put(1)
                puzzle_id = pos.get("PuzzleId") or typed_puzzle_id(
                    id_prefix, start_index + seq_count
                )
                record = {
                    "puzzle_id": puzzle_id,
                    "themes": pos.get("Themes", []),
                    "moves": seq,
                    "length": len(seq),
                }
                if result_queue is not None:
                    result_queue.put(record)
                else:
                    sequences.append(record)
                seq_count += 1
    finally:
        engine.quit()
    return seq_count if result_queue is not None else sequences


def _worker_pull_from_queue(args: tuple) -> None:
    """
    Worker: pull positions from shared work queue until empty/sentinel.
    args: (work_queue, progress_queue, result_queue, engine_path, max_moves,
           max_saved_moves, depth, timeout, id_prefix)
    Work items: (pos, index) or None (sentinel to stop).
    """
    (
        work_queue,
        progress_queue,
        result_queue,
        engine_path,
        max_moves,
        max_saved_moves,
        depth,
        timeout,
        id_prefix,
    ) = args
    engine_path = _resolve_engine_path(engine_path)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        while True:
            item = work_queue.get()
            if item is None:
                break
            pos, idx = item
            board = chess.Board(pos["FEN"])
            if board.is_game_over():
                continue
            seq = generate_labeled_sequence(
                board,
                engine,
                max_moves=max_moves,
                depth=depth,
                timeout=timeout,
            )
            if seq and _sequence_ends_in_checkmate(seq, seq[0]["fen"]):
                if progress_queue:
                    progress_queue.put(1)
                saved_moves = seq[-max_saved_moves:] if len(seq) > max_saved_moves else seq
                puzzle_id = pos.get("PuzzleId") or typed_puzzle_id(id_prefix, idx)
                record = {
                    "puzzle_id": puzzle_id,
                    "themes": pos.get("Themes", []),
                    "moves": saved_moves,
                    "length": len(saved_moves),
                }
                result_queue.put(record)
    finally:
        engine.quit()


class _SequenceCount:
    """Lightweight container for count when we write incrementally (don't store sequences)."""
    def __init__(self, n: int = 0):
        self.n = n
    def __len__(self) -> int:
        return self.n
    def __bool__(self) -> bool:
        return self.n > 0


def _write_single_record(seq: dict, out_handle, first: bool) -> bool:
    """Append one sequence to open file in mate_in_5 format. Only writes checkmate sequences."""
    moves_list = seq.get("moves", [])
    if not moves_list:
        return False
    if not _sequence_ends_in_checkmate(moves_list, moves_list[0]["fen"]):
        return False  # Omit draws and invalid sequences
    record = {
        "PuzzleId": seq.get("puzzle_id", ""),
        "GameId": None,
        "FEN": moves_list[0]["fen"],
        "Moves": " ".join(m["move"] for m in moves_list),
        "Rating": None,
        "RatingDeviation": None,
        "Popularity": None,
        "NbPlays": None,
        "Themes": seq.get("themes", []),
        "OpeningTags": None,
    }
    if not first:
        out_handle.write(",\n")
    out_handle.write(json.dumps(record))
    return True


def _to_mate_in_5_format(sequences: list[dict]) -> list[dict]:
    """Convert internal sequence format to mate_in_5.json style. Only includes checkmate sequences."""
    out = []
    for seq in sequences:
        moves_list = seq.get("moves", [])
        if not moves_list:
            continue
        if not _sequence_ends_in_checkmate(moves_list, moves_list[0]["fen"]):
            continue  # Omit draws
        uci_moves = " ".join(m["move"] for m in moves_list)
        out.append({
            "PuzzleId": seq.get("puzzle_id", ""),
            "GameId": None,
            "FEN": moves_list[0]["fen"],
            "Moves": uci_moves,
            "Rating": None,
            "RatingDeviation": None,
            "Popularity": None,
            "NbPlays": None,
            "Themes": seq.get("themes", []),
            "OpeningTags": None,
        })
    return out


def _count_existing_records(path: Path) -> int:
    """Count records in a JSON array file (format: [{"PuzzleId":...},...])."""
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().count('"PuzzleId"')
    except (OSError, UnicodeDecodeError):
        return 0


def _prep_file_for_append(path: Path) -> bool:
    """Truncate JSON array file before ] or orphan comma; ensure file ends with },\n for clean resume."""
    try:
        with open(path, "r+b") as f:
            f.seek(0, 2)
            size = f.tell()
            if size < 4:
                return False
            f.seek(size - 2)
            tail = f.read(2)
            if tail == b"]\n":
                f.seek(size - 2)
                f.truncate()
                new_size = size - 2
            elif tail == b",\n":
                f.seek(size - 2)
                f.truncate()
                new_size = size - 2
            else:
                return False
            if new_size < 2:
                return True
            # Replace trailing \n after last record with ,\n so next write is clean (no orphan comma)
            f.seek(new_size - 1)
            f.write(b",\n")
            return True
    except OSError:
        return False


def _write_sequences(sequences: list[dict], out_file: Path) -> None:
    """Write sequences to JSON file in mate_in_5.json format."""
    records = _to_mate_in_5_format(sequences)
    with open(out_file, "w") as f:
        lines = [json.dumps(r) for r in records]
        f.write("[\n" + ",\n".join(lines) + "\n]\n")
    print(f"Wrote {len(records)} sequences to {out_file}")


def _resolve_engine_path(engine_path: str) -> str:
    """Resolve Stockfish path. Try common locations if not in PATH."""
    resolved = shutil.which(engine_path)
    if resolved:
        return resolved
    if os.path.isfile(engine_path):
        return engine_path
    for candidate in (
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
    ):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Stockfish not found: {engine_path!r}\n"
        "Install with: brew install stockfish\n"
        "Or pass path explicitly: --engine /path/to/stockfish"
    ) from None


def collect_sequences_for_positions(
    positions: list[dict],
    engine_path: str = "stockfish",
    max_moves: int = 50,
    max_saved_moves: int = 30,
    depth: int = 20,
    timeout: float | None = None,
    id_prefix: str = "K_vs_KQ",
    start_index: int = 0,
    progress_task: tuple | None = None,
    out_file: Path | None = None,
    append_mode: bool = False,
    target_count: int | None = None,
) -> list[dict] | _SequenceCount:
    """
    For each position (dict with FEN), play Stockfish to mate and collect labeled sequence.
    progress_task: (Progress, task_id) to update an existing rich task, or None.
    out_file: when set, write each completed sequence immediately (saves memory); returns _SequenceCount.
    """
    engine_path = _resolve_engine_path(engine_path)
    sequences: list[dict] = [] if out_file is None else []
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    write_count = 0

    try:
        out_handle = None
        if out_file is not None:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            if append_mode and out_file.exists():
                _prep_file_for_append(out_file)
                out_handle = open(out_file, "a", encoding="utf-8")
            else:
                out_handle = open(out_file, "w", encoding="utf-8")
                out_handle.write("[\n")

        for i, pos in enumerate(positions):
            if target_count is not None and write_count >= target_count:
                break
            board = chess.Board(pos["FEN"])
            if board.is_game_over():
                continue

            seq = generate_labeled_sequence(
                board,
                engine,
                max_moves=max_moves,
                depth=depth,
                timeout=timeout,
            )
            if seq and _sequence_ends_in_checkmate(seq, seq[0]["fen"]):
                if progress_task:
                    progress, task_id = progress_task
                    progress.update(task_id, advance=1)
                saved_moves = seq[-max_saved_moves:] if len(seq) > max_saved_moves else seq
                puzzle_id = pos.get("PuzzleId") or typed_puzzle_id(
                    id_prefix, start_index + (write_count if out_handle else len(sequences))
                )
                record = {
                    "puzzle_id": puzzle_id,
                    "themes": pos.get("Themes", []),
                    "moves": saved_moves,
                    "length": len(saved_moves),
                }
                if out_handle is not None:
                    if _write_single_record(record, out_handle, first=(write_count == 0)):
                        write_count += 1
                else:
                    sequences.append(record)
                    if target_count is not None and len(sequences) >= target_count:
                        break

        if out_handle is not None:
            out_handle.write("\n]\n")
            out_handle.close()
            return _SequenceCount(write_count)
    finally:
        engine.quit()

    return sequences


DEFAULT_TYPES = ("KQ", "KR", "KBB", "KBN")


def generate_labeled_mating_sequences(
    count: int = 100,
    seed: int | None = 42,
    engine_path: str = "stockfish",
    max_moves: int = 50,
    max_saved_moves: int = 30,
    depth: int = 20,
    timeout: float | None = None,
    easy: bool = False,
    workers: int | None = None,
    use_multiprocess: bool = True,
    out_dir: Path | None = None,
    types: tuple[str, ...] | None = None,
) -> dict[str, list[dict]]:
    """
    Generate labeled mating sequences for minimal endgame types.
    count: positions per endgame type.
    max_saved_moves: truncate to last N moves in output (must be < max_moves).
    types: endgame types to generate (KQ, KR, KBB, KBN). Default: all four.
    out_dir: if set, write each type's JSON file as soon as that type completes.
    """
    if seed is not None:
        random.seed(seed)
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    type_list = types if types is not None else DEFAULT_TYPES
    valid = set(DEFAULT_TYPES)
    for t in type_list:
        if t not in valid:
            raise ValueError(f"Unknown type {t!r}; choose from {sorted(valid)}")

    generators = {
        "KQ": lambda n, si=0: generate_kq_vs_k(n, si, easy=easy),
        "KR": lambda n, si=0: generate_kr_vs_k(n, si, easy=easy),
        "KBB": lambda n, si=0: generate_kbb_vs_k(n, si, easy=easy),
        "KBN": lambda n, si=0: generate_kbn_vs_k(n, si, easy=easy),
    }

    suffix = "_easy" if easy else ""
    # (gen_name, n, start_index, out_file) - n=positions to generate, start_index for puzzle IDs
    active_configs: list[tuple[str, int, int, Path | None]] = []
    for g in type_list:
        if g not in generators:
            continue
        out_file = (out_dir / f"minimal_endgames_trajectories_K_vs_{g}{suffix}.json") if out_dir else None
        if out_file is not None:
            existing = _count_existing_records(out_file)
            remaining = max(0, count - existing)
            if remaining == 0:
                continue
            active_configs.append((g, remaining, existing, out_file))
        else:
            if count > 0:
                active_configs.append((g, count, 0, None))

    results: dict[str, list[dict] | _SequenceCount] = {}
    if not active_configs:
        return results

    BATCH_SIZE = 2000  # Generate positions in batches until we have enough successes

    def _run_single(
        gen_name: str,
        n: int,
        progress_task: tuple | None = None,
        out_file: Path | None = None,
        start_index: int = 0,
        append_mode: bool = False,
    ) -> list[dict] | _SequenceCount:
        collected = 0
        start_idx = start_index
        all_sequences: list[dict] = []
        first_batch = True

        while collected < n:
            need = n - collected
            batch_size = min(BATCH_SIZE, max(500, int(need * 1.25)))  # Oversample for failures
            positions = generators[gen_name](batch_size, start_idx)
            if not positions:
                break
            flip_indices = set(random.sample(range(len(positions)), len(positions) // 2))
            for i in flip_indices:
                positions[i] = {
                    **positions[i],
                    "FEN": swap_colors_only(positions[i]["FEN"]),
                }
            batch_append = append_mode or not first_batch  # Append for 2nd+ batches
            result = collect_sequences_for_positions(
                positions,
                engine_path=engine_path,
                max_moves=max_moves,
                max_saved_moves=max_saved_moves,
                depth=depth,
                timeout=timeout,
                id_prefix=TYPE_CONFIG[gen_name][0],
                start_index=start_idx,
                progress_task=progress_task,
                out_file=out_file,
                append_mode=batch_append,
                target_count=need,
            )
            cnt = len(result) if isinstance(result, _SequenceCount) else len(result)
            collected += cnt
            if not isinstance(result, _SequenceCount):
                all_sequences.extend(result)
            start_idx += len(positions)
            first_batch = False
            if cnt == 0 and len(positions) >= 100:
                break  # Avoid infinite loop if success rate is zero

        if out_file:
            return _SequenceCount(collected)
        return all_sequences[:n]

    def _run_parallel(
        gen_name: str,
        n: int,
        progress_queue: Any = None,
        progress_task_id: int | None = None,
        progress: "Progress | None" = None,
        out_file: Path | None = None,
        start_index: int = 0,
        append_mode: bool = False,
    ) -> list[dict] | _SequenceCount:
        # Over-generate positions to reach n successes (some positions fail: draw, max_moves, timeout)
        oversample = 2.0  # 2x ensures we usually get n even when success rate is ~50–60%
        num_to_try = int(n * oversample)
        positions = generators[gen_name](num_to_try, start_index)
        num_pos = len(positions)
        flip_indices = set(random.sample(range(num_pos), num_pos // 2))
        for i in flip_indices:
            positions[i] = {
                **positions[i],
                "FEN": swap_colors_only(positions[i]["FEN"]),
            }
        id_prefix = TYPE_CONFIG[gen_name][0]
        num_workers = min(workers, num_pos)
        if num_workers <= 1:
            return collect_sequences_for_positions(
                positions,
                engine_path=engine_path,
                max_moves=max_moves,
                max_saved_moves=max_saved_moves,
                depth=depth,
                timeout=timeout,
                id_prefix=id_prefix,
                start_index=start_index,
                progress_task=None,
                out_file=out_file,
                append_mode=append_mode,
                target_count=n,
            )

        # Shared work queue: each worker pulls next position until done
        _manager = mp.Manager()
        work_queue = _manager.Queue()
        result_queue = _manager.Queue()
        for i, pos in enumerate(positions):
            work_queue.put((pos, start_index + i))
        for _ in range(num_workers):
            work_queue.put(None)

        # When writing to file: workers do not report progress; writer advances only when a record is saved.
        # When not writing: workers report progress via queue, drain thread updates (with cap at n).
        worker_progress_queue = None if out_file else progress_queue

        worker_args = (
            work_queue,
            worker_progress_queue,
            result_queue,
            engine_path,
            max_moves,
            max_saved_moves,
            depth,
            timeout,
            id_prefix,
        )

        done_event = threading.Event()
        write_count = [0]  # use list for mutability in nested fn
        displayed = [0]  # for collector path: cap progress at n

        def _drain_progress() -> None:
            while not done_event.is_set():
                try:
                    count = progress_queue.get(timeout=0.2)
                    if progress and progress_task_id is not None:
                        advance_by = min(count, n - displayed[0])
                        if advance_by > 0:
                            progress.update(progress_task_id, advance=advance_by)
                            displayed[0] += advance_by
                except (queue.Empty, AttributeError):
                    pass
            while True:
                try:
                    count = progress_queue.get_nowait()
                    if progress and progress_task_id is not None:
                        advance_by = min(count, n - displayed[0])
                        if advance_by > 0:
                            progress.update(progress_task_id, advance=advance_by)
                            displayed[0] += advance_by
                except (queue.Empty, AttributeError, OSError):
                    break

        def _writer_thread() -> None:
            if not out_file or not result_queue:
                return
            out_file.parent.mkdir(parents=True, exist_ok=True)
            if append_mode and out_file.exists():
                _prep_file_for_append(out_file)
                f = open(out_file, "a", encoding="utf-8")
            else:
                f = open(out_file, "w", encoding="utf-8")
                f.write("[\n")
            try:
                first = True
                while not done_event.is_set():
                    try:
                        record = result_queue.get(timeout=0.2)
                        if write_count[0] < n and _write_single_record(record, f, first=first):
                            first = False
                            write_count[0] += 1
                            if progress and progress_task_id is not None:
                                progress.update(progress_task_id, advance=1)
                    except (queue.Empty, AttributeError):
                        pass
                while True:
                    try:
                        record = result_queue.get_nowait()
                        if write_count[0] < n and _write_single_record(record, f, first=first):
                            first = False
                            write_count[0] += 1
                            if progress and progress_task_id is not None:
                                progress.update(progress_task_id, advance=1)
                    except (queue.Empty, AttributeError, OSError):
                        break
                f.write("\n]\n")
            finally:
                f.close()

        def _collector_thread() -> None:
            while not done_event.is_set():
                try:
                    record = result_queue.get(timeout=0.2)
                    all_sequences.append(record)
                except (queue.Empty, AttributeError):
                    pass
            while True:
                try:
                    record = result_queue.get_nowait()
                    all_sequences.append(record)
                except (queue.Empty, AttributeError, OSError):
                    break

        updater = (
            threading.Thread(target=_drain_progress)
            if worker_progress_queue and progress
            else None
        )
        writer = threading.Thread(target=_writer_thread) if result_queue and out_file else None
        collector = threading.Thread(target=_collector_thread) if not out_file else None
        all_sequences: list[dict] = []
        if updater:
            updater.start()
        if writer:
            writer.start()
        if collector:
            collector.start()
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = [
                    pool.submit(_worker_pull_from_queue, worker_args)
                    for _ in range(num_workers)
                ]
                for future in as_completed(futures):
                    future.result()
        finally:
            done_event.set()
            if updater:
                updater.join()
            if writer:
                writer.join()
            if collector:
                collector.join()
        if out_file and result_queue:
            return _SequenceCount(write_count[0])
        return all_sequences

    if Progress:
        progress_columns = (
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        with Progress(*progress_columns) as progress:
            overall_id = progress.add_task("Overall", total=len(active_configs))
            progress_queue = None
            for gen_name, n, start_index, out_file in active_configs:
                task_label = TYPE_CONFIG[gen_name][2]
                task_id = progress.add_task(task_label, total=n)
                append_mode = out_file is not None and start_index > 0
                if append_mode and out_file:
                    print(f"Resuming {gen_name}: {start_index} existing, generating {n} more")
                if use_multiprocess and workers > 1 and n > 1:
                    if progress_queue is None:
                        _mgr = mp.Manager()
                        progress_queue = _mgr.Queue()
                    seqs = _run_parallel(
                        gen_name,
                        n,
                        progress_queue=progress_queue,
                        progress_task_id=task_id,
                        progress=progress,
                        out_file=out_file,
                        start_index=start_index,
                        append_mode=append_mode,
                    )
                else:
                    seqs = _run_single(
                        gen_name,
                        n,
                        progress_task=(progress, task_id),
                        out_file=out_file,
                        start_index=start_index,
                        append_mode=append_mode,
                    )
                progress.remove_task(task_id)
                progress.update(overall_id, advance=1)
                results[gen_name] = seqs
                if out_dir and seqs and not isinstance(seqs, _SequenceCount):
                    _write_sequences(
                        seqs,
                        out_dir / f"minimal_endgames_trajectories_K_vs_{gen_name}{suffix}.json",
                    )
                elif out_dir and isinstance(seqs, _SequenceCount) and seqs:
                    print(f"Wrote {len(seqs)} sequences to {out_file}")
    else:
        for gen_name, n, start_index, out_file in active_configs:
            append_mode = out_file is not None and start_index > 0
            if use_multiprocess and workers > 1 and n > 1:
                seqs = _run_parallel(
                    gen_name, n, out_file=out_file,
                    start_index=start_index, append_mode=append_mode,
                )
            else:
                seqs = _run_single(
                    gen_name, n, out_file=out_file,
                    start_index=start_index, append_mode=append_mode,
                )
            results[gen_name] = seqs
            if out_dir and seqs and not isinstance(seqs, _SequenceCount):
                _write_sequences(
                    seqs,
                    out_dir / f"minimal_endgames_trajectories_K_vs_{gen_name}{suffix}.json",
                )
            elif out_dir and isinstance(seqs, _SequenceCount) and seqs:
                print(f"Wrote {len(seqs)} sequences to {out_file}")

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate labeled mating sequences for minimal endgames (Stockfish)"
    )
    parser.add_argument(
        "--output-dir", "-o", 
        default="data",
        help="Directory for output JSON files",
    )
    parser.add_argument(
        "--count", "-n", type=int, default=10000,
        help="Sequences per endgame type",
    )
    parser.add_argument(
        "--types", "-t",
        nargs="+",
        default=list(DEFAULT_TYPES),
        choices=["KQ", "KR", "KBB", "KBN"],
        metavar="TYPE",
        help="Endgame types to generate (default: KQ KR KBB KBN)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--engine", "-e",
        default="stockfish",
        help="Path to Stockfish UCI executable",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=50,
        help="Max moves to play per sequence",
    )
    parser.add_argument(
        "--max-saved-moves",
        type=int,
        default=30,
        metavar="N",
        help="Max moves to save in output (last N moves only, must be < max-moves, default: 30)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Stockfish search depth",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        metavar="SEC",
        help="Max seconds per move; positions taking longer are skipped (0=no limit, default: 3)",
    )
    parser.add_argument(
        "--easy",
        action="store_true",
        help="Use easy positions (opponent king on edge)",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=None,
        help="Parallel workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--no-multiprocess",
        action="store_true",
        help="Run single-threaded (no parallelism)",
    )
    args = parser.parse_args()
    if args.max_saved_moves >= args.max_moves:
        parser.error("--max-saved-moves must be less than --max-moves")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = generate_labeled_mating_sequences(
        count=args.count,
        seed=args.seed,
        engine_path=args.engine,
        max_moves=args.max_moves,
        max_saved_moves=args.max_saved_moves,
        depth=args.depth,
        timeout=args.timeout,
        easy=args.easy,
        workers=args.workers,
        use_multiprocess=not args.no_multiprocess,
        out_dir=out_dir,
        types=tuple(args.types),
    )

    total = sum(len(s) for s in results.values())
    print(f"\nTotal: {total} labeled mating sequences")


if __name__ == "__main__":
    main()
