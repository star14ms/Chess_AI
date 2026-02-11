import argparse
import hashlib
import json
import os
import sys
import math
from pathlib import Path

import numpy as np
import chess
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import multiprocessing as mp

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.training_utils import iter_json_array

# Short forcing tactics: expand full trajectory; otherwise endgame uses first move only
FORCING_THEMES = frozenset({
    "fork", "pin", "skewer", "discoveredAttack",
    "backRankMate", "smotheredMate",
})

# Defaults for endgame expansion (parameterized)
DEFAULT_ENDGAME_EXPAND_MAX_PLIES = 3
DEFAULT_ENDGAME_EXPAND_MAX_MOVES = 3
DEFAULT_ENDGAME_EXPAND_REQUIRE_CRUSHING = True
DEFAULT_ENDGAME_EXPAND_REQUIRE_FORCING_THEME = True


def _is_mate_puzzle(themes):
    return "mate" in themes or any(t.startswith("mateIn") for t in themes)


def _should_expand_endgame(
    themes,
    moves,
    *,
    max_moves: int = DEFAULT_ENDGAME_EXPAND_MAX_MOVES,
    require_crushing: bool = DEFAULT_ENDGAME_EXPAND_REQUIRE_CRUSHING,
    require_forcing_theme: bool = DEFAULT_ENDGAME_EXPAND_REQUIRE_FORCING_THEME,
    forcing_themes: frozenset | None = None,
) -> bool:
    """Expand full trajectory only when conditions are met (e.g. short forcing tactics)."""
    if forcing_themes is None:
        forcing_themes = FORCING_THEMES
    theme_set = set(themes)
    if len(moves) > max_moves:
        return False
    if require_crushing and "crushing" not in theme_set:
        return False
    if require_forcing_theme and not (theme_set & forcing_themes):
        return False
    return True


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


def count_material(board):
    """Material difference from current player's perspective (in pawn units).
    Used for endgame pseudo-labels; piece values match common practice (Q=9,R=5,B=3,N=3,P=1)."""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    white = sum(
        piece_values.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == chess.WHITE
    )
    black = sum(
        piece_values.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == chess.BLACK
    )
    diff = white - black
    return diff if board.turn == chess.WHITE else -diff


def count_mobility(board):
    """Legal move count difference from current player's perspective (current side minus opponent)."""
    if board.turn == chess.WHITE:
        white_moves = board.legal_moves.count()
        board.push(chess.Move.null())  # no-op to flip turn for black count
        black_moves = board.legal_moves.count()
        board.pop()
        return white_moves - black_moves
    else:
        black_moves = board.legal_moves.count()
        board.push(chess.Move.null())
        white_moves = board.legal_moves.count()
        board.pop()
        return white_moves - black_moves


def _endgame_solver_value_target(entry, position_fen=None):
    """Binary value from current player's perspective: 1.0 if current player is the solver (wins), else -1.0.
    Solver = side to move in the puzzle's initial position. position_fen defaults to entry's FEN (initial position).
    Same convention as supervised and RL: value is always from side-to-move, not 'White win +1 / Black win -1'."""
    fen0 = entry.get("FEN") or entry.get("fen") or ""
    if not fen0:
        return 0.0
    board0 = chess.Board(fen0)
    solver_color = board0.turn
    fen = position_fen if position_fen is not None else fen0
    board = chess.Board(fen)
    return 1.0 if board.turn == solver_color else -1.0


def label_endgame_value(entry, rng=None, use_mobility=True):
    """Generate value label for any position (mate or heuristic).
    Value is from side-to-move perspective; positive = good for current player.
    Supports metadata['mate_in_n'] + metadata['winner'], or theme-based mate (mate/mateIn*).
    Heuristics: material (primary), additive theme (crushing/advantage/equality), optional mobility.
    """
    fen = entry.get("FEN") or entry.get("fen") or ""
    if not fen:
        return 0.0
    board = chess.Board(fen)
    themes = _normalize_themes(entry.get("Themes") or entry.get("themes"))

    # Mate: true label ±1.0 (mate_in_n or theme-based)
    if entry.get("mate_in_n"):
        winner = entry.get("winner")
        if winner is None:
            winner = "white" if board.turn == chess.WHITE else "black"
        return 1.0 if winner == "white" else -1.0
    if "mate" in themes or any(t.startswith("mateIn") for t in themes):
        winner = entry.get("winner")
        if winner is None:
            winner = "white" if board.turn == chess.WHITE else "black"
        return 1.0 if winner == "white" else -1.0

    # Heuristics: material (primary) + theme adjustment + optional mobility
    material_diff = count_material(board)
    material_term = np.tanh(material_diff / 8.0) * 0.6
    value = material_term

    # Theme-based additive adjustment (signed by who is better)
    sign = 1 if material_diff >= 0 else -1
    if "equality" in themes:
        value = 0.0
    elif "crushing" in themes:
        value += sign * 0.3
    elif "advantage" in themes:
        value += sign * 0.2

    if use_mobility:
        mobility_diff = count_mobility(board)
        value += np.tanh(mobility_diff / 20.0) * 0.2

    if rng is None:
        seed = int(hashlib.sha256(fen.encode()).hexdigest()[:12], 16)
        rng = np.random.default_rng(seed)
    value += rng.normal(0, 0.03)
    return float(np.clip(value, -1.0, 1.0))


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
    *,
    endgame_expand_max_plies: int = DEFAULT_ENDGAME_EXPAND_MAX_PLIES,
    endgame_expand_max_moves: int = DEFAULT_ENDGAME_EXPAND_MAX_MOVES,
    endgame_expand_require_crushing: bool = DEFAULT_ENDGAME_EXPAND_REQUIRE_CRUSHING,
    endgame_expand_require_forcing_theme: bool = DEFAULT_ENDGAME_EXPAND_REQUIRE_FORCING_THEME,
):
    output_path = Path(output_path)
    output_map = _build_output_map(inputs, output_path)
    for path in output_map.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    source_labels = [Path(path).stem for path in inputs]
    endgame_opts = {
        "endgame_expand_max_plies": endgame_expand_max_plies,
        "endgame_expand_max_moves": endgame_expand_max_moves,
        "endgame_expand_require_crushing": endgame_expand_require_crushing,
        "endgame_expand_require_forcing_theme": endgame_expand_require_forcing_theme,
    }
    if workers is None:
        workers = min(os.cpu_count() or 1, len(inputs))
    if len(inputs) > 1 and workers > 1:
        written = _build_value_dataset_mp(
            inputs,
            output_map,
            max_rows=max_rows,
            include_themes=include_themes,
            workers=workers,
            **endgame_opts,
        )
        return written
    return _build_value_dataset_single(
        inputs,
        output_map,
        max_rows=max_rows,
        include_themes=include_themes,
        **endgame_opts,
    )


def _build_value_dataset_single(
    inputs,
    output_map,
    max_rows=None,
    include_themes=True,
    *,
    endgame_expand_max_plies: int = DEFAULT_ENDGAME_EXPAND_MAX_PLIES,
    endgame_expand_max_moves: int = DEFAULT_ENDGAME_EXPAND_MAX_MOVES,
    endgame_expand_require_crushing: bool = DEFAULT_ENDGAME_EXPAND_REQUIRE_CRUSHING,
    endgame_expand_require_forcing_theme: bool = DEFAULT_ENDGAME_EXPAND_REQUIRE_FORCING_THEME,
):
    written = 0
    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )

    with Progress(*progress_columns, transient=False) as progress:
        for source_id, input_path in enumerate(inputs):
            total = _count_json_array_lines(input_path)
            if max_rows is not None:
                total = min(total, max_rows)
            task_id = progress.add_task(f"Processing {Path(input_path).name}", total=total)
            output_file = output_map[input_path]
            with output_file.open("w", encoding="utf-8") as out_handle:
                for idx, entry in enumerate(iter_json_array(input_path), start=1):
                    if max_rows is not None and idx > max_rows:
                        break
                    fen, moves, puzzle_id, themes = _extract_entry(entry)
                    if not fen:
                        progress.update(task_id, advance=1)
                        continue
                    # Endgame position (no solution line): one row with pseudo-labeled value and deterministic policy
                    if not moves:
                        try:
                            board = chess.Board(fen)
                        except ValueError:
                            progress.update(task_id, advance=1)
                            continue
                        value_target = label_endgame_value(entry)
                        # Use first legal move (UCI order) as policy label so every row has an actual move
                        legal_list = list(board.legal_moves)
                        first_move = min(legal_list, key=lambda m: m.uci()) if legal_list else None
                        policy_uci = first_move.uci() if first_move is not None else None
                        payload = {
                            "fen": fen,
                            "value_target": value_target,
                            "policy_uci": policy_uci,
                            "position_type": "endgame",  # no moves_to_mate; game does not finish in a fixed sequence
                        }
                        if puzzle_id:
                            payload["puzzle_id"] = puzzle_id
                        if include_themes and themes:
                            payload["themes"] = themes
                        out_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                        written += 1
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

                    if _is_mate_puzzle(themes):
                        # Mate puzzle: expand full trajectory with moves_to_mate
                        board = chess.Board(fen)
                        total_plies = len(parsed_moves)
                        for ply_index, move in enumerate(parsed_moves):
                            value_target = 1.0 if ply_index % 2 == 0 else -1.0
                            moves_to_mate = total_plies - ply_index
                            payload = {
                                "fen": board.fen(),
                                "value_target": value_target,
                                "policy_uci": move.uci(),
                                "moves_to_mate": moves_to_mate,
                                "position_type": "mate",
                            }
                            if puzzle_id:
                                payload["puzzle_id"] = puzzle_id
                            if include_themes and themes:
                                payload["themes"] = themes
                            out_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                            written += 1
                            board.push(move)
                    elif _should_expand_endgame(
                        themes,
                        moves,
                        max_moves=endgame_expand_max_moves,
                        require_crushing=endgame_expand_require_crushing,
                        require_forcing_theme=endgame_expand_require_forcing_theme,
                    ):
                        # Endgame: short forcing tactic — expand full trajectory (no moves_to_mate)
                        board = chess.Board(fen)
                        max_ply_idx = endgame_expand_max_plies - 1
                        for ply_index, move in enumerate(parsed_moves):
                            if ply_index > max_ply_idx:
                                break
                            value_target = 1.0 if ply_index % 2 == 0 else -1.0
                            payload = {
                                "fen": board.fen(),
                                "value_target": value_target,
                                "policy_uci": move.uci(),
                                "position_type": "endgame",
                                "endgame_ply": ply_index,
                            }
                            if puzzle_id:
                                payload["puzzle_id"] = puzzle_id
                            if include_themes and themes:
                                payload["themes"] = themes
                            out_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                            written += 1
                            board.push(move)
                    else:
                        # Endgame: first move only; binary value 1.0 (solver / side to move wins)
                        first_move = parsed_moves[0]
                        value_target = _endgame_solver_value_target(entry)
                        payload = {
                            "fen": fen,
                            "value_target": value_target,
                            "policy_uci": first_move.uci(),
                            "position_type": "endgame",
                            "endgame_ply": 0,
                        }
                        if puzzle_id:
                            payload["puzzle_id"] = puzzle_id
                        if include_themes and themes:
                            payload["themes"] = themes
                        out_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                        written += 1
                    progress.update(task_id, advance=1)
    return written


def _build_value_dataset_mp(
    inputs,
    output_map,
    max_rows=None,
    include_themes=True,
    workers=1,
    *,
    endgame_expand_max_plies: int = DEFAULT_ENDGAME_EXPAND_MAX_PLIES,
    endgame_expand_max_moves: int = DEFAULT_ENDGAME_EXPAND_MAX_MOVES,
    endgame_expand_require_crushing: bool = DEFAULT_ENDGAME_EXPAND_REQUIRE_CRUSHING,
    endgame_expand_require_forcing_theme: bool = DEFAULT_ENDGAME_EXPAND_REQUIRE_FORCING_THEME,
):
    per_file_total = {}
    total_all = 0
    for input_path in inputs:
        count = _count_json_array_lines(input_path)
        if max_rows is not None:
            count = min(count, max_rows)
        per_file_total[input_path] = count
        total_all += count

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )

    temp_dir = next(iter(output_map.values())).parent / ".tmp_value_build"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_paths = {}
    chunk_tasks = []
    for input_path in inputs:
        count = per_file_total.get(input_path, 0)
        if count == 0:
            continue
        file_share = count / total_all if total_all else 0
        alloc_workers = max(1, int(round(workers * file_share)))
        chunk_size = max(1, math.ceil(count / alloc_workers))
        num_chunks = math.ceil(count / chunk_size)
        temp_paths[input_path] = []
        for chunk_index in range(num_chunks):
            start_idx = chunk_index * chunk_size
            end_idx = min(start_idx + chunk_size, count)
            temp_path = temp_dir / f"{Path(input_path).stem}_part_{chunk_index:03d}.jsonl"
            temp_paths[input_path].append(temp_path)
            chunk_tasks.append(
                (
                    input_path,
                    temp_path,
                    start_idx,
                    end_idx,
                    max_rows,
                    include_themes,
                    endgame_expand_max_plies,
                    endgame_expand_max_moves,
                    endgame_expand_require_crushing,
                    endgame_expand_require_forcing_theme,
                )
            )

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = []
    for task in chunk_tasks:
        (
            input_path,
            temp_path,
            start_idx,
            end_idx,
            max_rows_value,
            include_themes_value,
            eg_max_plies,
            eg_max_moves,
            eg_require_crushing,
            eg_require_forcing,
        ) = task
        proc = ctx.Process(
            target=_process_input_chunk,
            args=(
                input_path,
                temp_path,
                start_idx,
                end_idx,
                max_rows_value,
                include_themes_value,
                progress_queue,
                1000,
                eg_max_plies,
                eg_max_moves,
                eg_require_crushing,
                eg_require_forcing,
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
        while completed < len(chunk_tasks):
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

    for input_path in inputs:
        output_file = output_map[input_path]
        with output_file.open("w", encoding="utf-8") as out_handle:
            for temp_path in temp_paths.get(input_path, []):
                if not temp_path.exists():
                    continue
                with temp_path.open("r", encoding="utf-8") as in_handle:
                    for line in in_handle:
                        out_handle.write(line)
    for temp_list in temp_paths.values():
        for temp_path in temp_list:
            if temp_path.exists():
                temp_path.unlink()
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    return written_total


def _process_input_chunk(
    input_path,
    temp_path,
    start_idx,
    end_idx,
    max_rows,
    include_themes,
    progress_queue,
    progress_batch,
    endgame_expand_max_plies: int = DEFAULT_ENDGAME_EXPAND_MAX_PLIES,
    endgame_expand_max_moves: int = DEFAULT_ENDGAME_EXPAND_MAX_MOVES,
    endgame_expand_require_crushing: bool = DEFAULT_ENDGAME_EXPAND_REQUIRE_CRUSHING,
    endgame_expand_require_forcing_theme: bool = DEFAULT_ENDGAME_EXPAND_REQUIRE_FORCING_THEME,
):
    written = 0
    processed = 0
    with Path(temp_path).open("w", encoding="utf-8") as out_handle:
        for entry in _iter_json_array_chunk(input_path, start_idx, end_idx, max_rows):
            processed += 1
            if processed % progress_batch == 0:
                progress_queue.put(("progress", input_path, progress_batch))
            fen, moves, puzzle_id, themes = _extract_entry(entry)
            if not fen:
                continue
            if not moves:
                try:
                    board = chess.Board(fen)
                except ValueError:
                    continue
                value_target = label_endgame_value(entry)
                legal_list = list(board.legal_moves)
                first_move = min(legal_list, key=lambda m: m.uci()) if legal_list else None
                policy_uci = first_move.uci() if first_move is not None else None
                payload = {
                    "fen": fen,
                    "value_target": value_target,
                    "policy_uci": policy_uci,
                    "position_type": "endgame",  # no moves_to_mate; game does not finish in a fixed sequence
                }
                if puzzle_id:
                    payload["puzzle_id"] = puzzle_id
                if include_themes and themes:
                    payload["themes"] = themes
                out_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                written += 1
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

            if _is_mate_puzzle(themes):
                board = chess.Board(fen)
                total_plies = len(parsed_moves)
                for ply_index, move in enumerate(parsed_moves):
                    value_target = 1.0 if ply_index % 2 == 0 else -1.0
                    moves_to_mate = total_plies - ply_index
                    payload = {
                        "fen": board.fen(),
                        "value_target": value_target,
                        "policy_uci": move.uci(),
                        "moves_to_mate": moves_to_mate,
                        "position_type": "mate",
                    }
                    if puzzle_id:
                        payload["puzzle_id"] = puzzle_id
                    if include_themes and themes:
                        payload["themes"] = themes
                    out_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                    written += 1
                    board.push(move)
            elif _should_expand_endgame(
                themes,
                moves,
                max_moves=endgame_expand_max_moves,
                require_crushing=endgame_expand_require_crushing,
                require_forcing_theme=endgame_expand_require_forcing_theme,
            ):
                board = chess.Board(fen)
                max_ply_idx = endgame_expand_max_plies - 1
                for ply_index, move in enumerate(parsed_moves):
                    if ply_index > max_ply_idx:
                        break
                    value_target = 1.0 if ply_index % 2 == 0 else -1.0
                    payload = {
                        "fen": board.fen(),
                        "value_target": value_target,
                        "policy_uci": move.uci(),
                        "position_type": "endgame",
                        "endgame_ply": ply_index,
                    }
                    if puzzle_id:
                        payload["puzzle_id"] = puzzle_id
                    if include_themes and themes:
                        payload["themes"] = themes
                    out_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                    written += 1
                    board.push(move)
            else:
                first_move = parsed_moves[0]
                value_target = _endgame_solver_value_target(entry)
                payload = {
                    "fen": fen,
                    "value_target": value_target,
                    "policy_uci": first_move.uci(),
                    "position_type": "endgame",
                    "endgame_ply": 0,
                }
                if puzzle_id:
                    payload["puzzle_id"] = puzzle_id
                if include_themes and themes:
                    payload["themes"] = themes
                out_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                written += 1
    remainder = processed % progress_batch
    if remainder:
        progress_queue.put(("progress", input_path, remainder))
    progress_queue.put(("done", input_path, written))


def _iter_json_array_chunk(path: str, start_idx: int, end_idx: int, max_rows: int | None):
    data_idx = 0
    max_idx = end_idx
    if max_rows is not None:
        max_idx = min(max_idx, max_rows)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped == "[" or stripped == "]":
                continue
            if stripped.endswith(","):
                stripped = stripped[:-1].rstrip()
            if data_idx < start_idx:
                data_idx += 1
                continue
            if data_idx >= max_idx:
                break
            data_idx += 1
            yield json.loads(stripped)


def _count_json_array_lines(path: str | Path) -> int:
    total_lines = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for _ in handle:
            total_lines += 1
    return max(total_lines - 2, 0)


def _build_output_map(inputs: list[str], output_path: Path) -> dict[str, Path]:
    output_dir = output_path if output_path.suffix != ".jsonl" else output_path.parent
    return {
        input_path: output_dir / f"{Path(input_path).stem}_expanded.jsonl"
        for input_path in inputs
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build value training dataset from mate puzzles and/or endgame positions. "
        "Mate puzzles: value_target from solution; endgame (entries with fen, no moves): pseudo-labels from themes + material (see puzzleTheme.xml)."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/mate_in_1_flipped.json",
            "data/mate_in_2_flipped.json",
            "data/mate_in_3_flipped.json",
            "data/mate_in_4_flipped.json",
            "data/mate_in_5_flipped.json",
            "data/endgame_without_mate_flipped.json",
        ],
        help="Input JSON array files. Mate files have FEN + Moves; endgame files have FEN + optional Themes, no Moves (one value_target row per entry).",
    )
    parser.add_argument(
        "--output",
        default="data",
        help=(
            "Output directory or JSONL path. "
            "If a directory (default), writes one *_processed.jsonl per input."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--include-themes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include themes field in output (default: True).",
    )
    parser.add_argument(
        "--endgame-expand-max-plies",
        type=int,
        default=DEFAULT_ENDGAME_EXPAND_MAX_PLIES,
        metavar="N",
        help=f"Number of plies to emit when expanding endgame (0..N-1). Default: {DEFAULT_ENDGAME_EXPAND_MAX_PLIES}.",
    )
    parser.add_argument(
        "--endgame-expand-max-moves",
        type=int,
        default=DEFAULT_ENDGAME_EXPAND_MAX_MOVES,
        metavar="N",
        help=f"Only expand endgame when puzzle has at most N moves. Default: {DEFAULT_ENDGAME_EXPAND_MAX_MOVES}.",
    )
    parser.add_argument(
        "--endgame-expand-require-crushing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require 'crushing' theme to expand endgame (default: True).",
    )
    parser.add_argument(
        "--endgame-expand-require-forcing-theme",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a forcing theme (fork/pin/skewer/etc.) to expand endgame (default: True).",
    )
    args = parser.parse_args()

    written = build_value_dataset(
        inputs=args.inputs,
        output_path=args.output,
        max_rows=args.max_rows,
        include_themes=args.include_themes,
        workers=args.workers,
        endgame_expand_max_plies=args.endgame_expand_max_plies,
        endgame_expand_max_moves=args.endgame_expand_max_moves,
        endgame_expand_require_crushing=args.endgame_expand_require_crushing,
        endgame_expand_require_forcing_theme=args.endgame_expand_require_forcing_theme,
    )
    output_path = Path(args.output)
    print(f"Wrote {written} rows across files in {output_path}/")


if __name__ == "__main__":
    main()
