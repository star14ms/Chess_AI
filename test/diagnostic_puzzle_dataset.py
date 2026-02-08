#!/usr/bin/env python3
"""
Evaluate puzzle datasets with MCTS.

For each position:
- Run MCTS for a fixed number of iterations per move.
- Play until checkmate or max_moves.
- Success if the initial side to move checkmates within max_moves.
"""

import argparse
import os
import sys
import multiprocessing as mp
from collections import defaultdict
import hashlib
import json
import random
import random

import chess
import torch
from omegaconf import OmegaConf
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from MCTS.mcts_algorithm import MCTS
from MCTS.mcts_node import MCTSNode
from MCTS.training_modules.chess import create_chess_network, create_board_from_fen
from utils.training_utils import iter_json_array


_WORKER_MCTS = None
_WORKER_CFG = None


def _init_worker(checkpoint_path: str, cfg_path: str, mcts_iterations: int) -> None:
    global _WORKER_MCTS, _WORKER_CFG
    torch.set_num_threads(1)
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network = create_chess_network(cfg, device)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    draw_reward = cfg.training.get("draw_reward", -0.1)
    cfg.runtime_mcts_iterations = int(mcts_iterations)
    _WORKER_CFG = cfg
    _WORKER_MCTS = MCTS(
        network=network,
        device=device,
        env=None,
        C_puct=cfg.mcts.c_puct,
        dirichlet_alpha=cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon=0.0,
        action_space_size=cfg.network.action_space_size,
        history_steps=cfg.env.history_steps,
        draw_reward=draw_reward,
    )


def _extract_moves(entry: dict) -> list[str]:
    moves_field = entry.get("Moves") or entry.get("moves")
    if not moves_field:
        return []
    if isinstance(moves_field, str):
        return [m for m in moves_field.split() if m.strip()]
    if isinstance(moves_field, list):
        return [str(m).strip() for m in moves_field if str(m).strip()]
    return []


def _extract_themes(entry: dict) -> list[str]:
    themes_field = entry.get("Themes") or entry.get("themes")
    if not themes_field:
        return []
    if isinstance(themes_field, list):
        return [str(t).strip() for t in themes_field if str(t).strip()]
    if isinstance(themes_field, str):
        return [t for t in themes_field.split() if t.strip()]
    return []


def _entry_matches_split(entry: dict, split: str, val_split: float, seed: int) -> bool:
    if val_split <= 0:
        return split == "train"
    if val_split >= 1:
        return split == "val"
    key = entry.get("PuzzleId") or entry.get("puzzle_id") or entry.get("FEN") or entry.get("fen")
    if not key:
        key = json.dumps(entry, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    is_val = bucket < val_split
    return is_val if split == "val" else not is_val


def _count_entries(
    paths: list[str],
    sample_size: int | None,
    val_only: bool,
    val_split: float,
    seed: int,
) -> int:
    total = 0
    for path in paths:
        count = 0
        for entry in iter_json_array(path):
            if val_only and not _entry_matches_split(entry, "val", val_split, seed):
                continue
            count += 1
            if sample_size is not None and sample_size > 0 and count >= sample_size:
                break
        total += count
    return total


def _select_best_action(root_node: MCTSNode) -> int | None:
    if not root_node.children:
        return None
    return max(root_node.children.items(), key=lambda kv: kv[1].N)[0]


def _play_position(
    mcts: MCTS,
    cfg: OmegaConf,
    fen: str,
    max_moves: int,
    mcts_iterations: int,
) -> tuple[bool, bool]:
    board = create_board_from_fen(fen)
    initial_turn = board.turn

    for _ in range(max_moves):
        if board.is_game_over():
            break

        root_node = MCTSNode(board.copy(stack=False))
        mcts.search(
            root_node,
            iterations=mcts_iterations,
            batch_size=cfg.mcts.batch_size,
            progress=None,
        )

        action_id = _select_best_action(root_node)
        if action_id is None:
            break

        move = root_node.board.action_id_to_move(action_id)
        if move is None:
            break

        board.push(move)

        if board.is_checkmate():
            winner = chess.WHITE if board.turn == chess.BLACK else chess.BLACK
            return winner == initial_turn, True

    return False, False


def _worker_play(task: tuple[str, str, int, list[str]]) -> tuple[str, bool, bool, list[str]]:
    path, fen, max_moves, themes = task
    success, checkmated = _play_position(
        mcts=_WORKER_MCTS,
        cfg=_WORKER_CFG,
        fen=fen,
        max_moves=max_moves,
        mcts_iterations=_WORKER_CFG.runtime_mcts_iterations,
    )
    return path, success, checkmated, themes


def _load_default_data(cfg: OmegaConf) -> tuple[list[str], dict]:
    data_files: list[str] = []
    per_file_max_moves: dict[str, int] = {}
    initial_board_fen = cfg.training.get("initial_board_fen", [])
    for entry in initial_board_fen or []:
        if isinstance(entry, dict) and entry.get("path"):
            data_files.append(entry["path"])
            if "max_game_moves" in entry:
                per_file_max_moves[entry["path"]] = int(entry["max_game_moves"])
        elif isinstance(entry, str):
            data_files.append(entry)
    return data_files, per_file_max_moves


def _build_config_max_moves(cfg: OmegaConf) -> dict:
    config_max_moves: dict[str, int] = {}
    initial_board_fen = cfg.training.get("initial_board_fen", [])
    for entry in initial_board_fen or []:
        if isinstance(entry, dict) and entry.get("path") and "max_game_moves" in entry:
            config_max_moves[entry["path"]] = int(entry["max_game_moves"])
    return config_max_moves


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate puzzle datasets with MCTS.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/model.pth",
        help="Path to checkpoint file (default: checkpoints/model.pth).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=256,
        help="MCTS iterations per move (default: 256).",
    )
    parser.add_argument(
        "--data",
        nargs="*",
        default=["data/mate_in_3.json", "data/mate_in_4.json", "data/mate_in_5.json"],
        help="One or more dataset files (JSON array). Defaults to config/train_mcts.yaml initial_board_fen.",
    )
    parser.add_argument(
        "--max-moves",
        nargs="*",
        type=int,
        default=[5, 7, 9],
        help=(
            "Optional per-file max_moves list (same order as --data). "
            "If omitted for a file, uses config max_game_moves or Moves length."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of CPU worker processes (default: CPU count - 1).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Sample size per file (default: 1000).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed for train/val split hashing (default: 1337).",
    )
    parser.add_argument(
        "--val-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample only validation split entries (default: True).",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle entries before sampling (default: True).",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, "config/train_mcts.yaml"))

    data_files = args.data
    per_file_max_moves: dict[str, int] = {}
    config_max_moves = _build_config_max_moves(cfg)

    if not data_files:
        data_files, per_file_max_moves = _load_default_data(cfg)
        if not data_files:
            raise SystemExit("No dataset files provided and config initial_board_fen is empty.")
    else:
        for path in data_files:
            if path in config_max_moves:
                per_file_max_moves[path] = config_max_moves[path]

    if args.max_moves:
        if len(args.max_moves) not in (1, len(data_files)):
            raise SystemExit("--max-moves must be length 1 or match --data length.")
        if len(args.max_moves) == 1:
            per_file_max_moves = {path: args.max_moves[0] for path in data_files}
        else:
            per_file_max_moves = {path: mm for path, mm in zip(data_files, args.max_moves)}

    gpu_device = None
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        gpu_device = torch.device("mps")

    cpu_workers = max(0, int(args.workers))
    mcts = None

    if gpu_device is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=gpu_device, weights_only=False)
        print("Building network...")
        network = create_chess_network(cfg, gpu_device)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.eval()

        draw_reward = cfg.training.get("draw_reward", -0.1)
        mcts = MCTS(
            network=network,
            device=gpu_device,
            env=None,
            C_puct=cfg.mcts.c_puct,
            dirichlet_alpha=cfg.mcts.dirichlet_alpha,
            dirichlet_epsilon=0.0,
            action_space_size=cfg.network.action_space_size,
            history_steps=cfg.env.history_steps,
            draw_reward=draw_reward,
        )
        if cpu_workers > 0:
            print(f"Using 1 GPU worker on {gpu_device} + {cpu_workers} CPU workers.")
        else:
            print(f"Using 1 GPU worker on {gpu_device}.")
    else:
        if cpu_workers <= 0:
            cpu_workers = 1
        print(f"Using {cpu_workers} CPU worker processes.")

    print("Counting total positions...")
    total_positions = _count_entries(
        data_files,
        args.sample_size,
        args.val_only,
        args.val_split,
        args.seed,
    )
    by_file = defaultdict(lambda: {"total": 0, "success": 0, "checkmates": 0, "skipped": 0})
    by_theme = defaultdict(lambda: {"total": 0, "success": 0})
    total_seen = 0
    total_success = 0

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn(
            "{task.fields[seen]}/{task.total} Done | {task.fields[success]}/{task.fields[seen]} Success ({task.fields[ratio]:.1f}%)"
        ),
    )

    def _iter_entries(path: str):
        if not args.shuffle:
            for entry in iter_json_array(path):
                if args.val_only and not _entry_matches_split(entry, "val", args.val_split, args.seed):
                    continue
                yield entry
            return
        entries = [
            entry
            for entry in iter_json_array(path)
            if (not args.val_only or _entry_matches_split(entry, "val", args.val_split, args.seed))
        ]
        random.shuffle(entries)
        for entry in entries:
            yield entry

    def task_generator():
        nonlocal total_seen
        for path in data_files:
            progress.print(f"\n--- Evaluating dataset: {path} ---")
            sampled = 0
            for entry in _iter_entries(path):
                if args.sample_size and sampled >= args.sample_size:
                    break
                by_file[path]["total"] += 1

                if not isinstance(entry, dict):
                    by_file[path]["skipped"] += 1
                    total_seen += 1
                    progress.update(task_id, advance=1, seen=total_seen)
                    continue

                fen = entry.get("FEN") or entry.get("fen")
                moves = _extract_moves(entry)
                if not fen or not moves:
                    by_file[path]["skipped"] += 1
                    total_seen += 1
                    progress.update(task_id, advance=1, seen=total_seen)
                    continue

                max_moves = per_file_max_moves.get(path, len(moves))
                themes = _extract_themes(entry)
                sampled += 1
                yield (path, fen, max_moves, themes)

    def apply_result(path: str, success: bool, checkmated: bool, themes: list[str]) -> None:
        nonlocal total_success, total_seen
        if themes:
            for theme in set(themes):
                by_theme[theme]["total"] += 1
                if success:
                    by_theme[theme]["success"] += 1
        if checkmated:
            by_file[path]["checkmates"] += 1
        if success:
            by_file[path]["success"] += 1
            total_success += 1
        total_seen += 1
        ratio = (total_success / total_seen * 100) if total_seen else 0.0
        progress.update(
            task_id,
            advance=1,
            success=total_success,
            seen=total_seen,
            ratio=ratio,
        )

    with progress:
        task_id = progress.add_task(
            "Evaluating",
            total=total_positions,
            success=0,
            seen=0,
            ratio=0.0,
        )

        if gpu_device is None and cpu_workers <= 1:
            for path, fen, max_moves, themes in task_generator():
                success, checkmated = _play_position(
                    mcts=mcts,
                    cfg=cfg,
                    fen=fen,
                    max_moves=max_moves,
                    mcts_iterations=args.iterations,
                )
                apply_result(path, success, checkmated, themes)
        elif gpu_device is None:
            ctx = mp.get_context("spawn")
            cfg_path = os.path.join(PROJECT_ROOT, "config/train_mcts.yaml")
            with ctx.Pool(
                processes=cpu_workers,
                initializer=_init_worker,
                initargs=(args.checkpoint, cfg_path, args.iterations),
            ) as pool:
                for path, success, checkmated, themes in pool.imap_unordered(
                    _worker_play, task_generator(), chunksize=8
                ):
                    apply_result(path, success, checkmated, themes)
        else:
            ctx = mp.get_context("spawn")
            cfg_path = os.path.join(PROJECT_ROOT, "config/train_mcts.yaml")
            pending = []
            max_pending = max(8, cpu_workers * 4)
            with ctx.Pool(
                processes=cpu_workers,
                initializer=_init_worker,
                initargs=(args.checkpoint, cfg_path, args.iterations),
            ) as pool:
                use_gpu = True
                for task in task_generator():
                    if cpu_workers == 0 or use_gpu:
                        path, fen, max_moves, themes = task
                        success, checkmated = _play_position(
                            mcts=mcts,
                            cfg=cfg,
                            fen=fen,
                            max_moves=max_moves,
                            mcts_iterations=args.iterations,
                        )
                        apply_result(path, success, checkmated, themes)
                    else:
                        pending.append(pool.apply_async(_worker_play, (task,)))

                    use_gpu = not use_gpu

                    while pending and len(pending) >= max_pending:
                        path, success, checkmated, themes = pending.pop(0).get()
                        apply_result(path, success, checkmated, themes)

                for async_result in pending:
                    path, success, checkmated, themes = async_result.get()
                    apply_result(path, success, checkmated, themes)

    print("\n=== Puzzle Dataset Evaluation ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"MCTS iterations per move: {args.iterations}")
    print("")
    overall_success = sum(stats["success"] for stats in by_file.values())
    overall_total = sum(stats["total"] for stats in by_file.values())
    overall_ratio = (overall_success / overall_total * 100) if overall_total else 0.0
    overall_skipped = sum(stats["skipped"] for stats in by_file.values())
    print(f"Overall success: {overall_success}/{overall_total} ({overall_ratio:.2f}%)")
    if overall_skipped:
        print(f"Skipped entries: {overall_skipped}")

    for path, stats in by_file.items():
        total = stats["total"]
        success = stats["success"]
        ratio = (success / total * 100) if total else 0.0
        print(f"- {path}: {success}/{total} ({ratio:.2f}%)")

    if by_theme:
        print("\n=== Theme Success (first-move) ===")
        for theme, stats in sorted(
            by_theme.items(), key=lambda item: (-item[1]["total"], item[0])
        ):
            total = stats["total"]
            success = stats["success"]
            ratio = (success / total * 100) if total else 0.0
            print(f"- {theme}: {success}/{total} ({ratio:.2f}%)")


if __name__ == "__main__":
    main()
