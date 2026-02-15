"""
Generate minimal endgame dataset: positions where mate is possible with minimal pieces.
Endgames: K vs K+Q, K vs K+R, K vs K+2B, K vs K+B+N.
Output format: JSON array with PuzzleId, FEN, Themes (similar to mate_in_5.json).
Writes each endgame type to a separate file.

Uses multiprocessing for speed and rich for progress bars.
"""

import json
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess

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


# Type key -> (ID prefix, theme label, progress label). K comes first (defender/loser side).
TYPE_CONFIG = {
    "KQ": ("K_vs_KQ", "K_vs_KQ", "K vs K+Q"),
    "KR": ("K_vs_KR", "K_vs_KR", "K vs K+R"),
    "KBB": ("K_vs_KBB", "K_vs_KBB", "K vs K+2B"),
    "KBN": ("K_vs_KBN", "K_vs_KBN", "K vs K+B+N"),
}

# Edge squares (rank 0/7 or file 0/7) - easier to checkmate
EDGE_SQUARES = frozenset(
    s for s in range(64)
    if chess.square_rank(s) in (0, 7) or chess.square_file(s) in (0, 7)
)


def typed_puzzle_id(prefix: str, index: int, width: int = 7) -> str:
    """ID with prefix (K_vs_...) and zero-padded index. E.g. K_vs_KQ0000000."""
    return f"{prefix}{index:0{width}d}"


def swap_colors_only(fen: str) -> str:
    """Swap piece colors only; keep board perspective fixed. Winning side becomes Black."""
    board = chess.Board(fen)
    new_board = chess.Board(None)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            new_board.set_piece_at(square, chess.Piece(piece.piece_type, not piece.color))
    new_board.turn = not board.turn
    return new_board.fen()


def generate_kq_vs_k(n: int, start_index: int = 0, easy: bool = False) -> list[dict]:
    """K+Q vs K: Queen vs bare king. Always winnable. easy: opponent king on edge."""
    positions = []
    attempts = 0
    max_attempts = n * 80 if easy else n * 50

    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        board = chess.Board(None)
        wk = random.choice(range(64))
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))

        valid = [s for s in range(64) if s != wk and chess.square_distance(s, wk) > 1]
        if not valid:
            continue
        wq = random.choice(valid)
        board.set_piece_at(wq, chess.Piece(chess.QUEEN, chess.WHITE))

        bk_valid = [
            s for s in range(64)
            if s not in (wk, wq)
            and chess.square_distance(s, wk) > 1
            and not board.is_attacked_by(chess.WHITE, s)
        ]
        if easy:
            bk_valid = [s for s in bk_valid if s in EDGE_SQUARES]
        if not bk_valid:
            continue
        bk = random.choice(bk_valid)
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE

        if board.is_checkmate() or board.is_stalemate():
            continue

        themes = ["minimalMate", "endgame", "K_vs_KQ"]
        if easy:
            themes.append("edgeKing")
        positions.append({
            "PuzzleId": typed_puzzle_id("K_vs_KQ", start_index + len(positions)),
            "FEN": board.fen(),
            "Themes": themes,
        })

    return positions


def generate_kr_vs_k(n: int, start_index: int = 0, easy: bool = False) -> list[dict]:
    """K+R vs K: Rook vs bare king. Always winnable. easy: opponent king on edge."""
    positions = []
    attempts = 0
    max_attempts = n * 80 if easy else n * 50

    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        board = chess.Board(None)
        wk = random.choice(range(64))
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))

        valid = [s for s in range(64) if s != wk and chess.square_distance(s, wk) > 1]
        if not valid:
            continue
        wr = random.choice(valid)
        board.set_piece_at(wr, chess.Piece(chess.ROOK, chess.WHITE))

        bk_valid = [
            s for s in range(64)
            if s not in (wk, wr)
            and chess.square_distance(s, wk) > 1
            and not board.is_attacked_by(chess.WHITE, s)
        ]
        if easy:
            bk_valid = [s for s in bk_valid if s in EDGE_SQUARES]
        if not bk_valid:
            continue
        bk = random.choice(bk_valid)
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE

        if board.is_checkmate() or board.is_stalemate():
            continue

        themes = ["minimalMate", "endgame", "K_vs_KR"]
        if easy:
            themes.append("edgeKing")
        positions.append({
            "PuzzleId": typed_puzzle_id("K_vs_KR", start_index + len(positions)),
            "FEN": board.fen(),
            "Themes": themes,
        })

    return positions


def _is_light_square(s: int) -> bool:
    """Light square = (file + rank) % 2 == 1. a1 is dark."""
    return (chess.square_file(s) + chess.square_rank(s)) % 2 == 1


def generate_kbb_vs_k(n: int, start_index: int = 0, easy: bool = False) -> list[dict]:
    """K+2B vs K: Two bishops vs bare king. Wins only with opposite-colored bishops. easy: opponent king on edge."""
    positions = []
    attempts = 0
    max_attempts = n * 120 if easy else n * 80
    light_squares = [s for s in range(64) if _is_light_square(s)]
    dark_squares = [s for s in range(64) if not _is_light_square(s)]

    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        board = chess.Board(None)
        wk = random.choice(range(64))
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))

        # Opposite-colored bishops: one on light, one on dark
        wb1 = random.choice(light_squares)
        if wb1 == wk:
            continue
        wb2 = random.choice(dark_squares)
        if wb2 == wk or wb2 == wb1:
            continue
        if chess.square_distance(wb1, wk) < 2 or chess.square_distance(wb2, wk) < 2:
            continue

        board.set_piece_at(wb1, chess.Piece(chess.BISHOP, chess.WHITE))
        board.set_piece_at(wb2, chess.Piece(chess.BISHOP, chess.WHITE))

        bk_valid = [
            s for s in range(64)
            if s not in (wk, wb1, wb2)
            and chess.square_distance(s, wk) > 1
            and not board.is_attacked_by(chess.WHITE, s)
        ]
        if easy:
            bk_valid = [s for s in bk_valid if s in EDGE_SQUARES]
        if not bk_valid:
            continue
        bk = random.choice(bk_valid)
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE

        if board.is_checkmate() or board.is_stalemate():
            continue

        themes = ["minimalMate", "endgame", "K_vs_KBB"]
        if easy:
            themes.append("edgeKing")
        positions.append({
            "PuzzleId": typed_puzzle_id("K_vs_KBB", start_index + len(positions)),
            "FEN": board.fen(),
            "Themes": themes,
        })

    return positions


def generate_kbn_vs_k(n: int, start_index: int = 0, easy: bool = False) -> list[dict]:
    """K+B+N vs K: Bishop and knight vs bare king. Always winnable. easy: opponent king on edge."""
    positions = []
    attempts = 0
    max_attempts = n * 120 if easy else n * 80

    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        board = chess.Board(None)
        wk = random.choice(range(64))
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))

        valid = [s for s in range(64) if s != wk and chess.square_distance(s, wk) > 1]
        if len(valid) < 2:
            continue
        wb, wn = random.sample(valid, 2)

        board.set_piece_at(wb, chess.Piece(chess.BISHOP, chess.WHITE))
        board.set_piece_at(wn, chess.Piece(chess.KNIGHT, chess.WHITE))

        bk_valid = [
            s for s in range(64)
            if s not in (wk, wb, wn)
            and chess.square_distance(s, wk) > 1
            and not board.is_attacked_by(chess.WHITE, s)
        ]
        if easy:
            bk_valid = [s for s in bk_valid if s in EDGE_SQUARES]
        if not bk_valid:
            continue
        bk = random.choice(bk_valid)
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE

        if board.is_checkmate() or board.is_stalemate():
            continue

        themes = ["minimalMate", "endgame", "K_vs_KBN"]
        if easy:
            themes.append("edgeKing")
        positions.append({
            "PuzzleId": typed_puzzle_id("K_vs_KBN", start_index + len(positions)),
            "FEN": board.fen(),
            "Themes": themes,
        })

    return positions


def _worker(args: tuple) -> tuple[str, list[dict]]:
    """Worker for multiprocessing. Returns (type_key, positions)."""
    gen_name, n, start_index, seed, easy = args
    random.seed(seed)
    generators = {
        "KQ": generate_kq_vs_k,
        "KR": generate_kr_vs_k,
        "KBB": generate_kbb_vs_k,
        "KBN": generate_kbn_vs_k,
    }
    positions = generators[gen_name](n, start_index, easy=easy)
    return (gen_name, positions)


def generate_minimal_endgames(
    kq: int = 100,
    kr: int = 100,
    kbb: int = 50,
    kbn: int = 50,
    seed: int | None = None,
    workers: int | None = None,
    use_multiprocess: bool = True,
    easy: bool = False,
) -> dict[str, list[dict]]:
    """Generate minimal endgame positions. easy: opponent king on edge (easier to mate)."""
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    configs = [("KQ", kq), ("KR", kr), ("KBB", kbb), ("KBN", kbn)]

    # Build chunk tasks: (gen_name, chunk_n, start_index, seed, easy)
    tasks: list[tuple[str, int, int, int, bool]] = []
    base_seed = (seed or 42) * 10000
    easy_offset = 50000 if easy else 0
    for gen_name, total in configs:
        if total <= 0:
            continue
        n_chunks = min(workers, total)
        chunk_size = (total + n_chunks - 1) // n_chunks
        for i in range(n_chunks):
            start = i * chunk_size
            n = min(chunk_size, total - start)
            if n > 0:
                tasks.append((gen_name, n, start, base_seed + hash(gen_name) % 10000 + i * 1000 + easy_offset, easy))

    positions_by_type: dict[str, list] = {"KQ": [], "KR": [], "KBB": [], "KBN": []}

    if use_multiprocess and len(tasks) > 1:
        task_ids = {}
        if Progress:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("•"),
                TextColumn("[cyan]{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                for gen_name, total in configs:
                    if total > 0:
                        label = f"{TYPE_CONFIG[gen_name][2]}{' (easy)' if easy else ''}"
                        task_ids[gen_name] = progress.add_task(label, total=total)

                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_worker, t): t for t in tasks}
                    for future in as_completed(futures):
                        gen_name, pos_list = future.result()
                        positions_by_type[gen_name].extend(pos_list)
                        if Progress and gen_name in task_ids:
                            progress.update(task_ids[gen_name], advance=len(pos_list))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_worker, t): t for t in tasks}
                for future in as_completed(futures):
                    gen_name, pos_list = future.result()
                    positions_by_type[gen_name].extend(pos_list)
    else:
        if seed is not None:
            random.seed(seed)
        generators = {
            "KQ": generate_kq_vs_k,
            "KR": generate_kr_vs_k,
            "KBB": generate_kbb_vs_k,
            "KBN": generate_kbn_vs_k,
        }
        for gen_name, total in configs:
            if total <= 0:
                continue
            gen = generators[gen_name]
            pos_list = gen(total, 0, easy=easy)
            if Progress:
                with Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=40),
                    TaskProgressColumn(),
                    TextColumn("•"),
                    TextColumn("[cyan]{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                ) as progress:
                    label = f"{TYPE_CONFIG[gen_name][2]}{' (easy)' if easy else ''}"
                    task_id = progress.add_task(label, total=total)
                    progress.update(task_id, advance=len(pos_list))
            positions_by_type[gen_name].extend(pos_list)

    # Flip color on half of each type (training on both perspectives; turn stays with winning side)
    for gen_name in positions_by_type:
        positions = positions_by_type[gen_name]
        n = len(positions)
        if n == 0:
            continue
        flip_indices = set(random.sample(range(n), n // 2))
        for i in flip_indices:
            positions[i] = {
                **positions[i],
                "FEN": swap_colors_only(positions[i]["FEN"]),
            }

    # Sort by PuzzleId (0000000, 0000001, ...)
    for gen_name in positions_by_type:
        positions_by_type[gen_name].sort(key=lambda p: p["PuzzleId"])
    return positions_by_type


def _write_positions(positions: list[dict], out_file: Path) -> None:
    """Write positions to JSON file."""
    with open(out_file, "w") as f:
        lines = [json.dumps(p, separators=(",", ":")) for p in positions]
        f.write("[\n" + ",\n".join(lines) + "\n]\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate minimal endgame dataset")
    parser.add_argument("--output-dir", "-o", default="data", help="Directory for output JSON files")
    parser.add_argument(
        "--count", "-n", type=int, default=100000,
        help="Positions per endgame type per file (KQ, KR, KBB, KBN)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", "-j", type=int, default=None, help="Parallel workers (default: cpu_count-1)")
    parser.add_argument("--no-multiprocess", action="store_true", help="Run single-threaded")
    parser.add_argument("--standard-only", action="store_true", help="Generate only standard (4 files)")
    parser.add_argument("--easy-only", action="store_true", help="Generate only easy (4 files)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_standard = not args.easy_only
    gen_easy = not args.standard_only

    total = 0
    for easy in [False, True]:
        if easy and not gen_easy:
            continue
        if not easy and not gen_standard:
            continue
        suffix = "_easy" if easy else ""
        print(f"\nGenerating {'easy (opponent king on edge)' if easy else 'standard'} positions...")
        positions_by_type = generate_minimal_endgames(
            kq=args.count,
            kr=args.count,
            kbb=args.count,
            kbn=args.count,
            seed=args.seed,
            workers=args.workers,
            use_multiprocess=not args.no_multiprocess,
            easy=easy,
        )
        for gen_name in ["KQ", "KR", "KBB", "KBN"]:
            positions = positions_by_type.get(gen_name, [])
            if not positions:
                continue
            out_file = out_dir / f"minimal_endgames_K_vs_{gen_name}{suffix}.json"
            _write_positions(positions, out_file)
            total += len(positions)
            print(f"  Wrote {len(positions)} positions to {out_file}")

    n_files = (4 if gen_standard else 0) + (4 if gen_easy else 0)
    print(f"\nTotal: {total} positions in {n_files} files")


if __name__ == "__main__":
    main()
