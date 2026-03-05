#!/usr/bin/env python3
"""
Test solution trajectory parsing for a specific puzzle position.
Verifies: (1) solution moves are legal, (2) action_id from move_to_action_id
is in legal_actions, (3) no mismatch between sol_board and env board.

Run from project root: python test/test_solution_trajectory.py
(Requires torch - use the same env as MCTS/train.py)

Root cause of follow_dataset_trajectory failure (fixed in chess_custom.py):
  BaseChessBoard.legal_moves was checking the wrong side after push(move).
  It used is_check() which checks the NEW side to move, but we need to verify
  the side that JUST moved (who was in check) escaped. This incorrectly
  filtered h5g4 when Black captured the checking pawn - after the capture,
  White is in check, so the old code saw is_check()=True and dropped the move.
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import chess


def test_legality_only(fen: str, moves_uci: list[str]) -> bool:
    """Verify moves are legal using plain python-chess (no action_id)."""
    board = chess.Board(fen)
    for i, move_str in enumerate(moves_uci):
        if board.is_game_over():
            return False
        try:
            move = chess.Move.from_uci(move_str)
        except ValueError:
            print(f"  Move {i+1} ({move_str}): Invalid UCI")
            return False
        if move not in board.legal_moves:
            print(f"  Move {i+1} ({move_str}): NOT LEGAL")
            return False
        board.push(move)
    return True


def test_with_legacy_board(fen: str, moves_uci: list[str]) -> None:
    """Test solution trajectory with LegacyChessBoard (action_id, legal_actions)."""
    from MCTS.training_modules.chess import create_board_from_fen

    print("=" * 60)
    print("Testing puzzle position (LegacyChessBoard / 4672 action space)")
    print("=" * 60)
    print(f"FEN: {fen}")
    print(f"Moves (UCI): {moves_uci}")
    print()

    sol_board = create_board_from_fen(fen)

    print("--- Move-by-move parsing (same logic as train.py) ---")
    solution_action_ids = []
    for i, move_str in enumerate(moves_uci):
        if sol_board.is_game_over():
            print(f"  Move {i+1}: Game over, stopping")
            break

        move = None
        try:
            if len(move_str) >= 4 and move_str[1].isdigit():
                move = chess.Move.from_uci(move_str)
            else:
                move = sol_board.parse_san(move_str)
        except Exception as e:
            print(f"  Move {i+1} ({move_str}): Parse failed: {e}")
            break

        in_legal = move in sol_board.legal_moves if move else False
        print(f"  Move {i+1} ({move_str}): move={move}, in_legal_moves={in_legal}")

        if not (move and in_legal):
            print(f"    -> FAIL: move not legal, breaking")
            break

        aid = sol_board.move_to_action_id(move)
        print(f"    action_id={aid}")

        if aid is None:
            print(f"    -> FAIL: move_to_action_id returned None, breaking")
            break

        solution_action_ids.append(aid)
        sol_board.push(move)

    print()
    print(f"solution_action_ids: {solution_action_ids}")
    print()

    # Verify first move's action_id is in legal_actions
    if solution_action_ids:
        fresh_board = create_board_from_fen(fen)
        legal_actions = fresh_board.legal_actions
        first_aid = solution_action_ids[0]

        print("--- Action ID vs legal_actions check ---")
        print(f"First solution action_id: {first_aid}")
        print(f"legal_actions count: {len(legal_actions)}")
        in_legal = first_aid in legal_actions
        print(f"first_aid in legal_actions: {in_legal}")

        if not in_legal:
            print()
            print("  *** MISMATCH: First solution action_id NOT in legal_actions! ***")
            # Show action_id for h5g4 (dataset move)
            h5g4 = chess.Move.from_uci("h5g4")
            if h5g4 in fresh_board.legal_moves:
                aid_h5g4 = fresh_board.move_to_action_id(h5g4)
                print(f"  h5g4 (dataset move) -> action_id: {aid_h5g4}")
                print(f"  h5g4 action_id in legal_actions: {aid_h5g4 in legal_actions if aid_h5g4 else 'N/A'}")
            # List legal action_ids for first few moves
            print("  Sample legal action_ids (first 10):", sorted(legal_actions)[:10])
        else:
            print("  OK: First solution action_id is in legal_actions")
    else:
        print("No solution_action_ids parsed - cannot verify.")


PUZZLES = [
    # Original (i7tBb) - was affected by the bug
    {
        "id": "i7tBb",
        "fen": "4r3/2B5/pR6/2p2kpp/2B3P1/2P4K/P2r4/6N1 b - - 0 30",
        "moves": ["h5g4", "h3g3", "e8e3", "g1f3", "e3f3"],
    },
    {"id": "KJ4wd", "fen": "r2q1k2/p1pbb1pn/1p5p/4N2Q/8/2N5/PPP3PP/R4rK1 w - - 0 18", "moves": ["a1f1", "f8g8", "h5f7", "g8h8", "e5g6"]},
    {"id": "8LAFw", "fen": "2Q5/pp4p1/1p2p3/2k2p2/2Pq3r/PB6/1P3PPb/3R1R1K b - - 1 29", "moves": ["h2c7", "h1g1", "h4h1", "g1h1", "d4h4", "h1g1", "h4h2"]},
    {"id": "cNzo4", "fen": "5B2/Q2R1p1k/1p2p2p/3p2N1/3PnK2/4P2P/PP4P1/7q b - - 0 26", "moves": ["h6g5", "f4e5", "h1h2", "g2g3", "h2g3"]},
    # 3sisn - underpromotion g2g1n vs capture g2f1n
    {"id": "3sisn", "fen": "1k4r1/p5r1/3R3p/4p2n/1NQ1P2b/P4P1K/1PP3p1/5N2 b - - 1 34", "moves": ["g2g1n", "h3h4", "g1f3", "h4h5", "g7g5", "h5h6", "g8h8"]},
]


def would_old_bug_filter_move(board, move: chess.Move) -> bool:
    """
    Simulate OLD bug: after push(move), old code used is_check() which checks
    the NEW side to move. If that side is in check, the move was filtered OUT.
    Returns True if the old bug would have incorrectly filtered this move.
    """
    if not board.is_check():
        return False  # No filtering when not in check
    test = board.copy()
    test.push(move)
    # Old logic: if is_check() after push -> filter out the move
    # is_check() checks the side to move (opponent after our move)
    return test.is_check()


def test_puzzle_brief(puzzle: dict) -> tuple[bool, str]:
    """Test one puzzle. Returns (success, message)."""
    from MCTS.training_modules.chess import create_board_from_fen

    fen = puzzle["fen"]
    moves = puzzle["moves"]
    first_move = chess.Move.from_uci(moves[0]) if moves else None

    # Check with plain chess first
    std = chess.Board(fen)
    if first_move not in std.legal_moves:
        return False, f"First move {moves[0]} not in chess.Board legal_moves"

    # Would OLD bug have filtered the first move?
    std_copy = chess.Board(fen)
    bug_affected = would_old_bug_filter_move(std_copy, first_move)

    # Check with LegacyChessBoard (after fix)
    lb = create_board_from_fen(fen)
    lb_legal = list(lb.legal_moves)
    if first_move not in lb_legal:
        return False, f"First move {moves[0]} NOT in LegacyChessBoard legal_moves"

    msg_suffix = " (WAS bug-affected)" if bug_affected else ""

    # Full parse
    sol_board = create_board_from_fen(fen)
    solution_action_ids = []
    for move_str in moves:
        if sol_board.is_game_over():
            break
        move = chess.Move.from_uci(move_str) if len(move_str) >= 4 and move_str[1].isdigit() else sol_board.parse_san(move_str)
        if not (move and move in sol_board.legal_moves):
            return False, f"Move {move_str} failed at parse/legal check"
        aid = sol_board.move_to_action_id(move)
        if aid is None:
            return False, f"move_to_action_id returned None for {move_str}"
        solution_action_ids.append(aid)
        sol_board.push(move)

    if not solution_action_ids:
        return False, "solution_action_ids empty"
    fresh = create_board_from_fen(fen)
    if solution_action_ids[0] not in fresh.legal_actions:
        return False, f"First action_id {solution_action_ids[0]} not in legal_actions"
    return True, f"OK (parsed {len(solution_action_ids)} moves){msg_suffix}"


if __name__ == "__main__":
    import sys

    print("Testing all puzzles (including those that may have been affected by the legal_moves bug)\n")

    all_ok = True
    for p in PUZZLES:
        try:
            ok, msg = test_puzzle_brief(p)
            status = "PASS" if ok else "FAIL"
            print(f"  {p['id']}: {status} - {msg}")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"  {p['id']}: ERROR - {e}")
            all_ok = False

    print()
    if all_ok:
        print("All puzzles pass with the fix.")
    else:
        print("Some puzzles failed.")
        sys.exit(1)

    # Also run the original detailed test for the first puzzle
    print("\n--- Detailed test for i7tBb ---")
    try:
        test_with_legacy_board(PUZZLES[0]["fen"], PUZZLES[0]["moves"])
    except ImportError as e:
        print(f"Cannot run detailed test: {e}")
