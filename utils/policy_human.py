# Cell content for sample_action function
import random
import chess
import numpy as np
import gymnasium as gym # Assuming gym is imported
from typing import Optional, List, Tuple, Union, overload, Literal, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Piece Values ---
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3.05,
    chess.BISHOP: 3.33,
    chess.ROOK: 5.63,
    chess.QUEEN: 9.5,
    chess.KING: 0 # King value is effectively infinite, 0 works for comparisons here
}
# ---

# --- Policy Titles ---
POLICY_TITLES = {
    0: "No Legal Moves",
    1: "Deliver Checkmate",
    2: "Avoid Immediate Checkmate Threat", # Implicitly used via filtering
    3: "Good Capture (Safe/Profitable)",
    4: "Attack Trapped Piece",
    5: "Avoid Attack (Evasion/Capture Attacker)",
    6: "Conditional Attack",
    7: "Maximize Min Mobility",
    8: "Fallback (Random from Considered)"
}
# ---

# --- Simple transposition cache for sample_action_v2 (negamax) ---
# Keyed by (transposition_key, depth, color)
_V2_TT: Dict[Tuple[int, int, int], float] = {}
_V2_TT_MAX_ENTRIES = 200_000


def _material_score_board(b: chess.Board) -> float:
    score = 0.0
    for pt in chess.PIECE_TYPES:
        score += len(b.pieces(pt, chess.WHITE)) * float(PIECE_VALUES.get(pt, 0))
        score -= len(b.pieces(pt, chess.BLACK)) * float(PIECE_VALUES.get(pt, 0))
    return score


def _evaluate_terminal_board(b: chess.Board, root_color: chess.Color) -> float:
    if b.is_checkmate():
        return -1e12 if b.turn == root_color else 1e12
    return _material_score_board(b)


def _negamax_worker(fen: str, move_uci: str, depth: int, color: int, root_color_is_white: bool) -> Tuple[str, float]:
    b = chess.Board(fen)
    mv = chess.Move.from_uci(move_uci)

    def negamax_local(board: chess.Board, d: int, c: int, alpha: float, beta: float) -> float:
        if d == 0 or board.is_game_over():
            rc = chess.WHITE if root_color_is_white else chess.BLACK
            return c * _evaluate_terminal_board(board, rc)
        best_val = -1e15
        for m in list(board.legal_moves):
            try:
                board.push(m)
                v = -negamax_local(board, d - 1, -c, -beta, -alpha)
                board.pop()
            except Exception:
                if board.move_stack and board.peek() == m:
                    board.pop()
                continue
            if v > best_val:
                best_val = v
            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                break
        return best_val

    try:
        b.push(mv)
        val = -negamax_local(b, max(0, depth), -color, -1e15, 1e15)
    except Exception:
        val = -1e15
    return move_uci, val

def get_piece_value(piece: Optional[chess.Piece]) -> int:
    """Gets the defined value of a piece, returns 0 if None."""
    if piece is None:
        return 0
    return PIECE_VALUES.get(piece.piece_type, 0)

def get_attacked_squares(board: chess.Board, player_color: chess.Color) -> set[chess.Square]:
    """Returns a set of squares occupied by `player_color`'s pieces that are attacked by the opponent."""
    attacked_squares = set()
    opponent_color = not player_color
    # Iterate through all squares where 'player_color' has pieces
    for piece_type in chess.PIECE_TYPES:
        for square in board.pieces(piece_type, player_color):
            if board.is_attacked_by(opponent_color, square):
                attacked_squares.add(square)
    return attacked_squares

# Helper to check if a square is safe after a move
def is_safe_after_move(board: chess.Board, move: chess.Move, opponent_color: chess.Color) -> bool:
    """Checks if the destination square of a move is safe from opponent attacks after the move."""
    try:
        board.push(move)
        is_safe = not board.is_attacked_by(opponent_color, move.to_square)
        board.pop()
        return is_safe
    except Exception:
        if board.move_stack and board.peek() == move: board.pop()
        return False # Assume unsafe if simulation fails

# Overload signatures for different return types based on return_info
@overload
def sample_action(board: chess.Board, avoid_attacks: bool = True, return_id: bool = False, return_move: bool = False, return_info: Literal[True] = True) -> Tuple[Union[np.ndarray, int, chess.Move, None], int, str]: ...
@overload
def sample_action(board: chess.Board, avoid_attacks: bool = True, return_id: bool = False, return_move: bool = False, return_info: Literal[False] = False) -> Union[np.ndarray, int, chess.Move, None]: ...

def sample_action(board: chess.Board, avoid_attacks: bool = True, return_id: bool = False, return_move: bool = False, return_info: bool = False) -> Union[Tuple[Union[np.ndarray, int, chess.Move, None], int, str], Union[np.ndarray, int, chess.Move, None]]:
    """
    Samples an action using a hierarchy of heuristics, operating directly on a board.

    Conditionally returns the action (as ID, numpy array, or Move object),
    or a tuple containing (action, policy_id, policy_title).

    Args:
        board: The current chess.Board state.
        avoid_attacks: Whether to include the 'Avoid Attack' heuristic (policy 5).
        return_id: If True and return_move is False, return action as integer ID.
                   Requires board to have a 'move_to_action_id' method.
        return_move: If True, return the chosen chess.Move object directly.
        return_info: If True, return (action, policy_id, policy_title), else return just action.

    Returns:
        Union[Tuple[Union[np.ndarray, int, chess.Move, None], int, str], Union[np.ndarray, int, chess.Move, None]]:
            - If return_info is True: (action, policy_id, policy_title)
            - If return_info is False: action
            The action type depends on return_id and return_move.
            Returns None for action if no legal moves exist.
    """
    # Board is now passed directly
    initial_legal_moves = list(board.legal_moves)

    # --- Handle No Legal Moves --- 
    if not initial_legal_moves:
        policy_id = 0
        # Return None for move, 0 for ID, or attempt zeros array if shape available
        action_repr: Union[np.ndarray, int, chess.Move, None] = None
        if not return_move:
             if return_id:
                 action_repr = 0 # Or None? Convention needed.
             # Cannot create np.zeros without action space shape. Return None or 0?
             # Let's return None if not returning move or ID.

        if return_info:
            return action_repr, policy_id, POLICY_TITLES[policy_id]
        else:
            return action_repr

    current_player_color = board.turn
    opponent_color = not current_player_color
    
    # --- Initialize variables --- 
    chosen_move: Optional[chess.Move] = None 
    policy_id = 8 # Default to fallback

    # --- 1. Deliver Checkmate ---    
    mating_moves = []
    for move in initial_legal_moves:
        try:
            board.push(move)
            if board.is_checkmate(): mating_moves.append(move)
            board.pop()
        except Exception:
             if board.move_stack and board.peek() == move: board.pop()
             continue
    if mating_moves:
        chosen_move = random.choice(mating_moves)
        policy_id = 1
        # --- Go to Final Return --- 

    # --- 2. Filter moves to avoid being checkmated --- 
    if chosen_move is None: # Only proceed if no mating move found
        safe_from_mate_moves = []
        can_opponent_mate_after_some_move = False
        for my_move in initial_legal_moves:
            opponent_can_mate_after_this_move = False
            try:
                board.push(my_move)
                # Check all opponent replies
                for opp_move in board.legal_moves:
                    try:
                        board.push(opp_move)
                        if board.is_checkmate(): opponent_can_mate_after_this_move = True
                        board.pop()
                        if opponent_can_mate_after_this_move: break
                    except Exception:
                         if board.move_stack and board.peek() == opp_move: board.pop()
                         continue # Skip inner simulation if fails
                board.pop()
            except Exception:
                 if board.move_stack and board.peek() == my_move: board.pop()
                 opponent_can_mate_after_this_move = True # Treat simulation failure as potentially unsafe

            if not opponent_can_mate_after_this_move: safe_from_mate_moves.append(my_move)
            else: can_opponent_mate_after_some_move = True
        
        # Determine which list of moves to use
        moves_to_consider = initial_legal_moves
        if safe_from_mate_moves and can_opponent_mate_after_some_move: moves_to_consider = safe_from_mate_moves
        elif not safe_from_mate_moves and can_opponent_mate_after_some_move: moves_to_consider = initial_legal_moves
        # Implicit else: if !can_opponent_mate_after_some_move, all initial moves are safe, use initial_legal_moves
        
        if not moves_to_consider: moves_to_consider = initial_legal_moves # Fallback if filtering fails
        # --- Filtering Complete --- 

        # --- 3. Prioritize Good Captures ---    
        if not chosen_move: # Check again (redundant if structure is followed, but safe)
            good_capture_moves = []
            capture_moves = [move for move in moves_to_consider if board.is_capture(move)]
            for move in capture_moves:
                capturing_piece = board.piece_at(move.from_square)
                captured_piece = board.piece_at(move.to_square)
                if is_safe_after_move(board, move, opponent_color) or (get_piece_value(captured_piece) > get_piece_value(capturing_piece)):
                    good_capture_moves.append(move)
            if good_capture_moves:
                best_captures = []; max_val = -1
                for move in good_capture_moves:
                    val = get_piece_value(board.piece_at(move.to_square))
                    if val > max_val: max_val = val; best_captures = [move]
                    elif val == max_val: best_captures.append(move)
                if best_captures:
                    chosen_move = random.choice(best_captures)
                    policy_id = 3
                    # --- Go to Final Return --- 
        
        # --- 4. Attack Trapped Piece (Revised: Find moves that *will* trap a piece) --- 
        if not chosen_move: 
            candidate_trapping_moves: List[Tuple[chess.Move, int]] = []
            for my_move in moves_to_consider:
                highest_value_trapped_by_this_move = -1
                try:
                    board.push(my_move)
                    player_attacked_squares_after_move = set().union(*(board.attacks(sq) for sq in board.pieces(chess.ANY, current_player_color)))
                    for opp_sq in board.pieces(chess.ANY, opponent_color):
                        if opp_sq in player_attacked_squares_after_move:
                            opponent_piece = board.piece_at(opp_sq)
                            if not opponent_piece: continue
                            is_trapped = True; has_any_escape = False
                            for opp_escape_move in board.generate_legal_moves(from_mask=chess.BB_SQUARES[opp_sq]):
                                has_any_escape = True
                                try:
                                    board.push(opp_escape_move)
                                    escape_dest_safe = not board.is_attacked_by(current_player_color, opp_escape_move.to_square)
                                    board.pop()
                                    if escape_dest_safe: is_trapped = False; break
                                except Exception:
                                    if board.move_stack and board.peek() == opp_escape_move: board.pop()
                                    is_trapped = False; break 
                            if not has_any_escape: is_trapped = True
                            if is_trapped:
                                highest_value_trapped_by_this_move = max(highest_value_trapped_by_this_move, get_piece_value(opponent_piece))
                    board.pop()
                    if highest_value_trapped_by_this_move > 0:
                        candidate_trapping_moves.append((my_move, highest_value_trapped_by_this_move))
                except Exception:
                    if board.move_stack and board.peek() == my_move: board.pop()
                    continue
            if candidate_trapping_moves:
                best_trapping_moves = []; max_trapped_val = -1
                for move, value in candidate_trapping_moves:
                    if value > max_trapped_val: max_trapped_val = value; best_trapping_moves = [move]
                    elif value == max_trapped_val: best_trapping_moves.append(move)
                if best_trapping_moves:
                    chosen_move = random.choice(best_trapping_moves)
                    policy_id = 4
                    # --- Go to Final Return --- 

        # --- 5. Avoid Attacks Logic ---    
        if not chosen_move and avoid_attacks: 
            attacked_own_squares = get_attacked_squares(board, current_player_color)
            if attacked_own_squares:
                candidate_safe_evasions: List[chess.Move] = []
                candidate_attacker_captures: List[Tuple[chess.Move, int]] = []
                all_attackers_squares = set().union(*(board.attackers(opponent_color, sq) for sq in attacked_own_squares))
                for move in moves_to_consider: 
                    is_safe_evasion = False; is_good_attacker_capture = False; captured_attacker_value = -1
                    if move.from_square in attacked_own_squares:
                        if is_safe_after_move(board, move, opponent_color):
                            is_safe_evasion = True
                            candidate_safe_evasions.append(move)
                    if board.is_capture(move) and move.to_square in all_attackers_squares:
                        capturing_piece = board.piece_at(move.from_square)
                        captured_piece = board.piece_at(move.to_square)
                        captured_attacker_value = get_piece_value(captured_piece)
                        if is_safe_after_move(board, move, opponent_color) or (captured_attacker_value > get_piece_value(capturing_piece)):
                            is_good_attacker_capture = True
                            candidate_attacker_captures.append((move, captured_attacker_value))
                if candidate_safe_evasions:
                    chosen_move = random.choice(list(set(candidate_safe_evasions)))
                    policy_id = 5
                    # --- Go to Final Return --- 
                elif candidate_attacker_captures:
                    best_captures = []; max_val = -1
                    for move, value in candidate_attacker_captures:
                        if value > max_val: max_val = value; best_captures = [move]
                        elif value == max_val: best_captures.append(move)
                    if best_captures:
                        chosen_move = random.choice(best_captures)
                        policy_id = 5
                        # --- Go to Final Return --- 
        
        # --- 6. Conditional Attack --- 
        if not chosen_move:
            candidate_conditional_attacks: List[Tuple[chess.Move, int]] = [] 
            capture_moves = [move for move in moves_to_consider if board.is_capture(move)]
            for move in capture_moves:
                attacking_piece = board.piece_at(move.from_square); captured_piece = board.piece_at(move.to_square)
                attacker_value = get_piece_value(attacking_piece); captured_value = get_piece_value(captured_piece)
                attacker_safe_cond = False; is_dest_safe = False; attacked_only_by_lesser = False
                try:
                    board.push(move)
                    is_dest_safe = not board.is_attacked_by(opponent_color, move.to_square)
                    if not is_dest_safe:
                        attackers = board.attackers(opponent_color, move.to_square)
                        if attackers: 
                            all_lesser = True
                            for attacker_sq in attackers:
                                if get_piece_value(board.piece_at(attacker_sq)) >= attacker_value: all_lesser = False; break
                            if all_lesser: attacked_only_by_lesser = True
                    board.pop()
                except Exception: 
                     if board.move_stack and board.peek() == move: board.pop()
                attacker_safe_cond = is_dest_safe or attacked_only_by_lesser
                target_cond = (captured_value > attacker_value) or (not board.is_attacked_by(opponent_color, move.to_square))
                if attacker_safe_cond and target_cond: candidate_conditional_attacks.append((move, captured_value))
            if candidate_conditional_attacks:
                best_conditional_attacks = []; max_val = -1
                for move, value in candidate_conditional_attacks:
                    if value > max_val: max_val = value; best_conditional_attacks = [move]
                    elif value == max_val: best_conditional_attacks.append(move)
                if best_conditional_attacks:
                    chosen_move = random.choice(best_conditional_attacks)
                    policy_id = 6
                    # --- Go to Final Return --- 
        
        # --- 7. Maximize Min(Mobility) ---    
        if not chosen_move:
            best_mobility_moves = [] # Renamed for clarity
            max_resulting_min_mobility = -float('inf') 
            for my_move in moves_to_consider:
                current_min_mobility_after_opponent = float('inf')
                try:
                    board.push(my_move)
                    opponent_legal_moves = list(board.legal_moves)
                    if not opponent_legal_moves:
                        if board.is_checkmate(): current_min_mobility_after_opponent = float('inf')
                        else: current_min_mobility_after_opponent = -float('inf')
                    else:
                        min_mobility_found_for_this_branch = float('inf') 
                        for opp_move in opponent_legal_moves:
                            try:
                                board.push(opp_move)
                                min_mobility_found_for_this_branch = min(min_mobility_found_for_this_branch, len(list(board.legal_moves)))
                                board.pop()
                            except Exception:
                                if board.move_stack and board.peek() == opp_move: board.pop()
                                min_mobility_found_for_this_branch = min(min_mobility_found_for_this_branch, 0)
                                continue
                        current_min_mobility_after_opponent = min_mobility_found_for_this_branch
                    board.pop()
                except Exception:
                    if board.move_stack and board.peek() == my_move: board.pop()
                    continue
                if current_min_mobility_after_opponent > max_resulting_min_mobility:
                    max_resulting_min_mobility = current_min_mobility_after_opponent
                    best_mobility_moves = [my_move]
                elif current_min_mobility_after_opponent == max_resulting_min_mobility:
                    best_mobility_moves.append(my_move)
            if best_mobility_moves:
                mating_moves_mobility = [m for m in best_mobility_moves if max_resulting_min_mobility == float('inf')]
                if mating_moves_mobility: chosen_move = random.choice(mating_moves_mobility) # Prefer mating move if found here
                else: chosen_move = random.choice(best_mobility_moves)
                policy_id = 7
                # --- Go to Final Return --- 

    # --- 8. Fallback --- 
    if chosen_move is None: # If no heuristic selected a move
        # Choose randomly from the filtered list (moves_to_consider)
        # If moves_to_consider is somehow empty, fall back to initial_legal_moves
        fallback_list = moves_to_consider if moves_to_consider else initial_legal_moves
        if fallback_list: # Ensure fallback list is not empty
            chosen_move = random.choice(fallback_list)
            policy_id = 8
        else: # Extremely unlikely edge case: no moves left at all
             policy_id = 0 # Revert to no moves
             action_repr = None # No move possible
             # Handle other return types if needed (e.g., ID 0)
             if not return_move and return_id: action_repr = 0

             if return_info:
                 return action_repr, policy_id, POLICY_TITLES[policy_id]
             else:
                 return action_repr
    
    # --- Final Return Logic --- 
    action_repr: Union[np.ndarray, int, chess.Move, None] = None # Initialize
    if return_move:
        action_repr = chosen_move # Return the move object directly
    elif return_id:
        # Requires board to have move_to_action_id method
        try:
            action_id = board.move_to_action_id(chosen_move)
            action_repr = action_id if action_id is not None else 0 # Return 0 if conversion fails?
        except AttributeError:
             print("Warning: board object lacks 'move_to_action_id' method needed for return_id=True.")
             action_repr = 0 # Fallback
    else:
        # Returning numpy array is problematic without action space shape/dtype.
        # Returning None as we cannot represent it accurately.
        # Alternatively, could try returning move UCI string?
        print("Warning: Returning numpy array action representation is not supported when passing board directly.")
        action_repr = None

    if return_info:
        return action_repr, policy_id, POLICY_TITLES[policy_id]
    else:
        return action_repr 


# Overloads for v2 lookahead sampler
@overload
def sample_action_v2(board: chess.Board, lookahead_moves: int = 2, return_id: bool = False, return_move: bool = False, return_info: Literal[True] = True) -> Tuple[Union[int, chess.Move, None], int, str]: ...

@overload
def sample_action_v2(board: chess.Board, lookahead_moves: int = 2, return_id: bool = False, return_move: bool = False, return_info: Literal[False] = False) -> Union[int, chess.Move, None]: ...


def sample_action_v2(board: chess.Board,
                     lookahead_moves: int = 3,
                     return_id: bool = False,
                     return_move: bool = False,
                     return_info: bool = False,
                     use_multiprocessing: bool = False,
                     max_workers: int = 0) -> Union[Tuple[Union[int, chess.Move, None], int, str], Union[int, chess.Move, None]]:
    """
    Depth-limited lookahead policy that chooses the move maximizing material sum
    after simulating the next N plies (half-moves), assuming the opponent also
    plays optimally.

    - State value = sum(piece values for White) - sum(piece values for Black)
    - Avoids checkmate by treating own-checkmate lines as -infinity and
      delivering checkmate as +infinity.

    Args:
        board: Current chess.Board
        lookahead_moves: Number of plies to search (default 2)
        return_id: If True, return action id via board.move_to_action_id
        return_move: If True, return chess.Move instead of id
        return_info: If True, return (action, policy_id, policy_title)

    Returns:
        action or (action, policy_id, policy_title)
    """
    n_cases = 0
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        action: Union[int, chess.Move, None] = None
        if return_id and not return_move:
            action = 0
        if return_info:
            return action, 9, POLICY_TITLES.get(9, "Lookahead")
        return action

    def material_score(b: chess.Board) -> float:
        return _material_score_board(b)

    root_color = chess.WHITE if board.turn == chess.WHITE else chess.BLACK

    def evaluate_terminal(b: chess.Board) -> float:
        return _evaluate_terminal_board(b, root_color)

    def negamax(b: chess.Board, depth: int, color: int, alpha: float, beta: float, mvs_str: str) -> float:
        nonlocal n_cases
        if depth == 0:
            n_cases += 1
        if depth == 0 or b.is_game_over():
            return color * evaluate_terminal(b)
        # Transposition cache lookup
        cache_key = None
        try:
            cache_key = (b.transposition_key(), depth, color)
        except Exception:
            cache_key = None
        if cache_key is not None:
            cached = _V2_TT.get(cache_key)
            if cached is not None:
                return cached
        best = -1e15
        for mv in list(b.legal_moves):
            try:
                b.push(mv)
                val = -negamax(b, depth - 1, -color, -beta, -alpha, mvs_str + " " + mv.uci())
                # print("Depth:", search_depth-depth, " Moves, ", n_cases, "val: ", round(val, 2), "mvs_str: ", mvs_str + " " + mv.uci())
                b.pop()
            except Exception:
                if b.move_stack and b.peek() == mv:
                    b.pop()
                continue
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        # Store in cache with simple size cap
        if cache_key is not None:
            if len(_V2_TT) >= _V2_TT_MAX_ENTRIES:
                # Remove approximately 10%
                for _ in range(_V2_TT_MAX_ENTRIES // 10):
                    _V2_TT.pop(next(iter(_V2_TT)))
            _V2_TT[cache_key] = best
        return best

    best_val = -1e15
    best_move: Optional[chess.Move] = None
    color = 1 if board.turn == chess.WHITE else -1
    search_depth = max(0, lookahead_moves)
    scored_moves: List[Tuple[chess.Move, float]] = []
    if use_multiprocessing and search_depth > 0 and len(legal_moves) > 1:
        fen = board.fen()
        root_color_is_white = (board.turn == chess.WHITE)
        worker_args = [(fen, mv.uci(), search_depth, color, root_color_is_white) for mv in legal_moves]
        workers = max_workers if max_workers and max_workers > 0 else None
        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_to_move = {ex.submit(_negamax_worker, *args): args[1] for args in worker_args}
            for future in as_completed(future_to_move):
                move_uci, val = future.result()
                mv = chess.Move.from_uci(move_uci)
                scored_moves.append((mv, val))
                if val > best_val:
                    best_val = val
                    best_move = mv
    else:
        for mv in legal_moves:
            try:
                board.push(mv)
                val = -negamax(board, max(0, search_depth), -color, -1e15, 1e15, mv.uci())
                board.pop()
            except Exception:
                if board.move_stack and board.peek() == mv:
                    board.pop()
                continue
            scored_moves.append((mv, val))
            if val > best_val:
                best_val = val
                best_move = mv
            
    # Tie-break by maximizing min mobility after opponent reply
    if scored_moves:
        eps = 1e-9
        top_moves = [mv for mv, v in scored_moves if abs(v - best_val) <= eps]

        if len(top_moves) > 1:
            def min_mobility_after_opponent(b: chess.Board, moved_to_sq: int, moved_piece_val: float) -> int:
                # b is position where opponent to move
                opp_legal = list(b.legal_moves)
                if not opp_legal:
                    # Checkmate: prefer this move by returning huge mobility
                    if b.is_checkmate():
                        return 10**9
                    # Stalemate: treat as very low
                    return -10**9
                min_mob = 10**9
                any_considered = False
                for opp_mv in opp_legal:
                    try:
                        b.push(opp_mv)
                        my_mob = len(list(b.legal_moves))
                        b.pop()
                    except Exception:
                        if b.move_stack and b.peek() == opp_mv:
                            b.pop()
                        my_mob = 0
                    any_considered = True
                    if my_mob < min_mob:
                        min_mob = my_mob
                if not any_considered:
                    # If every opponent reply was filtered out, treat as very favorable
                    return 10**9
                return min_mob

            best_mob = -10**9
            best_tb_move = best_move
            for mv in top_moves:
                try:
                    board.push(mv)
                    moved_piece = board.piece_at(mv.to_square)
                    moved_val = float(PIECE_VALUES.get(moved_piece.piece_type, 0)) if moved_piece else 0.0
                    mob = min_mobility_after_opponent(board, mv.to_square, moved_val)
                    board.pop()
                except Exception:
                    if board.move_stack and board.peek() == mv:
                        board.pop()
                    mob = -10**9
                if mob > best_mob:
                    best_mob = mob
                    best_tb_move = mv
            best_move = best_tb_move

    # Build return
    if return_move:
        action_out: Union[int, chess.Move, None] = best_move
    elif return_id:
        action_id: Optional[int] = None
        try:
            # Requires chess_gym board with move_to_action_id
            action_id = board.move_to_action_id(best_move) if best_move is not None else None
        except Exception:
            action_id = None
        action_out = action_id if action_id is not None else 0
    else:
        # Default to returning move object if neither id nor move is specified
        action_out = best_move

    if return_info:
        return action_out, 9, POLICY_TITLES.get(9, "Lookahead")
    return action_out
