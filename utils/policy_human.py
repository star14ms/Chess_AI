# Cell content for sample_action function
import random
import chess
import numpy as np
import gymnasium as gym # Assuming gym is imported
from typing import Optional, List, Tuple, Union

# --- Piece Values ---
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0 # King value is effectively infinite, 0 works for comparisons here
}
# ---

# --- Policy Titles ---
POLICY_TITLES = {
    0: "No Legal Moves",
    1: "1. Deliver Checkmate",
    2: "2. Avoid Checkmate (Filtering Step)", # Technically doesn't choose, but useful info
    3: "3. Prioritize Good Captures",
    4: "4. Attack Trapped Piece",
    5: "5. Avoid Attacks",
    6: "6. Conditional Attack",
    7: "7. Maximize Min(Mobility)",
    8: "8. Fallback Random"
}
# ---

def get_piece_value(piece: Optional[chess.Piece]) -> int:
    """Gets the defined value of a piece, returns 0 if None."""
    if piece is None:
        return 0
    return PIECE_VALUES.get(piece.piece_type, 0)

def get_attacked_squares(board: chess.Board, color: chess.Color) -> set[chess.Square]:
    """Returns a set of squares occupied by `color`'s pieces that are attacked by the opponent."""
    attacked_squares = set()
    opponent_color = not color
    # Iterate through all squares where 'color' has pieces
    for piece_type in chess.PIECE_TYPES:
        for square in board.pieces(piece_type, color):
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

def sample_action(action_space: gym.spaces.Space, avoid_attacks: bool = True, return_id: bool = False) -> Tuple[Union[np.ndarray, int], int, str]:
    """
    Samples an action using a hierarchy of heuristics, returning the action, policy ID, and policy title.

    Priority Order & Actions (Lookahead Level):
    1. Deliver Checkmate: If possible, choose a checkmating move.
                        (Checks state after player's move: 1-ply)
    2. Avoid Checkmate: Filter moves to exclude those allowing opponent to mate
                       next turn. Use this safer subset for subsequent steps.
                       (Checks state after opponent's reply: 2-ply)
    3. Prioritize Good Captures: Using safe moves, select safe or profitable captures,
                                targeting the highest-value piece.
                                (Checks current state for values, after-state for safety: 1-ply)
    4. Attack Trapped Piece: Attack an opponent piece that is attacked and has no safe escape moves,
                             provided the attack is safe or profitable and targets the highest-value trapped piece.
                             (Checks opponent escapes: 2-ply; Checks own attack safety: 1-ply)
    5. Avoid Attacks: Using safe moves, prioritize safely moving attacked pieces,
                     then capturing the highest-value attacker.
                     (Checks current state for attacks, after-state for safety: 1-ply)
    6. Conditional Attack: Make a capture if attacker is safe OR only attacked by lesser-value pieces,
                           AND target is higher value OR undefended. Prioritize highest value target.
                           (Checks attacker safety after: 1-ply; Checks target defense before: 0-ply)
    7. Maximize Min(Mobility): Using remaining safe moves, choose the move that
                              maximizes the minimum possible mobility after any
                              opponent reply (minimax).
                              (Checks mobility after opponent's reply: 2-ply)
    8. Fallback: If heuristics fail, choose randomly from considered moves.
                 (Uses moves available in current state: 0-ply)

    Note on Priority Interplay: While checks proceed in order, the first heuristic
    to identify and select a *specific* move returns immediately. For example,
    if a high-value trapped piece is identified in step 5, attacking it might be chosen
    even if a lower-value friendly piece was under attack in step 3 (but no decisive
    saving move was chosen).
    
    Returns:
        Tuple[Union[np.ndarray, int], int, str]: A tuple containing:
            - The chosen action as a numpy array or integer ID (based on return_id).
            - The integer ID (1-8) of the policy that chose it (0 if no moves).
            - The string title of the policy.
    """
    board = action_space.board
    initial_legal_moves = list(board.legal_moves)
    if not initial_legal_moves:
        policy_id = 0
        # Handle return_id for no moves case (return 0 if ID requested, else array)
        action_repr = 0 if return_id else np.zeros(action_space.shape, dtype=action_space.dtype)
        return action_repr, policy_id, POLICY_TITLES[policy_id]

    current_player_color = board.turn
    opponent_color = not current_player_color

    # --- 1. Deliver Checkmate ---    
    mating_moves = []
    for move in initial_legal_moves:
        try:
            board.push(move)
            if board.is_checkmate():
                mating_moves.append(move)
            board.pop()
        except Exception:
             if board.move_stack and board.peek() == move: board.pop()
             continue # Skip if simulation fails
    
    if mating_moves:
        chosen_move = random.choice(mating_moves)
        policy_id = 1
        return action_space._move_to_action(chosen_move, return_id=return_id), policy_id, POLICY_TITLES[policy_id]
    # --- End Deliver Checkmate ---

    # --- 2. Avoid Being Checkmated ---    
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
                    if board.is_checkmate():
                        opponent_can_mate_after_this_move = True
                    board.pop()
                    if opponent_can_mate_after_this_move:
                        break # No need to check other opponent moves
                except Exception:
                     if board.move_stack and board.peek() == opp_move: board.pop()
                     continue # Skip inner simulation if fails
            board.pop()
        except Exception:
             if board.move_stack and board.peek() == my_move: board.pop()
             # If simulating my_move fails, assume it might be unsafe? Or skip?
             # Let's skip it from the safe list for now.
             opponent_can_mate_after_this_move = True # Treat simulation failure as potentially unsafe

        if not opponent_can_mate_after_this_move:
            safe_from_mate_moves.append(my_move)
        else:
             can_opponent_mate_after_some_move = True # Record that at least one move leads to mate threat

    # Determine which list of moves to use for subsequent heuristics
    # If ALL moves lead to opponent checkmate, we can't avoid it, use all moves.
    # Otherwise, use only the moves that prevent immediate mate.
    moves_to_consider = initial_legal_moves
    if safe_from_mate_moves and can_opponent_mate_after_some_move:
         moves_to_consider = safe_from_mate_moves
    elif not safe_from_mate_moves and can_opponent_mate_after_some_move:
         moves_to_consider = initial_legal_moves # Can't avoid, consider all options
    # If !can_opponent_mate_after_some_move, all moves are safe, use initial_legal_moves
    
    if not moves_to_consider: moves_to_consider = initial_legal_moves
    if not moves_to_consider:
        policy_id = 0 # Treat as no legal moves if filtering failed unexpectedly
        action_repr = 0 if return_id else np.zeros(action_space.shape, dtype=action_space.dtype)
        return action_repr, policy_id, POLICY_TITLES[policy_id]
    # --- End Avoid Being Checkmated ---

    # --- Initialize variable before first use --- 
    heuristic_chosen_move: Optional[chess.Move] = None 
    policy_id = 8 # Default to fallback policy ID

    # --- 3. Prioritize Good Captures ---    
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
            heuristic_chosen_move = random.choice(best_captures)
            policy_id = 3
            return action_space._move_to_action(heuristic_chosen_move, return_id=return_id), policy_id, POLICY_TITLES[policy_id]

    # --- 4. Attack Trapped Piece (Revised: Find moves that *will* trap a piece) --- 
    if not heuristic_chosen_move: 
        candidate_trapping_moves: List[Tuple[chess.Move, int]] = [] # Store (trapping_move, value_of_trapped_piece)

        for my_move in moves_to_consider:
            highest_value_trapped_by_this_move = -1
            try:
                board.push(my_move)
                
                # Find all squares attacked by the player *after* the move
                player_attacked_squares_after_move = set().union(*(board.attacks(sq) for sq in board.pieces(chess.ANY, current_player_color)))
                
                # Check each opponent piece
                for opp_sq in board.pieces(chess.ANY, opponent_color):
                    # Is this opponent piece attacked *now*?
                    if opp_sq in player_attacked_squares_after_move:
                        opponent_piece = board.piece_at(opp_sq)
                        if not opponent_piece: continue # Should not happen

                        # Check if this attacked opponent piece is trapped (no safe escapes)
                        is_trapped = True
                        has_any_escape = False
                        for opp_escape_move in board.generate_legal_moves(from_mask=chess.BB_SQUARES[opp_sq]):
                            has_any_escape = True
                            # Check if the opponent's escape destination is safe from the player
                            try:
                                board.push(opp_escape_move)
                                escape_dest_safe = not board.is_attacked_by(current_player_color, opp_escape_move.to_square)
                                board.pop() # Pop opponent escape simulation
                                if escape_dest_safe:
                                    is_trapped = False
                                    break # Found a safe escape for this opponent piece
                            except Exception:
                                if board.move_stack and board.peek() == opp_escape_move: board.pop()
                                # If simulation fails, assume escape might be possible (conservative)
                                is_trapped = False 
                                break 
                        
                        # Also trapped if it had no legal moves at all
                        if not has_any_escape: 
                            is_trapped = True

                        # If this opponent piece IS trapped by my_move...
                        if is_trapped:
                            current_opp_piece_value = get_piece_value(opponent_piece)
                            # Keep track of the highest value piece trapped by this specific my_move
                            highest_value_trapped_by_this_move = max(highest_value_trapped_by_this_move, current_opp_piece_value)
                
                # Finished checking all opponent pieces for this my_move
                board.pop() # Pop my_move simulation

                # If this move resulted in trapping *some* piece, record it
                if highest_value_trapped_by_this_move > 0:
                    candidate_trapping_moves.append((my_move, highest_value_trapped_by_this_move))

            except Exception:
                # Ensure board state is clean if simulation of my_move fails
                if board.move_stack and board.peek() == my_move: board.pop()
                continue # Skip this move if simulation failed

        # --- Selection after checking all trapping moves ---
        if candidate_trapping_moves:
            best_trapping_moves = []
            max_trapped_val = -1
            # Find the move(s) that trap the highest value piece
            for move, value in candidate_trapping_moves:
                if value > max_trapped_val:
                    max_trapped_val = value
                    best_trapping_moves = [move]
                elif value == max_trapped_val:
                    best_trapping_moves.append(move)
            
            if best_trapping_moves:
                heuristic_chosen_move = random.choice(best_trapping_moves)
                policy_id = 4 # Still Policy 4, but with revised logic
                return action_space._move_to_action(heuristic_chosen_move, return_id=return_id), policy_id, POLICY_TITLES[policy_id]

    # --- 5. Avoid Attacks Logic ---    
    if not heuristic_chosen_move and avoid_attacks: 
        attacked_own_squares = get_attacked_squares(board, current_player_color)
        if attacked_own_squares:
            candidate_safe_evasions: List[chess.Move] = []
            candidate_attacker_captures: List[Tuple[chess.Move, int]] = []
            all_attackers_squares = set().union(*(board.attackers(opponent_color, sq) for sq in attacked_own_squares))
            for move in moves_to_consider: 
                is_safe_evasion = False
                is_good_attacker_capture = False
                captured_attacker_value = -1

                # Check a) Is it a safe evasion by the attacked piece?
                if move.from_square in attacked_own_squares:
                    if is_safe_after_move(board, move, opponent_color):
                        is_safe_evasion = True
                        candidate_safe_evasions.append(move)
                        # Don't 'continue' here, let's see if it *also* captures an attacker below
                
                # Check b) Does it capture an attacker (by any piece)?
                if board.is_capture(move) and move.to_square in all_attackers_squares:
                    capturing_piece = board.piece_at(move.from_square)
                    captured_piece = board.piece_at(move.to_square) # The attacker being captured
                    captured_attacker_value = get_piece_value(captured_piece)
                    # Is the capture safe or profitable?
                    if is_safe_after_move(board, move, opponent_color) or (captured_attacker_value > get_piece_value(capturing_piece)):
                        is_good_attacker_capture = True
                        candidate_attacker_captures.append((move, captured_attacker_value))

            # --- Selection within Avoid Attacks ---            
            if candidate_safe_evasions:
                chosen_move = random.choice(list(set(candidate_safe_evasions)))
                heuristic_chosen_move = chosen_move 
                policy_id = 5
                return action_space._move_to_action(heuristic_chosen_move, return_id=return_id), policy_id, POLICY_TITLES[policy_id]
            elif candidate_attacker_captures:
                best_captures = []
                max_val = -1
                for move, value in candidate_attacker_captures:
                    if value > max_val: max_val = value; best_captures = [move]
                    elif value == max_val: best_captures.append(move)
                if best_captures:
                    chosen_move = random.choice(best_captures)
                    heuristic_chosen_move = chosen_move 
                    policy_id = 5
                    return action_space._move_to_action(heuristic_chosen_move, return_id=return_id), policy_id, POLICY_TITLES[policy_id]

    # --- 6. Conditional Attack --- 
    if not heuristic_chosen_move:
        candidate_conditional_attacks: List[Tuple[chess.Move, int]] = [] 
        capture_moves = [move for move in moves_to_consider if board.is_capture(move)]

        for move in capture_moves:
            attacking_piece = board.piece_at(move.from_square)
            captured_piece = board.piece_at(move.to_square)
            attacker_value = get_piece_value(attacking_piece)
            captured_value = get_piece_value(captured_piece)

            # Check Condition 1: Attacker Safety
            attacker_safe_cond = False
            is_dest_safe = False
            attacked_only_by_lesser = False
            try:
                board.push(move)
                is_dest_safe = not board.is_attacked_by(opponent_color, move.to_square)
                if not is_dest_safe:
                    attackers = board.attackers(opponent_color, move.to_square)
                    if attackers: # Check if set is not empty
                        all_lesser = True
                        for attacker_sq in attackers:
                            attacker_piece = board.piece_at(attacker_sq)
                            if get_piece_value(attacker_piece) >= attacker_value:
                                all_lesser = False; break
                        if all_lesser: attacked_only_by_lesser = True
                board.pop()
            except Exception:
                 if board.move_stack and board.peek() == move: board.pop()
            attacker_safe_cond = is_dest_safe or attacked_only_by_lesser

            # Check Condition 2: Target Value/Safety
            target_cond = False
            is_profitable = captured_value > attacker_value
            is_target_undefended = not board.is_attacked_by(opponent_color, move.to_square) # Before move
            target_cond = is_profitable or is_target_undefended

            # Add if both conditions met
            if attacker_safe_cond and target_cond:
                 candidate_conditional_attacks.append((move, captured_value))
        
        # Select best conditional attack
        if candidate_conditional_attacks:
            best_conditional_attacks = []; max_val = -1
            for move, value in candidate_conditional_attacks:
                if value > max_val: max_val = value; best_conditional_attacks = [move]
                elif value == max_val: best_conditional_attacks.append(move)
            if best_conditional_attacks:
                heuristic_chosen_move = random.choice(best_conditional_attacks)
                policy_id = 6
                return action_space._move_to_action(heuristic_chosen_move, return_id=return_id), policy_id, POLICY_TITLES[policy_id]

    # --- 7. Maximize Min(Mobility) ---    
    # This is the final heuristic before fallback, no need to check heuristic_chosen_move
    best_moves = []
    max_resulting_min_mobility = -float('inf') 

    for my_move in moves_to_consider: # Use filtered list
        current_min_mobility_after_opponent = float('inf')
        try:
            board.push(my_move)
            opponent_legal_moves = list(board.legal_moves)
            if not opponent_legal_moves:
                if board.is_checkmate(): current_min_mobility_after_opponent = float('inf')
                else: current_min_mobility_after_opponent = -float('inf') # Avoid draws
            else:
                min_mobility_found_for_this_branch = float('inf') 
                for opp_move in opponent_legal_moves:
                    try:
                        board.push(opp_move)
                        my_next_mobility = len(list(board.legal_moves))
                        board.pop()
                        min_mobility_found_for_this_branch = min(min_mobility_found_for_this_branch, my_next_mobility)
                    except Exception:
                        if board.move_stack and board.peek() == opp_move: board.pop()
                        min_mobility_found_for_this_branch = min(min_mobility_found_for_this_branch, 0) # Assume worst
                        continue
                current_min_mobility_after_opponent = min_mobility_found_for_this_branch
            board.pop()
        except Exception:
            if board.move_stack and board.peek() == my_move: board.pop()
            continue

        if current_min_mobility_after_opponent > max_resulting_min_mobility:
            max_resulting_min_mobility = current_min_mobility_after_opponent
            best_moves = [my_move]
        elif current_min_mobility_after_opponent == max_resulting_min_mobility:
            best_moves.append(my_move)

    if best_moves:
        mating_moves = [m for m in best_moves if max_resulting_min_mobility == float('inf')]
        if mating_moves: chosen_move = random.choice(mating_moves)
        else: chosen_move = random.choice(best_moves)
        policy_id = 7 # Policy 7
        return action_space._move_to_action(chosen_move, return_id=return_id), policy_id, POLICY_TITLES[policy_id]
    else:
        # --- 8. Fallback ---        
        policy_id = 8 # Fallback Policy 8
        chosen_move = None
        if moves_to_consider: 
            chosen_move = random.choice(moves_to_consider)
        elif initial_legal_moves: 
            chosen_move = random.choice(initial_legal_moves)
        
        if chosen_move:
            return action_space._move_to_action(chosen_move, return_id=return_id), policy_id, POLICY_TITLES[policy_id]
        else: 
            # Final fallback if absolutely no moves are left (should be caught earlier)
            policy_id = 0
            return np.zeros(action_space.shape, dtype=action_space.dtype), policy_id, POLICY_TITLES[policy_id] 