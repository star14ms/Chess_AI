# Cell content for sample_action function
import random
import chess
import numpy as np
import gymnasium as gym # Assuming gym is imported
from typing import Optional, List, Tuple # Added for type hinting

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

def sample_action(action_space: gym.spaces.Space, avoid_attacks: bool = True) -> Tuple[np.ndarray, int, str]:
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
        Tuple[np.ndarray, int, str]: A tuple containing:
            - The chosen action as a numpy array.
            - The integer ID (1-8) of the policy that chose it (0 if no moves).
            - The string title of the policy.
    """
    board = action_space.board
    initial_legal_moves = list(board.legal_moves)
    if not initial_legal_moves:
        policy_id = 0
        return np.zeros(action_space.shape, dtype=action_space.dtype), policy_id, POLICY_TITLES[policy_id]

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
        return action_space._move_to_action(chosen_move), policy_id, POLICY_TITLES[policy_id]
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
        return np.zeros(action_space.shape, dtype=action_space.dtype), policy_id, POLICY_TITLES[policy_id]
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
            return action_space._move_to_action(heuristic_chosen_move), policy_id, POLICY_TITLES[policy_id]

    # --- 4. Attack Trapped Piece --- 
    if not heuristic_chosen_move: 
        good_attacks_on_trapped: List[Tuple[chess.Move, int]] = []        
        opponent_attacked_squares = get_attacked_squares(board, opponent_color) # Squares opponent occupies attacked by us

        for opp_sq in opponent_attacked_squares:
            opponent_piece = board.piece_at(opp_sq)
            if not opponent_piece: continue # Should not happen

            # Check if opponent piece is trapped (has no safe escape moves)
            is_trapped = True
            has_any_escape = False
            for opp_escape_move in board.generate_legal_moves(from_mask=chess.BB_SQUARES[opp_sq]):
                has_any_escape = True
                if is_safe_after_move(board, opp_escape_move, current_player_color): # Is opp destination safe from us? 
                    is_trapped = False
                    break # Found a safe escape, piece is not trapped
            
            # If opponent piece has NO legal moves at all, it's also trapped (or stalemated/checkmated) 
            if not has_any_escape: is_trapped = True

            if is_trapped:
                # Find our good moves that capture this trapped piece
                for my_attack_move in moves_to_consider: # Use filtered list
                    if my_attack_move.to_square == opp_sq: # If this move captures the piece
                        # Check if this attack is safe or profitable for us
                        capturing_piece = board.piece_at(my_attack_move.from_square)
                        trapped_piece_value = get_piece_value(opponent_piece)
                        if is_safe_after_move(board, my_attack_move, opponent_color) or (trapped_piece_value > get_piece_value(capturing_piece)):
                             good_attacks_on_trapped.append((my_attack_move, trapped_piece_value))
        
        # Select the best attack among those found
        if good_attacks_on_trapped:
            best_trapped_attacks = []
            max_trapped_val = -1
            for move, value in good_attacks_on_trapped:
                if value > max_trapped_val: max_trapped_val = value; best_trapped_attacks = [move]
                elif value == max_trapped_val: best_trapped_attacks.append(move)
            
            if best_trapped_attacks:
                heuristic_chosen_move = random.choice(best_trapped_attacks)
                policy_id = 4
                return action_space._move_to_action(heuristic_chosen_move), policy_id, POLICY_TITLES[policy_id]

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
                return action_space._move_to_action(heuristic_chosen_move), policy_id, POLICY_TITLES[policy_id]
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
                    return action_space._move_to_action(heuristic_chosen_move), policy_id, POLICY_TITLES[policy_id]

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
                return action_space._move_to_action(heuristic_chosen_move), policy_id, POLICY_TITLES[policy_id]

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
        return action_space._move_to_action(chosen_move), policy_id, POLICY_TITLES[policy_id]
    else:
        # --- 8. Fallback ---        
        policy_id = 8 # Fallback Policy 8
        chosen_move = None
        if moves_to_consider: 
            chosen_move = random.choice(moves_to_consider)
        elif initial_legal_moves: 
            chosen_move = random.choice(initial_legal_moves)
        
        if chosen_move:
            return action_space._move_to_action(chosen_move), policy_id, POLICY_TITLES[policy_id]
        else: 
            # Final fallback if absolutely no moves are left (should be caught earlier)
            policy_id = 0
            return np.zeros(action_space.shape, dtype=action_space.dtype), policy_id, POLICY_TITLES[policy_id] 