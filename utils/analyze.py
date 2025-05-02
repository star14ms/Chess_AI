from typing import Dict, Tuple, List, Iterator, Optional, Union
import chess
from collections import defaultdict
from chess_gym.chess_custom import FullyTrackedBoard
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Universal Chess Interface (UCI) to Local Action ID (1-based)
def uci_to_relative_rook_action_id(uci_move: str) -> Optional[int]:
    """Calculates the RELATIVE action ID (1-28) for a rook move pattern."""
    # --- Same logic as previous uci_to_rook_action_id ---
    # --- Returns the ID between 1-28 or None ---
    if len(uci_move) != 4: return None
    try:
        start_sq = chess.parse_square(uci_move[0:2])
        end_sq = chess.parse_square(uci_move[2:4])
    except ValueError: return None
    start_file, start_rank = chess.square_file(start_sq), chess.square_rank(start_sq)
    end_file, end_rank = chess.square_file(end_sq), chess.square_rank(end_sq)
    file_change, rank_change = end_file - start_file, end_rank - start_rank

    if not ((file_change == 0 and rank_change != 0) or \
            (rank_change == 0 and file_change != 0)): return None

    distance = max(abs(file_change), abs(rank_change))
    if distance > 7: return None # Max distance is 7

    rel_id = 0
    if rank_change > 0: rel_id = 1 + distance - 1 # N (1-7)
    elif file_change > 0: rel_id = 8 + distance - 1 # E (8-14)
    elif rank_change < 0: rel_id = 15 + distance - 1 # S (15-21)
    elif file_change < 0: rel_id = 22 + distance - 1 # W (22-28)

    return rel_id if 1 <= rel_id <= 28 else None

def uci_to_relative_knight_action_id(uci_move: str) -> Optional[int]:
    """Calculates the RELATIVE action ID (1-8) for a knight move pattern."""
    # --- Same logic as previous uci_to_knight_action_id ---
    # --- Returns the ID between 1-8 or None ---
    if len(uci_move) != 4: return None
    try:
        start_sq, end_sq = chess.parse_square(uci_move[0:2]), chess.parse_square(uci_move[2:4])
    except ValueError: return None
    file_change = chess.square_file(end_sq) - chess.square_file(start_sq)
    rank_change = chess.square_rank(end_sq) - chess.square_rank(start_sq)
    knight_move_patterns = {
        (1, 2): 1, (2, 1): 2, (2, -1): 3, (1, -2): 4,
        (-1, -2): 5, (-2, -1): 6, (-2, 1): 7, (-1, 2): 8
    }
    return knight_move_patterns.get((file_change, rank_change))

def uci_to_relative_bishop_action_id(uci_move: str) -> Optional[int]:
    """Calculates the RELATIVE action ID (1-28) for a bishop move pattern."""
     # --- Same logic as previous uci_to_bishop_action_id ---
     # --- Returns the ID between 1-28 or None ---
    if len(uci_move) != 4: return None
    try:
        start_sq, end_sq = chess.parse_square(uci_move[0:2]), chess.parse_square(uci_move[2:4])
    except ValueError: return None
    file_change = chess.square_file(end_sq) - chess.square_file(start_sq)
    rank_change = chess.square_rank(end_sq) - chess.square_rank(start_sq)

    if abs(file_change) != abs(rank_change) or file_change == 0: return None
    distance = abs(file_change)
    if distance > 7: return None

    rel_id = 0
    if file_change > 0 and rank_change > 0: rel_id = 1 + distance - 1 # NE (1-7)
    elif file_change > 0 and rank_change < 0: rel_id = 8 + distance - 1 # SE (8-14)
    elif file_change < 0 and rank_change < 0: rel_id = 15 + distance - 1 # SW (15-21)
    elif file_change < 0 and rank_change > 0: rel_id = 22 + distance - 1 # NW (22-28)

    return rel_id if 1 <= rel_id <= 28 else None

def uci_to_relative_queen_action_id(uci_move: str) -> Optional[int]:
    """Calculates the RELATIVE action ID (1-56) for a queen move pattern."""
    # Check rook pattern first
    rook_rel_id = uci_to_relative_rook_action_id(uci_move)
    if rook_rel_id is not None:
        return rook_rel_id # Relative IDs 1-28

    # Check bishop pattern
    bishop_rel_id = uci_to_relative_bishop_action_id(uci_move)
    if bishop_rel_id is not None:
        # Offset bishop relative IDs (1-28) to follow rook IDs (29-56)
        return 28 + bishop_rel_id # Relative IDs 29-56

    return None

def uci_to_relative_king_action_id(uci_move: str) -> Optional[int]:
    """Calculates the RELATIVE action ID (1-10) for a king move pattern."""
    if len(uci_move) != 4: return None
    try:
        start_sq = chess.parse_square(uci_move[0:2])
        end_sq = chess.parse_square(uci_move[2:4])
    except ValueError: return None

    start_file = chess.square_file(start_sq)
    start_rank = chess.square_rank(start_sq)
    end_file = chess.square_file(end_sq)
    end_rank = chess.square_rank(end_sq)

    file_change = end_file - start_file
    rank_change = end_rank - start_rank

    # --- Corrected Castling Check ---
    # Check if it's a standard castling move pattern (King moves 2 squares horizontally)
    # Note: This assumes the UCI represents a valid king move. More robust validation
    # might involve checking start/end ranks, but this captures the UCI pattern.
    if abs(file_change) == 2 and rank_change == 0:
        # Check if the start square is a typical king starting square for castling
        if start_file == chess.E1 or start_file == chess.E8:
             if file_change == 2: return 9 # Kingside (e.g., e1g1 or e8g8)
             if file_change == -2: return 10 # Queenside (e.g., e1c1 or e8c8)
        # If the start square isn't E1/E8 but it's a 2-square horizontal move,
        # it's invalid for our standard definition, return None or handle as error.
        return None


    # --- Standard Moves Check ---
    if abs(file_change) > 1 or abs(rank_change) > 1 or (file_change == 0 and rank_change == 0):
        # If it wasn't castling and isn't a single step move, return None
        # This condition might be slightly redundant now but keeps the logic clear.
         if not (abs(file_change) == 2 and rank_change == 0): # Only return None if not already handled castling
             return None

    # Standard single-step moves (relative IDs 1-8)
    king_move_patterns = { # Clockwise from N
        (0, 1): 1, (1, 1): 2, (1, 0): 3, (1, -1): 4,
        (0, -1): 5, (-1, -1): 6, (-1, 0): 7, (-1, 1): 8
    }
    relative_id = king_move_patterns.get((file_change, rank_change))

    # It should have returned for castling or invalid moves earlier.
    # If we get here, it must be a standard 1-8 move or something unexpected.
    return relative_id

def uci_to_relative_pawn_action_id(uci_move: str, color: chess.Color) -> Optional[int]:
    """
    Calculates the RELATIVE action ID (1-18) for a pawn move pattern,
    using the provided pawn color and handling all 4 promotion types.

    ID Scheme (1-based):
     1: Fwd1
     2: Fwd2 (Initial move)      <- Updated Order
     3: CapL (Diagonal Left)    <- Updated Order
     4: CapR (Diagonal Right)   <- Updated Order
     5: EPL (En Passant Left)  -> Needs distinction by caller using board state
     6: EPR (En Passant Right) -> Needs distinction by caller using board state
     --- Promotions (Fwd, CapL, CapR) ---
     7-9:   Queen Promo (Fwd, CapL, CapR)
    10-12: Knight Promo (Fwd, CapL, CapR)
    13-15: Bishop Promo (Fwd, CapL, CapR)
    16-18: Rook Promo (Fwd, CapL, CapR)

    NOTE: This function cannot distinguish En Passant (5,6) from regular
    Captures (3,4) using only UCI. The caller needs board context for EP.
    It currently maps diagonal moves to 3 (CapL) and 4 (CapR).
    """
    # Basic UCI cleaning and parsing
    clean_uci = uci_move.replace('x', '').replace('+', '').replace('#', '')
    promotion_char = None
    if len(clean_uci) == 5:
        promotion_char = clean_uci[4].lower()
        move_part = clean_uci[:4]
    elif len(clean_uci) == 4:
        move_part = clean_uci
    else:
        return None # Invalid length

    try:
        start_sq = chess.parse_square(move_part[0:2])
        end_sq = chess.parse_square(move_part[2:4])
        if promotion_char and promotion_char not in ['q', 'n', 'b', 'r']:
             return None # Invalid promotion piece
    except ValueError:
        return None # Invalid square notation

    start_file, start_rank = chess.square_file(start_sq), chess.square_rank(start_sq)
    end_file, end_rank = chess.square_file(end_sq), chess.square_rank(end_sq)
    file_change, rank_change = end_file - start_file, end_rank - start_rank

    is_white_move = (color == chess.WHITE)
    expected_rank_change = 1 if is_white_move else -1
    promo_rank = 7 if is_white_move else 0
    relative_id = None

    if promotion_char:
        # --- Promotion Logic (IDs 7-18, unchanged) ---
        if end_rank != promo_rank: return None
        if rank_change != expected_rank_change: return None

        promo_base_id = 0
        if promotion_char == 'q': promo_base_id = 7
        elif promotion_char == 'n': promo_base_id = 10
        elif promotion_char == 'b': promo_base_id = 13
        elif promotion_char == 'r': promo_base_id = 16
        else: return None

        if file_change == 0: relative_id = promo_base_id + 0       # Fwd Promo
        elif file_change == -1: relative_id = promo_base_id + 1    # Cap Left Promo
        elif file_change == 1: relative_id = promo_base_id + 2     # Cap Right Promo
        else: return None
    else:
        # --- Non-Promotion Logic (IDs 1-4, Updated Order) ---
        if end_rank == promo_rank: return None

        if file_change == 0: # Forward moves
            start_rank_for_2 = 1 if is_white_move else 6
            if rank_change == expected_rank_change:
                relative_id = 1 # Fwd 1
            elif rank_change == (2 * expected_rank_change) and start_rank == start_rank_for_2:
                relative_id = 2 # Fwd 2  <- Changed from 4
            else: return None
        elif abs(file_change) == 1: # Diagonal moves (Capture or EP)
            if rank_change == expected_rank_change:
                 # Map diagonal moves based on new order
                 if file_change == -1:
                     relative_id = 3 # Cap Left <- Changed from 2
                 else: # file_change == 1
                     relative_id = 4 # Cap Right <- Changed from 3
            else: return None
        else: return None

    # Final check on range (1-18)
    return relative_id if relative_id is not None and 1 <= relative_id <= 18 else None


def get_base_action_id(color: chess.Color, piece_type: chess.PieceType, instance_index: int, combine_color_ranges: bool = True) -> Optional[int]:
    """Calculates the base action ID for a specific piece instance (1-based start).

    Args:
        color: The color of the piece.
        piece_type: The type of the piece.
        instance_index: The 0-based index of the piece instance (e.g., 0 for the first rook, 1 for the second).
        combine_color_ranges: If True (default), Black's action IDs start after all of White's ranges.
                             If False, both White's and Black's ranges effectively start from 1,
                             considering only the pieces of their own color.

    Returns:
        The 1-based starting action ID for the piece's action range, or None if invalid input.
    """
    # instance_index is 0-based
    if not (0 <= instance_index < 8): return None # Basic check for pawn index validity

    # Relative action counts per piece type
    rel_counts = {
        chess.ROOK: 28,
        chess.KNIGHT: 8,
        chess.BISHOP: 28,
        chess.QUEEN: 56,
        chess.KING: 10,
        chess.PAWN: 82
    }

    # Check instance index validity for other pieces
    # Max instances: R=2, N=2, B=2, Q=1, K=1 (standard game)
    # Instance index is 0-based, so max allowed is count-1
    max_instance_map = {
        chess.ROOK: 1, chess.KNIGHT: 1, chess.BISHOP: 1,
        chess.QUEEN: 0, chess.KING: 0, chess.PAWN: 7
    }
    max_expected_index = max_instance_map.get(piece_type, -1)
    if instance_index > max_expected_index:
         logging.warning(f"Invalid instance index {instance_index} for piece type {piece_type}. Max expected: {max_expected_index}")
         return None

    base_id = 1  # Action IDs are 1-based
    # Consistent order of pieces for calculating base IDs
    current_piece_order = [
        (chess.ROOK, 2), (chess.KNIGHT, 2), (chess.BISHOP, 2),
        (chess.QUEEN, 1), (chess.KING, 1), (chess.PAWN, 8)
    ]

    # --- Conditional Offset for Black Pieces ---
    if color == chess.BLACK and combine_color_ranges:
        # If combining ranges, add all white piece ranges first for black pieces
        for p_type, count in current_piece_order:
             base_id += count * rel_counts[p_type]
    # --- End Conditional Offset ---

    # Add ranges for preceding pieces of the *target color*
    for p_type, count in current_piece_order:
        if p_type == piece_type:
            # Add ranges for preceding instances of the same piece type
            base_id += instance_index * rel_counts[p_type]
            return base_id
        else:
            # Add full range for this preceding piece type of the same color
             base_id += count * rel_counts[p_type]

    # Should not be reached if piece_type is valid
    logging.error(f"Error: Piece type {piece_type} not found in order.") # Use logging
    return None

def get_absolute_action_id(
    uci: str,
    color: chess.Color,
    piece_type: chess.PieceType,
    instance_index: int
) -> Optional[int]:
    """
    Calculates the absolute action ID for a given move based on the piece type,
    color, and instance index, incorporating the 'Pawn-Forever' logic.

    Args:
        uci: The move in UCI format.
        color: The color of the piece making the move.
        piece_type: The type of the piece making the move.
        instance_index: The 0-based index for this instance of the piece/color.

    Returns:
        The absolute action ID (1-based) or None if the move is invalid
        or cannot be mapped.
    """
    # Get the base ID for this piece instance
    base_id = get_base_action_id(color, piece_type, instance_index)
    if base_id is None:
        # print(f"Debug: Base ID failed for {color} {piece_type} instance {instance_index}")
        return None

    relative_id = None
    final_absolute_id = None

    # --- Handle Pawn-Forever Logic ---
    if piece_type == chess.PAWN:
        # 1. Try pawn-specific relative moves (1-18)
        rel_pawn_id = uci_to_relative_pawn_action_id(uci, color)
        if rel_pawn_id is not None:
            final_absolute_id = base_id + rel_pawn_id - 1
        else:
            # 2. Try queen-like moves (relative 1-56)
            rel_queen_id = uci_to_relative_queen_action_id(uci)
            if rel_queen_id is not None:
                offset = 18 # Offset for the 18 pawn-specific moves
                final_absolute_id = base_id + offset + rel_queen_id - 1
            else:
                # 3. Try knight-like moves (relative 1-8)
                rel_knight_id = uci_to_relative_knight_action_id(uci)
                if rel_knight_id is not None:
                    offset = 18 + 56 # Offset for pawn (18) + queen (56) moves
                    final_absolute_id = base_id + offset + rel_knight_id - 1
                # else: No valid pattern found for this pawn instance & move

    # --- Handle Other Piece Types ---
    elif piece_type == chess.ROOK:
        relative_id = uci_to_relative_rook_action_id(uci)
    elif piece_type == chess.KNIGHT:
        relative_id = uci_to_relative_knight_action_id(uci)
    elif piece_type == chess.BISHOP:
        relative_id = uci_to_relative_bishop_action_id(uci)
    elif piece_type == chess.QUEEN:
        relative_id = uci_to_relative_queen_action_id(uci)
    elif piece_type == chess.KING:
        relative_id = uci_to_relative_king_action_id(uci) # Returns 1-10 including castling
    else:
        print(f"Warning: Unknown piece type {piece_type} in get_absolute_action_id")
        return None

    # Calculate absolute ID for non-pawn pieces if relative ID was found
    if piece_type != chess.PAWN and relative_id is not None:
        final_absolute_id = base_id + relative_id - 1

    # Validate final ID against theoretical max if needed (e.g., 1700)
    # if final_absolute_id is not None and final_absolute_id > 1700:
    #    print(f"Warning: Calculated absolute ID {final_absolute_id} exceeds expected max.")
    #    return None

    return final_absolute_id

def get_action_id_for_piece_abs(
    uci: str,
    color: chess.Color,
    piece_type: chess.PieceType,
    instance_index: int
) -> Optional[int]:
    """
    Calculates the absolute action ID for a given move based on the piece type,
    color, and instance index, incorporating the 'Pawn-Forever' logic.
    (This version merges the logic previously split into multiple functions).

    Args:
        uci: The move in UCI format.
        color: The color of the piece making the move.
        piece_type: The type of the piece making the move.
        instance_index: The 0-based index for this instance of the piece/color.

    Returns:
        The absolute action ID (1-based) or None if the move is invalid
        or cannot be mapped.
    """
    # Get the base ID for this piece instance
    base_id = get_base_action_id(color, piece_type, instance_index)
    if base_id is None:
        return None

    relative_id = None
    final_absolute_id = None

    # --- Handle Pawn-Forever Logic ---
    if piece_type == chess.PAWN:
        # 1. Try pawn-specific relative moves (1-18)
        rel_pawn_id = uci_to_relative_pawn_action_id(uci, color)
        if rel_pawn_id is not None:
            final_absolute_id = base_id + rel_pawn_id - 1
        else:
            # 2. Try queen-like moves (relative 1-56)
            rel_queen_id = uci_to_relative_queen_action_id(uci)
            if rel_queen_id is not None:
                offset = 18 # Offset for the 18 pawn-specific moves
                final_absolute_id = base_id + offset + rel_queen_id - 1
            else:
                # 3. Try knight-like moves (relative 1-8)
                rel_knight_id = uci_to_relative_knight_action_id(uci)
                if rel_knight_id is not None:
                    offset = 18 + 56 # Offset for pawn (18) + queen (56) moves
                    final_absolute_id = base_id + offset + rel_knight_id - 1
                # else: No valid pattern found for this pawn instance & move

    # --- Handle Other Piece Types ---
    elif piece_type == chess.ROOK:
        relative_id = uci_to_relative_rook_action_id(uci)
    elif piece_type == chess.KNIGHT:
        relative_id = uci_to_relative_knight_action_id(uci)
    elif piece_type == chess.BISHOP:
        relative_id = uci_to_relative_bishop_action_id(uci)
    elif piece_type == chess.QUEEN:
        relative_id = uci_to_relative_queen_action_id(uci)
    elif piece_type == chess.KING:
        relative_id = uci_to_relative_king_action_id(uci) # Returns 1-10 including castling
    else:
        print(f"Warning: Unknown piece type {piece_type} in get_action_id_for_piece_abs")
        return None

    # Calculate absolute ID for non-pawn pieces if relative ID was found
    if piece_type != chess.PAWN and relative_id is not None:
        final_absolute_id = base_id + relative_id - 1

    # You could add validation here if needed:
    # if final_absolute_id is not None and not (1 <= final_absolute_id <= 1700):
    #    print(f"Error: Calculated ID {final_absolute_id} out of expected range 1-1700")
    #    return None

    return final_absolute_id

def create_piece_instance_map(
    board: chess.Board,
    instance_map: Dict[chess.Square, Tuple[chess.Color, chess.PieceType, int]]
) -> Dict[chess.Square, Tuple[chess.Color, chess.PieceType, int]]:
    """
    Populates a given map from square to (color, piece_type, instance_index)
    for the current board state, assigning indices based on a fixed iteration order.
    This provides an arbitrary but consistent mapping for boards without tracking.

    Args:
        board: The current chess.Board object.
        instance_map: The dictionary to populate. It will be cleared first.

    Returns:
        The populated instance_map dictionary.
    """
    instance_map.clear()
    counters: Dict[Tuple[chess.Color, chess.PieceType], int] = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = piece.color
            piece_type = piece.piece_type # Note: This is the CURRENT piece type
            key = (color, piece_type)
            current_count = counters.get(key, 0)
            instance_index = current_count
            instance_map[square] = (color, piece_type, instance_index)
            counters[key] = current_count + 1
    return instance_map


def get_move_action_id(move: chess.Move, board: Union[chess.Board, FullyTrackedBoard]) -> Optional[int]:
    """
    Calculates the absolute action ID for a move, robustly handling both standard
    chess.Board and FullyTrackedBoard, including its theoretically impossible state.

    - If FullyTrackedBoard and possible state: Uses tracker for original piece ID.
    - If FullyTrackedBoard and impossible state: Logs warning, uses positional mapping.
    - If standard chess.Board: Uses positional mapping.

    Args:
        move: The chess.Move object.
        board: The chess.Board or FullyTrackedBoard object before the move.

    Returns:
        The absolute action ID (1-based) or None.
    """
    piece_details: Optional[Tuple[chess.Color, chess.PieceType, int]] = None
    use_tracker = False

    if isinstance(board, FullyTrackedBoard):
        if board.is_theoretically_possible_state:
            # Use tracker for original piece identity
            piece_id = board.get_piece_instance_id_at(move.from_square)
            if piece_id:
                piece_details = piece_id # (color, original_type, original_index)
                use_tracker = True
            else:
                 logging.warning(f"Robust Get ID: Tracker failed for possible state! Square: {chess.square_name(move.from_square)}, Move: {move.uci()}. Falling back to mapping.")
                 # Fall through to mapping logic below
        else:
            logging.warning("Robust Get ID: Board state theoretically impossible. Using positional mapping, not tracker.")
            # Fall through to mapping logic below

    # Fallback/Default: Use positional mapping (create_piece_instance_map)
    if piece_details is None: # If not using tracker or tracker failed
        instance_map = create_piece_instance_map(board, {}) # Pass board directly
        instance_details_from_map = instance_map.get(move.from_square)
        if instance_details_from_map:
             # Note: piece_type here is the CURRENT type on the board
            piece_details = instance_details_from_map # (color, current_type, mapped_index)
        else:
            # If mapping also fails (e.g., no piece at source sq, though move implies one)
             logging.error(f"Robust Get ID: Positional mapping failed for move {move.uci()} from {chess.square_name(move.from_square)}. Board FEN: {board.fen()}")
             return None

    # We should have piece_details now, either from tracker or map
    color, piece_type_to_use, instance_index_to_use = piece_details

    # Calculate action ID using the determined details
    # get_action_id_for_piece_abs uses the piece_type to decide pawn logic,
    # so passing original_type (from tracker) or current_type (from map) is correct.
    action_id = get_action_id_for_piece_abs(
        move.uci(), color, piece_type_to_use, instance_index_to_use
    )

    if action_id is None:
         logging.warning(f"Robust Get ID: get_action_id_for_piece_abs returned None. Move: {move.uci()}, Details: {piece_details}, Used Tracker: {use_tracker}")

    return action_id

def action_id_to_move(action_id: int, board: Union[chess.Board, FullyTrackedBoard]) -> Optional[chess.Move]:
    """
    Finds the legal chess.Move for an action ID, robustly handling both board types
    and the theoretically impossible state flag of FullyTrackedBoard.

    - If FullyTrackedBoard and possible state: Uses tracker for original piece ID.
    - If FullyTrackedBoard and impossible state: Logs warning, uses positional mapping.
    - If standard chess.Board: Uses positional mapping.

    Args:
        action_id: The absolute action ID (1-based).
        board: The current chess.Board or FullyTrackedBoard state.

    Returns:
        The corresponding chess.Move if legal and found, otherwise None.
    """
    if not (1 <= action_id <= 1700): # Adjust max ID if needed
        logging.warning(f"Robust ID to Move: action_id {action_id} is outside expected range (1-1700).")
        return None

    # Decide which method to use based on board type and state
    use_tracker = False
    positional_instance_map: Optional[Dict[chess.Square, Tuple[chess.Color, chess.PieceType, int]]] = None

    if isinstance(board, FullyTrackedBoard):
        if board.is_theoretically_possible_state:
            use_tracker = True
        else:
            logging.warning("Robust ID to Move: Board state theoretically impossible. Using positional mapping, not tracker.")
            positional_instance_map = create_piece_instance_map(board, {})
    else:
        # Standard chess.Board, must use positional mapping
        positional_instance_map = create_piece_instance_map(board, {})

    # Iterate through legal moves and calculate their ID using the chosen method
    for move in board.legal_moves:
        piece_details: Optional[Tuple[chess.Color, chess.PieceType, int]] = None

        if use_tracker:
            piece_id = board.get_piece_instance_id_at(move.from_square)
            if piece_id:
                piece_details = piece_id # (color, original_type, original_index)
            else:
                logging.warning(f"Robust ID to Move: Tracker failed for move {move.uci()} in possible state. Skipping move.")
                continue # Skip this move if tracker fails unexpectedly
        else:
            # Use pre-calculated positional map
            if positional_instance_map is not None:
                instance_details_from_map = positional_instance_map.get(move.from_square)
                if instance_details_from_map:
                    piece_details = instance_details_from_map # (color, current_type, mapped_index)
                else:
                    logging.warning(f"Robust ID to Move: Positional map lookup failed for move {move.uci()}. Skipping move.")
                    continue # Skip if mapping failed
            else:
                 logging.error("Robust ID to Move: Positional map was expected but not created.")
                 return None # Should not happen

        # If we got piece details (either way)
        if piece_details:
            color, piece_type_to_use, instance_index_to_use = piece_details
            calculated_id = get_action_id_for_piece_abs(
                move.uci(), color, piece_type_to_use, instance_index_to_use
            )

            if calculated_id == action_id:
                return move # Found the match

    # If loop finishes without finding the action_id
    return None

def get_legal_moves_with_action_ids(
    board: Union[chess.Board, FullyTrackedBoard],
    return_squares_to_ids: bool = False
) -> Union[Optional[Dict[chess.Square, List[str]]], Optional[List[str]]]:
    """
    Calculates representations of absolute action IDs for all legal moves.

    Returns None if the board contains piece instances that are incompatible
    with the standard action space definition (e.g., more queens than expected).

    Args:
        board: The current chess.Board or FullyTrackedBoard object.
        return_squares_to_ids: If True, returns a dictionary mapping destination
            squares to sorted lists of action IDs. If False (default), returns a
            single sorted list containing all unique valid action IDs.

    Returns:
        - If return_squares_to_ids is True: Optional[Dict[chess.Square, List[str]]]
        - If return_squares_to_ids is False: Optional[List[str]]]
        Returns None if an incompatible piece instance is found.
    """
    dest_square_to_action_ids: Dict[chess.Square, List[int]] = defaultdict(list)
    # final_map: Dict[chess.Square, List[str]] = {} # Only needed if returning dict

    # Decide which method to use based on board type and state
    use_tracker = False
    positional_instance_map: Optional[Dict[chess.Square, Tuple[chess.Color, chess.PieceType, int]]] = None
    log_warning_issued = False # To avoid spamming logs for impossible state
    log_action_id_fail_issued = False # To avoid spamming logs for action_id failures

    # Define max instances expected by the action space (0-based index)
    max_instance_map = {
        chess.ROOK: 1, chess.KNIGHT: 1, chess.BISHOP: 1,
        chess.QUEEN: 0, chess.KING: 0, chess.PAWN: 7
    }

    if isinstance(board, FullyTrackedBoard):
        if board.is_theoretically_possible_state:
            use_tracker = True
        else:
            # Impossible state, use positional mapping
            positional_instance_map = create_piece_instance_map(board, {})
            if not log_warning_issued:
                logging.warning("Get Legal Moves w/ IDs: Board state theoretically impossible. Using positional mapping for all moves.")
                log_warning_issued = True
    else:
        # Standard chess.Board, must use positional mapping
        positional_instance_map = create_piece_instance_map(board, {})

    for move in board.legal_moves:
        piece_details: Optional[Tuple[chess.Color, chess.PieceType, int]] = None

        if use_tracker:
            # Use tracker for original piece identity
            piece_id = board.get_piece_instance_id_at(move.from_square)
            if piece_id:
                piece_details = piece_id # (color, original_type, original_index)
            else:
                # This case might indicate an issue with the tracker itself
                logging.error(f"Get Legal Moves w/ IDs: Tracker failed for move {move.uci()} in a theoretically possible state. Returning None.")
                return None # Treat tracker failure in possible state as critical
        else:
            # Use pre-calculated positional map
            if positional_instance_map is not None:
                instance_details_from_map = positional_instance_map.get(move.from_square)
                if instance_details_from_map:
                    piece_details = instance_details_from_map # (color, current_type, mapped_index)
                else:
                    # Should not happen for a legal move's source square
                    logging.error(f"Get Legal Moves w/ IDs: Positional map lookup failed unexpectedly for move {move.uci()}. Returning None.")
                    return None # Treat map failure as critical
            else:
                logging.error("Get Legal Moves w/ IDs: Positional map logic error. Map not available when expected. Returning None.")
                return None # Critical internal error

        # If we successfully got piece details
        if piece_details:
            color, piece_type_to_use, instance_index_to_use = piece_details

            # --- Check: Validate instance index against action space limits ---            
            max_expected_index = max_instance_map.get(piece_type_to_use, -1)
            if instance_index_to_use > max_expected_index:
                logging.warning(f"Get Legal Moves w/ IDs: Found invalid instance index {instance_index_to_use} (max: {max_expected_index}) for piece type {piece_type_to_use} (Move: {move.uci()}). Board state incompatible with standard action space. Returning None.")
                return None # Return None from the outer function
            # --- End Check ---

            # --- Proceed with action_id calculation only if instance index is valid ---            
            action_id = get_action_id_for_piece_abs(
                move.uci(), color, piece_type_to_use, instance_index_to_use
            )

            if action_id is not None:
                dest_square_to_action_ids[move.to_square].append(action_id)
            else:
                # Log if the specific action ID calculation failed, but only once per call
                if not log_action_id_fail_issued:
                    logging.warning(f"Get Legal Moves w/ IDs: get_action_id_for_piece_abs returned None for at least one valid move. Example Move: {move.uci()}, Details: {piece_details}, Used Tracker: {use_tracker}. (Further failures for this call will not be logged)")
                    log_action_id_fail_issued = True
        else:
            # This case should ideally not be reached if the logic above is sound
            logging.error(f"Get Legal Moves w/ IDs: Failed to determine piece details for move {move.uci()}. Returning None.")
            return None

    # If loop completes without returning None (i.e., all pieces compatible)

    if return_squares_to_ids:
        # Return the dictionary mapping squares to action ID lists
        final_map: Dict[chess.Square, List[str]] = {}
        for square, id_list in dest_square_to_action_ids.items():
            final_map[square] = [str(action_id) for action_id in sorted(id_list)]
        return final_map
    else:
        # Return a flat list of all unique action IDs
        all_action_ids_int: List[int] = []
        for id_list in dest_square_to_action_ids.values():
            all_action_ids_int.extend(id_list)
        # Sort and convert to string - ensure uniqueness if necessary (using set)
        # Using set ensures uniqueness, sorting happens after conversion to list
        unique_sorted_ids_str = [str(aid) for aid in sorted(list(set(all_action_ids_int)))]
        return unique_sorted_ids_str
