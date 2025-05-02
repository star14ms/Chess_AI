from typing import Dict, Tuple, List, Iterator, Optional
import chess
from collections import defaultdict

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


def get_base_action_id(color: chess.Color, piece_type: chess.PieceType, instance_index: int) -> Optional[int]:
    """Calculates the base action ID for a specific piece instance (1-based start)."""
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
    if instance_index > max_instance_map.get(piece_type, -1): # Check against max allowed index
         print(f"Warning: Invalid instance index {instance_index} for {piece_type}")
         return None

    base_id = 1  # Action IDs are 1-based
    # Consistent order of pieces for calculating base IDs
    current_piece_order = [
        (chess.ROOK, 2), (chess.KNIGHT, 2), (chess.BISHOP, 2),
        (chess.QUEEN, 1), (chess.KING, 1), (chess.PAWN, 8)
    ]

    if color == chess.BLACK:
        # Add all white piece ranges first
        for p_type, count in current_piece_order:
             base_id += count * rel_counts[p_type]

    # Add ranges for preceding pieces of the target color
    for p_type, count in current_piece_order:
        if p_type == piece_type:
            # Add ranges for preceding instances of the same piece type
            # instance_index is 0-based, rel_counts is the size of the range
            base_id += instance_index * rel_counts[p_type]
            return base_id
        else:
            # Add full range for this piece type
             base_id += count * rel_counts[p_type]

    # Should not be reached if piece_type is valid
    print(f"Error: Piece type {piece_type} not found in order.")
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


def moves_to_action_ids_abs(
    moves: Iterator[chess.Move],
    piece_instance_map: Dict[chess.Square, Tuple[chess.Color, chess.PieceType, int]],
    target_piece_type: chess.PieceType
) -> Dict[chess.Square, int]:
    """
    Generates a map from destination squares to ABSOLUTE action IDs for legal moves
    of a specific piece type, using a piece instance map.

    Args:
        moves: An iterator of legal chess.Move objects.
        piece_instance_map: A dictionary mapping the FROM square of a potential move
                            to a tuple: (color, piece_type, instance_index).
                            This map must be maintained externally based on game state.
        target_piece_type: The chess.PieceType to filter moves for.

    Returns:
        A dictionary mapping the destination square (chess.Square) of each valid move
        by the target piece type to its calculated absolute action ID (int).
    """
    dest_square_to_action_id: Dict[chess.Square, int] = {}

    for move in moves:
        # Look up piece details from the *starting* square using the map
        instance_details = piece_instance_map.get(move.from_square)

        if instance_details:
            color, piece_type, instance_index = instance_details

            # Check if the piece type matches the target type
            if piece_type == target_piece_type:
                # Get the absolute action ID using the correct function
                action_id = get_action_id_for_piece_abs(
                    move.uci(), color, piece_type, instance_index
                )

                if action_id is not None:
                    # Map the destination square to the action ID
                    dest_square_to_action_id[move.to_square] = action_id
        # else: Move starts from a square not in the instance map (shouldn't happen for legal moves)

    return dest_square_to_action_id

def action_id_to_move(action_id: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Finds the legal chess.Move corresponding to a given absolute action ID
    for the current board state.

    This function works by iterating through all legal moves, calculating their
    action IDs, and returning the move that matches the input ID.

    Args:
        action_id: The absolute action ID (1-based, e.g., 1-1700) to find the move for.
        board: The current chess.Board state.

    Returns:
        The chess.Move object corresponding to the action_id if it represents
        a legal move in the current position, otherwise None.
    """
    # Optional: Validate the input action ID range
    if not (1 <= action_id <= 1700): # Adjust max ID if needed
        print(f"Warning: action_id {action_id} is outside the expected range (1-1700).")
        return None

    # Generate the mapping from squares to piece instance details for the current board
    piece_instance_map: Dict[chess.Square, Tuple[chess.Color, chess.PieceType, int]] = {}
    create_piece_instance_map(board, piece_instance_map)

    # Iterate through all legal moves for the current player
    for move in board.legal_moves:
        # Find the details of the piece making the move
        instance_details = piece_instance_map.get(move.from_square)

        if instance_details:
            color, piece_type, instance_index = instance_details

            # Calculate the absolute action ID for this specific legal move
            calculated_id = get_absolute_action_id(
                move.uci(), color, piece_type, instance_index
            )

            # Check if the calculated ID matches the target ID
            if calculated_id == action_id:
                return move # Found the corresponding legal move
        else:
            # This should ideally not happen for a legal move
            print(f"Warning: Could not find instance details for from_square {chess.square_name(move.from_square)} "
                  f"of legal move {move.uci()}")

    # If the loop completes without finding a match, the action ID is not legal
    # in the current position (or potentially invalid).
    return None

def create_piece_instance_map(
    board: chess.Board,
    instance_map: Dict[chess.Square, Tuple[chess.Color, chess.PieceType, int]]
) -> Dict[chess.Square, Tuple[chess.Color, chess.PieceType, int]]:
    """
    Populates a given map from square to (color, piece_type, instance_index)
    for the current board state.

    Args:
        board: The current chess.Board object.
        instance_map: The dictionary to populate. It will be cleared first
                      if you want to ensure it only contains current board state.
                      Or you can pass a pre-populated dictionary if needed.

    Returns:
        The populated (or updated) instance_map dictionary.
    """
    # Clear the provided map to ensure it only reflects the current board state?
    # Or assume the user wants to add/update? For now, let's assume it should
    # represent ONLY the current board, so we clear it.
    # If you want to update/merge, remove the line below.
    instance_map.clear()

    # Use counters indexed by (color, piece_type)
    counters: Dict[Tuple[chess.Color, chess.PieceType], int] = {}

    # Iterate through squares in a fixed order (0-63) to ensure consistent indexing
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = piece.color
            piece_type = piece.piece_type
            key = (color, piece_type)

            # Get the current count for this piece type/color, default to 0 if not seen yet
            current_count = counters.get(key, 0)
            instance_index = current_count # 0-based index

            # Store the details in the map *passed as argument*
            instance_map[square] = (color, piece_type, instance_index)

            # Increment the counter for the next instance of this type/color
            counters[key] = current_count + 1

    # Return the modified dictionary (optional, as it's modified in-place, but good practice)
    return instance_map

def get_legal_moves_with_action_ids(board: chess.Board) -> Dict[chess.Square, List[str]]: # <-- Returns Dict[Square, List[str]]
    """
    Calculates a map from destination squares to a LIST of string representations
    of absolute action IDs for all legal moves on the current board state.

    Args:
        board: The current chess.Board object.

    Returns:
        A dictionary mapping the destination square (chess.Square) of each
        legal move to a list of its calculated absolute action IDs (as strings).
    """
    # Use defaultdict(list) to easily append to lists
    dest_square_to_action_ids: Dict[chess.Square, List[int]] = defaultdict(list) # <-- Store ints temporarily
    final_map: Dict[chess.Square, List[str]] = {} # <-- Final map with strings

    piece_instance_map = {} # Create map locally
    create_piece_instance_map(board, piece_instance_map) # Populate it

    for move in board.legal_moves:
        instance_details = piece_instance_map.get(move.from_square)

        if instance_details:
            color, piece_type, instance_index = instance_details
            action_id = get_action_id_for_piece_abs(
                move.uci(), color, piece_type, instance_index
            )

            if action_id is not None:
                # Append the integer action ID to the list for the destination square
                dest_square_to_action_ids[move.to_square].append(action_id) # <-- Append, don't overwrite
            else:
                    print(f"Warning: Could not get action ID for legal move {move.uci()} "
                        f"with details: {instance_details}")

    # Convert the lists of integers to lists of strings
    for square, id_list in dest_square_to_action_ids.items():
        final_map[square] = [str(action_id) for action_id in sorted(id_list)] # Sort for consistency

    return final_map # <-- Return the map with lists of strings
