from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
import chess
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


def interpret_tile(
    tile_input: Union[np.ndarray, str, torch.Tensor],
    observation_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> str:
    """
    Interprets a 10-element observation tile vector (numpy, tensor, or square string)
    and returns a descriptive sentence. Determines start square from numeric ID.

    Args:
        tile_input: A 1D numpy array (length 10), a 1D torch.Tensor (length 10),
                    OR a string with algebraic notation (e.g., "a1", "h8").
        observation_array: The full 10x8x8 observation array (numpy or tensor).
                           REQUIRED if tile_input is a string. Should have shape (10, 8, 8).

    Returns:
        A string describing the piece and state on the tile.

    Raises:
        ValueError: If tile_input is a string but observation_array is None or has wrong shape/type.
        ValueError: If tile_input is an invalid string square notation.
        TypeError: If tile_input is not a string, numpy array, or torch.Tensor.
    """
    tile: np.ndarray
    square_index: Optional[chess.Square] = None
    # start_square_name will be determined from numeric ID later

    if isinstance(tile_input, str):
        if observation_array is None:
            raise ValueError("observation_array must be provided when tile_input is a string.")

        # --- Handle observation_array being numpy or tensor ---
        obs_array_np: np.ndarray
        if isinstance(observation_array, torch.Tensor):
            if observation_array.shape != (10, 8, 8):
                raise ValueError(f"observation_array tensor must have shape (10, 8, 8), got {observation_array.shape}")
            obs_array_np = observation_array.detach().cpu().numpy()
        elif isinstance(observation_array, np.ndarray):
            if observation_array.shape != (10, 8, 8):
                raise ValueError(f"observation_array numpy array must have shape (10, 8, 8), got {observation_array.shape}")
            obs_array_np = observation_array
        else:
             raise TypeError(f"observation_array must be a numpy array or torch.Tensor, got {type(observation_array)}")
        # --- End observation_array handling ---
            
        try:
            square_index = chess.parse_square(tile_input.lower())
            rank = chess.square_rank(square_index)
            file = chess.square_file(square_index)
            if not (0 <= rank < 8 and 0 <= file < 8):
                 raise ValueError
            # Use the converted numpy array for indexing
            tile = obs_array_np[:, rank, file]

        except ValueError:
            raise ValueError(f"Invalid square notation: '{tile_input}'")
        except IndexError:
             raise IndexError(f"Could not access observation_array[:, {rank}, {file}] for input '{tile_input}'.")

    # --- Handle Tensor or Numpy Array Input for tile_input ---
    elif isinstance(tile_input, torch.Tensor):
        # Convert tensor to numpy array
        if tile_input.ndim != 1 or tile_input.shape[0] != 10:
            raise ValueError(f"Input torch.Tensor must have shape (10,), got {tile_input.shape}")
        # Ensure it's on CPU and detached from graph before converting
        tile = tile_input.detach().cpu().numpy()
    
    elif isinstance(tile_input, np.ndarray):
        if tile_input.shape != (10,):
             raise ValueError(f"Input numpy array must have shape (10,), got {tile_input.shape}")
        tile = tile_input
    else:
        raise TypeError(f"tile_input must be a string, numpy array, or torch.Tensor, got {type(tile_input)}")


    # --- Interpretation Logic (Updated for 10 channels + Start Square from ID) ---

    description_parts = []
    start_square_name: Optional[str] = None # Initialize here

    # 1. Piece and Color (Channels 0-6)
    if tile[0] == 0:
        description_parts.append("Empty square")
    else:
        color = "White" if tile[0] == 1 else "Black"
        piece_type_names = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]
        try:
            piece_type_index = np.argmax(tile[1:7])
            if tile[1 + piece_type_index] == 1:
                piece_type_name = piece_type_names[piece_type_index]
                description_parts.append(f"{color} {piece_type_name}")
            else:
                 description_parts.append(f"{color} piece (Unknown type)")
        except (ValueError, IndexError):
             description_parts.append(f"{color} piece (Error reading type)")

        # Piece ID (Channel 9) and Determining Start Square
        piece_id_numeric = int(tile[9])
        id_string = f"(ID: {piece_id_numeric}"

        # --- Determine Start Square from Numeric ID (1-32) ---
        if 1 <= piece_id_numeric <= 32:
            start_sq_idx: Optional[chess.Square] = None
            nid_color = chess.WHITE if piece_id_numeric <= 16 else chess.BLACK
            offset_id = piece_id_numeric if nid_color == chess.WHITE else piece_id_numeric - 16

            # Map offset_id (1-16 for each color) to original square
            if offset_id == 1:   start_sq_idx = chess.A1 if nid_color == chess.WHITE else chess.A8 # R0
            elif offset_id == 2: start_sq_idx = chess.H1 if nid_color == chess.WHITE else chess.H8 # R1
            elif offset_id == 3: start_sq_idx = chess.B1 if nid_color == chess.WHITE else chess.B8 # N0
            elif offset_id == 4: start_sq_idx = chess.G1 if nid_color == chess.WHITE else chess.G8 # N1
            elif offset_id == 5: start_sq_idx = chess.C1 if nid_color == chess.WHITE else chess.C8 # B0
            elif offset_id == 6: start_sq_idx = chess.F1 if nid_color == chess.WHITE else chess.F8 # B1
            elif offset_id == 7: start_sq_idx = chess.D1 if nid_color == chess.WHITE else chess.D8 # Q
            elif offset_id == 8: start_sq_idx = chess.E1 if nid_color == chess.WHITE else chess.E8 # K
            elif 9 <= offset_id <= 16: # Pawns P0-P7 -> IDs 9-16 / 25-32
                pawn_file_index = offset_id - 9 # 0-7 maps to file a-h
                start_rank = 1 if nid_color == chess.WHITE else 6 # Rank 2 or 7 (0-indexed)
                start_sq_idx = chess.square(pawn_file_index, start_rank)

            if start_sq_idx is not None:
                start_square_name = chess.square_name(start_sq_idx)
        # --- End Determine Start Square ---

        if start_square_name:
            id_string += f", Start: {start_square_name})"
        else:
            id_string += ")" # Close parenthesis if no start square found

        if piece_id_numeric == 0: # Handle case where ID in vector is 0
             if id_string.endswith(')'):
                 id_string = id_string[:-1] + " - Untracked)"
             else: # Should not happen
                 id_string += " (Untracked)"

        description_parts.append(id_string)


    # 2. Square Status (Channels 7-8)
    ep_can_be_captured = int(tile[7]) # Flag is now on the pawn itself
    castling_target = int(tile[8])

    # Update interpretation for channel 7
    if ep_can_be_captured == 1:
        description_parts.append("Can be captured EP.") # Changed text slightly

    if castling_target == 1:
        description_parts.append("King can land here via castling.")


    # 3. Construct Final Sentence
    if not description_parts:
        return "Error: Could not interpret tile." # Fallback

    final_description = description_parts[0] # Start with piece/empty description
    if len(description_parts) > 1:
        # Join remaining parts, handling piece ID placement
        piece_id_part = ""
        other_parts = []
        for part in description_parts[1:]:
            # Check if it's the specific ID string we constructed
            if part.startswith("(ID: "):
                piece_id_part = " " + part
            else:
                other_parts.append(part)

        final_description += piece_id_part # Add ID right after piece name
        if other_parts:
            final_description += ". " + " ".join(other_parts) # Add status parts


    # Ensure final sentence ends with a period if not already there.
    if not final_description.endswith('.'):
        final_description += '.'

    return final_description