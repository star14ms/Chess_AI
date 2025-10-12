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
    instance_map: Dict[chess.Square, Tuple[chess.Color, chess.PieceType, int]] = {}
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
    Interprets a 26-element observation tile vector (numpy, tensor, or square string)
    and returns a descriptive sentence.

    Args:
        tile_input: A 1D numpy array (length 26), a 1D torch.Tensor (length 26),
                    OR a string with algebraic notation (e.g., "a1", "h8").
        observation_array: The full 26x8x8 observation array (numpy or tensor).
                           REQUIRED if tile_input is a string. Should have shape (26, 8, 8).

    Returns:
        A string describing the piece and state on the tile.
    """
    tile: np.ndarray
    square_index: Optional[chess.Square] = None

    if isinstance(tile_input, str):
        if observation_array is None:
            raise ValueError("observation_array must be provided when tile_input is a string.")

        obs_array_np: np.ndarray
        if isinstance(observation_array, torch.Tensor):
            if observation_array.shape != (26, 8, 8):
                raise ValueError(f"observation_array tensor must have shape (26, 8, 8), got {observation_array.shape}")
            obs_array_np = observation_array.detach().cpu().numpy()
        elif isinstance(observation_array, np.ndarray):
            if observation_array.shape != (26, 8, 8):
                raise ValueError(f"observation_array numpy array must have shape (26, 8, 8), got {observation_array.shape}")
            obs_array_np = observation_array
        else:
            raise TypeError(f"observation_array must be a numpy array or torch.Tensor, got {type(observation_array)}")

        try:
            square_index = chess.parse_square(tile_input.lower())
            rank = chess.square_rank(square_index)
            file = chess.square_file(square_index)
            if not (0 <= rank < 8 and 0 <= file < 8):
                raise ValueError
            tile = obs_array_np[:, rank, file]

        except ValueError:
            raise ValueError(f"Invalid square notation: '{tile_input}'")
        except IndexError:
            raise IndexError(f"Could not access observation_array[:, {rank}, {file}] for input '{tile_input}'.")

    elif isinstance(tile_input, torch.Tensor):
        if tile_input.ndim != 1 or tile_input.shape[0] != 26:
            raise ValueError(f"Input torch.Tensor must have shape (26,), got {tile_input.shape}")
        tile = tile_input.detach().cpu().numpy()

    elif isinstance(tile_input, np.ndarray):
        if tile_input.shape != (26,):
            raise ValueError(f"Input numpy array must have shape (26,), got {tile_input.shape}")
        tile = tile_input
    else:
        raise TypeError(f"tile_input must be a string, numpy array, or torch.Tensor, got {type(tile_input)}")

    description_parts = []
    start_square_name: Optional[str] = None

    # 1. Piece and Color (Channels 0-6)
    piece_color_val = tile[0]
    if piece_color_val == 0:
        description_parts.append("Empty square")
    else:
        color = "White" if piece_color_val == 1 else "Black"
        piece_type_names = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]
        current_piece_type_name = "Unknown"
        try:
            piece_type_index = np.argmax(tile[1:7])
            if tile[1 + piece_type_index] == 1:
                current_piece_type_name = piece_type_names[piece_type_index]
                description_parts.append(f"{color} {current_piece_type_name}")
            else:
                description_parts.append(f"{color} piece (Unknown type)")
        except (ValueError, IndexError):
            description_parts.append(f"{color} piece (Error reading type)")

        # Get piece ID from one-hot encoding (channels 10-25)
        piece_id = np.argmax(tile[10:26])
        if tile[10 + piece_id] == 1:
            id_string = f"(ID: {piece_id}"
            
            # Determine Start Square from Piece ID (0-15) and Color
            actual_color_chess = chess.WHITE if piece_color_val == 1 else chess.BLACK
            start_sq_idx: Optional[chess.Square] = None

            if piece_id == 0: # King
                start_sq_idx = chess.E1 if actual_color_chess == chess.WHITE else chess.E8
            elif piece_id == 1: # Queen
                start_sq_idx = chess.D1 if actual_color_chess == chess.WHITE else chess.D8
            elif piece_id == 2: # Rook 0
                start_sq_idx = chess.A1 if actual_color_chess == chess.WHITE else chess.A8
            elif piece_id == 3: # Rook 1
                start_sq_idx = chess.H1 if actual_color_chess == chess.WHITE else chess.H8
            elif piece_id == 4: # Knight 0
                start_sq_idx = chess.B1 if actual_color_chess == chess.WHITE else chess.B8
            elif piece_id == 5: # Knight 1
                start_sq_idx = chess.G1 if actual_color_chess == chess.WHITE else chess.G8
            elif piece_id == 6: # Bishop 0
                start_sq_idx = chess.C1 if actual_color_chess == chess.WHITE else chess.C8
            elif piece_id == 7: # Bishop 1
                start_sq_idx = chess.F1 if actual_color_chess == chess.WHITE else chess.F8
            elif 8 <= piece_id <= 15: # Pawns
                pawn_instance_index = piece_id - 8
                start_rank_for_pawn = 1 if actual_color_chess == chess.WHITE else 6
                start_sq_idx = chess.square(pawn_instance_index, start_rank_for_pawn)

            if start_sq_idx is not None:
                start_square_name = chess.square_name(start_sq_idx)
                id_string += f", Start: {start_square_name})"
            else:
                id_string += ")"

            description_parts.append(id_string)

    # 2. Square Status (Channels 7-8)
    ep_can_be_captured = int(tile[7])
    castling_target = int(tile[8])

    if ep_can_be_captured == 1:
        description_parts.append("Can be captured EP.")

    if castling_target == 1:
        description_parts.append("King can land here via castling.")

    # 3. Current Player (Channel 9)
    current_player_val = int(tile[9])
    player_desc = "(Turn: White)" if current_player_val == 1 else "(Turn: Black)"

    # 4. Construct Final Sentence
    if not description_parts:
        return "Error: Could not interpret tile."

    final_description = description_parts[0]
    if len(description_parts) > 1:
        piece_id_part = ""
        status_parts = []
        for part in description_parts[1:]:
            if part.startswith("(ID: "):
                piece_id_part = " " + part
            else:
                status_parts.append(part)

        final_description += piece_id_part
        if status_parts:
            final_description += ". " + " ".join(status_parts)

    final_description += " " + player_desc

    if not final_description.endswith('.'):
        final_description += '.'

    return final_description

# --- Action ID Interpretation ---

def _interpret_relative_rook_action(relative_id: int) -> Optional[Tuple[str, str]]:
    """Interprets a relative rook action ID (1-28)."""
    if not (1 <= relative_id <= 28): return None

    distance = (relative_id - 1) % 7 + 1
    direction_code = (relative_id - 1) // 7

    directions = [("North", "N"), ("East", "E"), ("South", "S"), ("West", "W")]
    direction_name, direction_symbol = directions[direction_code]

    move_type = f"Rook {direction_name} dist {distance}"
    uci_pattern = f"{direction_symbol}_dist_{distance}"
    return move_type, uci_pattern

def _interpret_relative_knight_action(relative_id: int) -> Optional[Tuple[str, str]]:
    """Interprets a relative knight action ID (1-8)."""
    if not (1 <= relative_id <= 8): return None

    # Reverse of knight_move_patterns in uci_to_relative_knight_action_id
    patterns_map = {
        1: ((1, 2), "(1,2)"), 2: ((2, 1), "(2,1)"), 3: ((2, -1), "(2,-1)"), 4: ((1, -2), "(1,-2)"),
        5: ((-1, -2), "(-1,-2)"), 6: ((-2, -1), "(-2,-1)"), 7: ((-2, 1), "(-2,1)"), 8: ((-1, 2), "(-1,2)")
    }
    _delta, pattern_desc = patterns_map[relative_id]
    move_type = f"Knight move {pattern_desc}"
    uci_pattern = f"K{pattern_desc.replace('(', '_').replace(')','').replace(',','_')}" # e.g. K_1_2
    return move_type, uci_pattern

def _interpret_relative_bishop_action(relative_id: int) -> Optional[Tuple[str, str]]:
    """Interprets a relative bishop action ID (1-28)."""
    if not (1 <= relative_id <= 28): return None

    distance = (relative_id - 1) % 7 + 1
    direction_code = (relative_id - 1) // 7

    directions = [("NorthEast", "NE"), ("SouthEast", "SE"), ("SouthWest", "SW"), ("NorthWest", "NW")]
    direction_name, direction_symbol = directions[direction_code]

    move_type = f"Bishop {direction_name} dist {distance}"
    uci_pattern = f"{direction_symbol}_dist_{distance}"
    return move_type, uci_pattern

def _interpret_relative_queen_action(relative_id: int) -> Optional[Tuple[str, str]]:
    """Interprets a relative queen action ID (1-56)."""
    if not (1 <= relative_id <= 56): return None

    if 1 <= relative_id <= 28: # Rook-like part
        move_type, uci_pattern = _interpret_relative_rook_action(relative_id)
        return f"Queen as Rook: {move_type}", f"Q_as_R_{uci_pattern}"
    else: # Bishop-like part (relative_id 29-56 maps to bishop 1-28)
        move_type, uci_pattern = _interpret_relative_bishop_action(relative_id - 28)
        return f"Queen as Bishop: {move_type}", f"Q_as_B_{uci_pattern}"

def _interpret_relative_king_action(relative_id: int) -> Optional[Tuple[str, str]]:
    """Interprets a relative king action ID (1-10)."""
    if not (1 <= relative_id <= 10): return None

    if relative_id == 9: return "King Kingside Castle", "K_Castle_KS"
    if relative_id == 10: return "King Queenside Castle", "K_Castle_QS"

    # Standard moves (1-8)
    # Reverse of king_move_patterns
    king_patterns_map = {
        1: ((0, 1), "N (0,1)"), 2: ((1, 1), "NE (1,1)"), 3: ((1, 0), "E (1,0)"), 4: ((1, -1), "SE (1,-1)"),
        5: ((0, -1), "S (0,-1)"), 6: ((-1, -1), "SW (-1,-1)"), 7: ((-1, 0), "W (-1,0)"), 8: ((-1, 1), "NW (-1,1)")
    }
    _delta, pattern_desc = king_patterns_map[relative_id]
    move_type = f"King step {pattern_desc}"
    uci_pattern = f"K_step_{pattern_desc.split(' ')[0]}" # e.g. K_step_N
    return move_type, uci_pattern

def _interpret_relative_pawn_action(relative_id: int) -> Optional[Tuple[str, str]]:
    """Interprets a relative pawn action ID (1-82 for Pawn-Forever)."""
    if not (1 <= relative_id <= 82): return None

    # Standard Pawn Moves (1-18)
    if 1 <= relative_id <= 18:
        promo_types = {'Q': "Queen", 'N': "Knight", 'B': "Bishop", 'R': "Rook"}
        move_directions = ["Forward", "Capture Left", "Capture Right"]

        if relative_id == 1: return "Pawn Forward 1", "P_Fwd1"
        if relative_id == 2: return "Pawn Forward 2", "P_Fwd2"
        if relative_id == 3: return "Pawn Capture Left", "P_CapL" # Assuming EP maps here
        if relative_id == 4: return "Pawn Capture Right", "P_CapR"# Assuming EP maps here
        # IDs 5, 6 were for specific EP flags, not directly used by get_action_id_for_piece_abs output range for base pawn moves

        # Promotions (IDs 7-18)
        if 7 <= relative_id <= 18:
            promo_offset = relative_id - 7
            promo_piece_code = (promo_offset // 3)
            move_direction_code = promo_offset % 3

            promo_char_map = ['Q', 'N', 'B', 'R']
            promo_char = promo_char_map[promo_piece_code]
            promo_name = promo_types[promo_char]
            direction_name = move_directions[move_direction_code]

            move_type = f"Pawn {promo_name} Promo {direction_name}"
            uci_pattern = f"P_Promo{promo_char}_{direction_name.replace(' ','')}"
            return move_type, uci_pattern
        return None # Should be covered by above

    # Pawn-as-Queen Moves (relative_id 19-74, maps to queen 1-56)
    elif 19 <= relative_id <= 74:
        queen_rel_id = relative_id - 18
        queen_move_type, queen_uci_pattern = _interpret_relative_queen_action(queen_rel_id)
        return f"Pawn as Queen: {queen_move_type}", f"P_as_Q_{queen_uci_pattern}"

    # Pawn-as-Knight Moves (relative_id 75-82, maps to knight 1-8)
    elif 75 <= relative_id <= 82:
        knight_rel_id = relative_id - 18 - 56
        knight_move_type, knight_uci_pattern = _interpret_relative_knight_action(knight_rel_id)
        return f"Pawn as Knight: {knight_move_type}", f"P_as_K_{knight_uci_pattern}"

    return None # Should not be reached

def interpret_action(action_id: int) -> Optional[Dict[str, Union[str, int, bool]]]:
    """Interprets an absolute action ID (1-1700) and returns its details."""
    if not (1 <= action_id <= 1700):
        logging.warning(f"Action ID {action_id} is out of expected range (1-1700).")
        return None

    # Constants from get_base_action_id
    rel_counts = {
        chess.ROOK: 28, chess.KNIGHT: 8, chess.BISHOP: 28,
        chess.QUEEN: 56, chess.KING: 10, chess.PAWN: 82
    }
    piece_order = [
        (chess.ROOK, 2, "Rook"), (chess.KNIGHT, 2, "Knight"), (chess.BISHOP, 2, "Bishop"),
        (chess.QUEEN, 1, "Queen"), (chess.KING, 1, "King"), (chess.PAWN, 8, "Pawn")
    ]
    colors = [(chess.WHITE, "White"), (chess.BLACK, "Black")]

    current_base = 1
    for color_val, color_name in colors:
        for piece_type_val, num_instances, piece_type_name_str in piece_order:
            for instance_idx in range(num_instances):
                instance_action_space_size = rel_counts[piece_type_val]
                action_id_upper_bound = current_base + instance_action_space_size - 1

                if current_base <= action_id <= action_id_upper_bound:
                    relative_action_id = action_id - current_base + 1
                    interpretation_result = None

                    if piece_type_val == chess.ROOK:
                        interpretation_result = _interpret_relative_rook_action(relative_action_id)
                    elif piece_type_val == chess.KNIGHT:
                        interpretation_result = _interpret_relative_knight_action(relative_action_id)
                    elif piece_type_val == chess.BISHOP:
                        interpretation_result = _interpret_relative_bishop_action(relative_action_id)
                    elif piece_type_val == chess.QUEEN:
                        interpretation_result = _interpret_relative_queen_action(relative_action_id)
                    elif piece_type_val == chess.KING:
                        interpretation_result = _interpret_relative_king_action(relative_action_id)
                    elif piece_type_val == chess.PAWN:
                        interpretation_result = _interpret_relative_pawn_action(relative_action_id)

                    if interpretation_result:
                        move_type, uci_pattern = interpretation_result
                        return {
                            "action_id": action_id,
                            "piece_color": color_name,
                            "piece_type_str": piece_type_name_str,
                            "instance_index": instance_idx, # 0-based
                            "relative_id_in_piece_space": relative_action_id,
                            "move_type": move_type,
                            "uci_pattern": uci_pattern
                        }
                    else:
                        logging.warning(f"Could not interpret relative_id {relative_action_id} for {color_name} {piece_type_name_str} instance {instance_idx} (Abs ID: {action_id})")
                        return None # Error in relative interpretation

                current_base += instance_action_space_size

    logging.warning(f"Action ID {action_id} did not fall into any known piece range.")
    return None # Should not be reached if action_id is 1-1700 and logic is correct


def get_base_action_id_4672(square: chess.Square) -> int:
    """Calculates the base action ID for a specific square in the 4672-action space.
    
    Args:
        square: The chess square (0-63) to get the base action ID for.
        
    Returns:
        The 1-based starting action ID for the square's action range.
    """
    # Each square has 73 possible actions
    # Base ID for square 0 (a1) is 1, square 1 (b1) is 74, etc.
    return square * 73 + 1

def get_absolute_action_id_4672(
    uci: str,
    color: chess.Color,
    piece_type: chess.PieceType,
    instance_index: int
) -> Optional[int]:
    """
    Calculates the absolute action ID for a given move in the 4672-action space.
    
    Args:
        uci: The move in UCI format.
        color: The color of the piece making the move.
        piece_type: The type of the piece making the move.
        instance_index: The 0-based index for this instance of the piece/color.
        
    Returns:
        The absolute action ID (1-based) or None if the move is invalid.
    """
    if len(uci) != 4 and len(uci) != 5:
        return None
        
    try:
        start_sq = chess.parse_square(uci[0:2])
        end_sq = chess.parse_square(uci[2:4])
    except ValueError:
        return None
        
    # Get base ID for the starting square
    base_id = get_base_action_id_4672(start_sq)
    
    # Calculate relative action ID based on end square and promotion
    file_change = chess.square_file(end_sq) - chess.square_file(start_sq)
    rank_change = chess.square_rank(end_sq) - chess.square_rank(start_sq)
    
    # Handle promotions: underpromotions use 65-73; queen promotions fall back to queen-like indices
    if len(uci) == 5:
        promo_char = uci[4].lower()
        if promo_char in ['n', 'b', 'r']:
            # Promotion moves are encoded as 65-73 using N (65-67), B (68-70), R (71-73)
            promo_offset = {'n': 65, 'b': 68, 'r': 71}[promo_char]
            return base_id + promo_offset - 1  # Subtract 1 because base_id is 1-based
        elif promo_char == 'q':
            # Fall through to queen-like mapping below (treat as normal direction/distance)
            pass
        else:
            return None
    
    # Note: Castling is represented as a standard king move (E/W by 2 squares)
    # and therefore falls under the queen-like move encoding below.
    
    # Regular moves
    if abs(file_change) > 7 or abs(rank_change) > 7:
        return None
        
    # Calculate direction (0-7) and distance (1-7)
    if file_change == 0 and rank_change == 0:
        return None  # No move
        
    # Determine direction (0-7)
    if rank_change > 0:  # North
        if file_change > 0: direction = 1  # NE
        elif file_change < 0: direction = 7  # NW
        else: direction = 0  # N
    elif rank_change < 0:  # South
        if file_change > 0: direction = 3  # SE
        elif file_change < 0: direction = 5  # SW
        else: direction = 4  # S
    else:  # East or West
        if file_change > 0: direction = 2  # E
        else: direction = 6  # W
        
    # Calculate distance (1-7)
    distance = max(abs(file_change), abs(rank_change))
    if distance > 7:
        return None
        
    # Check if it's a knight move
    is_knight_move = (abs(file_change) == 2 and abs(rank_change) == 1) or (abs(file_change) == 1 and abs(rank_change) == 2)
    
    if is_knight_move:
        # Knight moves are encoded as 57-64
        # Map the knight move pattern to 1-8
        knight_patterns = {
            (2, 1): 1, (1, 2): 2, (-1, 2): 3, (-2, 1): 4,
            (-2, -1): 5, (-1, -2): 6, (1, -2): 7, (2, -1): 8
        }
        move_index = 56 + knight_patterns.get((file_change, rank_change), 0)
    else:
        # Queen-like moves are encoded as 1-56
        # direction * 7 + distance
        move_index = direction * 7 + distance
    
    if 1 <= move_index <= 73:  # Total of 73 move types per square
        return base_id + move_index - 1  # Subtract 1 because base_id is 1-based
    
    return None

def get_action_id_for_piece_abs_4672(
    uci: str,
    color: chess.Color,
    piece_type: chess.PieceType,
    instance_index: int
) -> Optional[int]:
    """
    Alias for get_absolute_action_id_4672 for backward compatibility.
    """
    return get_absolute_action_id_4672(uci, color, piece_type, instance_index)

def create_piece_instance_map_4672(
    board: chess.Board,
    instance_map: Dict[chess.Square, Tuple[chess.Color, chess.PieceType, int]] = {}
) -> Dict[chess.Square, Tuple[chess.Color, chess.PieceType, int]]:
    """
    Creates a map from square to (color, piece_type, instance_index) for the 4672-action space.
    This is similar to the original but simplified since we don't need instance tracking
    for the 4672-action space.
    
    Args:
        board: The current chess.Board object.
        instance_map: The dictionary to populate. It will be cleared first.
        
    Returns:
        The populated instance_map dictionary.
    """
    instance_map.clear()
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # For 4672-action space, instance_index is always 0
            instance_map[square] = (piece.color, piece.piece_type, 0)
    return instance_map

def interpret_tile_4672(
    tile_input: Union[np.ndarray, str, torch.Tensor],
    observation_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> str:
    """
    Interprets a tile in the 4672-action space observation format.
    This is similar to the original but simplified since we don't need
    piece instance tracking.
    
    Args:
        tile_input: A 1D numpy array, a 1D torch.Tensor, or a square string.
        observation_array: The full observation array (required if tile_input is a string).
        
    Returns:
        A string describing the piece and state on the tile.
    """
    # Implementation is similar to interpret_tile but without piece instance tracking
    # This is a placeholder - you'll need to implement the specific observation format
    # for your 4672-action space
    return "Not implemented yet"

def interpret_action_4672(action_id: int) -> Optional[Dict[str, Union[str, int, bool]]]:
    """Interprets an absolute action ID in the 4672-action space."""
    if not (1 <= action_id <= 4672):
        logging.warning(f"Action ID {action_id} is out of expected range (1-4672).")
        return None
        
    # Calculate which square this action belongs to
    square = (action_id - 1) // 73
    relative_action = (action_id - 1) % 73 + 1  # Convert to 1-based
    
    # Get square name
    square_name = chess.square_name(square)
    
    # Interpret the relative action
    if 1 <= relative_action <= 56:
        # Queen-like moves (8 directions × 7 distances)
        direction = (relative_action - 1) // 7
        distance = (relative_action - 1) % 7 + 1
        
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        direction_name = directions[direction]
        
        # Calculate end square
        file_change = 0
        rank_change = 0
        if direction == 0:  # N
            rank_change = distance
        elif direction == 1:  # NE
            file_change = distance
            rank_change = distance
        elif direction == 2:  # E
            file_change = distance
        elif direction == 3:  # SE
            file_change = distance
            rank_change = -distance
        elif direction == 4:  # S
            rank_change = -distance
        elif direction == 5:  # SW
            file_change = -distance
            rank_change = -distance
        elif direction == 6:  # W
            file_change = -distance
        elif direction == 7:  # NW
            file_change = -distance
            rank_change = distance
            
        end_file = chess.square_file(square) + file_change
        end_rank = chess.square_rank(square) + rank_change
        
        if 0 <= end_file < 8 and 0 <= end_rank < 8:
            end_square = chess.square(end_file, end_rank)
            end_square_name = chess.square_name(end_square)
            move_type = f"Queen-like move {direction_name} distance {distance}"
            uci_pattern = f"{square_name}{end_square_name}"
        else:
            return None
            
    elif 57 <= relative_action <= 64:
        # Knight moves (8 patterns)
        knight_patterns = {
            1: (2, 1), 2: (1, 2), 3: (-1, 2), 4: (-2, 1),
            5: (-2, -1), 6: (-1, -2), 7: (1, -2), 8: (2, -1)
        }
        file_change, rank_change = knight_patterns[relative_action - 56]
        
        end_file = chess.square_file(square) + file_change
        end_rank = chess.square_rank(square) + rank_change
        
        if 0 <= end_file < 8 and 0 <= end_rank < 8:
            end_square = chess.square(end_file, end_rank)
            end_square_name = chess.square_name(end_square)
            move_type = f"Knight move pattern {relative_action - 56}"
            uci_pattern = f"{square_name}{end_square_name}"
        else:
            return None
            
    elif 65 <= relative_action <= 73:
        # Promotions (3 directions × 3 underpromotions)
        # Order: N (65-67), B (68-70), R (71-73)
        promo_types = ['n', 'b', 'r']
        move_directions = ["forward", "left", "right"]
        
        promo_idx = (relative_action - 65) // 3
        direction_idx = (relative_action - 65) % 3
        
        promo_char = promo_types[promo_idx]
        direction = move_directions[direction_idx]
        
        # Calculate end square based on direction
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        if rank == 6:  # White pawn
            end_rank = 7
            if direction == "forward":
                end_file = file
            elif direction == "left":
                end_file = file - 1
            else:  # right
                end_file = file + 1
        elif rank == 1:  # Black pawn
            end_rank = 0
            if direction == "forward":
                end_file = file
            elif direction == "left":
                end_file = file + 1  # From mover's POV, left for black is file+1
            else:  # right
                end_file = file - 1  # From mover's POV, right for black is file-1
        else:
            return None
            
        if 0 <= end_file < 8:
            end_square = chess.square(end_file, end_rank)
            end_square_name = chess.square_name(end_square)
            move_type = f"Promotion to {promo_char.upper()} {direction}"
            uci_pattern = f"{square_name}{end_square_name}{promo_char}"
        else:
            return None
            
    else:
        return None
        
    return {
        "action_id": action_id,
        "square": square_name,
        "relative_action": relative_action,
        "move_type": move_type,
        "uci_pattern": uci_pattern
    }
