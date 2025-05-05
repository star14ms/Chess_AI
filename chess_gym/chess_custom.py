import chess
import copy
import logging
from typing import Dict, Tuple, Optional, Union, List, ClassVar, Mapping
from collections import Counter, defaultdict
import numpy as np

from utils.analyze import create_piece_instance_map, get_action_id_for_piece_abs

# Define type aliases for clarity
PieceInstanceId = Tuple[chess.Color, chess.PieceType, int] # (Color, OriginalPieceType, OriginalInstanceIndex 0-7/0-1)
PieceTrackingInfo = Dict[str, Optional[Union[chess.Square, chess.PieceType]]]
PieceTracker = Dict[PieceInstanceId, PieceTrackingInfo]

class FullyTrackedBoard(chess.Board):
    """
    A chess.Board subclass that tracks the original identity and current state
    of all 32 starting pieces throughout the game. Includes a flag indicating
    if the current state (set via FEN/map) is theoretically possible based on
    initial piece counts.

    Attributes:
        piece_tracker (PieceTracker): Maps original piece ID to current state.
        is_theoretically_possible_state (bool): Flag indicating if the piece counts
            found during the last FEN/map initialization are consistent with
            a state reachable from the standard start (ignoring position/legality).
            Set to False if counts exceed standard initial numbers (e.g., >8 pawns,
            >1 Queen, >2 R/N/B per color). Note: Multiple Q/R/N/B *can* exist via
            promotion, but this flag reflects impossibility based *only* on the counts
            found when initializing from a state snapshot.
    """

    INITIAL_INSTANCE_MAP: ClassVar[Dict[chess.Square, Tuple[chess.PieceType, int]]] = {
        chess.A1: (chess.ROOK, 0), chess.B1: (chess.KNIGHT, 0), chess.C1: (chess.BISHOP, 0),
        chess.D1: (chess.QUEEN, 0), chess.E1: (chess.KING, 0),
        chess.F1: (chess.BISHOP, 1), chess.G1: (chess.KNIGHT, 1), chess.H1: (chess.ROOK, 1),
        chess.A2: (chess.PAWN, 0), chess.B2: (chess.PAWN, 1), chess.C2: (chess.PAWN, 2), chess.D2: (chess.PAWN, 3),
        chess.E2: (chess.PAWN, 4), chess.F2: (chess.PAWN, 5), chess.G2: (chess.PAWN, 6), chess.H2: (chess.PAWN, 7),
        chess.A7: (chess.PAWN, 0), chess.B7: (chess.PAWN, 1), chess.C7: (chess.PAWN, 2), chess.D7: (chess.PAWN, 3),
        chess.E7: (chess.PAWN, 4), chess.F7: (chess.PAWN, 5), chess.G7: (chess.PAWN, 6), chess.H7: (chess.PAWN, 7),
        chess.A8: (chess.ROOK, 0), chess.B8: (chess.KNIGHT, 0), chess.C8: (chess.BISHOP, 0),
        chess.D8: (chess.QUEEN, 0), chess.E8: (chess.KING, 0),
        chess.F8: (chess.BISHOP, 1), chess.G8: (chess.KNIGHT, 1), chess.H8: (chess.ROOK, 1),
    }
    MAX_INITIAL_COUNTS: ClassVar[Dict[chess.PieceType, int]] = {
        chess.PAWN: 8, chess.ROOK: 2, chess.KNIGHT: 2, chess.BISHOP: 2, chess.QUEEN: 1, chess.KING: 1
    }

    def __init__(self, 
                 fen: Optional[str] = chess.STARTING_FEN, 
                 *, 
                 chess960: bool = False, 
                 piece_tracker_override: Optional[PieceTracker] = None):
        """Initializes the board and the piece tracker. 
        
        Args:
            fen: The FEN string to initialize from. Defaults to STARTING_FEN.
            chess960: Whether the board is in Chess960 mode.
            piece_tracker_override: If provided, use this tracker state instead
                                     of populating from the FEN. Assumes the
                                     provided tracker is consistent with the FEN.
        """
        self.piece_tracker: PieceTracker = {}
        self._piece_tracker_stack: List[Tuple[PieceTracker, bool]] = [] # Store flag state too
        self.is_theoretically_possible_state: bool = True # Assume possible initially
        
        if piece_tracker_override is not None:
            self.piece_tracker = copy.deepcopy(piece_tracker_override)
            # Assume the override is valid, no need to check possibility flag here
            # Set the board state *after* setting the tracker
            super().__init__(fen, chess960=chess960)
        else:
            # Original logic: Pre-initialize, set board, then populate/check
            self._pre_initialize_tracker_keys()
            super().__init__(fen, chess960=chess960)
            # Populate tracker based on the board state set by super().__init__
            self._post_init_populate_tracker(fen) # This sets is_theoretically_possible_state


    def _post_init_populate_tracker(self, initial_fen: Optional[str]):
         """Decides which population method to call after board is set."""
         if initial_fen == chess.STARTING_FEN:
             self.is_theoretically_possible_state = True # Standard start is possible
             self._populate_tracker_from_standard_start()
         else: # Handles None (empty) or arbitrary FEN
             # Reset flag, it will be set by the population method if issues found
             self.is_theoretically_possible_state = True
             self._populate_tracker_from_current_state()


    def _pre_initialize_tracker_keys(self):
        """Creates entries for all 32 potential starting pieces with None values."""
        self.piece_tracker.clear()
        for start_sq, (original_pt, original_instance_idx) in self.INITIAL_INSTANCE_MAP.items():
            color = chess.WHITE if chess.square_rank(start_sq) < 4 else chess.BLACK
            piece_id: PieceInstanceId = (color, original_pt, original_instance_idx)
            self.piece_tracker[piece_id] = {
                'start_sq': start_sq,
                'current_sq': None,
                'promoted_to': None
            }

    def _populate_tracker_from_standard_start(self):
        """Populates the pre-initialized tracker based on standard start positions."""
        # Assumes self.is_theoretically_possible_state is already True
        for piece_id, info in self.piece_tracker.items():
            start_sq = info['start_sq']
            current_piece = self.piece_at(start_sq)
            if current_piece and current_piece.color == piece_id[0] and current_piece.piece_type == piece_id[1]:
                info['current_sq'] = start_sq
            else:
                info['current_sq'] = None
                # This case implies the board wasn't actually standard start, flag it.
                self.is_theoretically_possible_state = False
                print(f"Warning: Standard start population failed for {piece_id} at {chess.square_name(start_sq)}")


    def _populate_tracker_from_current_state(self):
        """
        Populates the pre-initialized tracker based on pieces currently on the board.
        Sets `is_theoretically_possible_state` to False if piece counts exceed
        standard initial numbers.
        """
        # Reset current squares from pre-initialization
        for piece_id in self.piece_tracker:
            self.piece_tracker[piece_id]['current_sq'] = None
            self.piece_tracker[piece_id]['promoted_to'] = None # Reset promotion

        found_piece_counts = Counter() # Tracks counts for (Color, PieceType)
        assigned_tracker_slots = set() # Track which slots we've assigned a piece to

        # Iterate through squares to find pieces and assign them to tracker slots
        for square in chess.SQUARES:
            piece = self.piece_at(square)
            if piece:
                color = piece.color
                piece_type = piece.piece_type
                count_key = (color, piece_type)
                found_piece_counts[count_key] += 1
                current_found_count = found_piece_counts[count_key]

                # --- Check for Impossible Counts (compared to INITIAL set) ---
                max_initial_count = self.MAX_INITIAL_COUNTS.get(piece_type, 0)
                if current_found_count > max_initial_count:
                    # Found more pieces of this type than started the game.
                    # This *might* be due to promotion, but we flag it as theoretically
                    # impossible/ambiguous from the FEN snapshot perspective.
                    self.is_theoretically_possible_state = False
                    print(f"Warning: Found {current_found_count} instance(s) of {piece.symbol()} for "
                          f"{chess.COLOR_NAMES[color]}, exceeding initial max of {max_initial_count}. "
                          f"Setting is_theoretically_possible_state=False.")
                    # We don't try to assign this extra piece to a standard slot.

                else:
                    # --- Try to assign to a standard tracker slot ---
                    instance_idx = -1 # Sentinel value

                    if piece_type == chess.PAWN:
                        instance_idx = current_found_count - 1
                    elif piece_type == chess.KING:
                        instance_idx = 0
                    elif piece_type == chess.QUEEN:
                         instance_idx = 0 # Standard initial queen is instance 0
                    elif piece_type in [chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.PAWN]:
                        # Assign index 0 for the first found, 1 for the second
                        instance_idx = current_found_count - 1 # 0-based index

                    if instance_idx != -1:
                        piece_id_to_update: PieceInstanceId = (color, piece_type, instance_idx)

                        if piece_id_to_update in self.piece_tracker:
                            # Check if this specific slot has already been assigned
                            if piece_id_to_update in assigned_tracker_slots:
                                # This means we found another piece that maps to the same
                                # initial ID (e.g., two white pawns on the A-file in the FEN). Impossible state.
                                self.is_theoretically_possible_state = False
                                print(f"Warning: Multiple pieces found mapping to initial ID {piece_id_to_update}. "
                                      f"Piece at {chess.square_name(square)} conflicts. Setting is_theoretically_possible_state=False.")
                            else:
                                # Assign piece to this tracker slot
                                self.piece_tracker[piece_id_to_update]['current_sq'] = square
                                assigned_tracker_slots.add(piece_id_to_update)
                                # Note: Promotion status ('promoted_to') isn't reliably set here from FEN.
                        else:
                             # This case should theoretically not happen if MAX_INITIAL_COUNTS is correct
                             print(f"Error: Generated piece_id {piece_id_to_update} not found in tracker keys.")
                             self.is_theoretically_possible_state = False

    def get_piece_instance_id_at(self, square: chess.Square) -> Optional[PieceInstanceId]:
        """Finds the original PieceInstanceId associated with the piece currently at the given square."""
        for piece_id, info in self.piece_tracker.items():
            if info.get('current_sq') == square:
                return piece_id
        return None

    def reset(self):
        """Resets the board to the standard starting position and populates the tracker."""
        self.is_theoretically_possible_state = True # Reset flag
        self._pre_initialize_tracker_keys()
        super().reset()
        self._populate_tracker_from_standard_start()

    def set_fen(self, fen: str, piece_tracker_override: Optional[PieceTracker] = None):
        """Sets the board from FEN and populates the tracker based on the FEN state
           or uses the provided tracker override.
           
        Args:
            fen: The FEN string.
            piece_tracker_override: If provided, use this tracker state instead
                                     of populating from the FEN.
        """
        if piece_tracker_override is not None:
            self.piece_tracker = copy.deepcopy(piece_tracker_override)
            self.is_theoretically_possible_state = True # Assume override is valid
            super().set_fen(fen)
        else:
            # Original logic
            self.is_theoretically_possible_state = True # Reset flag before check
            self._pre_initialize_tracker_keys()
            super().set_fen(fen)
            self._populate_tracker_from_current_state() # This method will set flag if needed

    def set_board_fen(self, fen: str, piece_tracker_override: Optional[PieceTracker] = None):
        """Sets the board part from FEN and populates the tracker based on the FEN state
           or uses the provided tracker override.

        Args:
            fen: The board part of the FEN string.
            piece_tracker_override: If provided, use this tracker state instead
                                     of populating from the FEN.
        """
        if piece_tracker_override is not None:
            self.piece_tracker = copy.deepcopy(piece_tracker_override)
            self.is_theoretically_possible_state = True # Assume override is valid
            super().set_board_fen(fen)
        else:
            # Original logic
            self.is_theoretically_possible_state = True # Reset flag
            self._pre_initialize_tracker_keys()
            super().set_board_fen(fen)
            self._populate_tracker_from_current_state()

    def set_piece_map(self, pieces: Mapping[chess.Square, chess.Piece], piece_tracker_override: Optional[PieceTracker] = None):
        """Sets the board from a piece map and populates the tracker based on the map state
           or uses the provided tracker override.

        Args:
            pieces: The piece map.
            piece_tracker_override: If provided, use this tracker state instead
                                     of populating from the FEN.
        """
        if piece_tracker_override is not None:
            self.piece_tracker = copy.deepcopy(piece_tracker_override)
            self.is_theoretically_possible_state = True # Assume override is valid
            super().set_piece_map(pieces)
        else:
            # Original logic
            self.is_theoretically_possible_state = True # Reset flag
            self._pre_initialize_tracker_keys()
            super().set_piece_map(pieces)
            self._populate_tracker_from_current_state()

    # Override push to store the flag state
    def push(self, move: chess.Move):
        # --- Store current tracker state AND flag before applying the move ---
        current_tracker_copy = copy.deepcopy(self.piece_tracker)
        current_flag_state = self.is_theoretically_possible_state
        self._piece_tracker_stack.append((current_tracker_copy, current_flag_state))

        # --- (Previous pre-move analysis logic to find moving/captured IDs) ---
        moving_piece_id: Optional[PieceInstanceId] = None
        captured_piece_id: Optional[PieceInstanceId] = None
        is_capture = self.is_capture(move)
        captured_square = move.to_square
        if self.is_en_passant(move) and self.ep_square is not None:
             captured_square = self.ep_square + (-8 if self.turn == chess.WHITE else 8)
        elif not self.is_en_passant(move) and not is_capture: # Check captures missed by is_capture (e.g. pawn promotion)
            if self.piece_at(move.to_square):
                 is_capture = True

        moving_piece_id = self.get_piece_instance_id_at(move.from_square)
        if is_capture:
            captured_piece_id = self.get_piece_instance_id_at(captured_square)
        # --- End Pre-move analysis ---

        super().push(move) # Apply move to board

        # --- Post-move tracker update (Same as before, simplified castling) ---
        if captured_piece_id and captured_piece_id in self.piece_tracker:
            self.piece_tracker[captured_piece_id]['current_sq'] = None
            self.piece_tracker[captured_piece_id]['promoted_to'] = None

        if moving_piece_id and moving_piece_id in self.piece_tracker:
            original_piece_type = moving_piece_id[1]
            self.piece_tracker[moving_piece_id]['current_sq'] = move.to_square
            if original_piece_type == chess.PAWN and move.promotion:
                self.piece_tracker[moving_piece_id]['promoted_to'] = move.promotion
            elif original_piece_type != chess.PAWN:
                 self.piece_tracker[moving_piece_id]['promoted_to'] = None

        # Handle Castling tracker update (Simplified standard chess logic)
        if move.uci() in ["e1g1", "e1c1", "e8g8", "e8c8"]:
            # This logic needs access to the state BEFORE the push to be fully correct for 960/general case
            # Reverting to simplified standard logic for now
             is_ks = move.uci() in ["e1g1", "e8g8"]
             color = not self.turn # Color of player who just moved
             king_id = (color, chess.KING, 0)

             if is_ks:
                 rook_id = (color, chess.ROOK, 1) # Kingside rook instance
                 king_dest_sq = chess.G1 if color == chess.WHITE else chess.G8
                 rook_dest_sq = chess.F1 if color == chess.WHITE else chess.F8
             else: # Queenside
                 rook_id = (color, chess.ROOK, 0) # Queenside rook instance
                 king_dest_sq = chess.C1 if color == chess.WHITE else chess.C8
                 rook_dest_sq = chess.D1 if color == chess.WHITE else chess.D8

             if king_id in self.piece_tracker: self.piece_tracker[king_id]['current_sq'] = king_dest_sq
             if rook_id in self.piece_tracker: self.piece_tracker[rook_id]['current_sq'] = rook_dest_sq

        # Flag doesn't change during push/pop unless explicitly set by an impossible state calc later
        # self.is_theoretically_possible_state remains as it was before the push

    # Override pop to restore the flag state
    def pop(self) -> chess.Move:
        """Pops a move and restores the piece tracker state and possibility flag."""
        move = super().pop()
        if self._piece_tracker_stack:
            # Restore both tracker and flag state
            restored_tracker, restored_flag = self._piece_tracker_stack.pop()
            self.piece_tracker = restored_tracker
            self.is_theoretically_possible_state = restored_flag
        else:
            print("Warning: Popped board state but piece tracker stack was empty. Reinitializing tracker.")
            self.is_theoretically_possible_state = True # Assume possible on reinit
            self._pre_initialize_tracker_keys()
            self._populate_tracker_from_current_state() # Repopulate based on current board
        return move

    # Override copy to include the flag
    def copy(self, *, stack: Union[bool, int] = True) -> 'FullyTrackedBoard':
        """Creates a copy of the board, including the piece tracker state, stack, and possibility flag."""
        board_copy = super().copy(stack=stack)
        board_copy.piece_tracker = copy.deepcopy(self.piece_tracker)
        board_copy.is_theoretically_possible_state = self.is_theoretically_possible_state # Copy flag

        # Copy the tracker stack (now tuples) based on the 'stack' argument
        if stack is False:
            board_copy._piece_tracker_stack = []
        elif stack is True:
            board_copy._piece_tracker_stack = copy.deepcopy(self._piece_tracker_stack)
        else:
            stack_depth = max(0, len(self._piece_tracker_stack) - int(stack))
            board_copy._piece_tracker_stack = copy.deepcopy(self._piece_tracker_stack[stack_depth:])
        return board_copy
    

    def action_id_to_move(self, action_id: int) -> Optional[chess.Move]:
        """
        Finds the legal chess.Move for an action ID, robustly handling the 
        theoretically impossible state flag.

        - If possible state: Uses tracker for original piece ID.
        - If impossible state: Logs warning, uses positional mapping.

        Args:
            action_id: The absolute action ID (1-based).

        Returns:
            The corresponding chess.Move if legal and found, otherwise None.
        """
        if not (1 <= action_id <= 1700): # Adjust max ID if needed
            logging.warning(f"Robust ID to Move: action_id {action_id} is outside expected range (1-1700).")
            return None

        # Decide which method to use based on board state
        use_tracker = False
        positional_instance_map: Optional[Dict[chess.Square, Tuple[chess.Color, chess.PieceType, int]]] = None

        if self.is_theoretically_possible_state:
            use_tracker = True
        else:
            logging.warning("Robust ID to Move: Board state theoretically impossible. Using positional mapping, not tracker.")
            # Pass self instead of board to create_piece_instance_map
            positional_instance_map = create_piece_instance_map(self, {})

        # Iterate through legal moves and calculate their ID using the chosen method
        for move in self.legal_moves:
            piece_details: Optional[Tuple[chess.Color, chess.PieceType, int]] = None

            if use_tracker:
                # Use self.get_piece_instance_id_at instead of board.get_piece_instance_id_at
                piece_id = self.get_piece_instance_id_at(move.from_square)
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
                # Assume get_action_id_for_piece_abs is imported
                calculated_id = get_action_id_for_piece_abs(
                    move.uci(), color, piece_type_to_use, instance_index_to_use
                )

                if calculated_id == action_id:
                    return move # Found the match

        # If loop finishes without finding the action_id
        return None
    
    def move_to_action_id(self, move: chess.Move) -> Optional[int]:
        """
        Calculates the absolute action ID for a move, robustly handling the 
        theoretically impossible state.

        - If possible state: Uses tracker for original piece ID.
        - If impossible state: Logs warning, uses positional mapping.

        Args:
            move: The chess.Move object.

        Returns:
            The absolute action ID (1-based) or None.
        """
        piece_details: Optional[Tuple[chess.Color, chess.PieceType, int]] = None
        use_tracker = False

        if self.is_theoretically_possible_state:
            # Use tracker for original piece identity
            # Use self.get_piece_instance_id_at
            piece_id = self.get_piece_instance_id_at(move.from_square)
            if piece_id:
                piece_details = piece_id # (color, original_type, original_index)
                use_tracker = True
            else:
                 # Use self.fen()
                 logging.warning(f"Robust Get ID: Tracker failed for possible state! Square: {chess.square_name(move.from_square)}, Move: {move.uci()}. Falling back to mapping. FEN: {self.fen()}")
                 # Fall through to mapping logic below
        else:
            logging.warning("Robust Get ID: Board state theoretically impossible. Using positional mapping, not tracker.")
            # Fall through to mapping logic below

        # Fallback/Default: Use positional mapping (create_piece_instance_map)
        if piece_details is None: # If not using tracker or tracker failed
            # Pass self instead of board
            instance_map = create_piece_instance_map(self, {}) 
            instance_details_from_map = instance_map.get(move.from_square)
            if instance_details_from_map:
                 # Note: piece_type here is the CURRENT type on the board
                piece_details = instance_details_from_map # (color, current_type, mapped_index)
            else:
                # If mapping also fails (e.g., no piece at source sq, though move implies one)
                 # Use self.fen()
                 logging.error(f"Robust Get ID: Positional mapping failed for move {move.uci()} from {chess.square_name(move.from_square)}. Board FEN: {self.fen()}")
                 return None

        # We should have piece_details now, either from tracker or map
        color, piece_type_to_use, instance_index_to_use = piece_details

        # Calculate action ID using the determined details
        # Assume get_action_id_for_piece_abs is imported
        action_id = get_action_id_for_piece_abs(
            move.uci(), color, piece_type_to_use, instance_index_to_use
        )

        if action_id is None:
             logging.warning(f"Robust Get ID: get_action_id_for_piece_abs returned None. Move: {move.uci()}, Details: {piece_details}, Used Tracker: {use_tracker}")

        return action_id

    def get_legal_moves_with_action_ids(
        self,
        return_squares_to_ids: bool = False
    ) -> Union[Optional[Dict[chess.Square, List[int]]], Optional[List[int]]]:
        """
        Calculates representations of absolute action IDs for all legal moves.

        Returns None if the board contains piece instances that are incompatible
        with the standard action space definition (e.g., more queens than expected).

        Args:
            return_squares_to_ids: If True, returns a dictionary mapping destination
                squares to sorted lists of action IDs. If False (default), returns a
                single sorted list containing all unique valid action IDs.

        Returns:
            - If return_squares_to_ids is True: Optional[Dict[chess.Square, List[str]]]
            - If return_squares_to_ids is False: Optional[List[str]]]
            Returns None if an incompatible piece instance is found.
        """
        dest_square_to_action_ids: Dict[chess.Square, List[int]] = defaultdict(list)

        # Decide which method to use based on board state
        use_tracker = False
        positional_instance_map: Optional[Dict[chess.Square, Tuple[chess.Color, chess.PieceType, int]]] = None
        log_warning_issued = False # To avoid spamming logs for impossible state
        log_action_id_fail_issued = False # To avoid spamming logs for action_id failures

        # Define max instances expected by the action space (0-based index)
        max_instance_map = {
            chess.ROOK: 1, chess.KNIGHT: 1, chess.BISHOP: 1,
            chess.QUEEN: 0, chess.KING: 0, chess.PAWN: 7
        }

        if self.is_theoretically_possible_state:
            use_tracker = True
        else:
            # Impossible state, use positional mapping
            positional_instance_map = create_piece_instance_map(self, {})
            if not log_warning_issued:
                logging.warning("Get Legal Moves w/ IDs: Board state theoretically impossible. Using positional mapping for all moves.")
                log_warning_issued = True

        for move in self.legal_moves:
            piece_details: Optional[Tuple[chess.Color, chess.PieceType, int]] = None

            if use_tracker:
                # Use tracker for original piece identity
                piece_id = self.get_piece_instance_id_at(move.from_square)
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
            final_map: Dict[chess.Square, List[int]] = {}
            for square, id_list in dest_square_to_action_ids.items():
                final_map[square] = [action_id for action_id in sorted(id_list)]
            return final_map
        else:
            # Return a flat list of all unique action IDs
            all_action_ids_int: List[int] = []
            for id_list in dest_square_to_action_ids.values():
                all_action_ids_int.extend(id_list)
            # Sort and convert to string - ensure uniqueness if necessary (using set)
            # Using set ensures uniqueness, sorting happens after conversion to list
            unique_sorted_ids_str = [aid for aid in sorted(list(set(all_action_ids_int)))]
            return unique_sorted_ids_str

    def get_board_vector(self) -> np.ndarray:
        """
        Generates an 11x8x8 numpy array representing the board state based on
        instructions/observation_human.md.

        Output shape: (11, 8, 8) -> (Channels, Rank, File)

        Channels (11 dimensions):
        0:   Color (-1 Black, 0 Empty, 1 White)
        1-6: Piece Type / BehaviorType (One-hot: P, N, B, R, Q, K for piece on square)
        7:   En Passant Target (1 if this square is the EP target, 0 otherwise)
        8:   Castling Target (1 if King could land here via castling this turn, 0 otherwise)
        9:   Current Player (-1 Black, 1 White - constant across the plane)
        10:  Piece ID (0 for empty, 1-32 for pieces - requires mapping logic)
        """
        # Initialize with shape (Channels, Height, Width) -> (Channels, Rank, File)
        board_vector = np.zeros((11, 8, 8), dtype=np.int8)

        # Global states needed
        ep_square = self.ep_square
        current_turn = self.turn
        castling_rights = self.castling_rights # Use full rights mask

        # Channel 9: Current Player (Constant plane)
        player_val = 1 if current_turn == chess.WHITE else -1
        board_vector[9, :, :] = player_val

        # Potential castling destination squares for the current player
        castling_target_squares = set()
        if current_turn == chess.WHITE:
            if castling_rights & chess.BB_A1: # White Queenside (Rook on a1)
                 # Check if King can castle Queenside (requires King on e1, clear path b1-d1)
                 # Note: python-chess has helper functions, but implementing logic directly:
                 if self.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE) and \
                    self.piece_at(chess.B1) is None and self.piece_at(chess.C1) is None and self.piece_at(chess.D1) is None:
                    # Check if squares are attacked - python-chess board.has_castling_rights covers this implicitly if used
                    # Simplified check based *only* on rights mask for this example:
                     if self.has_queenside_castling_rights(chess.WHITE):
                         castling_target_squares.add(chess.C1) # King lands on c1
            if castling_rights & chess.BB_H1: # White Kingside (Rook on h1)
                 if self.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE) and \
                    self.piece_at(chess.F1) is None and self.piece_at(chess.G1) is None:
                     # Simplified check based *only* on rights mask for this example:
                      if self.has_kingside_castling_rights(chess.WHITE):
                          castling_target_squares.add(chess.G1) # King lands on g1
        else: # BLACK's turn
            if castling_rights & chess.BB_A8: # Black Queenside (Rook on a8)
                 if self.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK) and \
                    self.piece_at(chess.B8) is None and self.piece_at(chess.C8) is None and self.piece_at(chess.D8) is None:
                     if self.has_queenside_castling_rights(chess.BLACK):
                        castling_target_squares.add(chess.C8) # King lands on c8
            if castling_rights & chess.BB_H8: # Black Kingside (Rook on h8)
                 if self.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK) and \
                    self.piece_at(chess.F8) is None and self.piece_at(chess.G8) is None:
                     if self.has_kingside_castling_rights(chess.BLACK):
                        castling_target_squares.add(chess.G8) # King lands on g8

        # Determine the square of the pawn that moved two steps, if any
        pawn_advanced_two_sq: Optional[chess.Square] = None
        if ep_square is not None:
            # If the ep_square is on rank 2 (0-indexed), a White pawn just moved 2 squares.
            if chess.square_rank(ep_square) == 2: # White pawn moved (e.g., e2-e4), ep_square is rank 2 (e.g., e3)
                pawn_advanced_two_sq = ep_square + 8 # Pawn landed on rank 3 (e.g., e4)
            # If the ep_square is on rank 5 (0-indexed), a Black pawn just moved 2 squares.
            elif chess.square_rank(ep_square) == 5: # Black pawn moved (e.g., e7-e5), ep_square is rank 5 (e.g., e6)
                pawn_advanced_two_sq = ep_square - 8 # Pawn landed on rank 4 (e.g., e5)

        for sq in chess.SQUARES:
            rank = chess.square_rank(sq) # 0-7
            file = chess.square_file(sq) # 0-7
            piece = self.piece_at(sq)

            # Channel 7: Pawn Advanced Two Squares (New Meaning)
            # Set flag on the PAWN's square, not the target square
            if pawn_advanced_two_sq == sq:
                board_vector[7, rank, file] = 1
            # else: remains 0

            # Channel 8: Castling Target
            if sq in castling_target_squares:
                board_vector[8, rank, file] = 1

            if piece:
                # Channel 0: Color
                color_val = 1 if piece.color == chess.WHITE else -1
                board_vector[0, rank, file] = color_val

                # Channels 1-6: Piece Type / BehaviorType (One-hot)
                piece_type_idx = piece.piece_type - 1 # 0-5 for P,N,B,R,Q,K
                board_vector[1 + piece_type_idx, rank, file] = 1

                # Channel 10: Piece ID
                piece_instance_id_tuple = self.get_piece_instance_id_at(sq)
                if piece_instance_id_tuple:
                    # --- Implement consistent mapping logic --- 
                    color, original_type, original_index = piece_instance_id_tuple
                    mapped_id = 0 # Default invalid ID
                    offset = 0 if color == chess.WHITE else 16 # White: 0-15, Black: 16-31 -> Add 1 later

                    if original_type == chess.ROOK:         # IDs 1,2 / 17,18
                        mapped_id = offset + 1 + original_index
                    elif original_type == chess.KNIGHT:       # IDs 3,4 / 19,20
                        mapped_id = offset + 3 + original_index
                    elif original_type == chess.BISHOP:       # IDs 5,6 / 21,22
                        mapped_id = offset + 5 + original_index
                    elif original_type == chess.QUEEN:        # IDs 7 / 23
                        mapped_id = offset + 7 + original_index
                    elif original_type == chess.KING:         # IDs 8 / 24
                        mapped_id = offset + 8 + original_index
                    elif original_type == chess.PAWN:         # IDs 9-16 / 25-32
                        mapped_id = offset + 9 + original_index
                    
                    if not (1 <= mapped_id <= 32):
                        # Handle error case: Calculated ID is out of bounds
                        # This might happen if original_index is unexpected
                        print(f"Warning: Calculated invalid piece ID {mapped_id} for {piece_instance_id_tuple} at {chess.square_name(sq)}. Setting to 0.")
                        mapped_id = 0 # Assign 0 if mapping failed
                    # --- End mapping logic ---
                    
                    board_vector[10, rank, file] = mapped_id
                else:
                    board_vector[10, rank, file] = 0 # 0 if no specific instance ID tracked

            else:
                # Empty Square Defaults
                board_vector[0, rank, file] = 0 # Color
                # Channels 1-6 (Piece Type) remain 0
                board_vector[10, rank, file] = 0 # Piece ID

        return board_vector

    # --- Express Move in Notation ---
    def san(self, move_or_action_id: Union[chess.Move, int]) -> str:
        """Returns the SAN notation for a move."""
        if isinstance(move_or_action_id, int):
            move = self.action_id_to_move(move_or_action_id)
        else:
            move = move_or_action_id
        return super().san(move)
    
    def lan(self, move_or_action_id: Union[chess.Move, int]) -> str:
        """Returns the LAN notation for a move."""
        if isinstance(move_or_action_id, int):
            move = self.action_id_to_move(move_or_action_id)
        else:
            move = move_or_action_id
        return super().lan(move)
    
    def uci(self, move_or_action_id: Union[chess.Move, int]) -> str:
        """Returns the UCI notation for a move."""
        if isinstance(move_or_action_id, int):
            move = self.action_id_to_move(move_or_action_id)
        else:
            move = move_or_action_id
        return super().uci(move)

    def get_numeric_id_to_instance_id_map(self) -> Dict[int, PieceInstanceId]:
        """Creates a mapping from numeric ID (1-32) back to PieceInstanceId tuple."""
        mapping = {}
        for num_id in range(1, 33):
            color = chess.WHITE if num_id <= 16 else chess.BLACK
            offset_id = num_id if color == chess.WHITE else num_id - 16

            if offset_id <= 2:    # Rooks (1, 2)
                original_type = chess.ROOK
                original_index = offset_id - 1
            elif offset_id <= 4:  # Knights (3, 4)
                original_type = chess.KNIGHT
                original_index = offset_id - 3
            elif offset_id <= 6:  # Bishops (5, 6)
                original_type = chess.BISHOP
                original_index = offset_id - 5
            elif offset_id == 7:  # Queen (7)
                original_type = chess.QUEEN
                original_index = 0
            elif offset_id == 8:  # King (8)
                original_type = chess.KING
                original_index = 0
            elif offset_id <= 16: # Pawns (9-16)
                original_type = chess.PAWN
                original_index = offset_id - 9
            else:
                continue # Should not happen for 1-32
            
            instance_id: PieceInstanceId = (color, original_type, original_index)
            mapping[num_id] = instance_id
        return mapping
    
    def get_instance_id_to_numeric_id_map(self) -> Dict[PieceInstanceId, int]:
        """Creates a mapping from PieceInstanceId tuple to numeric ID (1-32)."""
        mapping = {}
        for piece_id_tuple, info in self.piece_tracker.items():
            color, original_type, original_index = piece_id_tuple
            offset = 0 if color == chess.WHITE else 16
            num_id = 0

            if original_type == chess.ROOK:      num_id = offset + 1 + original_index
            elif original_type == chess.KNIGHT:  num_id = offset + 3 + original_index
            elif original_type == chess.BISHOP:  num_id = offset + 5 + original_index
            elif original_type == chess.QUEEN:   num_id = offset + 7 + original_index
            elif original_type == chess.KING:    num_id = offset + 8 + original_index
            elif original_type == chess.PAWN:    num_id = offset + 9 + original_index

            if 1 <= num_id <= 32:
                mapping[piece_id_tuple] = num_id
            else:
                 print(f"Warning: Failed to map {piece_id_tuple} to a valid numeric ID (1-32).")
        return mapping

    def get_start_square_from_numeric_id(self, numeric_id: int) -> Optional[chess.Square]:
        """Finds the starting square for a given numeric piece ID (1-32)."""
        if not (1 <= numeric_id <= 32):
            logging.warning(f"get_start_square_from_numeric_id: Invalid numeric_id {numeric_id}. Must be 1-32.")
            return None

        # Reverse the mapping from numeric ID to PieceInstanceId tuple
        color = chess.WHITE if numeric_id <= 16 else chess.BLACK
        offset_id = numeric_id if color == chess.WHITE else numeric_id - 16 # Value relative to start of color range (1-16)
        instance_id: Optional[PieceInstanceId] = None

        if offset_id <= 2:    # Rooks (1, 2)
            original_type = chess.ROOK
            original_index = offset_id - 1
            instance_id = (color, original_type, original_index)
        elif offset_id <= 4:  # Knights (3, 4)
            original_type = chess.KNIGHT
            original_index = offset_id - 3
            instance_id = (color, original_type, original_index)
        elif offset_id <= 6:  # Bishops (5, 6)
            original_type = chess.BISHOP
            original_index = offset_id - 5
            instance_id = (color, original_type, original_index)
        elif offset_id == 7:  # Queen (7)
            original_type = chess.QUEEN
            original_index = 0
            instance_id = (color, original_type, original_index)
        elif offset_id == 8:  # King (8)
            original_type = chess.KING
            original_index = 0
            instance_id = (color, original_type, original_index)
        elif offset_id <= 16: # Pawns (9-16)
            original_type = chess.PAWN
            original_index = offset_id - 9
            instance_id = (color, original_type, original_index)
        else:
            logging.error(f"get_start_square_from_numeric_id: Logic error reconstructing PieceInstanceId for numeric_id {numeric_id}.")
            return None # Should not happen for valid numeric_id

        # Look up the reconstructed instance_id in the tracker
        if instance_id in self.piece_tracker:
            start_square = self.piece_tracker[instance_id].get('start_sq')
            if start_square is None:
                logging.warning(f"get_start_square_from_numeric_id: Piece ID {instance_id} found in tracker, but 'start_sq' is None.")
            return start_square # Return square index (or None if not set)
        else:
            logging.warning(f"get_start_square_from_numeric_id: Reconstructed Piece ID {instance_id} (from numeric {numeric_id}) not found in piece_tracker.")
            return None
