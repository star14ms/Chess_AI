import chess
import copy
from typing import Dict, Tuple, Optional, Union, List, Any, ClassVar, Mapping
from collections import Counter # Import Counter

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

    def __init__(self, fen: Optional[str] = chess.STARTING_FEN, *, chess960: bool = False):
        """Initializes the board and the piece tracker."""
        self.piece_tracker: PieceTracker = {}
        self._piece_tracker_stack: List[Tuple[PieceTracker, bool]] = [] # Store flag state too
        self.is_theoretically_possible_state: bool = True # Assume possible initially
        self._pre_initialize_tracker_keys()
        super().__init__(fen, chess960=chess960)
        # Populate tracker based on the board state set by super().__init__
        self._post_init_populate_tracker(fen)


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
                        instance_idx = chess.square_file(square) # Use file index for pawns
                    elif piece_type == chess.KING:
                        instance_idx = 0
                    elif piece_type == chess.QUEEN:
                         instance_idx = 0 # Standard initial queen is instance 0
                    elif piece_type in [chess.ROOK, chess.KNIGHT, chess.BISHOP]:
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

    def set_fen(self, fen: str):
        """Sets the board from FEN and populates the tracker based on the FEN state."""
        self.is_theoretically_possible_state = True # Reset flag before check
        self._pre_initialize_tracker_keys()
        super().set_fen(fen)
        self._populate_tracker_from_current_state() # This method will set flag if needed

    def set_board_fen(self, fen: str):
        """Sets the board part from FEN and populates the tracker based on the FEN state."""
        self.is_theoretically_possible_state = True # Reset flag
        self._pre_initialize_tracker_keys()
        super().set_board_fen(fen)
        self._populate_tracker_from_current_state()

    def set_piece_map(self, pieces: Mapping[chess.Square, chess.Piece]):
        """Sets the board from a piece map and populates the tracker based on the map state."""
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
