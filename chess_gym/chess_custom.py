import chess
import copy
import logging
from typing import Dict, Tuple, Optional, Union, List, ClassVar, Mapping
from collections import Counter, defaultdict
import numpy as np
import enum

from utils.analyze import create_piece_instance_map, get_action_id_for_piece_abs, create_piece_instance_map_4672, get_action_id_for_piece_abs_4672

# Define type aliases for clarity
PieceInstanceId = Tuple[chess.Color, chess.PieceType, int] # (Color, OriginalPieceType, OriginalInstanceIndex 0-7/0-1)
PieceTrackingInfo = Dict[str, Optional[Union[chess.Square, chess.PieceType]]]
PieceTracker = Dict[PieceInstanceId, PieceTrackingInfo]

chess.Termination.ILLEGAL_MOVE = enum.auto()

class BaseChessBoard(chess.Board):
    """Base class for chess boards with vector representation capability."""
    
    def __init__(self, fen: Optional[str] = chess.STARTING_FEN, *, chess960: bool = False):
        """Initialize the board with foul state tracking."""
        self.foul: bool = False
        super().__init__(fen, chess960=chess960)
    
    def get_board_vector(self, history_steps: int = 8) -> np.ndarray:
        """
        Generates a 26x8x8 numpy array representing the board state.

        Output shape: (26, 8, 8) -> (Channels, Rank, File)

        Channels (26 dimensions):
        Features (10 channels):
        0:   Color (-1 Black, 0 Empty, 1 White)
        1-6: Piece Type / BehaviorType (One-hot: P, N, B, R, Q, K for piece on square)
        7:   EnPassantTarget (1 if this square is the EP target for the next move, 0 otherwise)
        8:   CastlingTarget (1 if King could land on this square via castling this turn, 0 otherwise)
        9:   Current Player (-1 Black, 1 White - constant across the plane)

        Piece Type Information (16 channels):
        10-25: Piece Identity (16 channels, one-hot encoding for each piece)
        """
        # Initialize with shape (Channels, Height, Width) -> (Channels, Rank, File)
        board_vector = np.zeros((26, 8, 8), dtype=np.int8)

        # Global states needed
        ep_square = self.ep_square
        current_turn = self.turn
        castling_rights = self.castling_rights

        # Channel 9: Current Player (Constant plane)
        player_val = 1 if current_turn == chess.WHITE else -1
        board_vector[9, :, :] = player_val

        # Potential castling destination squares for the current player
        castling_target_squares = set()
        if current_turn == chess.WHITE:
            if castling_rights & chess.BB_A1: # White Queenside
                if self.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE) and \
                   self.piece_at(chess.B1) is None and self.piece_at(chess.C1) is None and self.piece_at(chess.D1) is None:
                    if self.has_queenside_castling_rights(chess.WHITE):
                        castling_target_squares.add(chess.C1)
            if castling_rights & chess.BB_H1: # White Kingside
                if self.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE) and \
                   self.piece_at(chess.F1) is None and self.piece_at(chess.G1) is None:
                    if self.has_kingside_castling_rights(chess.WHITE):
                        castling_target_squares.add(chess.G1)
        else: # BLACK's turn
            if castling_rights & chess.BB_A8: # Black Queenside
                if self.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK) and \
                   self.piece_at(chess.B8) is None and self.piece_at(chess.C8) is None and self.piece_at(chess.D8) is None:
                    if self.has_queenside_castling_rights(chess.BLACK):
                        castling_target_squares.add(chess.C8)
            if castling_rights & chess.BB_H8: # Black Kingside
                if self.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK) and \
                   self.piece_at(chess.F8) is None and self.piece_at(chess.G8) is None:
                    if self.has_kingside_castling_rights(chess.BLACK):
                        castling_target_squares.add(chess.G8)

        for sq in chess.SQUARES:
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            piece = self.piece_at(sq)

            # Channel 7: EnPassantTarget
            if ep_square is not None and sq == ep_square:
                board_vector[7, rank, file] = 1

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

                # Channels 10-25: Piece Identity (16 channels, one-hot)
                self._set_piece_identity_channels(board_vector, sq, rank, file, piece)

            else:
                # Empty Square Defaults
                board_vector[0, rank, file] = 0

        return board_vector

    def _set_piece_identity_channels(self, board_vector: np.ndarray, sq: chess.Square, rank: int, file: int, piece: chess.Piece):
        """Abstract method to be implemented by child classes for setting piece identity channels."""
        raise NotImplementedError("Child classes must implement _set_piece_identity_channels")

    def is_foul(self) -> bool:
        """Returns True if the board is in a foul state (illegal move)."""
        return self.foul

    def is_game_over(self, *, claim_draw: bool = False) -> bool:
        return self.outcome(claim_draw=claim_draw) is not None

    def outcome(self, claim_draw: bool = False) -> chess.Outcome:
        """Returns the outcome of the game."""
        # Save a copy of the board before calling outcome to prevent reset
        # This is the safest way to restore the board state if it gets reset
        board_copy = self.copy(stack=True)
        
        if self.foul:
            return chess.Outcome(chess.Termination.ILLEGAL_MOVE, not self.turn)
        
        # Clean _stack before calling super().outcome() to prevent AttributeError
        # Only filter out items that are None - don't modify structure since _stack items
        # have _BoardState as first element, not Move, and modifying structure could break things
        if hasattr(self, '_stack') and self._stack:
            # Only remove top-level None items, don't modify tuple structure
            if any(item is None for item in self._stack):
                self._stack = [item for item in self._stack if item is not None]
        
        try:
            # #region agent log
            import json
            import time
            try:
                saved_fen = self.fen()
                saved_move_stack_len = len(self.move_stack) if hasattr(self, 'move_stack') else 0
                with open('/Users/minseo/Documents/Github/_star14ms/Chess_AI/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"chess_custom.py:141","message":"before super().outcome","data":{"saved_fen":saved_fen[:80],"saved_move_stack_len":saved_move_stack_len},"timestamp":int(time.time()*1000)})+'\n')
            except: pass
            # #endregion
            outcome_result = super().outcome(claim_draw=claim_draw)
            # #region agent log
            try:
                current_fen_after = self.fen()
                current_move_stack_len_after = len(self.move_stack) if hasattr(self, 'move_stack') else 0
                with open('/Users/minseo/Documents/Github/_star14ms/Chess_AI/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"chess_custom.py:150","message":"after super().outcome","data":{"current_fen":current_fen_after[:80],"current_move_stack_len":current_move_stack_len_after,"saved_fen":saved_fen[:80],"saved_move_stack_len":saved_move_stack_len,"needs_restore":current_fen_after != saved_fen or current_move_stack_len_after != saved_move_stack_len},"timestamp":int(time.time()*1000)})+'\n')
            except: pass
            # #endregion
            # ALWAYS restore move_stack and _stack after super().outcome() if they were present
            # This is critical because super().outcome() may call is_repetition() multiple times,
            # and we need to preserve stacks for subsequent calls (e.g., is_fivefold_repetition())
            current_fen = self.fen()
            current_move_stack_len = len(self.move_stack) if hasattr(self, 'move_stack') else 0
            saved_fen = board_copy.fen()
            saved_move_stack_len = len(board_copy.move_stack) if hasattr(board_copy, 'move_stack') else 0
            needs_restore = current_fen != saved_fen or current_move_stack_len != saved_move_stack_len
            should_restore_stacks = saved_move_stack_len > 0  # Always restore if stacks were present
            
            # #region agent log
            try:
                with open('/Users/minseo/Documents/Github/_star14ms/Chess_AI/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"chess_custom.py:162","message":"checking if restore needed","data":{"current_fen":current_fen[:80],"saved_fen":saved_fen[:80],"current_move_stack_len":current_move_stack_len,"saved_move_stack_len":saved_move_stack_len,"needs_restore":needs_restore,"should_restore_stacks":should_restore_stacks},"timestamp":int(time.time()*1000)})+'\n')
            except: pass
            # #endregion
            
            if needs_restore:
                # #region agent log
                try:
                    with open('/Users/minseo/Documents/Github/_star14ms/Chess_AI/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F,G","location":"chess_custom.py:175","message":"restoring board state from copy","data":{"current_fen":current_fen[:80],"saved_fen":saved_fen[:80]},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                # Board was reset, restore it from the copy
                # Use _restore_chess_stack to properly restore move_stack and _stack
                from MCTS.mcts_algorithm import _restore_chess_stack
                self.set_fen(saved_fen)
                _restore_chess_stack(self, board_copy)
            elif should_restore_stacks:
                # Always restore stacks if they were present, even if FEN didn't change
                # This is necessary because super().outcome() may clear stacks without changing FEN
                # and we need stacks preserved for subsequent calls (e.g., is_fivefold_repetition())
                # #region agent log
                try:
                    with open('/Users/minseo/Documents/Github/_star14ms/Chess_AI/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"chess_custom.py:188","message":"restoring stacks only (FEN unchanged, but stacks were present)","data":{"current_move_stack_len":current_move_stack_len,"saved_move_stack_len":saved_move_stack_len},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                from MCTS.mcts_algorithm import _restore_chess_stack
                _restore_chess_stack(self, board_copy)
            
            # #region agent log
            try:
                fen_after_restore = self.fen()
                move_stack_len_after_restore = len(self.move_stack) if hasattr(self, 'move_stack') else 0
                stack_len_after_restore = len(self._stack) if hasattr(self, '_stack') else 0
                with open('/Users/minseo/Documents/Github/_star14ms/Chess_AI/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F,G","location":"chess_custom.py:194","message":"after restore from copy","data":{"fen":fen_after_restore[:80],"move_stack_len":move_stack_len_after_restore,"stack_len":stack_len_after_restore,"restored_correctly":fen_after_restore == saved_fen and move_stack_len_after_restore == saved_move_stack_len},"timestamp":int(time.time()*1000)})+'\n')
            except: pass
            # #endregion
            return outcome_result
        except AttributeError as e:
            # If _stack has None values causing AttributeError, try to handle it without clearing _stack
            # Clearing _stack causes python-chess to reset the board
            if "'NoneType'" in str(e) or "from_square" in str(e):
                # Instead of clearing _stack, try to filter it more carefully
                # Only clear if absolutely necessary, and restore board state after
                if hasattr(self, '_stack') and self._stack:
                    # Try filtering _stack more carefully
                    filtered_stack = []
                    for item in self._stack:
                        if item is None:
                            continue
                        # Check if it's a tuple/list and if the first element (move) is None
                        if isinstance(item, (tuple, list)) and len(item) > 0:
                            if item[0] is None:
                                continue
                        filtered_stack.append(item)
                    self._stack = filtered_stack
                # Retry once after filtering _stack
                try:
                    outcome_result = super().outcome(claim_draw=claim_draw)
                    # Check if board was reset and restore if needed
                    current_fen = self.fen()
                    current_move_stack_len = len(self.move_stack) if hasattr(self, 'move_stack') else 0
                    if current_fen != saved_fen or current_move_stack_len != len(saved_move_stack):
                        self.set_fen(saved_fen)
                        # Restore move_stack by replaying moves
                        if hasattr(self, 'move_stack') and saved_move_stack:
                            self.move_stack.clear()
                            for move in saved_move_stack:
                                if move is not None:
                                    try:
                                        self.push(move)
                                    except Exception:
                                        break
                        # Restore _stack if it was saved
                        if hasattr(self, '_stack') and saved_stack:
                            self._stack = list(saved_stack)
                    return outcome_result
                except (AttributeError, Exception):
                    # If retry also fails, return None (no outcome) to allow game to continue
                    # Restore board state before returning
                    current_fen = self.fen()
                    if current_fen != saved_fen:
                        self.set_fen(saved_fen)
                        # Restore move_stack by replaying moves
                        if hasattr(self, 'move_stack') and saved_move_stack:
                            self.move_stack.clear()
                            for move in saved_move_stack:
                                if move is not None:
                                    try:
                                        self.push(move)
                                    except Exception:
                                        break
                        # Restore _stack if it was saved
                        if hasattr(self, '_stack') and saved_stack:
                            self._stack = list(saved_stack)
                    return None  # Return None instead of ILLEGAL_MOVE to allow game to continue
            raise
    
    def is_repetition(self, count: int = 3) -> bool:
        """Check for repetition, handling None moves in the stack."""
        # Save a copy of the board before calling super().is_repetition() to prevent stack clearing
        # This is necessary because super().is_repetition() may clear move_stack and _stack
        board_copy = self.copy(stack=True)
        
        # #region agent log
        import json
        import time
        try:
            move_stack_len = len(self.move_stack) if hasattr(self, 'move_stack') else 0
            stack_len = len(self._stack) if hasattr(self, '_stack') else 0
            copy_move_stack_len = len(board_copy.move_stack) if hasattr(board_copy, 'move_stack') else 0
            copy_stack_len = len(board_copy._stack) if hasattr(board_copy, '_stack') else 0
            with open('/Users/minseo/Documents/Github/_star14ms/Chess_AI/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"chess_custom.py:270","message":"is_repetition called","data":{"count":count,"move_stack_len":move_stack_len,"stack_len":stack_len,"copy_move_stack_len":copy_move_stack_len,"copy_stack_len":copy_stack_len},"timestamp":int(time.time()*1000)})+'\n')
        except: pass
        # #endregion
        # Call parent method with error handling for _stack issues
        # Don't modify move_stack here as it can break synchronization with _stack
        try:
            result = super().is_repetition(count)
            # #region agent log
            try:
                with open('/Users/minseo/Documents/Github/_star14ms/Chess_AI/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"chess_custom.py:280","message":"is_repetition result","data":{"count":count,"result":result},"timestamp":int(time.time()*1000)})+'\n')
            except: pass
            # #endregion
            
            # ALWAYS restore stacks after super().is_repetition() if they were present
            # This is critical because super().outcome() may call is_repetition() multiple times,
            # and we need to preserve stacks for subsequent calls
            current_move_stack_len = len(self.move_stack) if hasattr(self, 'move_stack') else 0
            saved_move_stack_len = len(board_copy.move_stack) if hasattr(board_copy, 'move_stack') else 0
            should_restore = saved_move_stack_len > 0
            # #region agent log
            try:
                with open('/Users/minseo/Documents/Github/_star14ms/Chess_AI/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"chess_custom.py:300","message":"checking if restore needed after is_repetition","data":{"current_move_stack_len":current_move_stack_len,"saved_move_stack_len":saved_move_stack_len,"should_restore":should_restore},"timestamp":int(time.time()*1000)})+'\n')
            except: pass
            # #endregion
            # Always restore if stacks were present in the copy (regardless of current state)
            # This ensures stacks are preserved for subsequent calls from super().outcome()
            if should_restore:
                # #region agent log
                try:
                    with open('/Users/minseo/Documents/Github/_star14ms/Chess_AI/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"chess_custom.py:308","message":"restoring stacks after is_repetition","data":{"current_move_stack_len":current_move_stack_len,"saved_move_stack_len":saved_move_stack_len,"reason":"always_restore_when_stacks_present"},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                from MCTS.mcts_algorithm import _restore_chess_stack
                _restore_chess_stack(self, board_copy)
                # #region agent log
                try:
                    after_restore_move_stack_len = len(self.move_stack) if hasattr(self, 'move_stack') else 0
                    after_restore_stack_len = len(self._stack) if hasattr(self, '_stack') else 0
                    with open('/Users/minseo/Documents/Github/_star14ms/Chess_AI/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"chess_custom.py:318","message":"after restore stacks in is_repetition","data":{"after_restore_move_stack_len":after_restore_move_stack_len,"after_restore_stack_len":after_restore_stack_len,"saved_move_stack_len":saved_move_stack_len,"restored_correctly":after_restore_move_stack_len == saved_move_stack_len},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
            
            return result
        except AttributeError as e:
            # If _stack has None values causing AttributeError, return False (no repetition detected)
            if "'NoneType'" in str(e) or "from_square" in str(e):
                # #region agent log
                try:
                    with open('/Users/minseo/Documents/Github/_star14ms/Chess_AI/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"chess_custom.py:298","message":"is_repetition AttributeError","data":{"count":count,"error":str(e)[:100]},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                # Restore stacks even on error
                current_move_stack_len = len(self.move_stack) if hasattr(self, 'move_stack') else 0
                saved_move_stack_len = len(board_copy.move_stack) if hasattr(board_copy, 'move_stack') else 0
                if current_move_stack_len != saved_move_stack_len:
                    from MCTS.mcts_algorithm import _restore_chess_stack
                    _restore_chess_stack(self, board_copy)
                return False
            raise

    def push(self, move: chess.Move | None):
        """Pushes a move to the board, handling illegal moves."""
        if move is None or move == chess.Move.null():
            self.foul = True
            return
        super().push(move)
        
    def pop(self):
        """Pops a move from the board, handling illegal moves."""
        if self.foul:
            # Last push was illegal and was not applied to the move stack.
            # Just clear foul state and do not pop a real move.
            self.foul = False
            return
        super().pop()
        
    def reset(self):
        """Resets the board to the standard starting position and populates the tracker."""
        self.foul = False
        super().reset()

    def copy(self, *, stack: Union[bool, int] = True) -> 'BaseChessBoard':
        """Creates a copy of the board, including the foul state."""
        board_copy = super().copy(stack=stack)
        board_copy.foul = self.foul
        # Filter None moves from move_stack to prevent propagation of corrupted state
        if stack and hasattr(board_copy, 'move_stack') and board_copy.move_stack:
            if any(move is None for move in board_copy.move_stack):
                board_copy.move_stack = [move for move in board_copy.move_stack if move is not None]
                # After filtering move_stack, we should also clear _stack to let python-chess rebuild it
                # This prevents _stack from having None values that don't match the filtered move_stack
                if hasattr(board_copy, '_stack'):
                    board_copy._stack = []
        return board_copy

    def action_id_to_move(self, action_id: int) -> Optional[chess.Move]:
        raise NotImplementedError("Subclasses must implement action_id_to_move")
    
    def move_to_action_id(self, move: chess.Move) -> Optional[int]:
        raise NotImplementedError("Subclasses must implement move_to_action_id")

    @property
    def legal_actions(self) -> List[int]:
        raise NotImplementedError("Subclasses must implement legal_actions")
    
    def get_squares_to_action_ids_map(self) -> Dict[chess.Square, List[int]]:
        raise NotImplementedError("Subclasses must implement get_squares_to_action_ids_map")
    
    def set_fen(self, fen: str, reset_foul: bool = False, *args, **kwargs):
        super().set_fen(fen)
        if reset_foul:
            self.foul = False

class LegacyChessBoard(BaseChessBoard):
    """Chess board class for the 4672 action space without piece tracking."""
    
    def _set_piece_identity_channels(self, board_vector: np.ndarray, sq: chess.Square, rank: int, file: int, piece: chess.Piece):
        """Sets piece identity channels based on current piece type and position."""
        # For legacy board, we use a simpler identity system based on current piece type
        piece_type = piece.piece_type
        color = piece.color
        
        # Map piece type to identity channel (0-15)
        if piece_type == chess.KING:
            id_0_15 = 0
        elif piece_type == chess.QUEEN:
            id_0_15 = 1
        elif piece_type == chess.ROOK:
            id_0_15 = 2 + (1 if chess.square_file(sq) > 3 else 0)  # Kingside/Queenside
        elif piece_type == chess.KNIGHT:
            id_0_15 = 4 + (1 if chess.square_file(sq) > 3 else 0)  # Kingside/Queenside
        elif piece_type == chess.BISHOP:
            id_0_15 = 6 + (1 if chess.square_file(sq) > 3 else 0)  # Kingside/Queenside
        elif piece_type == chess.PAWN:
            id_0_15 = 8 + chess.square_file(sq)  # File-based identity for pawns
        
        # Set the corresponding one-hot channel
        if 0 <= id_0_15 <= 15:
            board_vector[10 + id_0_15, rank, file] = 1

    def action_id_to_move(self, action_id: int) -> Optional[chess.Move]:
        """
        Finds the legal chess.Move for an action ID in the 4672 action space.

        Args:
            action_id: The absolute action ID (1-based, 1-4672).

        Returns:
            The corresponding chess.Move if legal and found, otherwise None.
            If None is returned, the move will be considered illegal and the game will terminate.
        """
        if not (1 <= action_id <= 4672):
            logging.warning(f"Legacy ID to Move: action_id {action_id} is outside expected range (1-4672).")
            return None

        # Get all legal moves
        for move in self.legal_moves:
            # Create piece instance map for current board state
            piece_instance_map = create_piece_instance_map_4672(self)
            
            # Get piece details from the map
            piece_details = piece_instance_map.get(move.from_square)
            if piece_details is None:
                continue
                
            color, piece_type, instance_index = piece_details
            
            # Calculate action ID for this move
            calculated_id = get_action_id_for_piece_abs_4672(
                move.uci(), color, piece_type, instance_index
            )
            
            if calculated_id == action_id:
                return move

        # If we get here, the action_id doesn't correspond to any legal move
        logging.warning(f"Legacy ID to Move: action_id {action_id} does not correspond to any legal move.")
        return None  # This will trigger the foul state in push()

    def move_to_action_id(self, move: chess.Move) -> Optional[int]:
        """
        Calculates the absolute action ID (1-4672) for a move.

        Args:
            move: The chess.Move object.

        Returns:
            The absolute action ID (1-based) or None.
        """
        # Create piece instance map for current board state
        piece_instance_map = create_piece_instance_map_4672(self)
        
        # Get piece details from the map
        piece_details = piece_instance_map.get(move.from_square)
        if piece_details is None:
            logging.warning(f"Move to ID: No piece found at source square {chess.square_name(move.from_square)}")
            return None
            
        color, piece_type, instance_index = piece_details
        
        # Calculate action ID using the 4672-specific function
        action_id = get_action_id_for_piece_abs_4672(
            move.uci(), color, piece_type, instance_index
        )
        
        if action_id is None:
            logging.warning(f"Move to ID: Failed to calculate action ID for move {move.uci()}")
            
        return action_id
    
    def set_fen(self, fen: str, piece_tracker_override: Optional[PieceTracker] = None, *args, **kwargs):
        super().set_fen(fen, *args, **kwargs)

    @property
    def legal_actions(self) -> List[int]:
        """
        Returns a sorted list of all unique legal action IDs for the current board state (4672 action space).
        """
        mapping = self.get_squares_to_action_ids_map()
        all_action_ids = []
        for id_list in mapping.values():
            all_action_ids.extend(id_list)
        return sorted(set(all_action_ids))

    def get_squares_to_action_ids_map(self) -> Dict[chess.Square, List[int]]:
        """
        Returns a dictionary mapping destination squares to sorted lists of action IDs (4672 action space).
        """
        dest_square_to_action_ids: Dict[chess.Square, List[int]] = defaultdict(list)
        for move in self.legal_moves:
            action_id = self.move_to_action_id(move)
            if action_id is not None:
                dest_square_to_action_ids[move.to_square].append(action_id)
        final_map: Dict[chess.Square, List[int]] = {}
        for square, id_list in dest_square_to_action_ids.items():
            final_map[square] = sorted(id_list)
        return final_map

    def get_board_vector(self, history_steps: int = 8) -> np.ndarray:
        """
        Generates a 119x8x8 numpy array following AlphaZero-style input planes.

        Layout: 112 history planes (8 steps × [6 own + 6 opp + 2 repetition])
                + 7 constant meta planes:
                  - current player color (1 for white to move, -1 for black)
                  - total move count normalized (fullmove number scaled to [0,1])
                  - castling rights: WK, WQ, BK, BQ (1 if legal else 0)
                  - halfmove clock normalized by 100 (number of moves without progress)
        """
        # Validate history_steps (minimum 1)
        HISTORY_STEPS = max(1, int(history_steps))
        PLANES_PER_STEP = 14  # 6 own + 6 opp + 2 repetition
        TOTAL_CHANNELS = HISTORY_STEPS * PLANES_PER_STEP + 7  # = 119 (14 * 8 (default history steps) + 7 (meta planes))

        obs = np.zeros((TOTAL_CHANNELS, 8, 8), dtype=np.float32)

        # --- Build history snapshots using a copy we can pop safely ---
        board_copy: chess.Board = self.copy(stack=True)
        history_keys: List[str] = []  # keys in backward order: T, T-1, ...
        snapshots: List[Tuple[bool, List[Tuple[int, int, chess.Piece]]]] = []
        # Each snapshot: (turn_is_white, list of (rank, file, piece))

        steps_collected = 0
        while steps_collected < HISTORY_STEPS:
            # Key for repetition check: first 4 FEN fields
            fen_parts = board_copy.fen().split(' ')
            rep_key = ' '.join(fen_parts[:4])
            history_keys.append(rep_key)

            # Collect piece listing for this snapshot
            pieces_state: List[Tuple[int, int, chess.Piece]] = []
            for sq in chess.SQUARES:
                piece = board_copy.piece_at(sq)
                if piece is None:
                    continue
                rank = chess.square_rank(sq)
                file = chess.square_file(sq)
                pieces_state.append((rank, file, piece))
            snapshots.append((board_copy.turn == chess.WHITE, pieces_state))

            steps_collected += 1
            if steps_collected >= HISTORY_STEPS:
                break
            if len(board_copy.move_stack) == 0:
                break
            board_copy.pop()

        # If fewer than HISTORY_STEPS, repeat the oldest snapshot
        if len(snapshots) < HISTORY_STEPS:
            oldest_turn, oldest_pieces = snapshots[-1]
            oldest_key = history_keys[-1]
            repeat_times = HISTORY_STEPS - len(snapshots)
            snapshots.extend([(oldest_turn, oldest_pieces)] * repeat_times)
            history_keys.extend([oldest_key] * repeat_times)

        # --- Compute repetition counts per step (backward order) ---
        # For step i, count = occurrences of the same rep_key in any earlier position (i+1..end)
        # We cap the count at 2 to form two binary repetition planes (==1) and (>=2)
        repetition_counts = [0] * HISTORY_STEPS
        seen_counts: dict[str, int] = {}
        for i in range(HISTORY_STEPS - 1, -1, -1):
            key = history_keys[i]
            count_in_suffix = seen_counts.get(key, 0)
            repetition_counts[i] = 2 if count_in_suffix >= 2 else count_in_suffix
            seen_counts[key] = count_in_suffix + 1

        # --- Fill history planes ---
        ch = 0
        for step_idx in range(HISTORY_STEPS):
            turn_is_white, pieces_state = snapshots[step_idx]
            own_color = chess.WHITE if turn_is_white else chess.BLACK

            # 6 own planes (P,N,B,R,Q,K) then 6 opp planes
            # Map piece type (1..6) -> index 0..5
            for rank, file, piece in pieces_state:
                pt_idx = piece.piece_type - 1
                if piece.color == own_color:
                    obs[ch + pt_idx, rank, file] = 1.0
                else:
                    obs[ch + 6 + pt_idx, rank, file] = 1.0

            ch += 12

            # Repetition planes: exactly once before (==1) and twice-or-more before (>=2)
            rep_count = repetition_counts[step_idx]
            obs[ch, :, :] = 1.0 if rep_count == 1 else 0.0; ch += 1
            obs[ch, :, :] = 1.0 if rep_count >= 2 else 0.0; ch += 1

        # --- Constant meta planes (for current position self) ---
        current_white_to_move = (self.turn == chess.WHITE)
        obs[ch, :, :] = 1.0 if current_white_to_move else -1.0  # current player color
        ch += 1

        # Total move count (fullmove number) normalized to [0,1]
        # Use a conservative scale to keep values in range during typical games
        fullmove_norm = min(float(self.fullmove_number) / 200.0, 1.0)
        obs[ch, :, :] = fullmove_norm
        ch += 1

        # Castling rights (WK, WQ, BK, BQ)
        obs[ch, :, :] = 1.0 if self.has_kingside_castling_rights(chess.WHITE) else 0.0; ch += 1
        obs[ch, :, :] = 1.0 if self.has_queenside_castling_rights(chess.WHITE) else 0.0; ch += 1
        obs[ch, :, :] = 1.0 if self.has_kingside_castling_rights(chess.BLACK) else 0.0; ch += 1
        obs[ch, :, :] = 1.0 if self.has_queenside_castling_rights(chess.BLACK) else 0.0; ch += 1

        # Halfmove clock normalized by 100 (for 50-move rule)
        obs[ch, :, :] = float(self.halfmove_clock) / 100.0
        ch += 1

        # Sanity check: ch should equal TOTAL_CHANNELS
        # (Avoid raising to keep compatibility; rely on shape)
        return obs

class FullyTrackedBoard(BaseChessBoard):
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
    def push(self, move: chess.Move | None):
        if move is None or move == chess.Move.null():
            self.foul = True
            return

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
        # We need to check the piece type *before* the move was made for castling.
        # The 'moving_piece_id' captured before super().push() contains the original piece type.
        is_king_move = moving_piece_id is not None and moving_piece_id[1] == chess.KING

        if is_king_move and move.uci() in ["e1g1", "e1c1", "e8g8", "e8c8"]:
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
        # If the last action was a foul (no move applied), just clear the flag and return a null move
        if self.foul:
            self.foul = False
            return chess.Move.null()

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
            # logging.warning("Robust ID to Move: Board state theoretically impossible. Using positional mapping, not tracker.")
            # Pass self instead of board to create_piece_instance_map
            positional_instance_map = create_piece_instance_map(self)

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
            instance_map = create_piece_instance_map(self) 
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

    @property
    def legal_actions(self) -> Optional[List[int]]:
        """
        Returns a sorted list of all unique legal action IDs for the current board state.
        Returns None if the board is in an impossible state and mapping fails.
        """
        mapping = self.get_squares_to_action_ids_map()
        if mapping is None:
            return None
        all_action_ids = []
        for id_list in mapping.values():
            all_action_ids.extend(id_list)
        return sorted(set(all_action_ids))

    def get_squares_to_action_ids_map(self) -> Optional[Dict[chess.Square, List[int]]]:
        """
        Returns a dictionary mapping destination squares to sorted lists of action IDs.
        Returns None if the board is in an impossible state and mapping fails.
        """
        dest_square_to_action_ids: Dict[chess.Square, List[int]] = defaultdict(list)
        use_tracker = self.is_theoretically_possible_state
        positional_instance_map = None
        log_warning_issued = False

        if not use_tracker:
            positional_instance_map = create_piece_instance_map(self)
            if not log_warning_issued:
                logging.warning("Board state theoretically impossible. Using positional mapping for all moves.")
                log_warning_issued = True

        for move in self.legal_moves:
            piece_details = None
            if use_tracker:
                piece_id = self.get_piece_instance_id_at(move.from_square)
                if piece_id:
                    piece_details = piece_id
                else:
                    logging.error(f"Tracker failed for move {move.uci()} in a theoretically possible state. Returning None.")
                    return None
            else:
                if positional_instance_map is not None:
                    instance_details_from_map = positional_instance_map.get(move.from_square)
                    if instance_details_from_map:
                        piece_details = instance_details_from_map
                    else:
                        logging.error(f"Positional map lookup failed unexpectedly for move {move.uci()}. Returning None.")
                        return None
                else:
                    logging.error("Positional map logic error. Map not available when expected. Returning None.")
                    return None

            if piece_details:
                color, piece_type_to_use, instance_index_to_use = piece_details
                action_id = get_action_id_for_piece_abs(
                    move.uci(), color, piece_type_to_use, instance_index_to_use
                )
                if action_id is not None:
                    dest_square_to_action_ids[move.to_square].append(action_id)
                # Optionally add error handling for action_id is None

        final_map: Dict[chess.Square, List[int]] = {}
        for square, id_list in dest_square_to_action_ids.items():
            final_map[square] = sorted(id_list)
        return final_map

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

    def _set_piece_identity_channels(self, board_vector: np.ndarray, sq: chess.Square, rank: int, file: int, piece: chess.Piece):
        """Sets piece identity channels based on tracked piece identity."""
        piece_instance_id_tuple = self.get_piece_instance_id_at(sq)
        if piece_instance_id_tuple:
            _color, original_type, original_index = piece_instance_id_tuple
            
            id_0_15 = -1

            if original_type == chess.KING:    # ID 0
                id_0_15 = 0
            elif original_type == chess.QUEEN:   # ID 1
                id_0_15 = 1
            elif original_type == chess.ROOK:    # IDs 2, 3
                id_0_15 = 2 + original_index
            elif original_type == chess.KNIGHT:  # IDs 4, 5
                id_0_15 = 4 + original_index
            elif original_type == chess.BISHOP:  # IDs 6, 7
                id_0_15 = 6 + original_index
            elif original_type == chess.PAWN:    # IDs 8-15
                id_0_15 = 8 + original_index
            
            if 0 <= id_0_15 <= 15:
                board_vector[10 + id_0_15, rank, file] = 1
            else:
                if id_0_15 != -1:
                    print(f"Warning: Calculated out-of-range piece ID {id_0_15} for {piece_instance_id_tuple} at {chess.square_name(sq)}. Piece ID bits set to 0.")
