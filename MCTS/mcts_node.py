import math
import chess
from typing import Optional, Dict, Type
import numpy as np
import random

# Assuming the environment class is available
# Replace this with the actual import path if different
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from chess_gym.chess_custom import FullyTrackedBoard, LegacyChessBoard

# --- MCTS Node ---
class MCTSNode:
    """Node in the MCTS tree. Stores state via FEN string."""
    def __init__(self, board: FullyTrackedBoard | LegacyChessBoard,
                 parent: Optional['MCTSNode'] = None, 
                 prior_p: float = 0.0, 
                 action_id_leading_here: Optional[int] = None
                ): # Default to standard board
        self.parent = parent
        self.action_id_leading_here = action_id_leading_here # Action ID that led to this node from parent
        self.children: Dict[int, MCTSNode] = {}  # Keyed by action_id
        self.N = 0  # Visit count
        self.W = 0.0  # Total action value
        self.prior_p = prior_p
        self.board = board.copy()

        # Lazy evaluation for terminal state if needed (calculated on demand)
        self._is_terminal = None 
        self.is_expanded = False # Tracks if children have been added

    def get_board(self) -> FullyTrackedBoard | LegacyChessBoard:
        """Creates and returns a board object from the stored FEN string."""
        return self.board

    def is_terminal(self) -> bool:
        """Checks if the node represents a terminal state."""
        if self._is_terminal is None:
            # Calculate on demand and cache
            board = self.get_board()
            self._is_terminal = board.is_game_over(claim_draw=True)
        return self._is_terminal

    def Q(self) -> float:
        """Calculates the mean action value (Q) for this node."""
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, C_puct: float) -> float:
        """Calculates the Upper Confidence Bound (U) term."""
        # Ensure parent visit count is at least 1 for stability in log
        parent_N = self.parent.N if self.parent else 1
        return C_puct * self.prior_p * (math.sqrt(parent_N) / (1 + self.N))

    def select_child_uct(self, C_puct: float) -> 'MCTSNode':
        """Selects the child with the highest UCT score (by action_id)."""
        best_score = -float('inf')
        best_child = None
        best_action_id = None

        current_board = self.get_board()
        for action_id in current_board.legal_actions:
            child = self.children.get(action_id)
            if child: # Only consider expanded children
                score = child.Q() + child.U(C_puct)
                if score > best_score:
                    best_score = score
                    best_child = child
                    best_action_id = action_id
        if best_child is None:
            print(f"Warning: No best child found via UCT for node {self.board.fen()}. Actions considered: {list(current_board.legal_actions)}. Children keys: {list(self.children.keys())}")
            legal_children = [self.children.get(aid) for aid in current_board.legal_actions if aid in self.children]
            if legal_children: return random.choice(legal_children)
            return self
        return best_child

    # Maybe add __repr__ for debugging
    def __repr__(self):
        return f"MCTSNode(fen='{self.board.fen()}', N={self.N}, W={self.W:.2f}, Q={self.Q():.2f}, P={self.prior_p:.3f})" 