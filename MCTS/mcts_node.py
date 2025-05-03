import math
import chess
from typing import Optional, Dict, Type
import copy # Import copy for deepcopy
import numpy as np
import random

# Assuming the environment class is available
# Replace this with the actual import path if different
import sys
import os
sys.path.append('.')
from chess_gym.chess_custom import FullyTrackedBoard

# --- MCTS Node ---
class MCTSNode:
    """Node in the MCTS tree. Stores state via FEN string."""
    def __init__(self, fen: str, 
                 parent: Optional['MCTSNode'] = None, 
                 prior_p: float = 0.0, 
                 move_leading_here: Optional[chess.Move] = None
                ): # Default to standard board
        self.fen = fen
        self.parent = parent
        self.move_leading_here = move_leading_here # Move that led to this node from parent
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.N = 0  # Visit count
        self.W = 0.0  # Total action value
        self.prior_p = prior_p
        self.board = FullyTrackedBoard(fen=fen)

        # Lazy evaluation for terminal state if needed (calculated on demand)
        self._is_terminal = None 
        self.is_expanded = False # Tracks if children have been added

    def get_board(self) -> chess.Board:
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
        """Selects the child with the highest UCT score."""
        best_score = -float('inf')
        best_child = None
        best_move = None

        # Need a temporary board to get legal moves for selection
        current_board = self.get_board()
        
        for move in current_board.legal_moves:
            child = self.children.get(move)
            if child: # Only consider expanded children
                score = child.Q() + child.U(C_puct)
                if score > best_score:
                    best_score = score
                    best_child = child
                    best_move = move # Keep track of best move too if needed
            # else: If move not in children, it hasn't been expanded/created yet

        if best_child is None:
             # This can happen if no children are expanded or if all legal moves were pruned
             # Fallback: Maybe return self? Or raise error? 
             # Let's handle potential lack of children in the calling MCTS search logic.
             # For now, return self indicates selection cannot proceed down.
             # Or, if we *know* children exist but selection failed, randomly choose? Requires care.
             print(f"Warning: No best child found via UCT for node {self.fen}. Moves considered: {list(current_board.legal_moves)}. Children keys: {list(self.children.keys())}")
             # Fallback: choose random legal move's child if available? Risky if not all children created.
             legal_children = [self.children.get(m) for m in current_board.legal_moves if m in self.children]
             if legal_children: return random.choice(legal_children) # Random choice among existing children
             return self # Cannot select further
             
        return best_child

    def get_observation(self, env_cls: Type, observation_mode: str) -> np.ndarray:
        """Generates the network observation for this node's state."""
        # Instantiate a temporary environment to get the observation
        # Ensure the env doesn't require complex state beyond the board FEN
        temp_env = env_cls(observation_mode=observation_mode, render_mode=None)
        temp_env.board = self.get_board() # Set the board state
        obs, _ = temp_env.reset(fen=self.fen) # Use reset with FEN if available, otherwise relies on board set
        temp_env.close() # Optional, depending on env implementation
        return obs

    # Maybe add __repr__ for debugging
    def __repr__(self):
        return f"MCTSNode(fen='{self.fen}', N={self.N}, W={self.W:.2f}, Q={self.Q():.2f}, P={self.prior_p:.3f})" 