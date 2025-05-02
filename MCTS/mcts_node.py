import math
import chess
from typing import Optional, Dict
import copy # Import copy for deepcopy

# Assuming the environment class is available
# Replace this with the actual import path if different
import sys
import os
sys.path.append('.')
from chess_gym.envs import ChessEnv

# --- MCTS Node ---
class MCTSNode:
    def __init__(self, env: ChessEnv, parent=None, prior_p=0.0, move_leading_here=None):
        self.env = env # Store the environment instance
        self.parent = parent
        self.move_leading_here = move_leading_here # Move taken by parent to reach this node
        self.children: Dict[chess.Move, 'MCTSNode'] = {}  # map from move to MCTSNode
        # self.untried_actions = list(state.legal_moves) # No longer needed with policy sampling
        self.N = 0  # visit count
        self.W = 0  # total value of this node (sum of simulation results)
        self.P = prior_p # Prior probability of selecting the move leading to this node

    @property
    def board(self) -> chess.Board:
        """Convenience property to access the board within the environment."""
        return self.env.board # Assuming env has a .board attribute

    @property
    def observation(self):
        """Convenience property to get the current observation from the environment."""
        # Assumes env has a way to get the current observation (e.g., internal state or method)
        # If env follows gymnasium standard, observation is usually returned by step/reset
        # We might need to store the last observation if env doesn't provide it on demand.
        # Let's assume env._observe() exists based on chess_env.py provided earlier
        return self.env._observe()

    def is_fully_expanded(self) -> bool:
        # With policy sampling, a node is considered fully expanded if all legal moves have children
        # For simplicity here, especially with opponent heuristic, we check if children dict covers legal moves
        # This might not be perfectly accurate if opponent heuristic doesn't explore all moves
        # A better check might be if N > 0, implying it has been visited and potentially expanded
        # return len(self.children) >= len(list(self.state.legal_moves))
        return self.N > 0 # Consider expanded if visited (as expansion happens on first visit of player node)

    def is_terminal(self) -> bool:
        # Use the board within the environment
        return self.board.is_game_over()

    def get_value(self) -> float:
        """Average value W/N."""
        if self.N == 0:
            return 0
        # The value W is stored from the perspective of the player *who arrived* at this node.
        return self.W / self.N

    def select_child_uct(self, C_puct: float) -> 'MCTSNode':
        """Selects child node using the UCT formula."""
        best_score = -float('inf')
        best_child = None

        for move, child in self.children.items():
            # UCT = Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            # Q(s,a) is the value from the child's perspective (-W/N of child, as W is stored for parent's arrival)
            q_value = -child.get_value() # Negate because child.W is relative to the player *at* the child node
            uct_score = q_value + C_puct * child.P * (math.sqrt(self.N) / (1 + child.N))

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        if best_child is None:
            if self.children:
                return list(self.children.values())[0]
            else:
                # Avoid raising error if called on a leaf during selection
                # Return self or handle upstream? MCTS search loop should stop before this.
                # Let's keep the error for now, it indicates a logic flaw in search if it hits this.
                raise RuntimeError("Cannot select child from a node with no children.")

        return best_child 