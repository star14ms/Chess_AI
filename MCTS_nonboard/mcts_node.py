import numpy as np
from typing import Optional, Dict

class MCTSNode:
    """Node in the MCTS tree for non-board environments. Stores raw observation."""
    def __init__(self, observation, parent: Optional['MCTSNode'] = None, prior_p: float = 0.0, action_id_leading_here: Optional[int] = None):
        self.parent = parent
        self.action_id_leading_here = action_id_leading_here
        self.children: Dict[int, MCTSNode] = {}
        self.N = 0  # Visit count
        self.W = 0.0  # Total action value
        self.prior_p = prior_p
        self.observation = np.copy(observation) if isinstance(observation, np.ndarray) else observation
        self.is_expanded = False
        self._is_terminal = None

    def is_terminal(self) -> bool:
        return self._is_terminal if self._is_terminal is not None else False

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, C_puct: float) -> float:
        parent_N = self.parent.N if self.parent else 1
        return C_puct * self.prior_p * (np.sqrt(parent_N) / (1 + self.N))

    def __repr__(self):
        return f"MCTSNode(N={self.N}, W={self.W:.2f}, Q={self.Q():.2f}, P={self.prior_p:.3f}, action_id={self.action_id_leading_here})" 