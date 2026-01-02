"""Utility functions for training that are not directly part of the training loop."""
import torch
import torch.nn as nn
import random
from omegaconf import OmegaConf
from typing import Optional


def select_fen_from_dict(fen_dict):
    """Select a FEN string from a dictionary based on weights.
    
    Supports two formats:
    1. Legacy: {"fen": weight} - returns (fen, None)
    2. New: {"fen": {"weight": w, "quality": q}} - returns (fen, quality)
    
    Args:
        fen_dict: Dictionary mapping FEN strings to weights (floats) or dicts with weight/quality
    
    Returns:
        Tuple of (selected FEN string, position quality) or (None, None) if dictionary is empty/invalid
    """
    if not fen_dict or not isinstance(fen_dict, dict):
        return None, None
    
    fens = []
    weights = []
    qualities = []
    
    for fen, value in fen_dict.items():
        fens.append(fen)
        # Handle both legacy format (float) and new format (dict with weight/quality)
        if isinstance(value, dict):
            weights.append(value.get('weight', 0.0))
            qualities.append(value.get('quality', None))
        else:
            # Legacy format: just a float weight
            weights.append(float(value))
            qualities.append(None)
    
    # Normalize weights to ensure they sum to 1
    total_weight = sum(weights)
    if total_weight == 0:
        return None, None
    
    normalized_weights = [w / total_weight for w in weights]
    
    # Use random.choices to select based on weights
    selected_idx = random.choices(range(len(fens)), weights=normalized_weights, k=1)[0]
    selected_fen = fens[selected_idx]
    selected_quality = qualities[selected_idx]
    
    return selected_fen, selected_quality


class RewardComputer:
    """Class to compute position-aware draw rewards based on position quality and termination type."""
    
    def __init__(self, cfg: OmegaConf, network: Optional[nn.Module] = None, device: Optional[torch.device] = None):
        """Initialize the reward computer.
        
        Args:
            cfg: Configuration object with reward settings
            network: Optional neural network for position evaluation
            device: Optional device for network inference
        """
        self.cfg = cfg
        self.network = network
        self.device = device
        self.draw_reward_table = cfg.training.get('draw_reward_table', None)
        self.default_draw_reward = cfg.training.get('draw_reward', -0.1)
        
        # Default rewards if table not available
        self.default_rewards = {
            'winning': -0.8,
            'equal': -0.1,
            'losing': 0.2
        }
    
    @staticmethod
    def is_endgame_position(fen: Optional[str]) -> bool:
        """Detect if a FEN position represents an endgame.
        
        Endgame is typically defined as having fewer pieces on the board.
        We count pieces (excluding kings) - if total < 10, it's likely an endgame.
        
        Args:
            fen: FEN string to analyze, or None
        
        Returns:
            bool: True if position appears to be an endgame
        """
        if not fen:
            return False
        
        # Standard starting position - not an endgame
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        if fen.startswith(starting_fen):
            return False
        
        # Count pieces in the board part of FEN (first part before space)
        board_part = fen.split()[0] if ' ' in fen else fen
        
        # Count non-empty squares (pieces)
        # FEN uses: / for rank separator, numbers for empty squares, letters for pieces
        piece_count = 0
        for char in board_part:
            if char.isalpha():  # Piece (K, Q, R, B, N, P or lowercase)
                piece_count += 1
        
        # Endgame typically has fewer pieces
        # Excluding the 2 kings, if we have < 10 other pieces, it's likely an endgame
        # This threshold can be adjusted, but 10 pieces (excluding kings) is a reasonable cutoff
        return piece_count < 12  # 2 kings + 10 other pieces = 12 total
    
    def evaluate_position_quality(self, state_obs, is_first_player: bool, 
                                  precomputed_value: Optional[float] = None) -> str:
        """Evaluate position quality using network or precomputed value.
        
        Args:
            state_obs: Observation vector for the position
            is_first_player: Whether position is from first player's perspective
            precomputed_value: Optional pre-computed value from MCTS
        
        Returns:
            str: Position quality ('winning', 'equal', or 'losing')
        """
        # Use precomputed value if available
        if precomputed_value is not None:
            predicted_value = precomputed_value
        elif self.network is not None:
            # Get network's value prediction
            if self.device is None:
                try:
                    device = next(self.network.parameters()).device
                except:
                    device = torch.device('cpu')
            else:
                device = self.device
            
            obs_tensor = torch.tensor(state_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                self.network.eval()
                _, value_pred = self.network(obs_tensor)
                predicted_value = value_pred.item()  # Range: -1 to +1
        else:
            # No network available - return 'equal' as default
            return 'equal'
        
        # Adjust predicted value to current player's perspective
        if not is_first_player:
            predicted_value = -predicted_value
        
        # Determine position quality category
        if predicted_value > 0.1:
            return 'winning'
        elif predicted_value > -0.1:
            return 'equal'
        else:
            return 'losing'
    
    def compute_draw_reward(self, state_obs, is_first_player: bool,
                           termination_type: Optional[str] = None,
                           precomputed_value: Optional[float] = None,
                           initial_position_quality: Optional[str] = None,
                           is_endgame: bool = False) -> float:
        """Compute draw reward based on position quality and termination type.
        
        Args:
            state_obs: Observation vector for the position
            is_first_player: Whether position is from first player's perspective
            termination_type: Optional termination type (e.g., "THREEFOLD_REPETITION", "STALEMATE")
            precomputed_value: Optional pre-computed value from MCTS
            initial_position_quality: Optional initial position quality (from white's perspective) - used for endgames
            is_endgame: Whether this is an endgame position
        
        Returns:
            float: Position-aware draw reward
        """
        # Use initial position quality if provided (for both endgames and full games with known quality)
        # This ensures consistent rewards based on the starting position quality from config
        if initial_position_quality is not None:
            # initial_position_quality is from white's perspective, so flip if current player is black
            if is_first_player:
                position_quality = initial_position_quality
            else:
                # Flip the quality: winning <-> losing, equal stays equal
                if initial_position_quality == 'winning':
                    position_quality = 'losing'
                elif initial_position_quality == 'losing':
                    position_quality = 'winning'
                else:  # equal
                    position_quality = 'equal'
        else:
            # No initial quality provided, evaluate current position
            # This should only happen if initial_position_quality was not set from config
            position_quality = self.evaluate_position_quality(state_obs, is_first_player, precomputed_value)
        
        # Get reward from table if available
        if self.draw_reward_table and termination_type:
            termination_rewards = self.draw_reward_table.get(termination_type, None)
            if termination_rewards:
                reward = termination_rewards.get(position_quality, None)
                if reward is not None:
                    return reward
        
        # Fallback: use default rewards
        return self.default_rewards.get(position_quality, self.default_draw_reward if self.default_draw_reward is not None else -0.1)
    
    def evaluate_initial_position(self, initial_obs, is_first_player: bool) -> Optional[str]:
        """Evaluate initial board position and return quality.
        
        Args:
            initial_obs: Initial observation vector
            is_first_player: Whether the first player is to move
        
        Returns:
            Optional[str]: Position quality ('winning', 'equal', 'losing') or None if network unavailable
        """
        if self.network is None:
            return None
        
        # Determine device
        device = self.device
        if device is None:
            try:
                device = next(self.network.parameters()).device
            except:
                device = torch.device('cpu')
        
        # Evaluate initial position
        obs_tensor = torch.tensor(initial_obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            self.network.eval()
            _, initial_value_pred = self.network(obs_tensor)
            initial_value = initial_value_pred.item()  # Range: -1 to +1
        
        # Determine initial position quality from first player's perspective
        if not is_first_player:
            initial_value = -initial_value
        
        # Categorize initial position quality
        if initial_value > 0.1:
            return 'winning'
        elif initial_value > -0.1:
            return 'equal'
        else:
            return 'losing'


# Backward compatibility: keep old function names
def is_endgame_position(fen: Optional[str]) -> bool:
    """Detect if a FEN position represents an endgame."""
    return RewardComputer.is_endgame_position(fen)


def compute_position_aware_draw_reward(network: nn.Module, board_state, state_obs, 
                                       is_first_player: bool, cfg: OmegaConf, device: torch.device,
                                       precomputed_value: float | None = None,
                                       termination_type: str | None = None,
                                       initial_position_quality: str | None = None,
                                       is_endgame: bool = False) -> float:
    """Compute draw reward (backward compatibility wrapper)."""
    reward_computer = RewardComputer(cfg, network, device)
    return reward_computer.compute_draw_reward(
        state_obs, is_first_player, termination_type,
        precomputed_value, initial_position_quality, is_endgame
    )

