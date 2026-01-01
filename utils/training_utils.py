"""Utility functions for training that are not directly part of the training loop."""
import torch
import torch.nn as nn
import random
from omegaconf import OmegaConf


def select_fen_from_dict(fen_dict):
    """Select a FEN string from a dictionary based on weights.
    
    Args:
        fen_dict: Dictionary mapping FEN strings to weights (floats)
    
    Returns:
        Selected FEN string, or None if dictionary is empty/invalid
    """
    if not fen_dict or not isinstance(fen_dict, dict):
        return None
    
    fens = list(fen_dict.keys())
    weights = list(fen_dict.values())
    
    # Normalize weights to ensure they sum to 1
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    
    normalized_weights = [w / total_weight for w in weights]
    
    # Use random.choices to select based on weights
    selected_fen = random.choices(fens, weights=normalized_weights, k=1)[0]
    return selected_fen


def compute_position_aware_draw_reward(network: nn.Module, board_state, state_obs, 
                                       is_first_player: bool, cfg: OmegaConf, device: torch.device,
                                       precomputed_value: float | None = None) -> float:
    """Compute draw reward based on position quality using network's value prediction.
    
    Args:
        network: The neural network to evaluate position
        board_state: The board state at this position
        state_obs: The observation vector for this position
        is_first_player: Whether this position is from first player's perspective
        cfg: Configuration object with position_aware_draw_reward settings
        device: Device to run network inference on
        precomputed_value: Optional pre-computed value from MCTS (avoids re-running network)
    
    Returns:
        float: Position-aware draw reward
    """
    # Use precomputed value if available (from MCTS root node)
    if precomputed_value is not None:
        predicted_value = precomputed_value
    elif network is not None:
        # Get network's value prediction for this position
        # state_obs should already be in the correct format
        obs_tensor = torch.tensor(state_obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            network.eval()
            _, value_pred = network(obs_tensor)
            predicted_value = value_pred.item()  # Range: -1 to +1
    else:
        # Fallback to base draw reward if network not available
        draw_reward = cfg.training.get('draw_reward', -0.1)
        return draw_reward if draw_reward is not None else -0.1
    
    # Adjust predicted value to current player's perspective
    if not is_first_player:
        predicted_value = -predicted_value
    
    # Get reward dictionary from config with defaults
    reward_dict = cfg.training.get('position_aware_draw_rewards', None)
    if reward_dict is None:
        # Default rewards if not specified
        reward_dict = {
            'winning': -0.8,
            'equal': -0.1,
            'losing': 0.2
        }
    
    # Determine reward based on predicted position quality
    if predicted_value > 0.1:
        # Was winning - strong penalty for throwing away advantage
        return reward_dict.get('winning', -0.8)
    elif predicted_value > -0.1:
        # Equal position - base draw penalty
        return reward_dict.get('equal', -0.1)
    else:
        # Was losing - reward for saving the game
        return reward_dict.get('losing', 0.2)

