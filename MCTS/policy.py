"""
MCTS-based execute_policy function for test.ipynb

This version uses MCTS (like during training) instead of raw policy network.
This solves the distribution shift problem where raw policy degrades quickly.

USAGE:
    Copy the execute_policy_mcts function below and replace execute_policy in test.ipynb
    OR run both functions side-by-side to compare performance
"""

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
from chess_gym.chess_custom import LegacyChessBoard

# Import MCTS components
import sys
import os
sys.path.append(os.path.abspath('..'))
from MCTS.mcts_algorithm import MCTS
from MCTS.mcts_node import MCTSNode


def execute_policy_mcts(
    cfg: DictConfig, 
    model: nn.Module, 
    board: LegacyChessBoard, 
    device: torch.device, 
    mcts_iterations: int = 800,
    temperature: float = 0.1,
    deterministic: bool = True
):
    """
    Execute policy using MCTS search (like during training).
    
    This version uses Monte Carlo Tree Search to select moves, which:
    - Only considers legal moves (inherently filtered)
    - Uses the policy network as guidance for search
    - Performs lookahead to find better moves
    - Matches the training distribution (no distribution shift)
    
    Args:
        cfg: Configuration with env.history_steps
        model: ChessNetwork model instance
        board: Chess board instance
        device: Device to run model on ('cpu' or 'cuda')
        mcts_iterations: Number of MCTS simulations per move (default: 800, same as training)
        temperature: Temperature for move selection (default: 0.1 for near-deterministic)
        deterministic: If True, select best move. If False, sample from MCTS policy.
        
    Returns:
        action_id: Selected chess move (guaranteed to be legal)
        mcts_policy: MCTS-refined policy distribution (numpy array)
    """
    model.eval()  # Set model to evaluation mode
    
    # Create MCTS root node
    root_node = MCTSNode(board.copy())
    
    # Initialize MCTS
    mcts_player = MCTS(
        network=model,
        device=device,
        env=None,  # No rendering during MCTS
        C_puct=cfg.mcts.c_puct,  # Exploration constant from config
        dirichlet_alpha=cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon=cfg.mcts.dirichlet_epsilon,
        action_space_size=cfg.network.action_space_size,
        history_steps=cfg.env.history_steps
    )
    
    # Run MCTS search
    mcts_player.search(
        root_node, 
        iterations=mcts_iterations,
        batch_size=cfg.mcts.batch_size,
        progress=None  # No progress bar for single moves
    )
    
    # Get policy distribution from MCTS
    mcts_policy = mcts_player.get_policy_distribution(root_node, temperature=temperature)
    
    # Select action
    if deterministic or temperature < 0.01:
        # Select highest probability move
        action_id = np.argmax(mcts_policy) + 1  # +1 for 1-indexed
    else:
        # Sample from MCTS policy distribution
        action_id = np.random.choice(len(mcts_policy), p=mcts_policy) + 1
    
    return action_id, mcts_policy


def execute_policy_raw(cfg: DictConfig, model: nn.Module, board: LegacyChessBoard, device: torch.device, deterministic=True):
    """
    Execute policy using raw network output (original version).
    
    WARNING: This suffers from distribution shift and will degrade quickly!
    Use this only for comparison or if you need speed over accuracy.
    
    Args:
        cfg: Configuration with env.history_steps
        model: ChessNetwork model instance
        board: Chess board instance
        device: Device to run model on ('cpu' or 'cuda')
        deterministic: If True, select highest probability move. If False, sample.
        
    Returns:
        action_id: Selected chess move (may be ILLEGAL!)
        policy_probs: Raw policy probability distribution
    """
    model.eval()
    observation = torch.from_numpy(board.get_board_vector(history_steps=cfg.env.history_steps)).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        policy, _ = model(observation)
        policy = policy.squeeze()
        
    policy_probs = torch.softmax(policy, dim=0)

    if deterministic:
        action_id = policy.argmax().item() + 1
    else:
        action_id = torch.multinomial(policy_probs, num_samples=1).item() + 1

    return action_id, policy_probs.cpu().numpy()


def execute_policy_raw_legal_filtered(cfg: DictConfig, model: nn.Module, board: LegacyChessBoard, device: torch.device, deterministic=True):
    """
    Execute policy using raw network output WITH legal move filtering.
    
    This prevents fouls but doesn't solve the underlying distribution shift problem.
    The model still makes poor predictions (high illegal probability), we just mask them.
    
    Args:
        cfg: Configuration with env.history_steps
        model: ChessNetwork model instance
        board: Chess board instance
        device: Device to run model on ('cpu' or 'cuda')
        deterministic: If True, select highest probability legal move. If False, sample from legal moves.
        
    Returns:
        action_id: Selected chess move (guaranteed legal)
        policy_probs: Raw policy probability distribution (before filtering)
    """
    model.eval()
    observation = torch.from_numpy(board.get_board_vector(history_steps=cfg.env.history_steps)).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        policy, _ = model(observation)
        policy = policy.squeeze()
        
    policy_probs = torch.softmax(policy, dim=0)
    
    # Get legal actions (1-indexed)
    legal_actions = list(board.legal_actions)
    
    # Convert to 0-indexed for policy array
    legal_indices = [action_id - 1 for action_id in legal_actions]
    
    if deterministic:
        # Select highest probability LEGAL action
        legal_policy_logits = policy[legal_indices]
        best_legal_idx = legal_policy_logits.argmax().item()
        action_id = legal_actions[best_legal_idx]
    else:
        # Sample from legal moves only, weighted by their probabilities
        legal_probs = policy_probs[legal_indices]
        legal_probs = legal_probs / legal_probs.sum()  # Renormalize
        sampled_idx = torch.multinomial(legal_probs, num_samples=1).item()
        action_id = legal_actions[sampled_idx]

    return action_id, policy_probs.cpu().numpy()


# ==============================================================================
# COMPARISON FUNCTION - Test all three approaches
# ==============================================================================

def compare_execution_methods(cfg, model, board, device):
    """
    Compare all three execution methods on the same position.
    
    This is useful for understanding the difference between:
    1. MCTS-based execution (strong, matches training)
    2. Raw policy (weak, distribution shift)
    3. Raw policy with legal filtering (prevents fouls but still weak predictions)
    """
    print("\n" + "="*80)
    print(f"COMPARISON: Position Analysis")
    print("="*80)
    print(f"Turn: {'White' if board.turn else 'Black'}")
    print(f"Legal moves: {len(list(board.legal_actions))}")
    print(f"FEN: {board.fen()}")
    
    # 1. MCTS
    print("\n--- Method 1: MCTS (Recommended) ---")
    action_mcts, policy_mcts = execute_policy_mcts(cfg, model, board, device, mcts_iterations=400)
    legal_actions = list(board.legal_actions)
    is_legal_mcts = action_mcts in legal_actions
    print(f"Selected action: {action_mcts} ({'LEGAL' if is_legal_mcts else 'ILLEGAL'})")
    print(f"MCTS policy entropy: {-np.sum(policy_mcts * np.log(policy_mcts + 1e-10)):.4f}")
    
    # 2. Raw policy
    print("\n--- Method 2: Raw Policy (Weak, may foul) ---")
    action_raw, policy_raw = execute_policy_raw(cfg, model, board, device)
    is_legal_raw = action_raw in legal_actions
    # Calculate illegal probability mass
    legal_indices = [a-1 for a in legal_actions]
    illegal_prob = 1.0 - policy_raw[legal_indices].sum()
    print(f"Selected action: {action_raw} ({'LEGAL' if is_legal_raw else 'ILLEGAL ⚠️'})")
    print(f"Illegal probability mass: {illegal_prob*100:.2f}%")
    
    # 3. Raw policy with legal filtering
    print("\n--- Method 3: Raw Policy + Legal Filter (Band-aid) ---")
    action_filtered, policy_filtered = execute_policy_raw_legal_filtered(cfg, model, board, device)
    is_legal_filtered = action_filtered in legal_actions
    illegal_prob_filtered = 1.0 - policy_filtered[legal_indices].sum()
    print(f"Selected action: {action_filtered} ({'LEGAL' if is_legal_filtered else 'ILLEGAL'})")
    print(f"Illegal probability mass: {illegal_prob_filtered*100:.2f}% (still high, just masked)")
    
    print("\n" + "="*80)
    print("RECOMMENDATION: Use Method 1 (MCTS) for best results")
    print("="*80)


# ==============================================================================
# USAGE INSTRUCTIONS
# ==============================================================================

"""
COPY THIS INTO YOUR test.ipynb:

# Replace the existing execute_policy function with this:

def execute_policy(cfg: DictConfig, model: nn.Module, board: LegacyChessBoard, device: torch.device, deterministic=True):
    '''
    Execute policy using MCTS search (like during training).
    This solves the distribution shift problem.
    '''
    from MCTS.mcts_algorithm import MCTS
    from MCTS.mcts_node import MCTSNode
    import numpy as np
    
    model.eval()
    
    # Create MCTS root node
    root_node = MCTSNode(board.copy())
    
    # Initialize MCTS
    mcts_player = MCTS(
        network=model,
        device=device,
        env=None,
        C_puct=cfg.mcts.c_puct,
        dirichlet_alpha=cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon=cfg.mcts.dirichlet_epsilon,
        action_space_size=cfg.network.action_space_size,
        history_steps=cfg.env.history_steps
    )
    
    # Run MCTS search (800 iterations like training, or reduce for speed)
    mcts_player.search(root_node, num_simulations=800, batch_size=cfg.mcts.batch_size, progress=None)
    
    # Get policy distribution
    temperature = 0.1  # Low temperature for near-deterministic play
    mcts_policy = mcts_player.get_policy_distribution(root_node, temperature=temperature)
    
    # Select action
    if deterministic or temperature < 0.01:
        action_id = np.argmax(mcts_policy) + 1
    else:
        action_id = np.random.choice(len(mcts_policy), p=mcts_policy) + 1
    
    return action_id, mcts_policy

# That's it! Now your test.ipynb will use MCTS and won't have distribution shift problems.
# The illegal probability should stay around 1% throughout the game instead of rising to 87%.
"""

