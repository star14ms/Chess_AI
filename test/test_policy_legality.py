"""
Test Model Policy Legality from Replay Buffer.

This script loads a trained model and tests the legality of its policy predictions
for all states in the replay buffer. It calculates statistics separately
for white and black positions.

Metrics calculated (matching train.py):
1. Illegal probability mass: Sum of probability assigned to all illegal moves
2. Illegal argmax ratio: Fraction of positions where the highest probability move is illegal
3. Median/std illegal ratio: Per-position statistics of illegal probability mass

Note: board.legal_actions() returns 1-indexed action IDs, but policy arrays are 0-indexed.
We convert to 0-indexed before comparing with policy predictions.
"""

import torch
import torch.nn.functional as F
import gzip
import pickle
import numpy as np
import os
import sys
import chess
from typing import Tuple, List, Dict
from collections import defaultdict
import hydra
from omegaconf import DictConfig

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess_gym.chess_custom import LegacyChessBoard
from MCTS.training_modules.chess import create_chess_network


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint from file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint


def decompress_replay_buffer(replay_buffer_state: dict) -> List[Tuple[np.ndarray, np.ndarray, str, float]]:
    """Decompress and extract replay buffer data."""
    if replay_buffer_state is None:
        raise ValueError("No replay buffer state found in checkpoint")
    
    # Check format
    if not replay_buffer_state.get('compressed', False):
        print("Warning: Replay buffer is in legacy uncompressed format")
        buffer = replay_buffer_state.get('buffer', [])
        # Convert to compact format (state, policy, board_str, value)
        compact_buffer = []
        for exp in buffer:
            state, policy, board, value = exp
            board_str = board.fen() if hasattr(board, 'fen') else str(board)
            compact_buffer.append((state, policy, board_str, value))
        return compact_buffer
    
    print("Decompressing replay buffer...")
    compressed_bytes = replay_buffer_state['buffer_compressed']
    buffer_bytes = gzip.decompress(compressed_bytes)
    compact_buffer = pickle.loads(buffer_bytes)
    
    print(f"Replay buffer size: {len(compact_buffer)} experiences")
    print(f"Environment type: {replay_buffer_state.get('env_type', 'unknown')}")
    
    return compact_buffer


def test_policy_legality(network: torch.nn.Module, training_data: List[Tuple[np.ndarray, np.ndarray, str, float]], 
                         device: torch.device, batch_size: int = 128) -> Dict:
    """
    Test policy legality for model predictions on all states in training data.
    
    Args:
        network: The neural network model
        training_data: List of (observation, policy_target, board_fen, value_target)
        device: Device to run inference on
        batch_size: Batch size for inference
    
    Returns statistics for white and black positions separately.
    """
    print("\n" + "="*60)
    print("TESTING MODEL POLICY LEGALITY")
    print("="*60)
    
    if not training_data:
        print("No training data to analyze")
        return {}
    
    network.eval()
    
    # Statistics containers
    white_stats = {
        'total_positions': 0,
        'total_policy_mass': 0.0,
        'legal_policy_mass': 0.0,
        'illegal_policy_mass': 0.0,
        'num_legal_moves': [],
        'num_policy_moves': [],
        'illegal_move_ratios': [],  # Track per-position illegal ratio (probability mass)
        'illegal_argmax_count': 0,  # Count positions where argmax is illegal
    }
    
    black_stats = {
        'total_positions': 0,
        'total_policy_mass': 0.0,
        'legal_policy_mass': 0.0,
        'illegal_policy_mass': 0.0,
        'num_legal_moves': [],
        'num_policy_moves': [],
        'illegal_move_ratios': [],
        'illegal_argmax_count': 0,
    }
    
    errors = 0
    total_positions = len(training_data)
    
    print(f"\nAnalyzing {total_positions} positions with model predictions...")
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    
    # Process in batches
    with torch.no_grad():
        for batch_start in range(0, total_positions, batch_size):
            batch_end = min(batch_start + batch_size, total_positions)
            batch = training_data[batch_start:batch_end]
            
            print(f"  Processed {batch_start}/{total_positions} positions...")
            
            # Prepare batch observations
            obs_batch = []
            boards = []
            is_white_list = []
            
            for obs, _, board_str, _ in batch:
                obs_batch.append(obs)
                try:
                    board = LegacyChessBoard(board_str)
                    boards.append(board)
                    
                    # Determine if white or black to move
                    if obs.shape[0] > 112:
                        player_color = obs[112, 0, 0]
                        is_white = (player_color == 1.0)
                    else:
                        is_white = (board.turn == chess.WHITE)
                    is_white_list.append(is_white)
                except Exception as e:
                    errors += 1
                    boards.append(None)
                    is_white_list.append(None)
            
            # Stack observations and run through model
            obs_tensor = torch.from_numpy(np.stack([o.astype(np.float32) for o in obs_batch])).to(device)
            
            try:
                policy_logits, _ = network(obs_tensor)
                policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()
            except Exception as e:
                print(f"  Error running model on batch: {e}")
                errors += len(batch)
                continue
            
            # Analyze each position in batch
            for i, (board, is_white, policy_pred) in enumerate(zip(boards, is_white_list, policy_probs)):
                if board is None or is_white is None:
                    continue
                
                try:
                    # Select stats container
                    stats = white_stats if is_white else black_stats
                    stats['total_positions'] += 1
                    
                    # Get legal action IDs from board (1-indexed)
                    legal_actions_1indexed = set(board.legal_actions)
                    # Convert to 0-indexed for policy array indexing
                    legal_indices = set(action_id - 1 for action_id in legal_actions_1indexed)
                    
                    # Check which policy actions are legal
                    policy_nonzero = np.where(policy_pred > 1e-6)[0]
                    
                    legal_mass = 0.0
                    illegal_mass = 0.0
                    
                    for action_idx in range(len(policy_pred)):
                        if policy_pred[action_idx] > 1e-6:
                            if action_idx in legal_indices:
                                legal_mass += policy_pred[action_idx]
                            else:
                                illegal_mass += policy_pred[action_idx]
                    
                    total_mass = legal_mass + illegal_mass
                    
                    stats['total_policy_mass'] += total_mass
                    stats['legal_policy_mass'] += legal_mass
                    stats['illegal_policy_mass'] += illegal_mass
                    stats['num_legal_moves'].append(len(legal_actions_1indexed))
                    stats['num_policy_moves'].append(len(policy_nonzero))
                    
                    # Track per-position illegal ratio (probability mass)
                    if total_mass > 0:
                        stats['illegal_move_ratios'].append(illegal_mass / total_mass)
                    
                    # Check if argmax is legal (like train.py does)
                    predicted_action_idx = np.argmax(policy_pred)
                    predicted_action_id = predicted_action_idx + 1  # Convert to 1-indexed
                    if predicted_action_id not in legal_actions_1indexed:
                        stats['illegal_argmax_count'] += 1
                    
                except Exception as e:
                    errors += 1
                    if errors <= 5:  # Only print first few errors
                        print(f"    Error processing position {batch_start + i}: {e}")
    
    print(f"\nProcessed {total_positions} positions (errors: {errors})")
    
    # Calculate statistics
    results = {}
    
    for color_name, stats in [('White', white_stats), ('Black', black_stats)]:
        if stats['total_positions'] == 0:
            print(f"\n{color_name}: No positions found")
            continue
        
        avg_legal_pct = (stats['legal_policy_mass'] / stats['total_policy_mass'] * 100) if stats['total_policy_mass'] > 0 else 0
        avg_illegal_pct = (stats['illegal_policy_mass'] / stats['total_policy_mass'] * 100) if stats['total_policy_mass'] > 0 else 0
        illegal_argmax_ratio = stats['illegal_argmax_count'] / stats['total_positions'] if stats['total_positions'] > 0 else 0
        
        results[color_name.lower()] = {
            'positions': stats['total_positions'],
            'avg_legal_pct': avg_legal_pct,
            'avg_illegal_pct': avg_illegal_pct,
            'illegal_argmax_ratio': illegal_argmax_ratio,
            'avg_num_legal_moves': np.mean(stats['num_legal_moves']) if stats['num_legal_moves'] else 0,
            'avg_num_policy_moves': np.mean(stats['num_policy_moves']) if stats['num_policy_moves'] else 0,
            'median_illegal_ratio': np.median(stats['illegal_move_ratios']) if stats['illegal_move_ratios'] else 0,
            'std_illegal_ratio': np.std(stats['illegal_move_ratios']) if stats['illegal_move_ratios'] else 0,
        }
    
    return results


def print_results(results: Dict) -> None:
    """Print formatted results."""
    print("\n" + "="*60)
    print("MODEL POLICY LEGALITY RESULTS")
    print("="*60)
    
    for color in ['White', 'Black']:
        color_key = color.lower()
        if color_key not in results:
            continue
        
        stats = results[color_key]
        print(f"\n{color} to move:")
        print(f"  Positions analyzed: {stats['positions']}")
        print(f"  Average illegal move probability mass: {stats['avg_illegal_pct']:.2f}%")
        print(f"  Average illegal argmax ratio:          {stats['illegal_argmax_ratio']*100:.2f}%")
        print(f"  Median illegal ratio per position:     {stats['median_illegal_ratio']*100:.2f}%")
        print(f"  Std illegal ratio:                     {stats['std_illegal_ratio']*100:.2f}%")
        print(f"  Average # legal moves:  {stats['avg_num_legal_moves']:.2f}")
        print(f"  Average # policy moves: {stats['avg_num_policy_moves']:.2f}")
    
    # Overall summary
    if 'white' in results and 'black' in results:
        total_positions = results['white']['positions'] + results['black']['positions']
        weighted_illegal_prob = (
            results['white']['avg_illegal_pct'] * results['white']['positions'] +
            results['black']['avg_illegal_pct'] * results['black']['positions']
        ) / total_positions
        weighted_illegal_argmax = (
            results['white']['illegal_argmax_ratio'] * results['white']['positions'] +
            results['black']['illegal_argmax_ratio'] * results['black']['positions']
        ) / total_positions
        
        print(f"\n{'='*60}")
        print(f"Overall:")
        print(f"  Total positions: {total_positions}")
        print(f"  Weighted average illegal probability mass: {weighted_illegal_prob:.2f}%")
        print(f"  Weighted average illegal argmax ratio:     {weighted_illegal_argmax*100:.2f}%")
        
        # Check if there's a significant difference between colors
        diff_prob = abs(results['white']['avg_illegal_pct'] - results['black']['avg_illegal_pct'])
        diff_argmax = abs(results['white']['illegal_argmax_ratio'] - results['black']['illegal_argmax_ratio']) * 100
        print(f"\n  Color balance:")
        if diff_prob < 5:
            print(f"    ✓ Illegal probability mass is balanced (diff: {diff_prob:.2f}%)")
        else:
            print(f"    ✗ Illegal probability mass differs between colors (diff: {diff_prob:.2f}%)")
        if diff_argmax < 5:
            print(f"    ✓ Illegal argmax ratio is balanced (diff: {diff_argmax:.2f}%)")
        else:
            print(f"    ✗ Illegal argmax ratio differs between colors (diff: {diff_argmax:.2f}%)")


def load_network(checkpoint: dict, cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Load the neural network from checkpoint."""
    print("\nLoading neural network...")
    
    network = create_chess_network(cfg, device)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.to(device)
    network.eval()
    
    print(f"Network loaded on {device}")
    print(f"Model config: num_res_blocks={cfg.network.num_residual_layers}, num_channels={cfg.network.initial_conv_block_out_channels}")
    print(f"Action space size: {cfg.network.action_space_size}")
    return network


@hydra.main(config_path="../config", config_name="train_mcts", version_base=None)
def main(cfg: DictConfig):
    # Default checkpoint path
    default_checkpoint = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model.pth"
    )
    
    # Allow command line argument for checkpoint path (via Hydra override)
    checkpoint_path = cfg.training.get('checkpoint_dir_load', default_checkpoint)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path)
        
        # Print checkpoint info
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
        if 'iteration' in checkpoint:
            print(f"Training iteration: {checkpoint['iteration']}")
        if 'total_games_simulated' in checkpoint:
            print(f"Total games simulated: {checkpoint['total_games_simulated']}")
        
        # Load network
        network = load_network(checkpoint, cfg, device)
        
        # Load replay buffer
        if 'replay_buffer_state' not in checkpoint:
            print("\n✗ No replay buffer state found in checkpoint!")
            return 1
        
        training_data = decompress_replay_buffer(checkpoint['replay_buffer_state'])
        
        # Test policy legality with model predictions
        results = test_policy_legality(network, training_data, device, batch_size=128)
        
        # Print results
        print_results(results)
        
        print("\n" + "="*60)
        print("Analysis complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()

