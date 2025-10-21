"""
Test Model Policy Legality from Replay Buffer.

This script loads a trained model and tests the legality of its policy predictions
for all states in the replay buffer. It calculates statistics separately
for white and black positions.
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

# Import network creation function
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MCTS'))
from training_modules.chess import create_chess_network


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


def get_legal_action_ids_4672(board: chess.Board) -> set:
    """Get legal action IDs for 4672 action space."""
    legal_ids = set()
    
    for move in board.legal_moves:
        from_square = move.from_square
        to_square = move.to_square
        
        from_rank = chess.square_rank(from_square)
        from_file = chess.square_file(from_square)
        to_rank = chess.square_rank(to_square)
        to_file = chess.square_file(to_square)
        
        rank_diff = to_rank - from_rank
        file_diff = to_file - from_file
        
        # Queen moves (8 directions × 7 distances = 56)
        if rank_diff == 0 and file_diff != 0:  # Horizontal
            direction_idx = 0 if file_diff > 0 else 4
            distance = abs(file_diff) - 1
        elif file_diff == 0 and rank_diff != 0:  # Vertical
            direction_idx = 1 if rank_diff > 0 else 5
            distance = abs(rank_diff) - 1
        elif rank_diff == file_diff and rank_diff != 0:  # Diagonal /
            direction_idx = 2 if rank_diff > 0 else 6
            distance = abs(rank_diff) - 1
        elif rank_diff == -file_diff and rank_diff != 0:  # Diagonal \
            direction_idx = 3 if rank_diff > 0 else 7
            distance = abs(rank_diff) - 1
        # Knight moves (8 possible moves)
        elif (abs(rank_diff), abs(file_diff)) in [(2, 1), (1, 2)]:
            knight_moves = [
                (2, 1), (1, 2), (-1, 2), (-2, 1),
                (-2, -1), (-1, -2), (1, -2), (2, -1)
            ]
            try:
                knight_idx = knight_moves.index((rank_diff, file_diff))
                direction_idx = knight_idx
                distance = 0
                action_id = from_square * 73 + 56 + knight_idx
                legal_ids.add(action_id)
                continue
            except ValueError:
                continue
        # Underpromotion (9 possible underpromotions per square)
        elif move.promotion and move.promotion != chess.QUEEN:
            # N, B, R for 3 forward directions (left, straight, right)
            promotions = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
            try:
                promo_idx = promotions.index(move.promotion)
            except ValueError:
                continue
            
            # Determine direction: left (-1), straight (0), right (+1)
            if file_diff == -1:
                dir_offset = 0
            elif file_diff == 0:
                dir_offset = 1
            elif file_diff == 1:
                dir_offset = 2
            else:
                continue
            
            underpromo_idx = promo_idx * 3 + dir_offset
            action_id = from_square * 73 + 64 + underpromo_idx
            legal_ids.add(action_id)
            continue
        else:
            continue
        
        # Queen moves and queen promotions
        if 0 <= direction_idx < 8 and 0 <= distance < 7:
            action_id = from_square * 73 + direction_idx * 7 + distance
            legal_ids.add(action_id)
    
    return legal_ids


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
        'illegal_move_ratios': [],  # Track per-position illegal ratio
    }
    
    black_stats = {
        'total_positions': 0,
        'total_policy_mass': 0.0,
        'legal_policy_mass': 0.0,
        'illegal_policy_mass': 0.0,
        'num_legal_moves': [],
        'num_policy_moves': [],
        'illegal_move_ratios': [],
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
            
            if batch_start % 1000 == 0:
                print(f"  Processed {batch_start}/{total_positions} positions...")
            
            # Prepare batch observations
            obs_batch = []
            boards = []
            is_white_list = []
            
            for obs, _, board_str, _ in batch:
                obs_batch.append(obs)
                try:
                    board = chess.Board(board_str)
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
                    
                    # Get legal action IDs
                    legal_ids = get_legal_action_ids_4672(board)
                    
                    # Check which policy actions are legal
                    policy_nonzero = np.where(policy_pred > 1e-6)[0]
                    
                    legal_mass = 0.0
                    illegal_mass = 0.0
                    
                    for action_id in range(len(policy_pred)):
                        if policy_pred[action_id] > 1e-6:
                            if action_id in legal_ids:
                                legal_mass += policy_pred[action_id]
                            else:
                                illegal_mass += policy_pred[action_id]
                    
                    total_mass = legal_mass + illegal_mass
                    
                    stats['total_policy_mass'] += total_mass
                    stats['legal_policy_mass'] += legal_mass
                    stats['illegal_policy_mass'] += illegal_mass
                    stats['num_legal_moves'].append(len(legal_ids))
                    stats['num_policy_moves'].append(len(policy_nonzero))
                    
                    # Track per-position illegal ratio
                    if total_mass > 0:
                        stats['illegal_move_ratios'].append(illegal_mass / total_mass)
                    
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
        
        results[color_name.lower()] = {
            'positions': stats['total_positions'],
            'avg_legal_pct': avg_legal_pct,
            'avg_illegal_pct': avg_illegal_pct,
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
        print(f"  Average legal move probability:   {stats['avg_legal_pct']:.2f}%")
        print(f"  Average illegal move probability: {stats['avg_illegal_pct']:.2f}%")
        print(f"  Median illegal ratio per position: {stats['median_illegal_ratio']*100:.2f}%")
        print(f"  Std illegal ratio: {stats['std_illegal_ratio']*100:.2f}%")
        print(f"  Average # legal moves:  {stats['avg_num_legal_moves']:.2f}")
        print(f"  Average # policy moves: {stats['avg_num_policy_moves']:.2f}")
    
    # Overall summary
    if 'white' in results and 'black' in results:
        total_positions = results['white']['positions'] + results['black']['positions']
        weighted_legal_pct = (
            results['white']['avg_legal_pct'] * results['white']['positions'] +
            results['black']['avg_legal_pct'] * results['black']['positions']
        ) / total_positions
        weighted_illegal_pct = 100 - weighted_legal_pct
        
        print(f"\n{'='*60}")
        print(f"Overall:")
        print(f"  Total positions: {total_positions}")
        print(f"  Weighted average legal probability:   {weighted_legal_pct:.2f}%")
        print(f"  Weighted average illegal probability: {weighted_illegal_pct:.2f}%")
        
        # Check if there's a significant difference between colors
        diff = abs(results['white']['avg_legal_pct'] - results['black']['avg_legal_pct'])
        if diff < 5:
            print(f"  ✓ Policy legality is balanced between colors (diff: {diff:.2f}%)")
        else:
            print(f"  ✗ Policy legality differs between colors (diff: {diff:.2f}%)")


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

