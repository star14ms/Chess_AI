"""
Final Diagnostic: Check what's actually in the replay buffer for checkmate positions.

This will:
1. Load a checkpoint with replay buffer
2. Find all checkmate positions in the buffer
3. Show their stored values
4. Verify if they have ±1.0 or wrong values
"""

import sys
import os
import torch
import chess
import gzip
import pickle
import numpy as np
from typing import List, Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from chess_gym.chess_custom import LegacyChessBoard


def decompress_replay_buffer(replay_buffer_state: dict) -> List[Tuple]:
    """Decompress replay buffer if needed."""
    compressed = replay_buffer_state.get('compressed', False)
    
    if compressed:
        compressed_bytes = replay_buffer_state.get('buffer_compressed')
        if compressed_bytes:
            buffer_bytes = gzip.decompress(compressed_bytes)
            buffer = pickle.loads(buffer_bytes)
            return buffer
    else:
        buffer = replay_buffer_state.get('buffer', [])
        return buffer
    
    return []


def _extract_board_and_value(exp):
    """Extract board (as object), FEN string, and value from an experience."""
    if not exp or len(exp) < 4:
        return None, None, None

    board_or_str = exp[2]
    value = exp[3]

    if isinstance(board_or_str, str):
        try:
            board = LegacyChessBoard()
            board.set_fen(board_or_str)
            return board, board_or_str, value
        except Exception:
            return None, None, None

    # Uncompressed buffers may store board objects directly
    if hasattr(board_or_str, "fen"):
        try:
            board_str = board_or_str.fen()
        except Exception:
            board_str = str(board_or_str)
        return board_or_str, board_str, value

    return None, None, None


def analyze_replay_buffer_for_checkmate(checkpoint_path: str):
    """Analyze replay buffer to find checkmate positions and their values."""
    print("=" * 80)
    print("ANALYZING REPLAY BUFFER FOR CHECKMATE POSITIONS")
    print("=" * 80)
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'replay_buffer_state' not in checkpoint:
        print("\n⚠ No replay buffer found in checkpoint!")
        print("  This means the model was trained without storing replay buffer state.")
        print("  The replay buffer is only used during training, not saved in checkpoints.")
        return
    
    print("\nDecompressing replay buffer...")
    replay_buffer_state = checkpoint['replay_buffer_state']
    buffer = decompress_replay_buffer(replay_buffer_state)
    
    if not buffer:
        print("⚠ Replay buffer is empty!")
        return
    
    print(f"✓ Replay buffer contains {len(buffer)} experiences")
    
    # Find checkmate positions
    print("\nSearching for checkmate positions...")
    checkmate_experiences = []
    
    for i, exp in enumerate(buffer):
        # Support legacy and prioritized formats (extra flags in tuple)
        board, board_str, value = _extract_board_and_value(exp)
        if board is None:
            continue

        # Newer prioritized format includes explicit checkmate flags
        is_checkmate_flag = False
        if len(exp) >= 6:
            is_checkmate_flag = bool(exp[4]) or bool(exp[5])
        elif len(exp) == 5:
            is_checkmate_flag = bool(exp[4])

        try:
            if is_checkmate_flag or board.is_checkmate():
                checkmate_experiences.append((i, board_str, value, board))
        except Exception:
            continue
    
    print(f"Found {len(checkmate_experiences)} checkmate positions in replay buffer")
    
    if len(checkmate_experiences) == 0:
        print("\n⚠ NO CHECKMATE POSITIONS FOUND IN REPLAY BUFFER!")
        print("  This is the problem! The model never sees checkmate positions during training.")
        print("  Possible reasons:")
        print("  1. Games rarely end in checkmate (most end in draws)")
        print("  2. Replay buffer is too small and checkmate positions get evicted")
        print("  3. Checkmate positions are filtered out somehow")
        return
    
    # Analyze the checkmate positions
    print("\n" + "-" * 80)
    print("CHECKMATE POSITIONS AND THEIR VALUES:")
    print("-" * 80)
    
    correct_values = 0
    wrong_values = 0
    
    for idx, (exp_idx, fen, value, board) in enumerate(checkmate_experiences[:20], 1):
        result = board.result()
        turn = board.turn
        
        # Expected value: from the losing player's perspective (they're in checkmate)
        if result == "1-0":
            expected = -1.0  # Black is in checkmate, value from black's perspective
        elif result == "0-1":
            expected = 1.0   # White is in checkmate, value from white's perspective
        else:
            expected = None
        
        print(f"\n{idx}. Experience #{exp_idx}")
        print(f"   FEN: {fen}")
        print(f"   Result: {result}")
        print(f"   Turn: {'White' if turn == chess.WHITE else 'Black'}")
        print(f"   Stored value: {value:.4f}")
        
        if expected is not None:
            print(f"   Expected value: {expected:.4f}")
            if abs(value - expected) < 0.01:
                print(f"   ✓ CORRECT: Value is ±1.0")
                correct_values += 1
            else:
                print(f"   ✗ WRONG: Expected {expected:.4f}, got {value:.4f}")
                wrong_values += 1
        else:
            print(f"   ⚠ Unexpected result: {result}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total checkmate positions found: {len(checkmate_experiences)}")
    print(f"Correct values (±1.0): {correct_values}")
    print(f"Wrong values: {wrong_values}")
    
    if wrong_values > 0:
        print(f"\n✗ PROBLEM FOUND: {wrong_values} checkmate positions have wrong values!")
        print("  The model is being trained on incorrect targets.")
    elif len(checkmate_experiences) == 0:
        print(f"\n✗ PROBLEM FOUND: No checkmate positions in replay buffer!")
        print("  The model never sees checkmate during training.")
    else:
        print(f"\n✓ All checkmate positions have correct ±1.0 values")
        print("  The issue might be:")
        print("  1. Too few checkmate positions in training data")
        print("  2. Model needs more training on these positions")
        print("  3. Value head architecture needs adjustment")


if __name__ == "__main__":
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    
    # Try different checkpoints
    checkpoint_paths = [
        os.path.join(checkpoint_dir, "model.pth"),
    ]
    
    for checkpoint_path in checkpoint_paths:
        if os.path.exists(checkpoint_path):
            print(f"\n{'='*80}")
            print(f"Checking: {os.path.basename(checkpoint_path)}")
            print(f"{'='*80}")
            analyze_replay_buffer_for_checkmate(checkpoint_path)
            break
    else:
        print("ERROR: No checkpoint found!")
        print(f"Tried: {checkpoint_paths}")






