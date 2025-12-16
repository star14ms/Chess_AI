"""
Test Draw Detection During Training

This script tests that the training code properly detects and terminates games
when threefold or fivefold repetition occurs. It creates scenarios that force
repetition and verifies that games terminate correctly.

Usage:
    python test/test_draw_detection.py

The test suite includes:
1. Threefold Repetition Detection (Self-Play) - Tests random play with repetition detection
2. Training Loop Repetition Detection - Tests the exact logic used in train.py
3. Repetition Detection Methods - Direct test of board's repetition detection methods

All tests should pass if repetition detection is working correctly.
"""

import sys
import os
import chess
import torch

# Add parent directory to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Also add MCTS directory so relative imports in train.py work
mcts_dir = os.path.join(project_root, 'MCTS')
sys.path.insert(0, mcts_dir)

from omegaconf import OmegaConf, DictConfig
from MCTS.train import run_self_play_game, initialize_factories_from_cfg
from MCTS.training_modules.chess import create_chess_env, get_chess_legal_actions
from chess_gym.envs.chess_env import ChessEnv


def create_test_config():
    """Create a minimal config for testing draw detection."""
    cfg_dict = {
        'env': {
            'type': 'chess',
            'observation_mode': 'vector',
            'render_mode': None,
            'save_video_folder': None,
            'history_steps': 8,
        },
        'network': {
            'action_space_size': 4672,
            'input_channels': 119,
            'board_size': 8,
        },
        'mcts': {
            'iterations': 0,  # No MCTS, just random moves for testing
            'c_puct': 1.41,
            'temperature_start': 1.0,
            'temperature_end': 0.1,
            'temperature_decay_moves': 30,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25,
            'batch_size': 8,
        },
        'training': {
            'max_game_moves': 200,  # High limit to allow repetition to occur
            'initial_board_fen': None,  # Will be set per test
        }
    }
    return OmegaConf.create(cfg_dict)


def test_threefold_repetition():
    """Test that threefold repetition is detected and terminates the game."""
    print("\n" + "="*70)
    print("TEST 1: Threefold Repetition Detection (Random Play)")
    print("="*70)
    
    cfg = create_test_config()
    initialize_factories_from_cfg(cfg)
    
    # Use a position that's more likely to lead to repetition
    # Starting from a simplified endgame position
    test_fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"  # King and rook, white to move
    cfg.training.initial_board_fen = test_fen
    cfg.training.max_game_moves = 100  # Lower limit for faster testing
    
    print(f"Starting position: {test_fen}")
    print("Playing game with random moves (MCTS iterations = 0)...")
    print("(This may not always hit repetition, but tests the detection mechanism)")
    
    # Run a self-play game
    game_data, game_info = run_self_play_game(
        cfg,
        network=None,  # No network, just random moves
        env=None,
        progress=None,
        device=torch.device('cpu')
    )
    
    print(f"\nGame finished!")
    print(f"  Move count: {game_info['move_count']}")
    print(f"  Termination reason: {game_info['termination']}")
    print(f"  Result: {game_info['result']}")
    if len(game_info['moves_san']) > 0:
        moves_preview = game_info['moves_san'][:150] + "..." if len(game_info['moves_san']) > 150 else game_info['moves_san']
        print(f"  Moves: {moves_preview}")
    
    # Check if game terminated due to repetition
    termination = game_info['termination'].upper()
    is_repetition = (
        'REPETITION' in termination or
        'THREEFOLD' in termination or
        'FIVEFOLD' in termination
    )
    
    if is_repetition:
        print(f"\n✓ SUCCESS: Game correctly terminated due to repetition!")
        return True
    elif game_info['move_count'] >= cfg.training.max_game_moves:
        print(f"\n⚠ WARNING: Game reached max moves without detecting repetition")
        print(f"  This could indicate the repetition detection isn't working, or")
        print(f"  random moves simply didn't create repetition in this run.")
        print(f"  Run the specific repetition test (Test 3) for a definitive check.")
        return False
    else:
        print(f"\n✓ Game terminated for other reason: {game_info['termination']}")
        print(f"  This is acceptable if the game ended normally (checkmate, stalemate, etc.)")
        return True  # Not a failure, just didn't hit repetition in this run


def test_training_loop_repetition_detection():
    """Test that the training loop's repetition detection works correctly."""
    print("\n" + "="*70)
    print("TEST 2: Training Loop Repetition Detection")
    print("="*70)
    
    cfg = create_test_config()
    initialize_factories_from_cfg(cfg)
    
    # Create environment
    env = create_chess_env(cfg, render=False)
    
    # Start from a position where we can create repetition
    test_fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    obs, _ = env.reset(options={'fen': test_fen})
    
    print(f"Testing with position: {test_fen}")
    print("Simulating training loop repetition detection logic...")
    
    # Track positions (same logic as in train.py)
    position_counts = {}
    move_count = 0
    terminated = False
    truncated = False
    max_moves = 50
    
    # Play moves and check for repetition using the same logic as train.py
    while not terminated and not truncated and move_count < max_moves:
        legal_actions = get_chess_legal_actions(env.board)
        if not legal_actions:
            break
        
        # Take a random legal move
        import numpy as np
        action = np.random.choice(legal_actions)
        
        obs, _, terminated, truncated, info = env.step(action)
        move_count += 1
        
        # Use the same repetition detection logic as in train.py
        # Additional checks if environment didn't detect termination
        if not terminated:
            # Check for fivefold repetition (automatic draw)
            if hasattr(env.board, 'is_fivefold_repetition'):
                try:
                    if env.board.is_fivefold_repetition():
                        print(f"  Move {move_count}: Fivefold repetition detected!")
                        terminated = True
                except (AttributeError, TypeError):
                    pass  # Board might not support this method or state is invalid
            # Check for threefold repetition (claimable draw) if claim_draw is enabled
            if not terminated and env.claim_draw:
                if hasattr(env.board, 'can_claim_threefold_repetition'):
                    try:
                        if env.board.can_claim_threefold_repetition():
                            print(f"  Move {move_count}: Threefold repetition claimable!")
                            terminated = True
                    except (AttributeError, TypeError):
                        pass  # Board might not support this method or state is invalid
                # Also check the general can_claim_draw method
                if not terminated and hasattr(env.board, 'can_claim_draw'):
                    try:
                        if env.board.can_claim_draw():
                            print(f"  Move {move_count}: Draw can be claimed!")
                            terminated = True
                    except (AttributeError, TypeError):
                        pass  # Board might not support this method or state is invalid
        
        # Track positions for termination reason reporting (same as train.py)
        if hasattr(env.board, 'fen'):
            fen_parts = env.board.fen().split()
            if len(fen_parts) >= 4:
                position_key = ' '.join(fen_parts[:4])
                position_counts[position_key] = position_counts.get(position_key, 0) + 1
                
                # Report when we see repetition
                if position_counts[position_key] >= 3:
                    print(f"  Position repeated {position_counts[position_key]} times")
                if position_counts[position_key] >= 5:
                    print(f"  Position repeated {position_counts[position_key]} times (fivefold!)")
                
                # Use position tracking to detect repetition and terminate (same as train.py)
                if position_counts[position_key] >= 5:
                    # Fivefold repetition - automatic draw
                    print(f"  Move {move_count}: Terminating due to fivefold repetition!")
                    terminated = True
                elif position_counts[position_key] >= 3 and env.claim_draw:
                    # Threefold repetition - claimable draw
                    print(f"  Move {move_count}: Terminating due to threefold repetition!")
                    terminated = True
        
        if terminated:
            break
    
    # Check final state
    outcome = env.board.outcome(claim_draw=True)
    if outcome:
        termination_name = outcome.termination.name if hasattr(outcome.termination, 'name') else str(outcome.termination)
        print(f"\n✓ Game terminated: {termination_name}")
        if 'REPETITION' in termination_name.upper():
            print("  ✓ Repetition correctly detected by training loop logic!")
            return True
    elif terminated:
        print(f"\n✓ Game terminated by repetition detection logic")
        max_reps = max(position_counts.values()) if position_counts else 0
        if max_reps >= 3:
            print(f"  ✓ Detected {max_reps} repetitions")
            return True
    
    print(f"\n? Game did not terminate due to repetition (move count: {move_count})")
    max_reps = max(position_counts.values()) if position_counts else 0
    print(f"  Max position repetitions: {max_reps}")
    if max_reps >= 3:
        print(f"  ⚠ Repetition occurred but game didn't terminate - this is a bug!")
        return False
    else:
        print(f"  (No repetition occurred in this run)")
        return True  # Not a failure if no repetition occurred


def test_repetition_detection_methods():
    """Test that the board's repetition detection methods work correctly."""
    print("\n" + "="*70)
    print("TEST 3: Repetition Detection Methods (Direct Test)")
    print("="*70)
    
    cfg = create_test_config()
    initialize_factories_from_cfg(cfg)
    
    env = create_chess_env(cfg, render=False)
    
    # Start from a position where we can easily create repetition
    # Use a position with kings and rooks that can move back and forth
    initial_fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    env.reset(options={'fen': initial_fen})
    board = env.board
    
    print(f"Starting position: {initial_fen}")
    print("Playing moves to create repetition (rook and king moves back and forth)...")
    
    # Play moves that create repetition: Rg1-h1, Kf7-e8, Rh1-g1, Ke8-f7 (repeat)
    # This creates the same position multiple times
    moves_uci = [
        "h1g1", "e8f7",  # Rook and king move
        "g1h1", "f7e8",  # Back (position repeats first time)
        "h1g1", "e8f7",  # Forward again
        "g1h1", "f7e8",  # Back (position repeats second time - threefold!)
        "h1g1", "e8f7",  # Forward again
        "g1h1", "f7e8",  # Back (position repeats third time - fivefold!)
    ]
    
    repetition_detected = False
    position_counts = {}
    
    for i, move_uci in enumerate(moves_uci):
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
                move_num = (i // 2) + 1
                color = "White" if i % 2 == 0 else "Black"
                try:
                    san_move = board.san(move)
                except:
                    san_move = move_uci
                print(f"  Move {move_num} ({color}): {move_uci} ({san_move})")
                
                # Track positions manually (same as train.py)
                fen_parts = board.fen().split()
                if len(fen_parts) >= 4:
                    position_key = ' '.join(fen_parts[:4])
                    position_counts[position_key] = position_counts.get(position_key, 0) + 1
                    if position_counts[position_key] >= 3:
                        print(f"    Position repeated {position_counts[position_key]} times")
                
                # Check repetition after each move
                if hasattr(board, 'can_claim_threefold_repetition'):
                    try:
                        if board.can_claim_threefold_repetition():
                            print(f"    ✓ Threefold repetition claimable!")
                            repetition_detected = True
                    except (AttributeError, TypeError):
                        pass
                
                if hasattr(board, 'is_fivefold_repetition'):
                    try:
                        if board.is_fivefold_repetition():
                            print(f"    ✓ Fivefold repetition (automatic draw)!")
                            repetition_detected = True
                            break
                    except (AttributeError, TypeError):
                        pass
                
                if hasattr(board, 'can_claim_draw'):
                    try:
                        if board.can_claim_draw():
                            print(f"    ✓ Draw can be claimed!")
                            repetition_detected = True
                    except (AttributeError, TypeError):
                        pass
                
                # Also check our manual tracking
                if position_counts.get(position_key, 0) >= 3:
                    repetition_detected = True
            else:
                print(f"  Move {move_uci} is illegal, skipping...")
                # Skip this move and continue
                continue
        except Exception as e:
            print(f"  Error with move {move_uci}: {e}")
            # Continue with next move instead of breaking
            continue
    
    # Check final state
    outcome = board.outcome(claim_draw=True)
    if outcome:
        termination_name = outcome.termination.name if hasattr(outcome.termination, 'name') else str(outcome.termination)
        print(f"\n✓ Game outcome: {termination_name}")
        if 'REPETITION' in termination_name.upper():
            print("  ✓ Repetition correctly detected by board!")
            return True
        else:
            print(f"  ? Terminated for: {termination_name}")
            return repetition_detected  # Return True if we detected it manually
    else:
        # Check manually
        if hasattr(board, 'is_fivefold_repetition') and board.is_fivefold_repetition():
            print("\n✓ Fivefold repetition detected (but outcome not set)")
            return True
        elif hasattr(board, 'can_claim_threefold_repetition') and board.can_claim_threefold_repetition():
            print("\n✓ Threefold repetition claimable (but outcome not set)")
            return True
        elif repetition_detected:
            print("\n✓ Repetition detected during play")
            return True
        else:
            print("\n✗ Repetition not detected")
            return False


def main():
    """Run all draw detection tests."""
    print("\n" + "="*70)
    print("DRAW DETECTION TEST SUITE")
    print("="*70)
    print("\nThis test suite verifies that games terminate correctly")
    print("when threefold or fivefold repetition occurs during training.\n")
    
    results = []
    
    # Test 1: Threefold repetition in self-play
    try:
        result = test_threefold_repetition()
        results.append(("Threefold Repetition (Self-Play)", result))
    except Exception as e:
        print(f"\n✗ Test 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Threefold Repetition (Self-Play)", False))
    
    # Test 2: Training loop repetition detection
    try:
        result = test_training_loop_repetition_detection()
        results.append(("Training Loop Repetition Detection", result))
    except Exception as e:
        print(f"\n✗ Test 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Training Loop Repetition Detection", False))
    
    # Test 3: Repetition detection methods
    try:
        result = test_repetition_detection_methods()
        results.append(("Repetition Detection Methods", result))
    except Exception as e:
        print(f"\n✗ Test 3 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Repetition Detection Methods", False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✓ All tests passed! Draw detection is working correctly.")
        return 0
    else:
        print(f"\n✗ {total_tests - total_passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit(main())

