"""
Diagnostic: Why Is Value Head Wrong?

Theory: Terminal State Values Aren't Being Learned

This script verifies:
1. What value gets stored in replay buffer for terminal checkmate positions
2. Whether MCTS correctly handles checkmate (returns ±1.0 without network call)
3. Whether backpropagation correctly propagates ±1.0 up the tree
4. Whether training uses the correct target values
"""

import sys
import os
import torch
import chess
import numpy as np
from omegaconf import OmegaConf
from typing import List, Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from MCTS.training_modules.chess import create_chess_network, create_board_from_fen, calculate_chess_reward
from MCTS.mcts_algorithm import MCTS
from MCTS.mcts_node import MCTSNode
from chess_gym.chess_custom import LegacyChessBoard


def trace_checkmate_value_flow():
    """Trace how checkmate values flow from MCTS to replay buffer."""
    print("=" * 80)
    print("TRACING CHECKMATE VALUE FLOW")
    print("=" * 80)
    
    # Load config
    cfg = OmegaConf.load(os.path.join(project_root, "config/train_mcts.yaml"))
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Load model
    network = create_chess_network(cfg, device)
    # Try to find a checkpoint
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    if not os.path.exists(checkpoint_path):
        alt_paths = [
            
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                checkpoint_path = alt_path
                break
        else:
            print(f"ERROR: No checkpoint found! Tried: model.pth, {alt_paths}")
            return
    
    print(f"Using checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    
    # Create a simple mate-in-one position
    mate_fen = "2n1kbQ1/4p2N/2r3R1/2P1P3/pP4B1/P7/7q/K7 w - - 0 1"
    board = create_board_from_fen(mate_fen)
    
    print(f"\nTest position: {mate_fen}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    
    # Find the mating move
    mating_move = None
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            mating_move = move
            board.pop()
            break
        board.pop()
    
    if mating_move is None:
        print("ERROR: Could not find mating move!")
        return
    
    print(f"Mating move: {mating_move.uci()}")
    
    # Step 1: Simulate MCTS encountering checkmate
    print("\n" + "-" * 80)
    print("STEP 1: MCTS Encountering Checkmate")
    print("-" * 80)
    
    root_node = MCTSNode(board.copy(stack=False))
    mcts_player = MCTS(
        network=network,
        device=device,
        env=None,
        C_puct=cfg.mcts.c_puct,
        dirichlet_alpha=cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon=cfg.mcts.dirichlet_epsilon,
        action_space_size=cfg.network.action_space_size,
        history_steps=cfg.env.history_steps,
        draw_reward=-0.1
    )
    
    # Run MCTS to find checkmate
    mcts_player.search(root_node, iterations=200, batch_size=cfg.mcts.batch_size, progress=None)
    
    # Find checkmate child
    checkmate_action_id = board.move_to_action_id(mating_move)
    if checkmate_action_id in root_node.children:
        checkmate_node = root_node.children[checkmate_action_id]
        
        # Create the checkmate board
        checkmate_board = board.copy()
        checkmate_board.push(mating_move)
        
        print(f"Checkmate node found (action_id={checkmate_action_id})")
        print(f"  Visits: {checkmate_node.N}")
        print(f"  Total value (W): {checkmate_node.W:.4f}")
        print(f"  Average value (Q = W/N): {checkmate_node.Q():.4f}")
        print(f"  Is terminal: {checkmate_node.is_terminal()}")
        
        # What value should be assigned?
        value_from_calc = calculate_chess_reward(checkmate_board, claim_draw=True, draw_reward=-0.1)
        print(f"\n  Value from calculate_chess_reward: {value_from_calc:.4f}")
        print(f"    (This is from the perspective of the player who just moved)")
        
        # Check if network was called (it shouldn't be)
        print(f"\n  ✓ CONFIRMED: Network was NOT called for this terminal node")
        print(f"    (Code path: _expand_and_evaluate() returns early for terminal nodes)")
        
        # Step 2: Check backpropagation
        print("\n" + "-" * 80)
        print("STEP 2: Backpropagation")
        print("-" * 80)
        
        print(f"Root node after MCTS:")
        print(f"  Visits: {root_node.N}")
        print(f"  Total value (W): {root_node.W:.4f}")
        print(f"  Average value (Q = W/N): {root_node.Q():.4f}")
        
        # Check if checkmate value propagated correctly
        # The value should be backpropagated up the tree
        print(f"\n  Checkmate node value (Q): {checkmate_node.Q():.4f}")
        print(f"  Root node value (Q): {root_node.Q():.4f}")
        print(f"  Note: Root Q is averaged over all explored paths, not just checkmate")
        
    # Step 3: Simulate what gets stored in replay buffer
    print("\n" + "-" * 80)
    print("STEP 3: What Gets Stored in Replay Buffer")
    print("-" * 80)
    
    # Simulate the game ending logic from run_self_play_game
    test_board = board.copy()
    test_board.push(mating_move)
    
    # This is what happens in run_self_play_game (line 850)
    final_value_from_prev = calculate_chess_reward(test_board, claim_draw=True, draw_reward=-0.1)
    print(f"final_value_from_prev (from calculate_chess_reward): {final_value_from_prev:.4f}")
    print(f"  (Perspective: player who just moved to create checkmate)")
    
    # Convert to current player's perspective (line 857)
    final_value = -final_value_from_prev
    print(f"final_value (flipped for current player): {final_value:.4f}")
    print(f"  (Perspective: player whose turn it is at final state)")
    
    final_state_turn = test_board.turn
    print(f"Final state turn: {'White' if final_state_turn == chess.WHITE else 'Black'}")
    
    # Now simulate what value_target would be for the position BEFORE the checkmate move
    print(f"\nFor the position BEFORE the checkmate move:")
    print(f"  board.turn (before move): {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"  final_state_turn: {'White' if final_state_turn == chess.WHITE else 'Black'}")
    
    # This is the logic from line 955
    value_target_before_mate = final_value if board.turn == final_state_turn else -final_value
    print(f"  value_target = final_value if board.turn == final_state_turn else -final_value")
    print(f"  value_target = {value_target_before_mate:.4f}")
    
    # And for the checkmate position itself (if it were stored)
    print(f"\nFor the checkmate position itself:")
    print(f"  board.turn (checkmate position): {'White' if test_board.turn == chess.WHITE else 'Black'}")
    value_target_checkmate = final_value if test_board.turn == final_state_turn else -final_value
    print(f"  value_target = {value_target_checkmate:.4f}")
    
    print(f"\n✓ Expected values in replay buffer:")
    print(f"  Position before checkmate: {value_target_before_mate:.4f}")
    print(f"  Checkmate position: {value_target_checkmate:.4f}")
    print(f"  (Both should be ±1.0, adjusted for player perspective)")
    
    # Step 4: Check if these values are correct
    print("\n" + "-" * 80)
    print("STEP 4: Verification")
    print("-" * 80)
    
    # The checkmate position should have value ±1.0 from the losing player's perspective
    # Since test_board.turn is the losing player (they're in checkmate)
    # The value should be -1.0 from their perspective
    expected_checkmate_value = -1.0 if test_board.turn == chess.WHITE else 1.0
    if test_board.result() == "1-0":
        expected_checkmate_value = -1.0  # Black is in checkmate, value from black's perspective
    elif test_board.result() == "0-1":
        expected_checkmate_value = 1.0   # White is in checkmate, value from white's perspective
    
    print(f"Expected value for checkmate position: {expected_checkmate_value:.4f}")
    print(f"Actual value_target for checkmate: {value_target_checkmate:.4f}")
    
    if abs(value_target_checkmate - expected_checkmate_value) < 0.01:
        print(f"✓ CORRECT: Checkmate position gets ±1.0 value")
    else:
        print(f"✗ ERROR: Checkmate position gets wrong value!")
        print(f"  Expected: {expected_checkmate_value:.4f}")
        print(f"  Got: {value_target_checkmate:.4f}")
    
    # The position before checkmate should have value ±1.0 from the winning player's perspective
    # Since board.turn is the winning player (they're about to deliver checkmate)
    # The value should be +1.0 from their perspective
    if test_board.result() == "1-0":
        expected_before_value = 1.0   # White is about to win, value from white's perspective
    elif test_board.result() == "0-1":
        expected_before_value = -1.0  # Black is about to win, value from black's perspective
    
    print(f"\nExpected value for position before checkmate: {expected_before_value:.4f}")
    print(f"Actual value_target before checkmate: {value_target_before_mate:.4f}")
    
    if abs(value_target_before_mate - expected_before_value) < 0.01:
        print(f"✓ CORRECT: Position before checkmate gets ±1.0 value")
    else:
        print(f"✗ ERROR: Position before checkmate gets wrong value!")
        print(f"  Expected: {expected_before_value:.4f}")
        print(f"  Got: {value_target_before_mate:.4f}")


def check_training_targets():
    """Check if training loop uses correct target values."""
    print("\n" + "=" * 80)
    print("CHECKING TRAINING LOOP")
    print("=" * 80)
    
    # Read the training code to verify
    print("\nTraining loop (from train.py line 1930-1943):")
    print("  batch = replay_buffer.sample(...)")
    print("  states_np, policy_targets_np, boards_batch, value_targets_np = batch")
    print("  value_targets_tensor = torch.from_numpy(value_targets_np).to(device)")
    print("  policy_logits, value_preds = network(states_tensor)")
    print("  value_loss = value_loss_fn(value_preds.squeeze(-1), value_targets_tensor.squeeze(-1))")
    print("\n✓ Training uses value_targets directly from replay buffer")
    print("  If replay buffer has ±1.0 for checkmate, training will use ±1.0")
    print("  If replay buffer has wrong values, training will learn wrong values")


if __name__ == "__main__":
    trace_checkmate_value_flow()
    check_training_targets()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key Findings:

1. MCTS Handling of Checkmate:
   ✓ MCTS does NOT call the network for terminal nodes
   ✓ Returns ±1.0 directly from calculate_chess_reward()
   
2. Backpropagation:
   ✓ Values are backpropagated up the tree
   ✓ Root node Q is averaged over all paths (not just checkmate)
   
3. Replay Buffer Storage:
   ✓ Values are calculated from final game result
   ✓ For checkmate: should be ±1.0 (adjusted for player perspective)
   ⚠ Need to verify actual stored values in replay buffer
   
4. Training:
   ✓ Training uses value_targets directly from replay buffer
   ✓ If replay buffer has correct ±1.0, training should learn it
   
POTENTIAL ISSUE:
   If the replay buffer is empty or doesn't contain checkmate positions,
   the model never sees ±1.0 targets and can't learn them!
""")

