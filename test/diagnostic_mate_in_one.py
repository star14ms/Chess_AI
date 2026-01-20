"""
Diagnostic test for mate-in-one positions:
- Load iteration 17 checkpoint
- Pick ONE mate-in-one position
- Run MCTS with 2000 iterations
- Print the top 10 moves by visit count
- Check if the mating move is in the top 10
"""

import sys
import os
import json
import torch
import chess
from omegaconf import OmegaConf
from typing import Optional, Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from MCTS.training_modules.chess import create_chess_network, create_board_from_fen
from MCTS.mcts_algorithm import MCTS
from MCTS.mcts_node import MCTSNode


def find_mating_move(board) -> Optional[chess.Move]:
    """Find the mating move from the current position."""
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()
    return None


def get_top_moves_by_visits(root_node: MCTSNode, top_k: int = 10) -> list[Tuple[int, int, Optional[chess.Move]]]:
    """
    Get top moves by visit count.
    Returns: list of (action_id, visit_count, move) tuples, sorted by visit count descending.
    """
    if not root_node.children:
        return []
    
    # Get all children with their visit counts
    moves_data = []
    for action_id, child_node in root_node.children.items():
        visit_count = child_node.N
        # Convert action_id to move
        try:
            move = root_node.board.action_id_to_move(action_id)
        except Exception:
            move = None
        moves_data.append((action_id, visit_count, move))
    
    # Sort by visit count descending
    moves_data.sort(key=lambda x: x[1], reverse=True)
    
    return moves_data[:top_k]


def run_diagnostic_test(
    checkpoint_path: str,
    fen_position: str,
    mcts_iterations: int = 2000,
    device: Optional[torch.device] = None
):
    """Run the diagnostic test."""
    
    # Load config
    cfg = OmegaConf.load(os.path.join(project_root, "config/train_mcts.yaml"))
    
    # Set device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print(f"Using device: {device}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"FEN position: {fen_position}")
    print(f"MCTS iterations: {mcts_iterations}")
    print("=" * 80)
    
    # Load model
    print("\n1. Loading model...")
    network = create_chess_network(cfg, device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    print(f"   Model loaded! Iteration: {checkpoint.get('iteration', 'unknown')}")
    
    # Create board from FEN
    print("\n2. Setting up board position...")
    board = create_board_from_fen(fen_position)
    print(f"   Board FEN: {board.fen()}")
    print(f"   Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    
    # Find the mating move
    print("\n3. Finding the mating move...")
    mating_move = find_mating_move(board)
    if mating_move is None:
        print("   ERROR: No mating move found! This is not a mate-in-one position.")
        return
    
    # Get action ID for the mating move
    mating_action_id = board.move_to_action_id(mating_move)
    print(f"   Mating move: {mating_move.uci()}")
    print(f"   Mating action ID: {mating_action_id}")
    
    # Run MCTS
    print(f"\n4. Running MCTS with {mcts_iterations} iterations...")
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
    
    mcts_player.search(root_node, iterations=mcts_iterations, batch_size=cfg.mcts.batch_size, progress=None)
    print(f"   MCTS search completed!")
    print(f"   Root node visits: {root_node.N}")
    print(f"   Root node value (Q): {root_node.Q():.4f}")
    
    # Get top 10 moves by visit count
    print("\n5. Top 10 moves by visit count:")
    print("-" * 80)
    top_moves = get_top_moves_by_visits(root_node, top_k=10)
    
    if not top_moves:
        print("   No moves explored!")
        return
    
    for rank, (action_id, visit_count, move) in enumerate(top_moves, 1):
        move_str = move.uci() if move else "N/A"
        is_mating = "*** MATING MOVE ***" if (mating_action_id is not None and action_id == mating_action_id) else ""
        print(f"   {rank:2d}. Action ID: {action_id:4d} | Visits: {int(visit_count):5d} | Move: {move_str:8s} {is_mating}")
    
    # Check if mating move is in top 10
    print("\n6. Result:")
    print("-" * 80)
    top_action_ids = [action_id for action_id, _, _ in top_moves]
    
    if mating_action_id is not None and mating_action_id in top_action_ids:
        rank = top_action_ids.index(mating_action_id) + 1
        print(f"   ✓ SUCCESS: Mating move is in top 10! (Rank #{rank})")
    else:
        print(f"   ✗ FAILURE: Mating move is NOT in top 10")
        if mating_action_id is not None:
            # Check if it was explored at all
            if mating_action_id in root_node.children:
                child_visits = root_node.children[mating_action_id].N
                print(f"      Mating move was explored with {child_visits} visits")
            else:
                print(f"      Mating move was NOT explored at all")
    
    print("=" * 80)


if __name__ == "__main__":
    # Configuration
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load one mate-in-one position
    mate_positions_file = os.path.join(project_root, "data/mate_in_one_positions.json")
    with open(mate_positions_file, 'r') as f:
        mate_positions = json.load(f)
    
    # Pick the first position
    test_fen = list(mate_positions.keys())[0]
    
    print("DIAGNOSTIC TEST: Mate-in-One Position")
    print("=" * 80)
    
    run_diagnostic_test(
        checkpoint_path=checkpoint_path,
        fen_position=test_fen,
        mcts_iterations=400
    )

