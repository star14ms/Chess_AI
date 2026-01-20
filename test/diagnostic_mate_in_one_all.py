"""
Diagnostic test for mate-in-one positions:
- Load checkpoint
- Test ALL mate-in-one positions from the JSON file
- Run MCTS with specified iterations for each position
- Collect statistics on whether mating moves are found in top 10
- Print summary statistics
"""

import sys
import os
import json
import torch
import chess
from omegaconf import OmegaConf
from typing import Optional, Tuple, List, Dict
from collections import defaultdict
import time
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

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


def test_single_position(
    network: torch.nn.Module,
    cfg: OmegaConf,
    fen_position: str,
    mcts_iterations: int,
    device: torch.device,
    verbose: bool = False
) -> Dict:
    """Test a single mate-in-one position and return results."""
    
    result = {
        'fen': fen_position,
        'success': False,
        'rank': None,
        'mating_move': None,
        'mating_action_id': None,
        'mating_visits': 0,
        'was_explored': False,
        'root_visits': 0,
        'root_value': 0.0,
        'top_move_visits': 0,
        'top_move_is_mating': False,  # Whether top move (by visits) is the mating move
        'error': None
    }
    
    try:
        # Create board from FEN
        board = create_board_from_fen(fen_position)
        
        # Find the mating move
        mating_move = find_mating_move(board)
        if mating_move is None:
            result['error'] = "No mating move found"
            return result
        
        result['mating_move'] = mating_move.uci()
        mating_action_id = board.move_to_action_id(mating_move)
        result['mating_action_id'] = mating_action_id
        
        # Run MCTS
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
        
        result['root_visits'] = root_node.N
        result['root_value'] = root_node.Q() if root_node.N > 0 else 0.0
        
        # Get top 10 moves by visit count
        top_moves = get_top_moves_by_visits(root_node, top_k=10)
        
        if top_moves:
            top_action_id, top_visits, _ = top_moves[0]
            result['top_move_visits'] = top_visits
            
            # Check if top move is the mating move
            if top_action_id == mating_action_id:
                result['top_move_is_mating'] = True
                result['success'] = True
                result['rank'] = 1
            else:
                top_action_ids = [action_id for action_id, _, _ in top_moves]
                
                if mating_action_id in top_action_ids:
                    result['success'] = True
                    result['rank'] = top_action_ids.index(mating_action_id) + 1
                else:
                    # Check if it was explored at all
                    if mating_action_id in root_node.children:
                        result['was_explored'] = True
                        result['mating_visits'] = root_node.children[mating_action_id].N
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def run_batch_diagnostic_test(
    checkpoint_path: str,
    mate_positions: Dict,
    mcts_iterations: int = 1000,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    max_positions: Optional[int] = None
):
    """Run diagnostic test on all positions in batch."""
    
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
    
    print("=" * 80)
    print("BATCH DIAGNOSTIC TEST: Mate-in-One Positions")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"MCTS iterations per position: {mcts_iterations}")
    print(f"Total positions in file: {len(mate_positions)}")
    
    if max_positions:
        print(f"Testing first {max_positions} positions (limited)")
    print("=" * 80)
    
    # Load model once
    print("\nLoading model...")
    network = create_chess_network(cfg, device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    print(f"Model loaded! Iteration: {checkpoint.get('iteration', 'unknown')}")
    print()
    
    # Process all positions
    positions_to_test = list(mate_positions.keys())
    if max_positions:
        positions_to_test = positions_to_test[:max_positions]
    
    total_positions = len(positions_to_test)
    results = []
    start_time = time.time()
    
    # Statistics for progress bar
    top_move_is_mating_count = 0
    top_move_not_mating_count = 0
    
    # Use Rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=None,
        transient=False
    ) as progress:
        task = progress.add_task(
            "[cyan]Testing positions...",
            total=total_positions
        )
        
        for idx, fen_position in enumerate(positions_to_test, 1):
            if verbose:
                print(f"\n[{idx}/{total_positions}] Testing position...")
                print(f"FEN: {fen_position}")
            
            result = test_single_position(
                network=network,
                cfg=cfg,
                fen_position=fen_position,
                mcts_iterations=mcts_iterations,
                device=device,
                verbose=verbose
            )
            results.append(result)
            
            # Update statistics (only count positions without errors)
            if result['error'] is None:
                if result['top_move_is_mating']:
                    top_move_is_mating_count += 1
                else:
                    top_move_not_mating_count += 1
            
            # Calculate ratio and update description
            total_tested = top_move_is_mating_count + top_move_not_mating_count
            if total_tested > 0:
                ratio = top_move_is_mating_count / total_tested
                description = (
                    f"[cyan]Top1=Checkmate: {top_move_is_mating_count} | "
                    f"Top1≠Checkmate: {top_move_not_mating_count} | "
                    f"Ratio: {ratio:.1%}"
                )
            else:
                description = "[cyan]Testing positions... (initializing)"
            
            progress.update(task, advance=1, description=description)
    
    elapsed_time = time.time() - start_time
    
    # Calculate statistics
    successful = [r for r in results if r['success']]
    explored_but_not_top10 = [r for r in results if r['was_explored'] and not r['success']]
    not_explored = [r for r in results if not r['was_explored'] and not r['success'] and r['error'] is None]
    errors = [r for r in results if r['error'] is not None]
    
    # Rank distribution
    rank_distribution = defaultdict(int)
    for r in successful:
        rank_distribution[r['rank']] += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total positions tested: {total_positions}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/total_positions:.2f} sec/position)")
    print()
    
    # Top move statistics
    top_move_is_mating = [r for r in results if r.get('top_move_is_mating', False)]
    top_move_not_mating = [r for r in results if r.get('top_move_is_mating', False) == False and r['error'] is None]
    
    print(f"🎯 Top 1 Move = Checkmate: {len(top_move_is_mating)} ({100*len(top_move_is_mating)/total_positions:.1f}%)")
    print(f"❌ Top 1 Move ≠ Checkmate: {len(top_move_not_mating)} ({100*len(top_move_not_mating)/total_positions:.1f}%)")
    if len(top_move_is_mating) + len(top_move_not_mating) > 0:
        ratio = len(top_move_is_mating) / (len(top_move_is_mating) + len(top_move_not_mating))
        print(f"   Ratio: {ratio:.1%}")
    print()
    
    print(f"✓ SUCCESS (mating move in top 10): {len(successful)} ({100*len(successful)/total_positions:.1f}%)")
    if successful:
        print(f"   Average rank: {sum(r['rank'] for r in successful) / len(successful):.2f}")
        print(f"   Rank distribution:")
        for rank in sorted(rank_distribution.keys()):
            print(f"      Rank #{rank}: {rank_distribution[rank]} positions")
    
    print(f"✗ Explored but not in top 10: {len(explored_but_not_top10)} ({100*len(explored_but_not_top10)/total_positions:.1f}%)")
    if explored_but_not_top10:
        avg_visits = sum(r['mating_visits'] for r in explored_but_not_top10) / len(explored_but_not_top10)
        print(f"   Average mating move visits: {avg_visits:.1f}")
    
    print(f"✗ Not explored at all: {len(not_explored)} ({100*len(not_explored)/total_positions:.1f}%)")
    print(f"✗ Errors: {len(errors)} ({100*len(errors)/total_positions:.1f}%)")
    
    if errors:
        print("\nErrors encountered:")
        error_types = defaultdict(int)
        for r in errors:
            error_types[r['error']] += 1
        for error_msg, count in error_types.items():
            print(f"   {error_msg}: {count}")
    
    # Additional statistics
    if successful:
        print("\nAdditional statistics for successful positions:")
        avg_root_visits = sum(r['root_visits'] for r in successful) / len(successful)
        avg_root_value = sum(r['root_value'] for r in successful) / len(successful)
        print(f"   Average root node visits: {avg_root_visits:.1f}")
        print(f"   Average root node value: {avg_root_value:.4f}")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnostic test for mate-in-one positions")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (default: checkpoints/model.pth)")
    parser.add_argument("--iterations", type=int, default=200, help="MCTS iterations per position (default: 1000)")
    parser.add_argument("--max-positions", type=int, default=None, help="Maximum number of positions to test (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output for each position")
    parser.add_argument("--positions-file", type=str, default=None, help="Path to positions JSON file (default: data/mate_in_one_positions.json)")
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_dir = os.path.join(project_root, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load mate-in-one positions
    if args.positions_file:
        mate_positions_file = args.positions_file
    else:
        mate_positions_file = os.path.join(project_root, "data/mate_in_one_positions.json")
    
    if not os.path.exists(mate_positions_file):
        print(f"ERROR: Positions file not found: {mate_positions_file}")
        sys.exit(1)
    
    with open(mate_positions_file, 'r') as f:
        mate_positions = json.load(f)
    
    # Run batch test
    results = run_batch_diagnostic_test(
        checkpoint_path=checkpoint_path,
        mate_positions=mate_positions,
        mcts_iterations=args.iterations,
        verbose=args.verbose,
        max_positions=args.max_positions
    )

