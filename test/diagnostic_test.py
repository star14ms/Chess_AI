"""
Diagnostic script to track illegal move probability during actual gameplay.

This will help us understand if illegal move probability increases as the game progresses
(suggesting distribution shift / out-of-distribution positions).
"""

import sys
import os
sys.path.append(os.path.abspath('..'))

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

from MCTS.training_modules.chess import create_chess_network, create_chess_env

def calculate_illegal_move_stats(policy_probs, legal_actions):
    """
    Calculate illegal move probability statistics.
    
    Args:
        policy_probs: torch.Tensor of probabilities (0-indexed)
        legal_actions: list of legal action IDs (1-indexed)
    
    Returns:
        dict with statistics
    """
    # Convert legal actions to 0-indexed
    legal_indices = [action_id - 1 for action_id in legal_actions]
    
    # Calculate illegal probability mass
    legal_prob_mass = policy_probs[legal_indices].sum().item()
    illegal_prob_mass = 1.0 - legal_prob_mass
    
    # Get argmax action
    argmax_action_idx = policy_probs.argmax().item()
    argmax_action_id = argmax_action_idx + 1
    is_argmax_legal = argmax_action_id in legal_actions
    
    # Top-5 actions
    top5_probs, top5_indices = torch.topk(policy_probs, k=min(5, len(policy_probs)))
    top5_action_ids = [idx.item() + 1 for idx in top5_indices]
    top5_legality = [action_id in legal_actions for action_id in top5_action_ids]
    
    return {
        'illegal_prob_mass': illegal_prob_mass,
        'legal_prob_mass': legal_prob_mass,
        'is_argmax_legal': is_argmax_legal,
        'argmax_action': argmax_action_id,
        'num_legal_moves': len(legal_actions),
        'top5_actions': top5_action_ids,
        'top5_probs': top5_probs.tolist(),
        'top5_legality': top5_legality,
    }


def run_diagnostic_game(cfg, model, device, max_moves=100, deterministic=True):
    """
    Play a game and track illegal move probability at each step.
    """
    env = create_chess_env(cfg, render=False)
    env.reset()
    board = env.board
    
    model.eval()
    
    game_stats = []
    step = 0
    terminated = False
    truncated = False
    
    print(f"Starting diagnostic game (deterministic={deterministic})...")
    print(f"{'Step':<5} {'Color':<6} {'Illegal%':<10} {'ArgmaxLegal':<12} {'#Legal':<8} {'Result':<20}")
    print("-" * 80)
    
    while not terminated and not truncated and step < max_moves:
        # Get policy prediction
        observation = torch.from_numpy(
            board.get_board_vector(history_steps=cfg.env.history_steps)
        ).unsqueeze(0).to(device=device, dtype=torch.float32)
        
        with torch.no_grad():
            policy_logits, _ = model(observation)
            policy_logits = policy_logits.squeeze()
            policy_probs = F.softmax(policy_logits, dim=0)
        
        # Get legal actions
        legal_actions = list(board.legal_actions)
        
        # Calculate statistics
        stats = calculate_illegal_move_stats(policy_probs, legal_actions)
        stats['step'] = step
        stats['color'] = 'White' if board.turn else 'Black'
        
        # Select action (WITHOUT legal filtering to see if it causes foul)
        if deterministic:
            action = policy_logits.argmax().item() + 1
        else:
            action = torch.multinomial(policy_probs, num_samples=1).item() + 1
        
        stats['selected_action'] = action
        stats['selected_legal'] = action in legal_actions
        
        # Print step info
        result = "OK" if stats['selected_legal'] else "**FOUL**"
        print(f"{step:<5} {stats['color']:<6} {stats['illegal_prob_mass']*100:>8.2f}% "
              f"{'Yes' if stats['is_argmax_legal'] else 'NO':<12} {stats['num_legal_moves']:<8} {result:<20}")
        
        game_stats.append(stats)
        
        # Execute move
        observation, reward, terminated, truncated, info = env.step(action)
        board = env.board
        step += 1
        
        # Stop if foul
        if not stats['selected_legal']:
            print("\nGame ended due to FOUL!")
            break
    
    env.close()
    
    # Summary statistics
    print("\n" + "="*80)
    print("GAME SUMMARY")
    print("="*80)
    
    total_steps = len(game_stats)
    illegal_masses = [s['illegal_prob_mass'] for s in game_stats]
    argmax_illegals = [not s['is_argmax_legal'] for s in game_stats]
    actual_fouls = [not s['selected_legal'] for s in game_stats]
    
    print(f"Total steps: {total_steps}")
    print(f"\nIllegal probability mass:")
    print(f"  Mean:   {np.mean(illegal_masses)*100:.2f}%")
    print(f"  Median: {np.median(illegal_masses)*100:.2f}%")
    print(f"  Max:    {np.max(illegal_masses)*100:.2f}%")
    print(f"  Min:    {np.min(illegal_masses)*100:.2f}%")
    
    print(f"\nArgmax illegal rate: {np.mean(argmax_illegals)*100:.2f}%")
    print(f"Actual foul rate: {np.mean(actual_fouls)*100:.2f}% ({sum(actual_fouls)} fouls)")
    
    # Check if illegal probability increases over time
    if len(illegal_masses) > 10:
        first_half_mean = np.mean(illegal_masses[:len(illegal_masses)//2])
        second_half_mean = np.mean(illegal_masses[len(illegal_masses)//2:])
        print(f"\nTrend analysis:")
        print(f"  First half mean:  {first_half_mean*100:.2f}%")
        print(f"  Second half mean: {second_half_mean*100:.2f}%")
        if second_half_mean > first_half_mean * 1.2:
            print(f"  ⚠️  Illegal probability INCREASED by {((second_half_mean/first_half_mean)-1)*100:.1f}%")
            print(f"  → Suggests distribution shift / model degradation over time")
        else:
            print(f"  ✓ Illegal probability relatively stable")
    
    return game_stats


if __name__ == "__main__":
    # Load config and model
    cfg = OmegaConf.load("../config/train_mcts.yaml")
    checkpoint_path = cfg.training.get("checkpoint_dir_load") if cfg.training.get("checkpoint_dir_load") else "../model.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model = create_chess_network(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded!\n")
    
    # Run diagnostic games
    print("="*80)
    print("DIAGNOSTIC: Raw Policy (No Legal Filtering)")
    print("="*80)
    
    # Run deterministic game
    stats = run_diagnostic_game(cfg, model, device, max_moves=100, deterministic=True)
    
    # Optionally run a few more games
    print("\n\nWould you like to run more diagnostic games? (Ctrl+C to stop)")
    input("Press Enter to run another game...")
    stats2 = run_diagnostic_game(cfg, model, device, max_moves=100, deterministic=True)

