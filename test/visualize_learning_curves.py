#!/usr/bin/env python3
"""
Visualize learning curves from training checkpoint history.

Usage:
    python utils/visualize_learning_curves.py path/to/checkpoint/model.pth
"""

import torch
import matplotlib.pyplot as plt
import sys
import os


def plot_learning_curves(checkpoint_path, save_dir=None):
    """
    Load and plot learning curves from a training checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        save_dir: Optional directory to save plots (if None, just displays)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'history' not in checkpoint:
        print("Error: No training history found in checkpoint")
        return
    
    history = checkpoint['history']
    iteration = checkpoint.get('iteration', len(history['policy_loss']))
    
    # Handle backward compatibility: initialize draw statistics if missing
    if 'non_draw_count' not in history:
        history['non_draw_count'] = [0] * len(history['policy_loss'])
    if 'repetition_draw_count' not in history:
        history['repetition_draw_count'] = [0] * len(history['policy_loss'])
    if 'other_draw_count' not in history:
        history['other_draw_count'] = [0] * len(history['policy_loss'])
    
    print(f"Loaded training history with {iteration} iterations")
    print(f"Total games simulated: {checkpoint.get('total_games_simulated', 'N/A')}")
    
    # Create iterations axis
    iterations = list(range(1, len(history['policy_loss']) + 1))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Learning Curves (Iteration {iteration})', fontsize=16)
    
    # Plot 1: Policy + Value Loss
    ax = axes[0, 0]
    ax.plot(iterations, history['policy_loss'], 'b-', linewidth=2, label='Policy Loss')
    ax.plot(iterations, history['value_loss'], 'r-', linewidth=2, label='Value Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Policy + Value Loss Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)
    
    # Plot 2: Mate-in Success Ratios (Train vs Val)
    ax = axes[0, 1]
    mate_labels = history.get('mate_success_labels')
    if not mate_labels:
        train_mate = history.get('mate_success_train', {})
        val_mate = history.get('mate_success_val', {})
        mate_labels = sorted({*train_mate.keys(), *val_mate.keys()})
    if mate_labels:
        color_cycle = plt.cm.tab10.colors
        for idx, label in enumerate(mate_labels):
            train_series = history.get('mate_success_train', {}).get(label, [0.0] * len(iterations))
            val_series = history.get('mate_success_val', {}).get(label, [0.0] * len(iterations))
            if len(train_series) < len(iterations):
                train_series = train_series + [0.0] * (len(iterations) - len(train_series))
            if len(val_series) < len(iterations):
                val_series = val_series + [0.0] * (len(iterations) - len(val_series))
            color = color_cycle[idx % len(color_cycle)]
            ax.plot(iterations, [v * 100 for v in train_series], color=color, linewidth=2, label=f'{label} Train')
            ax.plot(iterations, [v * 100 for v in val_series], color=color, linewidth=2, linestyle='--', label=f'{label} Val')
        ax.legend(loc='best', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Mate success ratios unavailable', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Success Ratio (%)')
    ax.set_title('Mate-in Success Ratios Over Time')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 3: Combined Illegal Move Metrics
    ax = axes[1, 0]
    ax.plot(iterations, [r * 100 for r in history['illegal_move_ratio']], 'orange', linewidth=2, label='Illegal Move Ratio (%)', marker='o', markersize=3)
    ax.plot(iterations, [p * 100 for p in history['illegal_move_prob']], 'purple', linewidth=2, label='Illegal Move Probability (%)', marker='s', markersize=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Illegal Move Metrics Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)
    
    # Plot 4: Draw Statistics
    ax = axes[1, 1]
    # Handle backward compatibility: if draw stats don't exist, use zeros
    non_draw = history.get('non_draw_count', [0] * len(iterations))
    repetition_draw = history.get('repetition_draw_count', [0] * len(iterations))
    other_draw = history.get('other_draw_count', [0] * len(iterations))
    
    # Stacked bar chart for game outcomes
    bar_width = 0.8
    bottom = [0] * len(iterations)
    ax.bar(iterations, non_draw, color='green', label='Non-Draw Games', width=bar_width)
    bottom = [b + n for b, n in zip(bottom, non_draw)]
    ax.bar(iterations, repetition_draw, bottom=bottom, color='orange', label='Repetition Draws', width=bar_width)
    bottom = [b + r for b, r in zip(bottom, repetition_draw)]
    ax.bar(iterations, other_draw, bottom=bottom, color='red', label='Other Draws', width=bar_width)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Game Count')
    ax.set_title('Game Outcomes Over Time')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save or display
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'learning_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n=== Training Summary ===")
    print(f"Final Policy Loss: {history['policy_loss'][-1]:.4f}")
    print(f"Final Value Loss: {history['value_loss'][-1]:.4f}")
    print(f"Final Illegal Move Ratio: {history['illegal_move_ratio'][-1]:.2%}")
    print(f"Final Illegal Move Prob: {history['illegal_move_prob'][-1]:.2%}")
    
    # Print draw statistics if available
    if 'non_draw_count' in history and len(history['non_draw_count']) > 0:
        total_games = history['non_draw_count'][-1] + history.get('repetition_draw_count', [0])[-1] + history.get('other_draw_count', [0])[-1]
        if total_games > 0:
            print(f"\n=== Game Outcomes (Last Iteration) ===")
            print(f"Non-Draw Games: {history['non_draw_count'][-1]} ({history['non_draw_count'][-1]/total_games:.1%})")
            print(f"Repetition Draws: {history.get('repetition_draw_count', [0])[-1]} ({history.get('repetition_draw_count', [0])[-1]/total_games:.1%})")
            print(f"Other Draws: {history.get('other_draw_count', [0])[-1]} ({history.get('other_draw_count', [0])[-1]/total_games:.1%})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_learning_curves.py <checkpoint_path> [save_dir]")
        print("\nExample:")
        print("  python utils/visualize_learning_curves.py outputs/mcts_train/2025-01-01/12-00-00/checkpoints/model.pth")
        print("  python utils/visualize_learning_curves.py checkpoints/model.pth plots/")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    save_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_learning_curves(checkpoint_path, save_dir)


if __name__ == "__main__":
    main()

