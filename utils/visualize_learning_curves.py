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
    
    print(f"Loaded training history with {iteration} iterations")
    print(f"Total games simulated: {checkpoint.get('total_games_simulated', 'N/A')}")
    
    # Create iterations axis
    iterations = list(range(1, len(history['policy_loss']) + 1))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Training Learning Curves (Iteration {iteration})', fontsize=16)
    
    # Plot 1: Policy Loss
    ax = axes[0, 0]
    ax.plot(iterations, history['policy_loss'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Loss Over Time')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Value Loss
    ax = axes[0, 1]
    ax.plot(iterations, history['value_loss'], 'r-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value Loss')
    ax.set_title('Value Loss Over Time')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Illegal Move Ratio
    ax = axes[1, 0]
    ax.plot(iterations, [r * 100 for r in history['illegal_move_ratio']], 'orange', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Illegal Move Ratio (%)')
    ax.set_title('Illegal Move Ratio (Argmax) Over Time')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Illegal Move Probability Mass
    ax = axes[1, 1]
    ax.plot(iterations, [p * 100 for p in history['illegal_move_prob']], 'purple', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Illegal Move Probability (%)')
    ax.set_title('Illegal Move Probability Mass Over Time')
    ax.grid(True, alpha=0.3)
    
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

