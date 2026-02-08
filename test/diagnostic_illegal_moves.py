"""
Diagnostic script to track illegal move probability during actual gameplay.

This will help us understand if illegal move probability increases as the game progresses
(suggesting distribution shift / out-of-distribution positions).
"""

import sys
import os
import argparse
import io
from typing import Optional
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn.functional as F
import numpy as np
import pygame
import chess
import chess.svg
from omegaconf import OmegaConf

from MCTS.training_modules.chess import create_chess_network, create_chess_env

try:
    import cairosvg  # SVG -> PNG
except Exception:
    cairosvg = None
    print("cairosvg not available. Install with: pip install cairosvg")


def _board_surface_from_chess_svg(
    board: chess.Board, size_px: int, flipped: bool
) -> pygame.Surface:
    """Render board with python-chess.svg, convert to pygame Surface."""
    if cairosvg is None:
        raise RuntimeError("cairosvg is required to render chess.svg into pygame")

    svg_str = chess.svg.board(board=board, size=size_px, flipped=flipped)
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"))
    return pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()


def replay_diagnostic_pygame(
    game_stats: list[dict],
    boards: list[chess.Board],
    board_px: int = 768,
    fps: int = 60,
    white_perspective: bool = True,
):
    """
    Pygame replay for diagnostic stats.
    Use ←/→ to move back/forward, Esc to quit.
    """
    pygame_initialized = False
    screen = None
    quit_requested = False

    try:
        pygame.init()
        pygame_initialized = True
        pygame.display.set_caption("Diagnostic illegal move replay")

        margin_top = 140
        margin = 16
        width = board_px + margin * 2
        height = board_px + margin_top + margin

        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()

        ui_font = pygame.font.SysFont(None, 24)
        small_font = pygame.font.SysFont(None, 18)
        ui_bg = (20, 20, 20)
        ui_fg = (230, 230, 230)

        idx = 0
        running = True

        cached_idx = None
        cached_surf = None

        def draw(board: chess.Board, stat: Optional[dict], ply: int):
            nonlocal cached_idx, cached_surf

            screen.fill(ui_bg)
            title = "Diagnostic: Illegal Move Probabilities"
            line1 = ui_font.render(title, True, ui_fg)
            screen.blit(line1, (margin, 10))

            if ply == 0:
                status = "Ply 0 / Start position"
            else:
                status = f"Ply {ply} / {len(boards)-1}"
            line2 = ui_font.render(status, True, ui_fg)
            screen.blit(line2, (margin, 38))

            if stat:
                illegal_pct = stat["illegal_prob_mass"] * 100
                line3 = ui_font.render(
                    f"Illegal %: {illegal_pct:.2f} | #Legal: {stat['num_legal_moves']} | "
                    f"Argmax legal: {'Yes' if stat['is_argmax_legal'] else 'No'}",
                    True,
                    ui_fg,
                )
                screen.blit(line3, (margin, 64))
                selected = stat.get("selected_move_san") or stat.get("selected_move_uci") or "-"
                action_id = stat.get("selected_action")
                action_label = "-" if action_id is None else str(action_id)
                uci_move = stat.get("selected_move_uci") or "-"
                line4 = ui_font.render(
                    f"Selected: {selected} | Action ID: {action_label} | UCI: {uci_move} | "
                    f"Selected legal: {'Yes' if stat['selected_legal'] else 'No'}",
                    True,
                    ui_fg,
                )
                screen.blit(line4, (margin, 90))

            controls_text = "←/→: Navigate | Home/End: Jump | Esc: Quit"
            controls_surface = small_font.render(controls_text, True, (150, 150, 150))
            screen.blit(controls_surface, (margin, 116))

            if cached_idx != ply:
                cached_surf = _board_surface_from_chess_svg(
                    board,
                    size_px=board_px,
                    flipped=(not white_perspective),
                )
                cached_idx = ply

            screen.blit(cached_surf, (margin, margin_top))
            pygame.display.flip()

        pygame.key.set_repeat(200, 50)

        while running:
            clock.tick(fps)
            idx_changed = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        quit_requested = True
                        running = False
                        break
                    elif event.key == pygame.K_RIGHT:
                        idx = min(idx + 1, len(boards) - 1)
                        idx_changed = True
                    elif event.key == pygame.K_LEFT:
                        idx = max(idx - 1, 0)
                        idx_changed = True
                    elif event.key == pygame.K_HOME:
                        idx = 0
                        idx_changed = True
                    elif event.key == pygame.K_END:
                        idx = len(boards) - 1
                        idx_changed = True
                    elif event.key == pygame.K_d:
                        idx = min(idx + 1, len(boards) - 1)
                        idx_changed = True
                    elif event.key == pygame.K_a:
                        idx = max(idx - 1, 0)
                        idx_changed = True
                    elif event.key == pygame.K_SPACE:
                        idx = min(idx + 1, len(boards) - 1)
                        idx_changed = True
                    elif event.key == pygame.K_BACKSPACE:
                        idx = max(idx - 1, 0)
                        idx_changed = True

            if not running:
                break

            if idx_changed:
                cached_idx = None

            stat = game_stats[idx - 1] if idx > 0 and idx - 1 < len(game_stats) else None
            draw(boards[idx], stat, idx)

    finally:
        if pygame_initialized:
            try:
                if screen is not None:
                    pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
    return quit_requested

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
    boards = [board.copy(stack=False)]
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
        selected_move_uci = "-"
        selected_move_san = "-"
        move_obj = None
        board_mapped = None
        if hasattr(board, "action_id_to_move"):
            try:
                board_mapped = board.action_id_to_move(action)
            except Exception:
                board_mapped = None
        if action in legal_actions:
            try:
                move_obj = env.action_space._action_to_move(action)
            except Exception:
                move_obj = None
        if move_obj is not None:
            selected_move_uci = move_obj.uci()
            try:
                selected_move_san = board.san(move_obj)
            except Exception:
                selected_move_san = "-"
        stats["selected_move_uci"] = selected_move_uci
        stats["selected_move_san"] = selected_move_san
        
        # Print step info
        result = "OK" if stats['selected_legal'] else "**FOUL**"
        print(
            f"{step:<5} {stats['color']:<6} {stats['illegal_prob_mass']*100:>8.2f}% "
            f"{'Yes' if stats['is_argmax_legal'] else 'NO':<12} {stats['num_legal_moves']:<8} "
            f"act={stats['selected_action']} uci={stats['selected_move_uci']} {result:<20}"
        )
        
        game_stats.append(stats)
        
        # Execute move
        if move_obj is None:
            print("\nGame ended due to INVALID ACTION (no move mapping).")
            stats['selected_legal'] = False
            break
        observation, reward, terminated, truncated, info = env.step(action)
        board = env.board
        boards.append(board.copy(stack=False))
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
    
    return game_stats, boards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic illegal move analysis.")
    parser.add_argument(
        "--config",
        default="./config/train_mcts.yaml",
        help="Path to training config.",
    )
    parser.add_argument(
        "--checkpoint",
        default="../model.pth",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip pygame visualization.",
    )
    args = parser.parse_args()

    # Load config and model
    cfg = OmegaConf.load(args.config)
    checkpoint_path = args.checkpoint
    
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
    stats, boards = run_diagnostic_game(cfg, model, device, max_moves=100, deterministic=True)
    if not args.no_visualize:
        quit_requested = replay_diagnostic_pygame(
            stats,
            boards,
            board_px=768,
            fps=60,
            white_perspective=True,
        )
        if quit_requested:
            sys.exit(0)
    
    # Optionally run a few more games
    print("\n\nWould you like to run more diagnostic games? (Ctrl+C to stop)")
    input("Press Enter to run another game...")
    stats2, boards2 = run_diagnostic_game(cfg, model, device, max_moves=100, deterministic=True)
    if not args.no_visualize:
        quit_requested = replay_diagnostic_pygame(
            stats2,
            boards2,
            board_px=768,
            fps=60,
            white_perspective=True,
        )
        if quit_requested:
            sys.exit(0)
