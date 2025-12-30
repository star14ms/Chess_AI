#!/usr/bin/env python3
"""
Test script to diagnose repetition detection issues in training.
This script replays a game from the history and checks if repetition is detected.
"""

import chess
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from chess_gym.envs.chess_env import ChessEnv
from omegaconf import OmegaConf

def test_game_repetition(moves_san: list, game_num: int = 1):
    """Test if repetition is detected when replaying a game."""
    print(f"\n{'='*70}")
    print(f"Testing Game {game_num} for repetition detection")
    print(f"{'='*70}\n")
    
    # Create environment
    cfg = OmegaConf.create({
        'env': {
            'type': 'chess',
            'observation_mode': 'vector',
            'render_mode': None,
            'history_steps': 8
        }
    })
    
    env = ChessEnv(
        observation_mode=cfg.env.observation_mode,
        render_mode=cfg.env.render_mode,
        history_steps=cfg.env.history_steps
    )
    obs, _ = env.reset()
    
    position_counts = {}
    repetition_detected = False
    termination_reason = None
    
    print(f"Playing {len(moves_san)} moves...")
    print("-" * 70)
    
    for i, move_san in enumerate(moves_san):
        # Get legal actions
        legal_actions = env.board.legal_actions
        
        # Find the action ID for this move
        action_id = None
        try:
            move = env.board.parse_san(move_san)
            for aid in legal_actions:
                if env.action_space._action_to_move(aid) == move:
                    action_id = aid
                    break
        except:
            print(f"Error parsing move {i+1}: {move_san}")
            break
        
        if action_id is None:
            print(f"Move {i+1} ({move_san}) is not legal!")
            break
        
        # Check board state before move
        fen_parts = env.board.fen().split()
        position_key = ' '.join(fen_parts[:4])
        position_counts[position_key] = position_counts.get(position_key, 0) + 1
        count = position_counts[position_key]
        
        # Check python-chess repetition detection
        can_claim_3fold = env.board.can_claim_threefold_repetition() if hasattr(env.board, 'can_claim_threefold_repetition') else False
        is_5fold = env.board.is_fivefold_repetition() if hasattr(env.board, 'is_fivefold_repetition') else False
        is_game_over = env.board.is_game_over(claim_draw=True)
        
        # Make the move
        obs, reward, terminated, truncated, info = env.step(action_id)
        
        if count >= 3:
            print(f"Move {i+1:3d} ({move_san:5s}): Position repeated {count} times")
            if count == 3:
                print(f"  -> THREEfold repetition!")
            if count == 5:
                print(f"  -> FIVEfold repetition!")
        
        if can_claim_3fold:
            print(f"  -> can_claim_threefold_repetition() = True")
        if is_5fold:
            print(f"  -> is_fivefold_repetition() = True")
        if is_game_over:
            outcome = env.board.outcome(claim_draw=True)
            if outcome:
                termination_reason = outcome.termination.name
                print(f"  -> is_game_over() = True, termination: {termination_reason}")
                repetition_detected = True
                break
        
        if terminated:
            outcome = env.board.outcome(claim_draw=True)
            if outcome:
                termination_reason = outcome.termination.name
                print(f"  -> env.step() returned terminated=True, termination: {termination_reason}")
                repetition_detected = True
                break
    
    print("-" * 70)
    print(f"\nResults:")
    print(f"  Total moves played: {i+1}")
    print(f"  Unique positions: {len(position_counts)}")
    print(f"  Positions repeated 3+ times: {sum(1 for c in position_counts.values() if c >= 3)}")
    print(f"  Repetition detected: {repetition_detected}")
    print(f"  Termination reason: {termination_reason}")
    print(f"  Final move_stack length: {len(env.board.move_stack) if hasattr(env.board, 'move_stack') else 'N/A'}")
    print(f"  Final is_game_over(): {env.board.is_game_over(claim_draw=True)}")
    
    return repetition_detected, termination_reason

if __name__ == "__main__":
    # Test Game 1 from games_iter_101.txt
    game1_moves = 'a3 g5 e3 f6 d3 Nc6 b3 f5 Nh3 a6 f3 Kf7 c3 Ra7 e4 Ra8 Kd2 Kf6 d4 b6 a4 h6 Bb2 d6 Bc1 Nxd4 Qe2 Ra7 c4 Ra8 Ng1 Ra7 Nh3 Ra8 Bb2 Ra7 Bc1 Ra8 Ng1 Ra7 Kd3 Ra8 Nh3 Ra7 Ng1 Ra8 Bb2 Ra7 Nh3 Ra8 Ng1 Ra7 Nh3 Ra8 Bc1 Ra7 Ng1 Ra8 Nh3 Ra7 Bb2 Ra8 Bc1 Ra7 Bb2 Ra8 Ng1 Ra7 Bc1 Ra8 Bb2 Ra7 Nh3 Ra8 Ng1 Ra7 Nh3 Ra8 Ng1 Ra7 Bc1 Ra8 Nh3 Ra7 Ng1 Ra8 Nh3 Ra7 Ng1 Ra8 Nh3 Ra7 Ng1 Ra8 Nh3 Ra7 Ng1 Ra8 Nh3 Ra7'.split()
    
    test_game_repetition(game1_moves, game_num=1)

