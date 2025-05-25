import torch
import torch.nn as nn
import chess
import sys
import os

# Ensure utils and chess_gym can be found
# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from chess_gym.envs import ChessEnv
from models.network import ChessNetwork
from models.network_4672 import ChessNetwork4672

def create_chess_network(cfg, device):
    """Create and initialize the appropriate chess network based on config."""
    if cfg.network.action_space_mode == "4672":
        network = ChessNetwork4672(
            input_channels=cfg.network.input_channels,
            board_size=cfg.network.board_size,
            num_residual_layers=cfg.network.num_residual_layers,
            num_filters=cfg.network.num_filters,
            conv_blocks_channel_lists=cfg.network.conv_blocks_channel_lists,
            action_space_size=cfg.network.action_space_size,
            num_pieces=cfg.network.num_pieces,
            value_head_hidden_size=cfg.network.value_head_hidden_size
        ).to(device)
    else:
        network = ChessNetwork(
            input_channels=cfg.network.input_channels,
            dim_piece_type=cfg.network.dim_piece_type,
            board_size=cfg.network.board_size,
            num_residual_layers=cfg.network.num_residual_layers,
            num_filters=cfg.network.num_filters,
            conv_blocks_channel_lists=cfg.network.conv_blocks_channel_lists,
            action_space_size=cfg.network.action_space_size,
            num_pieces=cfg.network.num_pieces,
            value_head_hidden_size=cfg.network.value_head_hidden_size
        ).to(device)
    return network

def create_chess_env(cfg, render=False):
    """Create and initialize the chess environment."""
    return ChessEnv(
        observation_mode=cfg.env.observation_mode,
        render_mode=cfg.env.render_mode if render else None,
        save_video_folder=cfg.env.save_video_folder if render else None,
        action_space_mode=cfg.network.action_space_mode
    )

def get_chess_game_result(board):
    """Get the game result from a chess board."""
    result_str = board.result(claim_draw=True)
    if result_str == "1-0": return 1.0  # White won
    elif result_str == "0-1": return -1.0  # Black won
    else: return 0.0  # Draw

def is_white_turn(board):
    """Check if it's white's turn on the chess board."""
    return board.turn == chess.WHITE

def get_chess_legal_actions(board):
    """Get legal moves from a chess board."""
    return board.get_legal_moves_with_action_ids()

def action_id_to_move(board, action_id):
    """Convert action ID to move on chess board."""
    return board.action_id_to_move(action_id) 