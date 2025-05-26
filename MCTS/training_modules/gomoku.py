import torch
import torch.nn as nn
from gym_gomoku.envs import GomokuEnv
from gym_gomoku.envs.util import gomoku_util
from models.network_4672 import ChessNetwork4672

def create_gomoku_network(cfg, device):
    """Create and initialize the Gomoku network based on config."""
    # TODO: Implement Gomoku network creation
    # This should match your Gomoku network architecture
    return ChessNetwork4672(
        input_channels=cfg.network.input_channels,
        board_size=cfg.network.board_size,
        num_residual_layers=cfg.network.num_residual_layers,
        num_filters=cfg.network.num_filters,
        conv_blocks_channel_lists=cfg.network.conv_blocks_channel_lists,
        action_space_size=cfg.network.action_space_size,
        num_pieces=cfg.network.num_pieces,
        value_head_hidden_size=cfg.network.value_head_hidden_size
    ).to(device)

def create_gomoku_env(cfg, render=False):
    """Create and initialize the Gomoku environment."""
    return GomokuEnv(
        board_size=cfg.network.board_size,
    )

def get_gomoku_game_result(board):
    """Get the game result from a Gomoku board (Board instance)."""
    exist, win_color = gomoku_util.check_five_in_row(board.board_state)
    is_full = all(board.board_state[i][j] != 0 for i in range(board.size) for j in range(board.size))
    if exist:
        return 1.0 if win_color == 'black' else -1.0
    elif is_full:
        return 0.0  # Draw
    else:
        return None  # Game not finished

def is_first_player_turn(board):
    """Check if it's the first player's turn (black) on the Gomoku board (Board instance)."""
    # Black always goes first, and colors alternate
    # Count number of stones: if equal, it's black's turn; else white's turn
    black_count = sum(cell == 1 for row in board.board_state for cell in row)
    white_count = sum(cell == -1 for row in board.board_state for cell in row)
    return black_count == white_count

def get_gomoku_legal_actions(board):
    """Get legal moves (action IDs) from a Gomoku board (Board instance)."""
    return board.get_legal_action()

def gomoku_action_id_to_move(board, action_id):
    """Convert action ID to move on Gomoku board."""
    # TODO: Implement Gomoku action ID to move conversion
    raise NotImplementedError("Gomoku action ID to move not yet implemented") 