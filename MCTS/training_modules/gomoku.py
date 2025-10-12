import torch.nn as nn
from gym_gomoku.envs import GomokuEnv, Board
from gym_gomoku.envs.util import gomoku_util
from MCTS.models.network_4672 import ChessNetwork4672

def create_gomoku_network(cfg, device) -> nn.Module:
    """Create and initialize the Gomoku network based on config."""
    # TODO: Implement Gomoku network creation
    # This should match your Gomoku network architecture
    return ChessNetwork4672(
        input_channels=cfg.network.input_channels,
        board_size=cfg.network.board_size,
        num_residual_layers=cfg.network.num_residual_layers,
        initial_conv_block_out_channels=cfg.network.initial_conv_block_out_channels,
        residual_blocks_out_channels=cfg.network.residual_blocks_out_channels,
        action_space_size=cfg.network.action_space_size,
        num_pieces=cfg.network.num_pieces,
        policy_head_out_channels=cfg.network.policy_head_out_channels,
        value_head_hidden_size=cfg.network.value_head_hidden_size
    ).to(device)

def create_gomoku_env(cfg, render=False) -> GomokuEnv:
    """Create and initialize the Gomoku environment."""
    return GomokuEnv(
        board_size=cfg.network.board_size,
    )

def get_gomoku_game_result(board: Board) -> float | None:
    """Get the game result from a Gomoku board (Board instance)."""
    if getattr(board, 'foul', False):
        # The player who just moved made an illegal move, so the opponent wins
        # If board.move is even, black's turn, so white just won
        # If board.move is odd, white's turn, so black just won
        last_player = 'black' if board.move % 2 == 0 else 'white'
        return -1.0 if last_player == 'black' else 1.0
    exist, win_color = gomoku_util.check_five_in_row(board.board_state)
    is_full = all(board.board_state[i][j] != 0 for i in range(board.size) for j in range(board.size))
    if exist:
        return 1.0 if win_color == 'black' else -1.0
    elif is_full:
        return 0.0  # Draw
    else:
        return None  # Game not finished

def is_gomoku_first_player_turn(board: Board) -> bool:
    """Check if it's the first player's turn (black) on the Gomoku board (Board instance)."""
    # Black always goes first, and colors alternate
    # Count number of stones: if equal, it's black's turn; else white's turn
    black_count = sum(cell == 1 for row in board.board_state for cell in row)
    white_count = sum(cell == -1 for row in board.board_state for cell in row)
    return black_count == white_count

def get_gomoku_legal_actions(board: Board) -> list[int]:
    """Get legal moves (action IDs) from a Gomoku board (Board instance)."""
    return board.legal_actions
