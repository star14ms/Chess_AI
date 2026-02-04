import torch.nn as nn
import chess
import sys
import os

# Ensure utils and chess_gym can be found
# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from chess_gym.envs import ChessEnv
from chess_gym.chess_custom import BaseChessBoard, LegacyChessBoard
from MCTS.models.network import ChessNetwork
from MCTS.models.network_4672 import ChessNetwork4672

def create_chess_network(cfg, device) -> nn.Module:
    """Create and initialize the appropriate chess network based on config."""
    if cfg.network.action_space_size == 4672:
        network = ChessNetwork4672(
            input_channels=cfg.network.input_channels,
            board_size=cfg.network.board_size,
            num_residual_layers=cfg.network.num_residual_layers,
            initial_conv_block_out_channels=cfg.network.initial_conv_block_out_channels,
            residual_blocks_out_channels=cfg.network.residual_blocks_out_channels,
            action_space_size=cfg.network.action_space_size,
            num_pieces=cfg.network.num_pieces,
            value_head_hidden_size=cfg.network.value_head_hidden_size,
            policy_linear_out_features=cfg.network.policy_linear_out_features,
            conv_bias=cfg.network.conv_bias,
            policy_dropout=getattr(cfg.network, "policy_dropout", 0.0),
            value_dropout=getattr(cfg.network, "value_dropout", 0.0),
            conv_dropout=getattr(cfg.network, "conv_dropout", 0.0),
        ).to(device)
    else:
        network = ChessNetwork(
            input_channels=cfg.network.input_channels,
            dim_piece_type=cfg.network.dim_piece_type,
            board_size=cfg.network.board_size,
            num_residual_layers=cfg.network.num_residual_layers,
            initial_conv_block_out_channels=cfg.network.initial_conv_block_out_channels,
            residual_blocks_out_channels=cfg.network.residual_blocks_out_channels,
            action_space_size=cfg.network.action_space_size,
            num_pieces=cfg.network.num_pieces,
            value_head_hidden_size=cfg.network.value_head_hidden_size,
            policy_dropout=getattr(cfg.network, "policy_dropout", 0.0),
            value_dropout=getattr(cfg.network, "value_dropout", 0.0),
            conv_dropout=getattr(cfg.network, "conv_dropout", 0.0),
        ).to(device)
    return network

def create_chess_env(cfg, render=False, render_mode=None, show_possible_actions=False) -> ChessEnv:
    """Create and initialize the chess environment."""
    return ChessEnv(
        observation_mode=cfg.env.observation_mode,
        render_mode=cfg.env.render_mode if render and render_mode is None else render_mode if render else None,
        save_video_folder=cfg.env.save_video_folder if render else None,
        action_space_size=cfg.network.action_space_size,
        history_steps=cfg.env.history_steps,
        show_possible_actions=show_possible_actions
    )

def calculate_chess_reward(board: BaseChessBoard, claim_draw: bool = True, draw_reward: float = -0.1) -> float:
    """Calculate the reward for a chess game state from the previous player's perspective.
    
    Note: After board.push(move), board.turn changes to the next player. So the reward
    is calculated from the perspective of the player who just moved (previous turn = not board.turn).
    
    Args:
        board: The chess board state (after a move has been pushed)
        claim_draw: Whether to claim draws (for threefold repetition)
        draw_reward: Reward value for draws (applies to both players equally)
    
    Returns:
        float: Reward value from the previous player's perspective (who just moved)
        - +1.0 if previous player's side won
        - -1.0 if previous player's side lost
        - draw_reward for draws (equally penalized for both colors)
    """
    result_str = board.result(claim_draw=claim_draw)
    # After push(move), board.turn is the next player, so previous turn is the opposite
    previous_turn = not board.turn
    
    if result_str == "1-0":
        # White won - reward is +1.0 if previous turn was white, else -1.0
        return 1.0 if previous_turn == chess.WHITE else -1.0
    elif result_str == "0-1":
        # Black won - reward is +1.0 if previous turn was black, else -1.0
        return 1.0 if previous_turn == chess.BLACK else -1.0
    else:
        # Draw: use draw_reward for all draws
        return draw_reward

def get_chess_game_result(board: BaseChessBoard, draw_reward: float = -0.1) -> float:
    """Get the game result from a chess board."""
    return calculate_chess_reward(board, claim_draw=True, draw_reward=draw_reward)

def is_white_turn(board: BaseChessBoard) -> bool:
    """Check if it's white's turn on the chess board."""
    return board.turn == chess.WHITE

def get_chess_legal_actions(board: BaseChessBoard) -> list[int]:
    """Get legal moves from a chess board."""
    return board.legal_actions

def action_id_to_move(board: BaseChessBoard, action_id: int) -> chess.Move:
    """Convert action ID to move on chess board."""
    return board.action_id_to_move(action_id)

def create_board_from_fen(fen: str) -> BaseChessBoard:
    """Create a chess board from a FEN string."""
    board = LegacyChessBoard()
    board.set_fen(fen)
    return board 