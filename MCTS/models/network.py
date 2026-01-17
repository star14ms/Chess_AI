import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from chess_gym.chess_custom import FullyTrackedBoard
import hydra
from omegaconf import DictConfig, OmegaConf

# Default values based on common AlphaZero implementations for chess
DEFAULT_INPUT_CHANNELS = 26  # 10 (features) + 16 (piece type info)
DEFAULT_DIM_PIECE_TYPE = 16
DEFAULT_BOARD_SIZE = 8
DEFAULT_NUM_RESIDUAL_LAYERS = 0
DEFAULT_INITIAL_CONV_BLOCK_OUT_CHANNELS = [32, 32, 64, 64, 128, 128, 128]
DEFAULT_RESIDUAL_BLOCKS_OUT_CHANNELS = [] * DEFAULT_NUM_RESIDUAL_LAYERS
DEFAULT_ACTION_SPACE = 1700 # User-specified action space size
DEFAULT_NUM_PIECES = 32
DEFAULT_VALUE_HIDDEN_SIZE = 64

class ConvBlock(nn.Module):
    """A single convolutional block with Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, policy_only: bool = False):
        return F.relu(self.bn(self.conv(x)))

class ConvBlockInitial(nn.Module):
    """A multi-scale convolutional block that captures both local and long-range patterns."""
    def __init__(self, in_channels, initial_conv_block_out_channels=[16, 32, 64, 128, 128, 128, 128]):
        super().__init__()
        self.conv_paths = nn.ModuleList()
        self.bn_paths = nn.ModuleList()

        initial_conv_block_out_channels = [in_channels] + initial_conv_block_out_channels
        self.conv_blocks = nn.Sequential()
        for i in range(len(initial_conv_block_out_channels)-1):
            self.conv_blocks.append(ConvBlock(initial_conv_block_out_channels[i], initial_conv_block_out_channels[i+1], kernel_size=3, padding=1))
            
        self.bn = nn.BatchNorm2d(initial_conv_block_out_channels[-1])

    def forward(self, x):
        return F.relu(self.bn(self.conv_blocks(x)))

class ResidualConvBlock(nn.Module):
    def __init__(self, channel_list, kernel_size=3, padding=1):
        super().__init__()
        self.conv_blocks = nn.Sequential()
        for i in range(len(channel_list)-1):
            self.conv_blocks.append(ConvBlock(channel_list[i], channel_list[i+1], kernel_size, padding))

        self.last_block = nn.Sequential(
            nn.Conv2d(channel_list[-2], channel_list[-1], kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(channel_list[-1])
        )

    def forward(self, x):
        return F.relu(self.last_block(self.conv_blocks(x)) + x)

# --- Policy Head Module ---
class PolicyHead(nn.Module):
    """Calculates policy logits from the final piece vector."""
    def __init__(
        self,
        num_channels: int,
        dim_piece_type: int,
        action_space_size: int,
        board_height: int,
        board_width: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = ConvBlock(num_channels+dim_piece_type+1, 128)  # Directly reduce channels including piece type
        self.fc = nn.Sequential(
            nn.Linear(128*board_height*board_width, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(128, action_space_size)
        )

    def forward(self, conv_features: torch.Tensor, piece_type: torch.Tensor) -> torch.Tensor:
        # Input shape: (N, P, C)
        N = conv_features.size(0)

        # Merge piece type with position information
        conv_features = torch.cat([conv_features, piece_type], dim=1)
        conv_features = self.conv(conv_features)

        # Continue with normal processing
        conv_features = conv_features.view(N, -1)  # Shape: (N, P * C)
        policy_logits = self.fc(conv_features)
        return policy_logits

# --- Value Head Module ---
class ValueHead(nn.Module):
    """Calculates the value estimate from the final convolutional features."""
    def __init__(self, num_channels: int, board_height: int, board_width: int, hidden_size: int = 256):
        super().__init__()
        # Input channels = C
        # Use a 1x1 Conv block to reduce channels to 1
        self.value_conv = ConvBlock(num_channels, 1, kernel_size=1, padding=0) 
        value_input_size = 1 * board_height * board_width 
        self.value_fc1 = nn.Linear(value_input_size, hidden_size)
        self.value_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, conv_features: torch.Tensor) -> torch.Tensor:
        # Input shape: (N, C, H, W)
        # Apply 1x1 ConvBlock (returns 3D: N, H*W, 1)
        value_x_3d = self.value_conv(conv_features)
        
        N = value_x_3d.size(0)
        value_x = value_x_3d.view(N, -1) # Shape: (N, H*W * 1)
        
        value_x = F.relu(self.value_fc1(value_x))
        value = torch.tanh(self.value_fc2(value_x)) # Shape: (N, 1)
        return value

# --- Main Chess Network --- 
class ChessNetwork(nn.Module):
    """Neural network architecture inspired by AlphaZero for Chess.

    Input channels:
    - 26 input channels:
        - 10 channels for features:
            - Color (1)
            - BehaviorType (6)
            - EnPassantTarget (1)
            - CastlingTarget (1)
            - Current Player (1)
        - 16 channels for piece type information:
            - 16 channels for one-hot piece identity for each color (16)
    """
    def __init__(self,
                 input_channels=DEFAULT_INPUT_CHANNELS,
                 dim_piece_type=DEFAULT_DIM_PIECE_TYPE,
                 board_size=DEFAULT_BOARD_SIZE,
                 num_residual_layers=DEFAULT_NUM_RESIDUAL_LAYERS,
                 initial_conv_block_out_channels=DEFAULT_INITIAL_CONV_BLOCK_OUT_CHANNELS,
                 residual_blocks_out_channels=DEFAULT_RESIDUAL_BLOCKS_OUT_CHANNELS,
                 action_space_size=DEFAULT_ACTION_SPACE,
                 num_pieces=DEFAULT_NUM_PIECES,
                 value_head_hidden_size=DEFAULT_VALUE_HIDDEN_SIZE,
                 policy_dropout: float = 0.0,
                ):
        super().__init__()
        self.board_height = board_size
        self.board_width = board_size
        self.action_space_size = action_space_size
        self.input_channels = input_channels # Total input channels
        self.conv_input_channels = input_channels - dim_piece_type # Channels for conv body
        self.num_residual_layers = num_residual_layers # Total number of conv stages
        self.num_pieces = num_pieces
        self.initial_conv_block_out_channels_last_stage = initial_conv_block_out_channels[-1]

        # --- Convolutional Configuration --- 
        if residual_blocks_out_channels is None or len(residual_blocks_out_channels) != num_residual_layers:
            raise ValueError(f"residual_blocks_out_channels must be a list of length num_residual_layers ({num_residual_layers})")

        self.first_conv_block = ConvBlockInitial(self.conv_input_channels, initial_conv_block_out_channels=initial_conv_block_out_channels)

        if residual_blocks_out_channels:
            self.final_conv_channels = residual_blocks_out_channels[-1][-1]

            # --- Convolutional Body (All stages are ConvBlocks now) --- 
            self.residual_blocks = nn.ModuleList()
            current_stage_in_channels = self.initial_conv_block_out_channels_last_stage # Start with conv input channels
            for i in range(num_residual_layers): # Iterate num_residual_layers times
                ch_list = residual_blocks_out_channels[i]
                if not ch_list:
                    raise ValueError(f"Channel list for conv stage {i} cannot be empty")
                self.residual_blocks.append(
                    ResidualConvBlock(channel_list=[current_stage_in_channels] + ch_list)
                )
                current_stage_in_channels = ch_list[-1] # Output of this stage is input for next
        else:
            self.final_conv_channels = self.initial_conv_block_out_channels_last_stage

        # --- Instantiate Head Modules (using C_final) --- 
        self.policy_head = PolicyHead(
            self.final_conv_channels,
            dim_piece_type,
            self.action_space_size,
            self.board_height,
            self.board_width,
            dropout=policy_dropout,
        )
        self.value_head = ValueHead(self.final_conv_channels, self.board_height, self.board_width, hidden_size=value_head_hidden_size)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in_total, H, W)
                            Last 16 channels are piece type information

        Returns:
            Tuple[torch.Tensor, torch.Tensor | None]: policy_logits (N, action_space_size),
                                                     value (N, 1) or None if policy_only
        """
        N = x.shape[0]
        if x.shape[1] != self.input_channels:
             raise ValueError(f"Input tensor channel dimension ({x.shape[1]}) doesn't match expected total channels ({self.input_channels})")

        # Split input into features and piece type information
        features = x[:, :self.conv_input_channels, :, :]
        piece_type = torch.cat([x[:, :1, :, :], x[:, self.conv_input_channels:, :, :]], dim=1)  # Already in shape [N, 17, H, W]

        # --- Feature Extraction --- 
        conv_features = self.first_conv_block(features)

        num_stages = self.num_residual_layers
        for i in range(num_stages):
            residual_block = self.residual_blocks[i]
            conv_features = residual_block(conv_features)

        policy_logits = self.policy_head(conv_features, piece_type)
        if policy_only:
            return policy_logits, None
        value = self.value_head(conv_features)

        return policy_logits, value


def test_network(cfg: DictConfig):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torchview_custom.torchview import draw_graphs
    from utils.profile_model import profile_model
    # from utils.analyze import interpret_tile

    # Example instantiation with new config
    network = ChessNetwork(
        input_channels=cfg.network.input_channels,
        dim_piece_type=cfg.network.dim_piece_type,
        board_size=cfg.network.board_size,
        num_residual_layers=cfg.network.num_residual_layers,
        initial_conv_block_out_channels=cfg.network.initial_conv_block_out_channels,
        residual_blocks_out_channels=cfg.network.residual_blocks_out_channels,
        action_space_size=cfg.network.action_space_size,
        num_pieces=cfg.network.num_pieces,
        value_head_hidden_size=cfg.network.value_head_hidden_size
    )
    print("Network Initialized.")
    print(f"Using: num_residual_layers={network.num_residual_layers}, final_conv_channels={network.final_conv_channels}, action_space={network.action_space_size}")

    board = FullyTrackedBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    dummy_input_tensor = torch.from_numpy(board.get_board_vector()).to(dtype=torch.float32)
    dummy_input_tensor = dummy_input_tensor.unsqueeze(0).repeat(2, 1, 1, 1)
    # for i in range(dummy_input_tensor.shape[2]):
    #     for j in range(dummy_input_tensor.shape[3]):
    #         print(dummy_input_tensor[0, 10:, i, j], interpret_tile(dummy_input_tensor[0, :, i, j]))
    print("\nInput shape (before batching):", dummy_input_tensor.shape)

    network.eval()
    with torch.no_grad():
        # Test forward pass (forward expects tensor)
        policy_logits, value = network(dummy_input_tensor)

    print("\nOutput Policy Logits shape (single input):", policy_logits.shape) # (action_space_size)
    print("Output Value (single input):", value) # scalar

    # Profile requires batched input
    profile_model(network, (dummy_input_tensor,))
    draw_graphs(network, (dummy_input_tensor,), 
                min_depth=1, max_depth=3, 
                output_names=['Policy', 'Value', 'Piece IDs'], 
                input_names=['Input'], 
                directory='./model_viz/', 
                hide_module_functions=True, 
                hide_inner_tensors=True,
                print_code_path=False)
    print("\nModel graph saved to ./model_viz/")


# --- Hydra Entry Point --- 
# Ensure config_path points to the directory containing train_mcts.yaml
@hydra.main(config_path="../../config", config_name="train_mcts", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Configuration:\n")
    # Use OmegaConf.to_yaml for structured printing
    # print(OmegaConf.to_yaml(cfg))
    test_network(cfg)


if __name__ == "__main__":
    main()