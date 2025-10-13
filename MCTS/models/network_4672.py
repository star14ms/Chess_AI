import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import hydra
from omegaconf import DictConfig
from chess_gym.chess_custom import LegacyChessBoard

# Default values based on common AlphaZero implementations for chess
DEFAULT_INPUT_CHANNELS = 10  # 10 features (Color, Piece Type, EnPassant, Castling, Current Player)
DEFAULT_BOARD_SIZE = 8
DEFAULT_NUM_RESIDUAL_LAYERS = 8
DEFAULT_INITIAL_CONV_BLOCK_OUT_CHANNELS = [32] * 8
DEFAULT_RESIDUAL_BLOCKS_OUT_CHANNELS = [[32]] * DEFAULT_NUM_RESIDUAL_LAYERS
DEFAULT_ACTION_SPACE = 4672 # User-specified action space size
DEFAULT_NUM_PIECES = 32
DEFAULT_VALUE_HIDDEN_SIZE = 4
DEFAULT_POLICY_LINEAR_OUT_FEATURES = [4672]


class ConvBlock(nn.Module):
    """A single convolutional block with Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ConvBlockInitial(nn.Module):
    """A multi-scale convolutional block that captures both local and long-range patterns."""
    def __init__(self, in_channels, initial_conv_block_out_channels=[32, 32, 32, 32, 32, 32, 32, 32]):
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
            nn.Conv2d(channel_list[-2], channel_list[-1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel_list[-1])
        )

    def forward(self, x):
        return F.relu(self.last_block(self.conv_blocks(x)) + x)

# --- Convolutional Body Module ---
class ConvBody(nn.Module):
    """Encapsulates the initial conv stack and subsequent residual stages."""
    def __init__(self, input_channels: int, initial_conv_block_out_channels: list, residual_blocks_out_channels: list, num_residual_layers: int):
        super().__init__()
        if residual_blocks_out_channels is None or len(residual_blocks_out_channels) != num_residual_layers:
            raise ValueError(f"residual_blocks_out_channels must be a list of length num_residual_layers ({num_residual_layers})")

        self.first_conv_block = ConvBlockInitial(input_channels, initial_conv_block_out_channels=initial_conv_block_out_channels)
        self.initial_conv_block_out_channels_last_stage = initial_conv_block_out_channels[-1]

        self.residual_blocks = nn.ModuleList()
        if residual_blocks_out_channels:
            self.final_conv_channels = residual_blocks_out_channels[-1][-1]
            current_stage_in_channels = self.initial_conv_block_out_channels_last_stage
            for i in range(num_residual_layers):
                ch_list = residual_blocks_out_channels[i]
                if not ch_list:
                    raise ValueError(f"Channel list for conv stage {i} cannot be empty")
                self.residual_blocks.append(
                    ResidualConvBlock(channel_list=[current_stage_in_channels] + ch_list)
                )
                current_stage_in_channels = ch_list[-1]
        else:
            self.final_conv_channels = self.initial_conv_block_out_channels_last_stage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.first_conv_block(x)
        for block in self.residual_blocks:
            features = block(features)
        return features

# --- Policy Head Module ---
class PolicyHead(nn.Module):
    """Calculates policy logits from the final convolutional features.

    Supports either a single 1x1 conv reduction (legacy) or a configurable
    stack of residual convolutional blocks defined by channel lists.
    """
    def __init__(self,
                 in_channels: int,
                 action_space_size: int,
                 vec_height: int,
                 vec_width: int,
                 linear_out_features: list = DEFAULT_POLICY_LINEAR_OUT_FEATURES):
        super().__init__()
        # Default to [64] if not provided or empty
        self.use_residual_stack = True
        self.vec_height = vec_height
        self.vec_width = vec_width
        self.action_space_size = action_space_size
        
        # Residual conv stack for policy head
        final_channels = 2
        self.conv = ConvBlock(in_channels, final_channels, kernel_size=1, padding=0)

        # Build fully connected stack after conv features using provided out_features
        in_features = final_channels * vec_height * vec_width
        if not all(isinstance(f, int) and f > 0 for f in linear_out_features):
            raise ValueError(f"All elements of linear_out_features must be positive integers: {linear_out_features}")
        if linear_out_features[-1] != action_space_size:
            raise ValueError(f"The last out_features ({linear_out_features[-1]}) must equal action_space_size ({action_space_size})")
        layers = []
        layers.append(nn.Flatten())
        prev_features = in_features
        for idx, out_features in enumerate(linear_out_features):
            layers.append(nn.Linear(prev_features, out_features))
            # Add ReLU between layers, but not after the final output to action space
            if idx < len(linear_out_features) - 1:
                layers.append(nn.ReLU(inplace=True))
            prev_features = out_features
        self.fc = nn.Sequential(*layers)

    def forward(self, conv_features: torch.Tensor) -> torch.Tensor:
        x = self.conv(conv_features)
        policy_logits = self.fc(x)
        return policy_logits

# --- Value Head Module ---
class ValueHead(nn.Module):
    """Calculates the value estimate from the final convolutional features."""
    def __init__(self, num_channels: int, board_height: int, board_width: int, hidden_size: int = 256):
        super().__init__()
        # Input channels = C
        # Use a 1x1 Conv block to reduce channels to 1
        self.conv = ConvBlock(num_channels, 1, kernel_size=1, padding=0) 
        value_input_size = 1 * board_height * board_width 
        self.fc1 = nn.Linear(value_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (N, C, H, W)
        # Apply 1x1 ConvBlock (returns 3D: N, H*W, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1) # Shape: (N, H*W * 1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) # Shape: (N, 1)
        return x

# --- Main Chess Network --- 
class ChessNetwork4672(nn.Module):
    """Neural network architecture inspired by AlphaZero for Chess.

    Input channels:
    - 10 input channels:
        - Color (1)
        - BehaviorType (6)
        - EnPassantTarget (1)
        - CastlingTarget (1)
        - Current Player (1)
    """
    def __init__(self,
                 input_channels=DEFAULT_INPUT_CHANNELS,
                 board_size=DEFAULT_BOARD_SIZE,
                 num_residual_layers=DEFAULT_NUM_RESIDUAL_LAYERS,
                 initial_conv_block_out_channels=DEFAULT_INITIAL_CONV_BLOCK_OUT_CHANNELS,
                 residual_blocks_out_channels=DEFAULT_RESIDUAL_BLOCKS_OUT_CHANNELS,
                 action_space_size=DEFAULT_ACTION_SPACE,
                 num_pieces=DEFAULT_NUM_PIECES,
                 value_head_hidden_size=DEFAULT_VALUE_HIDDEN_SIZE,
                 policy_linear_out_features: list | None = DEFAULT_POLICY_LINEAR_OUT_FEATURES,
                ):
        super().__init__()
        self.board_height = board_size
        self.board_width = board_size
        self.action_space_size = action_space_size
        self.input_channels = input_channels
        self.num_residual_layers = num_residual_layers
        self.num_pieces = num_pieces
        # --- Convolutional Body ---
        self.body = ConvBody(self.input_channels, initial_conv_block_out_channels, residual_blocks_out_channels, num_residual_layers)
        self.final_conv_channels = self.body.final_conv_channels

        # --- Instantiate Head Modules (using C_final) --- 
        self.policy_head = PolicyHead(
            self.final_conv_channels,
            self.action_space_size,
            self.board_height,
            self.board_width,
            linear_out_features=policy_linear_out_features,
        )
        self.value_head = ValueHead(self.final_conv_channels, self.board_height, self.board_width, hidden_size=value_head_hidden_size)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 10, H, W)
                            10 channels for board features

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: policy_logits (N, action_space_size), 
                                              value (N, 1)
        """
        N = x.shape[0]
        if x.shape[1] != self.input_channels:
             raise ValueError(f"Input tensor channel dimension ({x.shape[1]}) doesn't match expected channels ({self.input_channels})")

        # --- Feature Extraction via body --- 
        conv_features = self.body(x)

        policy_logits = self.policy_head(conv_features)
        value = self.value_head(conv_features)

        return policy_logits, value


def test_network(cfg: DictConfig):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torchview_custom.torchview import draw_graphs
    from utils.profile import profile_model
    # from utils.analyze import interpret_tile

    # Example instantiation with new config
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
    )
    print("Network Initialized.")
    print(f"Using: num_residual_layers={network.num_residual_layers}, final_conv_channels={network.final_conv_channels}, action_space={network.action_space_size}")

    board = LegacyChessBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    # from gym_gomoku.envs import Board
    # board = Board(15)
    # Use env-configured history steps if available
    history_steps = getattr(cfg.env, 'history_steps', 8)
    dummy_input_tensor = torch.from_numpy(board.get_board_vector(history_steps=history_steps)).to(dtype=torch.float32)
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