import hydra
from omegaconf import DictConfig

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), '.'))
if project_root not in sys.path:
    sys.path.append(project_root)

from MCTS.models.network_4672 import ConvBlock, ResidualConvBlock, PolicyHead, ValueHead
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_INPUT_CHANNELS = 2
DEFAULT_BOARD_HEIGHT = 160
DEFAULT_BOARD_WIDTH = 144
DEFAULT_NUM_RESIDUAL_LAYERS = 8
DEFAULT_INITIAL_CONV_BLOCK_OUT_CHANNELS = [16, 32, 32, 32]
DEFAULT_RESIDUAL_BLOCKS_OUT_CHANNELS = [[32]] * DEFAULT_NUM_RESIDUAL_LAYERS
DEFAULT_ACTION_SPACE = 4
DEFAULT_POLICY_OUT_CHANNELS = 2
DEFAULT_VALUE_HEAD_HIDDEN_SIZE = 32


class ConvBlockInitial(nn.Module):
    """A multi-scale convolutional block that captures both local and long-range patterns."""
    def __init__(self, in_channels, initial_conv_block_out_channels=[16, 32, 32, 32]):
        super().__init__()
        self.conv_paths = nn.ModuleList()
        self.bn_paths = nn.ModuleList()

        initial_conv_block_out_channels = [in_channels] + initial_conv_block_out_channels
        self.conv_blocks = nn.ModuleList()
        self.pool_blocks = nn.ModuleList()
        
        for i in range(len(initial_conv_block_out_channels)-1):
            self.conv_blocks.append(ConvBlock(initial_conv_block_out_channels[i], initial_conv_block_out_channels[i+1], kernel_size=3, padding=1))
            self.pool_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.bn = nn.BatchNorm2d(initial_conv_block_out_channels[-1])

    def forward(self, x):
        for conv, pool in zip(self.conv_blocks, self.pool_blocks):
            x = conv(x)
            x = pool(x)
            # print(x.shape)
        return F.relu(self.bn(x))


class BreakoutNetwork(nn.Module):
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
                 board_height=DEFAULT_BOARD_HEIGHT,
                 board_width=DEFAULT_BOARD_WIDTH,
                 num_residual_layers=DEFAULT_NUM_RESIDUAL_LAYERS,
                 initial_conv_block_out_channels=DEFAULT_INITIAL_CONV_BLOCK_OUT_CHANNELS,
                 residual_blocks_out_channels=DEFAULT_RESIDUAL_BLOCKS_OUT_CHANNELS,
                 action_space_size=DEFAULT_ACTION_SPACE,
                 policy_head_out_channels=DEFAULT_POLICY_OUT_CHANNELS,
                 value_head_hidden_size=DEFAULT_VALUE_HEAD_HIDDEN_SIZE
                ):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        self.action_space_size = action_space_size
        self.input_channels = input_channels
        self.num_residual_layers = num_residual_layers
        self.initial_conv_block_out_channels_last_stage = initial_conv_block_out_channels[-1]

        # --- Convolutional Configuration --- 
        if residual_blocks_out_channels is None or len(residual_blocks_out_channels) != num_residual_layers:
            raise ValueError(f"residual_blocks_out_channels must be a list of length num_residual_layers ({num_residual_layers})")

        self.first_conv_block = ConvBlockInitial(self.input_channels, initial_conv_block_out_channels=initial_conv_block_out_channels)

        if residual_blocks_out_channels:
            self.final_conv_channels = residual_blocks_out_channels[-1][-1]

            # --- Convolutional Body (All stages are ConvBlocks now) --- 
            self.residual_blocks = nn.ModuleList()
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

        # --- Instantiate Head Modules (using C_final) --- 
        self.policy_head = PolicyHead(self.final_conv_channels, policy_head_out_channels, self.action_space_size, self.board_height//(2**len(initial_conv_block_out_channels)), self.board_width//(2**len(initial_conv_block_out_channels)))
        self.value_head = ValueHead(self.final_conv_channels, self.board_height//(2**len(initial_conv_block_out_channels)), self.board_width//(2**len(initial_conv_block_out_channels)), hidden_size=value_head_hidden_size)

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

        # --- Feature Extraction --- 
        conv_features = self.first_conv_block(x)

        num_stages = self.num_residual_layers
        for i in range(num_stages):
            residual_block = self.residual_blocks[i]
            conv_features = residual_block(conv_features)

        policy_logits = self.policy_head(conv_features)
        value = self.value_head(conv_features)

        return policy_logits, value


# --- Test/Visualization Entrypoint ---
def test_network(cfg):
    import torch
    import gymnasium as gym
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from MCTS_nonboard.training_modules.breakout import CroppedBreakoutEnv
    from torchview_custom.torchview import draw_graphs
    from utils.profile_model import profile_model

    # Example instantiation with new config
    network = BreakoutNetwork(
        input_channels=cfg.network.input_channels,
        board_height=cfg.network.get("board_height", DEFAULT_BOARD_HEIGHT),
        board_width=cfg.network.get("board_width", DEFAULT_BOARD_WIDTH),
        num_residual_layers=cfg.network.num_residual_layers,
        initial_conv_block_out_channels=cfg.network.initial_conv_block_out_channels,
        residual_blocks_out_channels=cfg.network.residual_blocks_out_channels,
        action_space_size=cfg.network.action_space_size,
        policy_head_out_channels=cfg.network.policy_head_out_channels,
        value_head_hidden_size=cfg.network.value_head_hidden_size
    )
    print("Network Initialized.")
    print(f"Using: num_residual_layers={network.num_residual_layers}, final_conv_channels={network.final_conv_channels}, action_space={network.action_space_size}")

    env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
    env = CroppedBreakoutEnv(env)
    observation, info = env.reset()

    dummy_input_tensor = torch.from_numpy(observation).to(dtype=torch.float32)
    dummy_input_tensor = dummy_input_tensor.unsqueeze(0).repeat(2, 1, 1, 1)
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
                directory='./../model_viz/', 
                hide_module_functions=True, 
                hide_inner_tensors=True,
                print_code_path=False)
    print("\nModel graph saved to ./../model_viz/")


@hydra.main(config_path="../../config", config_name="train_breakout", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Configuration:\n")
    test_network(cfg)


if __name__ == "__main__":
    main()