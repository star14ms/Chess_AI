import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# from utils.analyze import interpret_tile
from chess_gym.chess_custom import FullyTrackedBoard
from interaction import InteractionBlock, NUM_INTERACTION_LAYERS
import hydra
from omegaconf import DictConfig, OmegaConf

# Default values based on common AlphaZero implementations for chess
DEFAULT_INPUT_CHANNELS = 1
DEFAULT_BOARD_SIZE = 8
DEFAULT_NUM_RESIDUAL_LAYERS = 7 # Number of convolutional stages (now all are ConvBlocks)
DEFAULT_RESIDUAL_BLOCKS_OUT_CHANNELS = [[64]*7] * DEFAULT_NUM_RESIDUAL_LAYERS
DEFAULT_ACTION_SPACE = 1700 # User-specified action space size
DEFAULT_NUM_PIECES = 32
DEFAULT_NUM_ATTENTION_HEADS = 4 # Heads for MultiheadCrossAttentionLayer
DEFAULT_DECODER_FF_DIM_MULT = 4
DEFAULT_POLICY_HIDDEN_SIZE = 256

class ConvBlock(nn.Module):
    """A single convolutional block with Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResidualConvBlock(nn.Module):
    def __init__(self, channel_list, kernel_size=7, padding=3):
        super().__init__()
        self.conv_blocks = nn.Sequential()
        for i in range(len(channel_list)-1):
            self.conv_blocks.append(ConvBlock(channel_list[i], channel_list[i+1], kernel_size, padding))

        self.last_block = nn.Sequential(
            nn.Conv2d(channel_list[-2], channel_list[-1], kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_list[-1])
        )

    def forward(self, x):
        return F.relu(self.last_block(self.conv_blocks(x)) + x)

# --- PieceVectorExtractor with Internal Raw Extractor Module ---
class PieceVectorExtractor(nn.Module):
    """Extracts raw piece features using an internal module and then projects them."""
    
    # --- Define Inner Module for Raw Extraction --- 
    class _RawExtractor(nn.Module):
        """Internal module to handle raw feature extraction loop."""
        def __init__(self, num_pieces=DEFAULT_NUM_PIECES):
            super().__init__()
            self.num_pieces = num_pieces
            
        def forward(self, full_board_vector: torch.Tensor, piece_ids: torch.Tensor) -> torch.Tensor:
            batch_size, num_input_channels, height, width = full_board_vector.shape # Use num_input_channels
            device = full_board_vector.device
            raw_piece_vector = torch.zeros(batch_size, self.num_pieces, num_input_channels, device=device)
            
            for b in range(batch_size):
                for piece_idx in range(self.num_pieces):
                    target_piece_id = piece_idx + 1
                    locations = torch.where(piece_ids[b] == target_piece_id)
                    if locations[0].numel() > 0:
                        rank_idx, file_idx = locations[0][0], locations[1][0]
                        features = full_board_vector[b, :, rank_idx, file_idx]
                        raw_piece_vector[b, piece_idx, :] = features
            return raw_piece_vector # Shape: (N, P, C_in)
            
    # --- Outer Module Initialization --- 
    def __init__(self, num_pieces=DEFAULT_NUM_PIECES):
        super().__init__()
        self._raw_extractor = self._RawExtractor(num_pieces)

    def forward(self, full_board_vector: torch.Tensor, piece_ids: torch.Tensor) -> torch.Tensor:
        raw_piece_vector = self._raw_extractor(full_board_vector, piece_ids.long())
        return raw_piece_vector

# --- Policy Head Module ---
class PolicyHead(nn.Module):
    """Calculates policy logits from the final piece vector."""
    def __init__(self, num_pieces: int, num_channels: int, action_space_size: int, policy_hidden_size: int):
        super().__init__()
        policy_input_size = num_pieces * (num_channels) # P * C + C_in
        self.policy_fc1 = nn.Linear(policy_input_size, policy_hidden_size)
        self.policy_fc2 = nn.Linear(policy_hidden_size, action_space_size)

    def forward(self, piece_vector: torch.Tensor) -> torch.Tensor:
        # Input shape: (N, P, C)
        N = piece_vector.size(0)
        policy_input = piece_vector.view(N, -1) # Shape: (N, P * C)
        hidden = F.relu(self.policy_fc1(policy_input))
        policy_logits = self.policy_fc2(hidden)
        return policy_logits

# --- Value Head Module ---
class ValueHead(nn.Module):
    """Calculates the value estimate from the final convolutional features."""
    def __init__(self, num_channels: int, board_height: int, board_width: int):
        super().__init__()
        # Input channels = C
        # Use a 1x1 Conv block to reduce channels to 1
        self.value_conv = ConvBlock(num_channels, 1, kernel_size=1, padding=0) 
        value_input_size = 1 * board_height * board_width 
        self.value_fc1 = nn.Linear(value_input_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, conv_features_4d: torch.Tensor) -> torch.Tensor:
        # Input shape: (N, C, H, W)
        # Apply 1x1 ConvBlock (returns 3D: N, H*W, 1)
        value_x_3d = self.value_conv(conv_features_4d)
        
        N = value_x_3d.size(0)
        value_x = value_x_3d.view(N, -1) # Shape: (N, H*W * 1)
        
        value_x = F.relu(self.value_fc1(value_x))
        value = torch.tanh(self.value_fc2(value_x)) # Shape: (N, 1)
        return value

# --- Main Chess Network --- 
class ChessNetwork(nn.Module):
    """Neural network architecture inspired by AlphaZero for Chess.

    Interaction:
      - PieceVectorExtractor creates initial (N, P, C_final) vector.
      - Conv body processes spatial inputs through multiple stages with potentially varying channels.
      - At each stage i, InteractionBlock updates piece_vector and conv_features using C_final.
      - Policy head operates on the final piece_vector (input features P * C_final).
      - Value head operates on final conv_features (input channels C_final).
    """
    def __init__(self,
                 input_channels=DEFAULT_INPUT_CHANNELS,
                 board_size=DEFAULT_BOARD_SIZE,
                 num_residual_layers=DEFAULT_NUM_RESIDUAL_LAYERS,
                 residual_blocks_out_channels=DEFAULT_RESIDUAL_BLOCKS_OUT_CHANNELS,
                 action_space_size=DEFAULT_ACTION_SPACE,
                 num_pieces=DEFAULT_NUM_PIECES,
                 num_attention_heads=DEFAULT_NUM_ATTENTION_HEADS,
                 decoder_ff_dim_mult=DEFAULT_DECODER_FF_DIM_MULT,
                 policy_hidden_size=DEFAULT_POLICY_HIDDEN_SIZE,
                 num_interaction_layers=NUM_INTERACTION_LAYERS):
        super().__init__()
        if input_channels < 2: raise ValueError("Input channels must be >= 2")
        if num_residual_layers < 1: raise ValueError("Must have at least 1 conv layer stage")

        self.board_height = board_size
        self.board_width = board_size
        self.action_space_size = action_space_size
        self.input_channels = input_channels # Total input channels
        self.conv_input_channels = input_channels - 1 # Channels for conv body
        self.num_residual_layers = num_residual_layers # Total number of conv stages
        self.num_pieces = num_pieces

        # --- Convolutional Configuration --- 
        if residual_blocks_out_channels is None or len(residual_blocks_out_channels) != num_residual_layers:
            raise ValueError(f"residual_blocks_out_channels must be a list of length num_residual_layers ({num_residual_layers})")

        # Determine final conv channels for downstream modules
        if not residual_blocks_out_channels or not residual_blocks_out_channels[-1]:
             raise ValueError("Last channel list in residual_blocks_out_channels cannot be empty")
        final_conv_channels = residual_blocks_out_channels[-1][-1]
        
        self.final_conv_channels = final_conv_channels # C_final

        # --- Piece Vector Extraction & Projection (Projects to C_final) ---
        self.piece_vector_extractor = PieceVectorExtractor(num_pieces=self.num_pieces)

        # --- Convolutional Body (All stages are ConvBlocks now) --- 
        self.first_conv_block = ConvBlock(self.conv_input_channels, residual_blocks_out_channels[0][0]-1)
        self.residual_blocks = nn.ModuleList()
        current_stage_in_channels = residual_blocks_out_channels[0][0] # Start with conv input channels
        for i in range(num_residual_layers): # Iterate num_residual_layers times
            ch_list = residual_blocks_out_channels[i]
            if not ch_list:
                 raise ValueError(f"Channel list for conv stage {i} cannot be empty")
            self.residual_blocks.append(
                ResidualConvBlock(channel_list=[current_stage_in_channels] + ch_list)
            )
            current_stage_in_channels = ch_list[-1] # Output of this stage is input for next

        # --- Interaction Blocks (d_model = C_final) --- 
        self.interaction_blocks = nn.ModuleList()
        for _ in range(num_residual_layers): # One interaction per conv stage
            self.interaction_blocks.append(InteractionBlock(
                d_model=self.final_conv_channels, 
                nhead=num_attention_heads,
                dim_feedforward=self.final_conv_channels * decoder_ff_dim_mult,
                board_height=self.board_height,
                board_width=self.board_width,
                num_layers=num_interaction_layers
            ))

        # --- Instantiate Head Modules (using C_final) --- 
        self.policy_head = PolicyHead(self.num_pieces, 
                                      self.final_conv_channels, # Use final channels
                                      self.action_space_size, 
                                      policy_hidden_size=policy_hidden_size)
        self.value_head = ValueHead(self.final_conv_channels, # Use final channels
                                    self.board_height, 
                                    self.board_width)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in_total, H, W). 
                                Assumes the last channel is piece IDs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: policy_logits (N, action_space_size), 
                                              value (N, 1)
        """
        N = x.shape[0]
        if x.shape[1] != self.input_channels: # Check against total input channels
             raise ValueError(f"Input tensor channel dimension ({x.shape[1]}) doesn't match expected total channels ({self.input_channels})")

        # --- Feature Extraction --- 
        conv_input = x[:, :self.conv_input_channels, :, :] # Shape: (N, C_in, H, W)
        piece_ids = x[:, -1, :, :].long() # Shape: (N, H, W)

        # --- Conv + Interaction Loop --- 
        # Initialize conv_features_4d for the loop
        # The first conv layer takes conv_input
        conv_features_4d = self.first_conv_block(conv_input) # (N, C_final, H, W)
        conv_features_4d = torch.cat((conv_features_4d, piece_ids.view(N, 1, self.board_height, self.board_width)), dim=1)
        piece_vector = self.piece_vector_extractor(conv_features_4d, piece_ids) # (N, P, C_final)

        # Collect the last dimension of piece_vector
        N, P, _ = piece_vector.shape
        piece_ids = piece_vector[:, :, -1].view(N, 1, P)

        num_stages = self.num_residual_layers
        for i in range(num_stages):
            # 1. Apply Convolutional Stage
            residual_block = self.residual_blocks[i]
            conv_features_4d = residual_block(conv_features_4d) # Output: (N, C_stage_out, H, W)

            # 2. Apply Interaction Block
            interaction_block = self.interaction_blocks[i]
            piece_vector, conv_features_4d = interaction_block(piece_vector, conv_features_4d)
            
            piece_ids = torch.cat((piece_ids, piece_vector[:, :, -1].view(N, 1, P)), dim=1)

        # --- Final Features --- 
        # Final piece_vector (N, P, C_final) for policy
        # Final conv_features_4d (N, C_final, H, W) for value

        # --- Call Head Modules --- 
        policy_logits = self.policy_head(piece_vector)
        value = self.value_head(conv_features_4d)

        return policy_logits, value, piece_ids


def test_network(cfg: DictConfig):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torchview_custom.torchview import draw_graphs
    from utils.profile import profile_model

    # Example instantiation with new config
    network = ChessNetwork(
        input_channels=cfg.network.input_channels,
        board_size=cfg.network.board_size,
        num_residual_layers=cfg.network.num_residual_layers,
        residual_blocks_out_channels=cfg.network.residual_blocks_out_channels,
        action_space_size=cfg.network.action_space_size,
        num_pieces=cfg.network.num_pieces,
    )
    print("Network Initialized.")
    print(f"Using: num_residual_layers={network.num_residual_layers}, final_conv_channels={network.final_conv_channels}, action_space={network.action_space_size}")

    board = FullyTrackedBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    dummy_input_tensor = torch.from_numpy(board.get_board_vector()).to(dtype=torch.float32)
    dummy_input_tensor = dummy_input_tensor.unsqueeze(0).repeat(2, 1, 1, 1)
    print("\nInput shape (before batching):", dummy_input_tensor.shape)

    network.eval()
    with torch.no_grad():
        # Test forward pass (forward expects tensor)
        policy_logits, value, _ = network(dummy_input_tensor)

        # Test piece vector extraction + projection (now done internally)
        piece_ids_internal = dummy_input_tensor[:, -1, :, :].long()
        # The extractor now directly outputs the projected vector
        final_projected_pv = network.piece_vector_extractor(dummy_input_tensor, piece_ids_internal)
        print("\nProjected Piece Vector shape (output from extractor):", final_projected_pv.shape) # (1, P, C)

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