import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# from utils.analyze import interpret_tile
from chess_gym.chess_custom import FullyTrackedBoard
from models.cross_attn import MultiheadCrossAttentionLayer

# Default values based on common AlphaZero implementations for chess
DEFAULT_INPUT_CHANNELS = 11
DEFAULT_BOARD_SIZE = 8
DEFAULT_NUM_CONV_LAYERS = 5 # Number of convolutional stages (now all are ConvBlocks)
DEFAULT_CONV_BLOCKS_CHANNEL_LISTS = [[64]*7] * DEFAULT_NUM_CONV_LAYERS
DEFAULT_ACTION_SPACE = 1700 # User-specified action space size
DEFAULT_NUM_PIECES = 32
DEFAULT_NUM_ATTENTION_HEADS = 4 # Heads for MultiheadCrossAttentionLayer
DEFAULT_DECODER_FF_DIM_MULT = 4
DEFAULT_POLICY_HIDDEN_SIZE = 256

# --- Redefined ConvBlock (4D -> 4D) ---
class ConvBlock(nn.Module):
    """A single convolutional block with Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

# --- New ConvBlocks Module --- 
class ConvBlocks(nn.Module):
    """A sequence of ConvBlock layers with specified channel sizes."""
    def __init__(self, initial_in_channels: int, channel_list: list[int]):
        super().__init__()
        if not channel_list:
            raise ValueError("channel_list cannot be empty")
            
        layers = []
        num_layers = len(channel_list)
        current_in_channels = initial_in_channels
        
        for i in range(num_layers):
            out_channels = channel_list[i]
            layers.append(ConvBlock(current_in_channels, out_channels))
            current_in_channels = out_channels # Input for next layer is output of current
            
        self.blocks = nn.ModuleList(layers)
        
    def forward(self, x):
        # Input x is 4D, output is 4D
        for block in self.blocks:
            x = block(x)

        return x

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
    def __init__(self, num_pieces=DEFAULT_NUM_PIECES, input_channels=DEFAULT_INPUT_CHANNELS, output_channels=DEFAULT_CONV_BLOCKS_CHANNEL_LISTS[0][0]):
        super().__init__()
        self._raw_extractor = self._RawExtractor(num_pieces)
        # Projects from C_in to C_out
        self.projection = nn.Linear(input_channels, output_channels) 

    def forward(self, full_board_vector: torch.Tensor, piece_ids: torch.Tensor) -> torch.Tensor:
        raw_piece_vector = self._raw_extractor(full_board_vector, piece_ids.long())
        projected_piece_vector = self.projection(raw_piece_vector) # Shape (B, P, C)
        return projected_piece_vector

# --- Interaction Block (Keep as is) ---
class BiDirectionalInteractionBlock(nn.Module):
    """Performs forward and reverse attention between piece and conv features."""
    def __init__(self, d_model, nhead, dim_feedforward, board_height, board_width):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        
        # Forward layer (updates piece vec)
        self.interaction_layer_fwd = MultiheadCrossAttentionLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True
        )
        # Reverse layer (updates conv features)
        self.interaction_layer_rev = MultiheadCrossAttentionLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True
        )
        # Norm for reverse pass (applied to conv features, no residual)
        self.norm_rev = nn.LayerNorm(d_model)

    def forward(self, projected_piece_vector, kv_permuted):
        """
        Args:
            projected_piece_vector: (N, P, C)
            kv_permuted: (N, H*W, C) - Context features

        Returns:
            updated_piece_vector: (N, P, C)
            updated_conv_features: (N, C, H, W) - Note: Returns 4D spatial map
        """
        # Reshape 4D conv features to 3D for interaction block
        N, C_curr, H, W = kv_permuted.shape
        kv_permuted = kv_permuted.view(N, C_curr, -1).permute(0, 2, 1) # (N, H*W, C)

        updated_piece_vector = self.interaction_layer_fwd(projected_piece_vector, kv_permuted)
        interaction_output_rev = self.interaction_layer_rev(tgt=kv_permuted, memory=updated_piece_vector)
        updated_kv_permuted = self.norm_rev(interaction_output_rev) # Shape (N, H*W, C)

        N, L, C_out = updated_kv_permuted.shape 
        H, W = self.board_height, self.board_width 
        updated_conv_features = updated_kv_permuted.permute(0, 2, 1).view(N, C_out, H, W) # Shape (N, C, H, W)

        return updated_piece_vector, updated_conv_features

# --- Policy Head Module ---
class PolicyHead(nn.Module):
    """Calculates policy logits from the final piece vector."""
    def __init__(self, num_pieces: int, num_channels: int, action_space_size: int, policy_hidden_size: int):
        super().__init__()
        policy_input_size = num_pieces * num_channels # P * C
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
        self.value_fc1 = nn.Linear(value_input_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

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
      - At each stage i, BiDirectionalInteractionBlock updates piece_vector and conv_features using C_final.
      - Policy head operates on final piece_vector (input features P * C_final).
      - Value head operates on final conv_features (input channels C_final).
    """
    def __init__(self,
                 input_channels=DEFAULT_INPUT_CHANNELS,
                 board_size=DEFAULT_BOARD_SIZE,
                 num_conv_layers=DEFAULT_NUM_CONV_LAYERS,
                 conv_blocks_channel_lists=DEFAULT_CONV_BLOCKS_CHANNEL_LISTS,
                 action_space_size=DEFAULT_ACTION_SPACE,
                 num_pieces=DEFAULT_NUM_PIECES,
                 num_attention_heads=DEFAULT_NUM_ATTENTION_HEADS,
                 decoder_ff_dim_mult=DEFAULT_DECODER_FF_DIM_MULT,
                 policy_hidden_size=DEFAULT_POLICY_HIDDEN_SIZE):
        super().__init__()
        if input_channels < 2: raise ValueError("Input channels must be >= 2")
        if num_conv_layers < 1: raise ValueError("Must have at least 1 conv layer stage")

        self.board_height = board_size
        self.board_width = board_size
        self.action_space_size = action_space_size
        self.input_channels = input_channels # Total input channels
        self.conv_input_channels = input_channels - 1 # Channels for conv body
        self.num_conv_layers = num_conv_layers # Total number of conv stages
        self.num_pieces = num_pieces

        # --- Convolutional Configuration --- 
        if conv_blocks_channel_lists is None or len(conv_blocks_channel_lists) != num_conv_layers:
            raise ValueError(f"conv_blocks_channel_lists must be a list of length num_conv_layers ({num_conv_layers})")

        # Determine final conv channels for downstream modules
        if not conv_blocks_channel_lists or not conv_blocks_channel_lists[-1]:
             raise ValueError("Last channel list in conv_blocks_channel_lists cannot be empty")
        final_conv_channels = conv_blocks_channel_lists[-1][-1]
        
        self.final_conv_channels = final_conv_channels # C_final

        # --- Piece Vector Extraction & Projection (Projects to C_final) ---
        self.piece_vector_extractor = PieceVectorExtractor(
            num_pieces=self.num_pieces, 
            input_channels=self.input_channels, 
            output_channels=self.final_conv_channels # Output matches final conv channels
        )

        # --- Convolutional Body (All stages are ConvBlocks now) --- 
        self.conv_layers = nn.ModuleList()
        current_stage_in_channels = self.conv_input_channels # Start with conv input channels
        for i in range(num_conv_layers): # Iterate num_conv_layers times
            ch_list = conv_blocks_channel_lists[i]
            if not ch_list:
                 raise ValueError(f"Channel list for conv stage {i} cannot be empty")
            self.conv_layers.append(
                ConvBlocks(initial_in_channels=current_stage_in_channels, channel_list=ch_list)
            )
            current_stage_in_channels = ch_list[-1] # Output of this stage is input for next

        # --- Interaction Blocks (d_model = C_final) --- 
        self.interaction_blocks = nn.ModuleList()
        for _ in range(num_conv_layers): # One interaction per conv stage
            self.interaction_blocks.append(BiDirectionalInteractionBlock(
                d_model=self.final_conv_channels, 
                nhead=num_attention_heads,
                dim_feedforward=self.final_conv_channels * decoder_ff_dim_mult,
                board_height=self.board_height,
                board_width=self.board_width
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
        projected_piece_vector = self.piece_vector_extractor(x, piece_ids) # (N, P, C_final)

        # --- Conv + Interaction Loop --- 
        # Initialize conv_features_4d for the loop
        # The first conv layer takes conv_input
        conv_features_4d = None # Will be assigned in the first iteration
        
        num_stages = self.num_conv_layers
        for i in range(num_stages):
            # 1. Apply Convolutional Stage
            conv_stage = self.conv_layers[i]
            # Use conv_input for the first stage, output of previous interaction otherwise
            input_features = conv_input if i == 0 else conv_features_4d 
            conv_features_4d = conv_stage(input_features) # Output: (N, C_stage_out, H, W)
            
            # Ensure final channel size for interaction if intermediate stages vary
            # (This assumes interaction blocks always use final_conv_channels)
            # If channel sizes *within* a stage or *between* stages vary, 
            # the interaction d_model must match the *output* of the current conv stage.
            # Current design assumes d_model=final_conv_channels, which might be incorrect
            # if intermediate conv stages output different channel sizes. 
            # For now, we proceed assuming the interaction block's d_model matches the *final* conv channel size.
            # A check or projection might be needed if intermediate sizes differ from final size.
            if i < num_stages - 1 and conv_features_4d.shape[1] != self.final_conv_channels:
                # This indicates a potential mismatch if interaction d_model is fixed to final_conv_channels
                print(f"Warning: Conv stage {i} output channels ({conv_features_4d.shape[1]}) differ from final channels ({self.final_conv_channels}). Interaction block d_model might mismatch.")
                # Consider adding a projection layer here if needed
            elif i == num_stages - 1 and conv_features_4d.shape[1] != self.final_conv_channels:
                 raise ValueError(f"Final conv stage output channels ({conv_features_4d.shape[1]}) must match final_conv_channels ({self.final_conv_channels}) for heads.")

            # 2. Apply Interaction Block
            interaction_block = self.interaction_blocks[i]
            projected_piece_vector, conv_features_4d = interaction_block(
                projected_piece_vector, conv_features_4d
            ) # Interaction block outputs features with d_model channels (assumed = final_conv_channels)

        # --- Final Features --- 
        # Final projected_piece_vector (N, P, C_final) for policy
        # Final conv_features_4d (N, C_final, H, W) for value

        # --- Call Head Modules --- 
        policy_logits = self.policy_head(projected_piece_vector)
        value = self.value_head(conv_features_4d)

        return policy_logits, value


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torchview_custom.torchview import draw_graphs
    from utils.profile import profile_model

    # Example instantiation with new config
    network = ChessNetwork()
    print("Network Initialized.")
    print(f"Using: num_conv_layers={network.num_conv_layers}, final_conv_channels={network.final_conv_channels}, action_space={network.action_space_size}")

    board = FullyTrackedBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    dummy_input_tensor = torch.from_numpy(board.get_board_vector()).to(dtype=torch.float32)
    dummy_input_tensor = dummy_input_tensor.unsqueeze(0).repeat(2, 1, 1, 1)
    print("\nInput shape (before batching):", dummy_input_tensor.shape)

    network.eval()
    with torch.no_grad():
        # Test forward pass (forward expects tensor)
        policy_logits, value = network(dummy_input_tensor)

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
                output_names=['Policy', 'Value'], 
                input_names=['Input'], 
                directory='./model_viz/', 
                hide_module_functions=True, 
                hide_inner_tensors=True,
                print_code_path=False)
    print("\nModel graph saved to ./model_viz/")
