import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# from utils.analyze import interpret_tile
from chess_gym.chess_custom import FullyTrackedBoard

# Default values based on common AlphaZero implementations for chess
DEFAULT_INPUT_CHANNELS = 10
DEFAULT_BOARD_SIZE = 8
DEFAULT_NUM_CONV_LAYERS = 4
DEFAULT_FILTERS = 24
DEFAULT_ACTION_SPACE = 16 # User-specified action space size
DEFAULT_NUM_PIECES = 32
DEFAULT_NUM_ATTENTION_HEADS = 4 # Heads for TransformerDecoderLayer

class ConvBlock(nn.Module):
    """A single convolutional block with Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Standard 4D output
        x = F.relu(self.bn(self.conv(x)))
        
        # Reshaped 3D version for attention
        N, C_filt, H, W = x.shape
        x = x.view(N, C_filt, -1).permute(0, 2, 1) # Shape: (N, H*W, F)
        
        return x

class PieceVectorExtractor(nn.Module):
    """Extracts initial piece vectors and projects them to the target dimension."""
    def __init__(self, num_pieces=DEFAULT_NUM_PIECES, input_channels=DEFAULT_INPUT_CHANNELS, output_channels=DEFAULT_FILTERS):
        super().__init__()
        self.num_pieces = num_pieces
        self.projection = nn.Linear(input_channels, output_channels)

    def forward(self, full_board_vector, piece_ids):
        """Creates the initial (N, P, C) piece vector and projects it to (N, P, F)."""
        batch_size, num_channels, height, width = full_board_vector.shape
        device = full_board_vector.device

        # Initialize with zeros
        raw_piece_vector = torch.zeros(batch_size, self.num_pieces, num_channels, device=device)

        # Loop through batch and pieces
        for b in range(batch_size):
            for piece_idx in range(self.num_pieces):
                target_piece_id = piece_idx + 1
                # Find location(s) for this piece in this batch
                locations = torch.where(piece_ids[b] == target_piece_id)
                if locations[0].numel() > 0:
                    # Take the first location found
                    rank_idx, file_idx = locations[0][0], locations[1][0]
                    # Extract features for this piece
                    features = full_board_vector[b, :, rank_idx, file_idx] # Shape (C,)
                    # Assign features to the correct slot
                    raw_piece_vector[b, piece_idx, :] = features

        # Apply projection after collecting all features
        projected_piece_vector = self.projection(raw_piece_vector) # Shape (B, P, F)
        return projected_piece_vector
    
    # def forward(self, full_board_vector, piece_ids):
    #     """Creates the initial (N, P, C) piece vector and projects it to (N, P, F)."""
    #     batch_size, num_channels, height, width = full_board_vector.shape
    #     full_board_vector = full_board_vector.view(batch_size, num_channels, -1).transpose(1, 2)
    #     piece_ids = piece_ids.view(batch_size, -1)
    #     # Initial vector with input channels
    #     indices = []
        
    #     for b in range(batch_size):
    #         for piece_idx in range(self.num_pieces):
    #             target_piece_id = piece_idx + 1
    #             locations = torch.where(piece_ids[b] == target_piece_id)
    #             indices.append(locations)

    #         # Extract values directly from full_board_vector using advanced indexing
    #         raw_piece_vector = full_board_vector[b, indices]
        
    #     # Apply projection before returning
    #     projected_piece_vector = self.projection(raw_piece_vector.transpose(0, 1)) # Shape (N, P, F)

    #     # for b in range(batch_size):
    #     #     for p in range(self.num_pieces):
    #     #         print(interpret_tile(raw_piece_vector[p, b, :]))
    #     #     print()
    #     return projected_piece_vector

# --- New Interaction Block --- 
class BiDirectionalInteractionBlock(nn.Module):
    """Performs forward and reverse attention between piece and conv features."""
    def __init__(self, d_model, nhead, dim_feedforward, board_height, board_width):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        
        # Forward layer (updates piece vec)
        self.interaction_layer_fwd = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True
        )
        # Reverse layer (updates conv features)
        self.interaction_layer_rev = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True
        )
        # Norm for reverse pass (applied to conv features, no residual)
        self.norm_rev = nn.LayerNorm(d_model)

    def forward(self, projected_piece_vector, kv_permuted):
        """
        Args:
            projected_piece_vector: (N, P, F)
            kv_permuted: (N, H*W, F) - Output from previous conv block

        Returns:
            updated_piece_vector: (N, P, F)
            updated_conv_features: (N, F, H, W)
            updated_kv_permuted: (N, H*W, F)
        """
        # 1. Forward Pass (Update Piece Vector - No Residual)
        updated_piece_vector = self.interaction_layer_fwd(projected_piece_vector, kv_permuted)

        # 2. Reverse Pass (Update Conv Features - WITHOUT Residual)
        interaction_output_rev = self.interaction_layer_rev(tgt=kv_permuted, memory=updated_piece_vector)
        updated_kv_permuted = self.norm_rev(interaction_output_rev)

        # 3. Reshape updated conv features back to 4D
        N, L, C_filt = updated_kv_permuted.shape
        H, W = self.board_height, self.board_width # Use stored board dims
        updated_conv_features = updated_kv_permuted.permute(0, 2, 1).view(N, C_filt, H, W)

        return updated_piece_vector, updated_conv_features

class ChessNetwork(nn.Module):
    """Neural network architecture inspired by AlphaZero for Chess.

    Interaction:
      - PieceVectorExtractor creates initial (N, 32, C) vector and projects it to (N, 32, F).
      - Conv body (num_filters=F channels) processes first C-1 inputs.
      - At each stage i, BiDirectionalInteractionBlock updates both piece_vector and conv_features.
      - Policy head operates on final piece_vector.
      - Value head operates on final conv_features.
    """
    def __init__(self,
                 input_channels=DEFAULT_INPUT_CHANNELS,
                 board_size=DEFAULT_BOARD_SIZE,
                 num_conv_layers=DEFAULT_NUM_CONV_LAYERS,
                 num_filters=DEFAULT_FILTERS, # Dimension for interaction
                 action_space_size=DEFAULT_ACTION_SPACE,
                 num_pieces=DEFAULT_NUM_PIECES,
                 num_attention_heads=DEFAULT_NUM_ATTENTION_HEADS,
                 decoder_ff_dim_mult=4):
        super().__init__()
        if input_channels < 2: raise ValueError("Input channels must be >= 2")

        self.board_height = board_size
        self.board_width = board_size
        self.action_space_size = action_space_size
        self.input_channels = input_channels
        self.conv_input_channels = input_channels - 1
        self.num_filters = num_filters
        self.num_pieces = num_pieces

        # --- Piece Vector Extractor (now includes projection) ---
        self.piece_vector_extractor = PieceVectorExtractor(self.num_pieces, self.input_channels, self.num_filters)

        # --- Convolutional Body (Outputs num_filters=F) ---
        self.initial_conv = ConvBlock(self.conv_input_channels, num_filters) # C-1 -> F
        self.conv_layers = nn.ModuleList([ConvBlock(num_filters, num_filters) for _ in range(num_conv_layers - 1)])

        # --- Interaction Blocks --- 
        self.interaction_blocks = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.interaction_blocks.append(BiDirectionalInteractionBlock(
                d_model=num_filters,
                nhead=num_attention_heads,
                dim_feedforward=num_filters * decoder_ff_dim_mult,
                board_height=self.board_height,
                board_width=self.board_width
            ))

        # --- Policy Head (Input from final piece_vector = N, 32, F) ---
        policy_input_size = self.num_pieces * self.num_filters # 32 * F
        self.policy_fc = nn.Linear(policy_input_size, action_space_size)

        # --- Value Head (Input from final conv_features = F channels) ---
        self.value_conv = ConvBlock(num_filters, 1, kernel_size=1, padding=0)
        value_input_size = 1 * self.board_height * self.board_width
        self.value_fc1 = nn.Linear(value_input_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Args:
            board_vector (np.ndarray | torch.Tensor): The board state representation.
                 Expected shape: (10, 8, 8) or (N, 10, 8, 8)

        Returns:
            tuple[torch.Tensor, float | torch.Tensor]: Policy logits and Value estimate.
        """
        
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Move tensor to the device where the model parameters reside
        model_device = next(self.parameters()).device
        x = x.to(model_device)

        # --- Split Input ---
        conv_input = x[:, :self.conv_input_channels, :, :] # (N, C-1, H, W)
        piece_ids = x[:, -1, :, :].long()

        # --- Create Projected Piece Vector ---
        projected_piece_vector = self.piece_vector_extractor(x, piece_ids) # Shape: (N, 32, F)

        # --- Initial Conv (now returns 4D and 3D) ---
        conv_features = self.initial_conv(conv_input) # Shapes: (N, F, H, W), (N, H*W, F)

        # --- Iterative Interaction and Conv Loop ---
        num_stages = len(self.interaction_blocks)
        for i in range(num_stages):
            # 1. Apply Interaction Block
            interaction_block = self.interaction_blocks[i]
            # Takes current piece vec & kv_permuted, returns updated versions
            updated_piece_vector, conv_features = interaction_block(
                projected_piece_vector, conv_features
            )
            
            # Update states for next iteration or final heads
            projected_piece_vector = updated_piece_vector

            # 2. Apply next convolutional layer (if applicable)
            # Takes the updated conv_features from the interaction block
            if i < len(self.conv_layers):
                 conv_features = self.conv_layers[i](conv_features)
            # else: final states (projected_piece_vector, conv_features) are ready

        # --- Policy Head (Operates on final updated piece_vector) ---
        N_policy = projected_piece_vector.size(0)
        policy_input = projected_piece_vector.view(N_policy, -1) # Shape: (N, 32 * F)
        policy_logits = self.policy_fc(policy_input)

        # --- Value Head (Operates on final updated 4D conv_features) ---
        # Note: value_conv expects 4D input, but returns tuple (4D, 3D). We only need the 4D part.
        conv_features = self.value_conv(conv_features)
        value_x = conv_features.view(conv_features.size(0), -1)
        value_x = F.relu(self.value_fc1(value_x))
        value = torch.tanh(self.value_fc2(value_x))

        # NOTE: Value head uses conv_features, Policy head uses projected_piece_vector
        return policy_logits, value


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torchview_custom.torchview import draw_graphs
    from utils.profile import profile_model

    network = ChessNetwork(num_conv_layers=DEFAULT_NUM_CONV_LAYERS, 
                         num_filters=DEFAULT_FILTERS, 
                         action_space_size=DEFAULT_ACTION_SPACE,
                         num_attention_heads=DEFAULT_NUM_ATTENTION_HEADS)
    print("Network Initialized.")
    print(f"Using: num_conv_layers={len(network.conv_layers) + 1}, num_filters={network.num_filters}, action_space={network.action_space_size}")

    board = FullyTrackedBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    dummy_input_tensor = torch.from_numpy(board.get_board_vector()).to(dtype=torch.float32)
    print("\nInput shape (before batching):", dummy_input_tensor.shape)

    network.eval()
    with torch.no_grad():
        # Test forward pass (forward expects tensor)
        policy_logits, value = network(dummy_input_tensor)

        # Test piece vector extraction + projection (now done internally)
        x_batched = dummy_input_tensor.unsqueeze(0).to(next(network.parameters()).device)
        piece_ids_internal = x_batched[:, -1, :, :].long()
        # The extractor now directly outputs the projected vector
        final_projected_pv = network.piece_vector_extractor(x_batched, piece_ids_internal)
        print("\nProjected Piece Vector shape (output from extractor):", final_projected_pv.shape) # (1, 32, F)

    print("\nOutput Policy Logits shape (single input):", policy_logits.shape) # (action_space_size)
    print("Output Value (single input):", value) # scalar

    # Profile requires batched input
    profile_model(network, dummy_input_tensor.unsqueeze(0))
    draw_graphs(network, dummy_input_tensor.unsqueeze(0), 
                min_depth=1, max_depth=3, 
                output_names=['Policy', 'Value'], 
                input_names=['Input'], 
                directory='./model_viz/', 
                hide_module_functions=False, 
                hide_inner_tensors=True,
                print_code_path=False)
    print("\nModel graph saved to ./model_viz/")
