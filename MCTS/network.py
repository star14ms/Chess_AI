import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.analyze import interpret_tile

# Default values based on common AlphaZero implementations for chess
DEFAULT_INPUT_CHANNELS = 10
DEFAULT_BOARD_SIZE = 8
DEFAULT_NUM_CONV_LAYERS = 7
DEFAULT_FILTERS = 32 # Number of filters in conv layers
DEFAULT_ACTION_SPACE = 750 # User-specified action space size
DEFAULT_NUM_PIECES = 32

class ConvBlock(nn.Module):
    """A single convolutional block with Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class PieceVectorExtractor(nn.Module):
    """A non-trainable module to extract piece vectors from the input tensor."""
    def __init__(self, num_pieces=DEFAULT_NUM_PIECES):
        super().__init__()
        self.num_pieces = num_pieces

    def forward(self, full_board_vector, piece_ids):
        """Creates the (N, 32, C) piece vector from the full input vector."""
        batch_size, num_channels, height, width = full_board_vector.shape
        device = full_board_vector.device
        # Output tensor: (batch_size, num_pieces, num_input_channels)
        piece_vector = torch.zeros(batch_size, self.num_pieces, num_channels, device=device)

        for b in range(batch_size):
            # Find coordinates for each piece ID (1 to num_pieces)
            for piece_idx in range(self.num_pieces):
                target_piece_id = piece_idx + 1 # IDs are 1-based
                locations = torch.where(piece_ids[b] == target_piece_id)
                # locations is a tuple of tensors (rank_indices, file_indices)

                if locations[0].numel() > 0:
                    # If piece exists, take the first location found
                    rank_idx = locations[0][0]
                    file_idx = locations[1][0]
                    # Gather the full channel data from the original input vector
                    piece_vector[b, piece_idx, :] = full_board_vector[b, :, rank_idx, file_idx]
                # else: piece_vector[b, piece_idx, :] remains zeros

        return piece_vector

class ChessNetwork(nn.Module):
    """Neural network architecture inspired by AlphaZero for Chess.

    Input Processing:
      - Creates a Piece Vector (N, 32, C) directly from input using PieceVectorExtractor.
      - Convolutional body processes first C-1 input channels (excluding Piece ID).
      - Policy/Value heads operate on convolutional features.
    """
    def __init__(self,
                 input_channels=DEFAULT_INPUT_CHANNELS,
                 board_size=DEFAULT_BOARD_SIZE,
                 num_conv_layers=DEFAULT_NUM_CONV_LAYERS,
                 num_filters=DEFAULT_FILTERS,
                 action_space_size=DEFAULT_ACTION_SPACE,
                 num_pieces=DEFAULT_NUM_PIECES):
        super().__init__()
        if input_channels < 2:
            raise ValueError("Input channels must be at least 2 (features + Piece ID)")

        self.board_height = board_size
        self.board_width = board_size
        self.action_space_size = action_space_size
        self.input_channels = input_channels
        self.conv_input_channels = input_channels - 1 # Exclude the last channel (Piece IDs)
        self.num_filters = num_filters
        self.num_pieces = num_pieces

        # --- Piece Vector Extractor ---
        self.piece_vector_extractor = PieceVectorExtractor(self.num_pieces)

        # --- Convolutional Body (Operates on C-1 channels) ---
        self.initial_conv = ConvBlock(self.conv_input_channels, num_filters)
        self.conv_layers = nn.ModuleList(
            [ConvBlock(num_filters, num_filters) for _ in range(num_conv_layers - 1)]
        )

        # --- Policy Head (Operates on conv_features) ---
        self.policy_conv = ConvBlock(num_filters, 2, kernel_size=1, padding=0) # Reduce to 2 channels
        policy_input_size = 2 * self.board_height * self.board_width
        self.policy_fc = nn.Linear(policy_input_size, action_space_size)

        # --- Value Head (Operates on conv_features) ---
        self.value_conv = ConvBlock(num_filters, 1, kernel_size=1, padding=0) # Reduce to 1 channel
        value_input_size = 1 * self.board_height * self.board_width
        self.value_fc1 = nn.Linear(value_input_size, 128)
        self.value_fc2 = nn.Linear(128, 1) # Output a single value


    def forward(self, board_vector):
        """
        Args:
            board_vector (np.ndarray | torch.Tensor): The board state representation.
                 Expected shape: (input_channels, board_height, board_width) or
                 (batch_size, input_channels, board_height, board_width).
                 e.g., (10, 8, 8) or (N, 10, 8, 8)

        Returns:
            tuple[torch.Tensor, float | torch.Tensor]:
                - Policy logits tensor (shape [action_space_size] or [batch_size, action_space_size]).
                - Value estimate (-1 to 1). Scalar float if input was single state, Tensor if batched.
        """
        # --- Input Processing ---
        is_single_input = False
        if isinstance(board_vector, np.ndarray):
            expected_shape = (self.input_channels, self.board_height, self.board_width)
            if board_vector.ndim == 3:
                is_single_input = True
                if board_vector.shape != expected_shape:
                    raise ValueError(f"Input numpy array has wrong shape {board_vector.shape}, expected {expected_shape}")
                x = torch.tensor(board_vector, dtype=torch.float32)
            else:
                raise ValueError(f"Input numpy array has unexpected dimensions {board_vector.ndim}. Expected 3.")
        elif isinstance(board_vector, torch.Tensor):
            x = board_vector
            if x.ndim == 3:
                is_single_input = True
                expected_shape = (self.input_channels, self.board_height, self.board_width)
                if x.shape != expected_shape:
                     raise ValueError(f"Input tensor has wrong shape {x.shape}, expected {expected_shape}")
            elif x.ndim == 4:
                expected_shape_no_batch = (self.input_channels, self.board_height, self.board_width)
                if x.shape[1:] != expected_shape_no_batch:
                     raise ValueError(f"Input tensor has wrong shape {x.shape}, expected (N, {self.input_channels}, {self.board_height}, {self.board_width})")
            else:
                 raise ValueError(f"Input tensor has wrong dimensions {x.ndim}, expected 3 or 4.")
        else:
             raise TypeError(f"board_vector must be np.ndarray or torch.Tensor, got {type(board_vector)}")

        if x.dtype != torch.float32:
             x = x.float()
        if is_single_input:
            x = x.unsqueeze(0) # Shape becomes (1, C, H, W)

        # --- Split Input ---
        conv_input = x[:, :self.conv_input_channels, :, :] # (N, C-1, H, W)
        piece_ids = x[:, -1, :, :].long() # (N, H, W)

        # --- Create Piece Vector ---
        piece_vector = self.piece_vector_extractor(x, piece_ids) # Shape: (N, 32, C)
        # for i in range(32):
        #     print(interpret_tile(piece_vector[0, i, :]))

        # --- Convolutional Body ---
        conv_features = self.initial_conv(conv_input)
        for layer in self.conv_layers:
            conv_features = layer(conv_features) # Shape: (N, num_filters, H, W)

        # --- Policy Head ---
        policy_x = self.policy_conv(conv_features)
        policy_x = policy_x.view(policy_x.size(0), -1) # Flatten features
        policy_logits = self.policy_fc(policy_x)

        # --- Value Head ---
        value_x = self.value_conv(conv_features)
        value_x = value_x.view(value_x.size(0), -1) # Flatten features
        value_x = F.relu(self.value_fc1(value_x))
        value = torch.tanh(self.value_fc2(value_x)) # Squash value to [-1, 1]

        # --- Output Formatting ---
        if is_single_input:
             policy_logits = policy_logits.squeeze(0)
             value = value.squeeze(0).item() # Return scalar python float
        else:
            value = value.squeeze(1)

        # NOTE: piece_vector is created but not returned or used by heads in this version.
        return policy_logits, value


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torchview_custom.torchview import draw_graphs
    from utils.profile import profile_model

    # Test with default parameters
    network = ChessNetwork()
    print("Network Initialized:")
    # print(network) # Optional: Print network structure
    
    from chess_gym.chess_custom import FullyTrackedBoard
    board = FullyTrackedBoard()
    board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(board.get_board_vector())

    # Create dummy input (Batch size 1)
    dummy_input_tensor = torch.from_numpy(board.get_board_vector()).unsqueeze(0)
    print("\nInput shape:", dummy_input_tensor.shape)

    # # Test piece vector creation via the module
    # network.eval()
    # with torch.no_grad():
    #     piece_ids_internal = dummy_input_tensor[:, -1, :, :].long()
    #     # Access the module directly for testing
    #     piece_vector_internal = network.piece_vector_extractor(dummy_input_tensor, piece_ids_internal)
    #     print("\nCreated Piece Vector shape:", piece_vector_internal.shape) # Should be (1, 32, 10)
    #     # Check values for created pieces
    #     print("Piece Vector[0, 0, :]:", piece_vector_internal[0, 0, :]) # Should match input at (0,0)
    #     print("Piece Vector[0, 1, :]:", piece_vector_internal[0, 1, :]) # Should match input at (1,1)
    #     print("Piece Vector[0, 31, :]:", piece_vector_internal[0, 31, :]) # Should match input at (7,7)
    #     print("Piece Vector[0, 2, :]:", piece_vector_internal[0, 2, :]) # Should be zeros (ID 3 not placed)

    # # Forward pass
    # with torch.no_grad():
    #     policy_logits, value = network(dummy_input_tensor)

    # print("\nOutput Policy Logits shape:", policy_logits.shape) # Should be (750,)
    # print("Output Value:", value) # Should be scalar float

    # --- Optional: Profiling and Visualization ---
    profile_model(network, dummy_input_tensor) 
    draw_graphs(network, dummy_input_tensor, 
                min_depth=1, max_depth=5, 
                output_names=['Policy', 'Value'], 
                input_names=['Input'], 
                directory='./model_viz/', 
                hide_module_functions=True, 
                print_code_path=False)
    print("\nModel graph saved to ./model_viz/")
