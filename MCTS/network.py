import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Default values based on common AlphaZero implementations for chess
DEFAULT_INPUT_CHANNELS = 13
DEFAULT_BOARD_SIZE = 8
DEFAULT_NUM_CONV_LAYERS = 7
DEFAULT_FILTERS = 32 # Number of filters in conv layers
DEFAULT_ACTION_SPACE = 750 # User-specified action space size

class ConvBlock(nn.Module):
    """A single convolutional block with Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ChessNetwork(nn.Module):
    """Neural network architecture inspired by AlphaZero for Chess.

    Assumes input is a spatial tensor: (batch_size, input_channels, board_height, board_width)
    """
    def __init__(self,
                 input_channels=DEFAULT_INPUT_CHANNELS,
                 board_size=DEFAULT_BOARD_SIZE,
                 num_conv_layers=DEFAULT_NUM_CONV_LAYERS,
                 num_filters=DEFAULT_FILTERS,
                 action_space_size=DEFAULT_ACTION_SPACE):
        super().__init__()
        self.board_height = board_size
        self.board_width = board_size
        self.action_space_size = action_space_size
        self.input_channels = input_channels

        # --- Convolutional Body ---
        self.initial_conv = ConvBlock(input_channels, num_filters)
        self.conv_layers = nn.ModuleList(
            [ConvBlock(num_filters, num_filters) for _ in range(num_conv_layers - 1)]
        )

        # --- Policy Head ---
        # Uses a 1x1 convolution to reduce filters, then flattens and applies linear layer
        self.policy_conv = ConvBlock(num_filters, 2, kernel_size=1, padding=0) # Reduce to 2 channels
        policy_input_size = 2 * self.board_height * self.board_width
        self.policy_fc = nn.Linear(policy_input_size, action_space_size)

        # --- Value Head ---
        # Uses a 1x1 convolution, flattens, then uses fully connected layers
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
                 e.g., (13, 8, 8) or (N, 13, 8, 8)

        Returns:
            tuple[torch.Tensor, float | torch.Tensor]:
                - Policy logits tensor (shape [action_space_size] or [batch_size, action_space_size]).
                - Value estimate (-1 to 1). Scalar float if input was single state, Tensor if batched.
        """
        # --- Input Processing ---
        is_single_input = False
        if isinstance(board_vector, np.ndarray):
            # Check shape (Allow C, H, W)
            expected_shape = (self.input_channels, self.board_height, self.board_width)
            if board_vector.ndim == 3:
                is_single_input = True
                if board_vector.shape != expected_shape:
                    raise ValueError(f"Input numpy array has wrong shape {board_vector.shape}, expected {expected_shape}")
            elif board_vector.ndim == 1: # Handle potential flat array (less common)
                 is_single_input = True
                 expected_elements = self.input_channels * self.board_height * self.board_width
                 if board_vector.size == expected_elements:
                      board_vector = board_vector.reshape(expected_shape)
                 else:
                      raise ValueError(f"Input flat numpy array has wrong size ({board_vector.size}), expected {expected_elements} for shape {expected_shape}")
            else:
                raise ValueError(f"Input numpy array has unexpected dimensions {board_vector.ndim}. Expected 1 or 3.")
            # Convert to tensor
            x = torch.tensor(board_vector, dtype=torch.float32)

        elif isinstance(board_vector, torch.Tensor):
            x = board_vector # Assume tensor is already correct
            # Basic dimension checks
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

        # Ensure correct dtype
        if x.dtype != torch.float32:
             x = x.float()

        # Add batch dimension if processing a single state
        if is_single_input:
            x = x.unsqueeze(0) # Shape becomes (1, C, H, W)

        # --- Convolutional Body ---
        x = self.initial_conv(x)
        for layer in self.conv_layers:
            x = layer(x)

        # --- Policy Head ---
        policy_x = self.policy_conv(x)
        policy_x = policy_x.view(policy_x.size(0), -1) # Flatten features
        policy_logits = self.policy_fc(policy_x)

        # --- Value Head ---
        value_x = self.value_conv(x)
        value_x = value_x.view(value_x.size(0), -1) # Flatten features
        value_x = F.relu(self.value_fc1(value_x))
        value = torch.tanh(self.value_fc2(value_x)) # Squash value to [-1, 1]

        # Remove batch dimension if input was a single state & return scalar value
        if is_single_input:
             policy_logits = policy_logits.squeeze(0)
             value = value.squeeze(0).item() # Return scalar python float
        else:
            # If input was batched, return batched logits and value tensor
            value = value.squeeze(1) # Shape (batch_size, 1) -> (batch_size)

        return policy_logits, value


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torchview_custom.torchview import draw_graphs
    from utils.profile import profile_model

    network = ChessNetwork()
    inputs = torch.zeros(1, 13, 8, 8)
    profile_model(network, inputs)

    # Input size should now be (Batch, Channels, Height, Width) -> (1, 13, 8, 8)
    draw_graphs(network, (torch.zeros(1, 13, 8, 8),), min_depth=1, max_depth=5, output_names=['Policy', 'Value'], input_names=['Input'], directory='./model_viz/', hide_module_functions=True, print_code_path=False)