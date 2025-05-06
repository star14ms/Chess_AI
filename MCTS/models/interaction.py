import torch.nn as nn

DEFAULT_NUM_ATTENTION_HEADS = 4 # Heads for TransformerDecoderLayer
DEFAULT_DECODER_FF_DIM_MULT = 4
NUM_INTERACTION_LAYERS = 2

# --- Interaction Block (Keep as is) ---
class InteractionBlock(nn.Module):
    """Performs forward and reverse attention between piece and conv features."""
    def __init__(self, d_model, nhead, dim_feedforward, board_height, board_width, num_layers=NUM_INTERACTION_LAYERS):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        
        # Forward layer (updates piece vec)
        self.interaction_layer_fwd = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, batch_first=True
            ),
            num_layers=num_layers
        )
        # Reverse layer (updates conv features)
        self.interaction_layer_rev = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, batch_first=True
            ),
            num_layers=num_layers
        )
        # Norm for reverse pass (applied to conv features, no residual)
        self.norm_rev = nn.LayerNorm(d_model)

    def forward(self, projected_piece_vector, conv_features):
        """
        Args:
            projected_piece_vector: (N, P, C)
            conv_features: (N, H*W, C) - Context features

        Returns:
            updated_piece_vector: (N, P, C)
            updated_conv_features: (N, C, H, W) - Note: Returns 4D spatial map
        """
        
        # Reshape 4D conv features to 3D for interaction block
        # conv_features = torch.cat((conv_input, conv_features), dim=1)

        N, C_curr, H, W = conv_features.shape
        conv_features = conv_features.view(N, C_curr, -1).permute(0, 2, 1) # (N, H*W, C)

        updated_piece_vector = self.interaction_layer_fwd(projected_piece_vector, conv_features)
        interaction_output_rev = self.interaction_layer_rev(tgt=conv_features, memory=updated_piece_vector)
        updated_conv_features = self.norm_rev(interaction_output_rev) # Shape (N, H*W, C)

        N, L, C_out = updated_conv_features.shape 
        H, W = self.board_height, self.board_width 
        updated_conv_features = updated_conv_features.permute(0, 2, 1).view(N, C_out, H, W) # Shape (N, C, H, W)
        # updated_conv_features = updated_conv_features[:, conv_input.shape[1]:, :, :]

        return updated_piece_vector, updated_conv_features
