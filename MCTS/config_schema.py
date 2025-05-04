from dataclasses import dataclass, field
from typing import Optional

# Note: These defaults should ideally match the YAML defaults

@dataclass
class NetworkConfig:
    # Defaults matching network.py where possible
    input_channels: int = 10
    board_size: int = 8
    num_conv_layers: int = 4
    num_filters: int = 24 # Corresponds to d_model for interaction layers in the current reverted state
    num_attention_heads: int = 4
    decoder_ff_dim_mult: int = 4
    # Add piece_embedding_dim, interaction_dim if the network architecture requires them again
    # action_space_size: int = 16 # This should likely be derived or set based on env

@dataclass
class MCTSConfig:
    iterations: int = 100
    c_puct: float = 1.41
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    temperature_decay_moves: int = 30

@dataclass
class OptimizerConfig:
    type: str = "Adam" # Adam or SGD
    learning_rate: float = 0.001
    momentum: Optional[float] = 0.9 # Only used for SGD
    weight_decay: float = 1e-4

@dataclass
class TrainingConfig:
    device: str = "auto" # auto, cuda, mps, cpu
    replay_buffer_size: int = 50000
    batch_size: int = 128
    self_play_games_per_epoch: int = 50
    training_epochs: int = 10 # Renamed from training_steps_per_iteration for clarity
    num_training_iterations: int = 1000
    checkpoint_dir: str = "checkpoints" # Will be relative to hydra output dir
    save_interval: int = 10
    max_game_moves: int = 200
    board_cls_str: str = "chess_gym.chess_custom.FullyTrackedBoard"
    action_space_size: int = 850 # Define action space size needed by Network
    # Environment parameters
    observation_mode: str = "vector"
    render_mode: Optional[str] = "human" # Can be None
    save_video_folder: Optional[str] = "./videos" # Can be None

@dataclass
class Config:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Hydra specific config is usually handled separately in the yaml 