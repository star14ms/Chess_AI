from dataclasses import dataclass, field
from typing import Optional, List

# Note: These defaults should ideally match the YAML defaults

@dataclass
class NetworkConfig:
    # Defaults matching network.py where possible
    input_channels: int = 11
    board_size: int = 8
    num_conv_layers: int = 4
    conv_blocks_channel_lists: Optional[List[List[int]]] = None 
    num_attention_heads: int = 4
    decoder_ff_dim_mult: int = 4
    action_space_size: int = 1700 # Updated based on latest yaml
    num_pieces: int = 32 # Standard number of pieces
    policy_hidden_size: int = 128 # Example default

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

# New dataclass for environment settings
@dataclass
class EnvConfig:
    observation_mode: str = "vector"
    render_mode: Optional[str] = None # Default to None
    save_video_folder: Optional[str] = None # Default to None

@dataclass
class TrainingConfig:
    device: str = "auto" # auto, cuda, mps, cpu
    replay_buffer_size: int = 50000
    num_training_iterations: int = 1000
    training_epochs: int = 10 # Renamed from training_steps_per_iteration for clarity
    batch_size: int = 128
    use_multiprocessing: bool = True # Flag to enable/disable multiprocessing for self-play
    self_play_workers: int = 0 # Number of parallel workers for self-play. 0 means use default heuristic (e.g., half CPU cores).
    self_play_games_per_epoch: int = 50
    max_game_moves: int = 200
    checkpoint_dir: str = "checkpoints" # Will be relative to hydra output dir
    save_interval: int = 10
    board_cls_str: str = "chess_gym.chess_custom.FullyTrackedBoard"
    # Removed env parameters

@dataclass
class Config:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    env: EnvConfig = field(default_factory=EnvConfig) # Add EnvConfig
    # Hydra specific config is usually handled separately in the yaml 