from dataclasses import dataclass, field
from typing import Optional, List

# Note: These defaults should ideally match the YAML defaults

@dataclass
class NetworkConfig:
    # Defaults matching network.py where possible
    input_channels: int = 26
    dim_piece_type: int = 16 # Standard number of piece types
    board_size: int = 8
    num_filters: List[int] = [32, 64, 64, 128, 128, 128, 128]
    num_residual_layers: int = 0
    conv_blocks_channel_lists: Optional[List[List[int]]] = [] * 0
    action_space_size: int = 4672 # Updated based on latest yaml
    num_pieces: int = 32 # Standard number of pieces
    value_head_hidden_size: int = 256
    policy_linear_out_features: Optional[List[int]] = field(default_factory=lambda: [4672])
    action_space_mode: str = "4672" # "1700" or "4672" - controls which action space and board class to use

@dataclass
class MCTSConfig:
    iterations: int = 100
    c_puct: float = 1.41
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    temperature_decay_moves: int = 30
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

@dataclass
class OptimizerConfig:
    type: str = "Adam" # Adam or SGD
    learning_rate: float = 0.001
    momentum: Optional[float] = 0.9 # Only used for SGD
    weight_decay: float = 1e-4

# New dataclass for environment settings
@dataclass
class EnvConfig:
    type: str = "chess"  # "chess" or "gomoku" - specifies which environment to use
    observation_mode: str = "vector"
    render_mode: Optional[str] = None # Default to None
    save_video_folder: Optional[str] = None # Default to None
    history_steps: int = 8 # Number of history snapshots for observation; must be >= 1

@dataclass
class TrainingConfig:
    device: str = "auto" # auto, cuda, mps, cpu
    replay_buffer_size: int = 50000
    num_training_iterations: int = 1000
    num_training_steps: int = 10 # Renamed from training_steps_per_iteration for clarity
    batch_size: int = 128
    use_multiprocessing: bool = True # Flag to enable/disable multiprocessing for self-play
    self_play_workers: int = 0 # Number of parallel workers for self-play. 0 means use default heuristic (e.g., half CPU cores).
    self_play_games_per_epoch: int = 50
    continual_training: bool = True # If True, continue training from the last checkpoint
    continual_queue_maxsize: int = 64 # Maximum size of the queue for continual training
    max_game_moves: int = 200
    checkpoint_dir: str = "checkpoints" # Will be relative to hydra output dir
    checkpoint_dir_load: Optional[str] = None # Optional separate directory to load checkpoints from (defaults to checkpoint_dir if None or empty)
    progress_bar: bool = True # If True, show bars only when not multiprocessing; if False, never show
    save_interval: int = 10
    initial_board_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" #except pawns

@dataclass
class Config:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    env: EnvConfig = field(default_factory=EnvConfig) # Add EnvConfig
    # Hydra specific config is usually handled separately in the yaml 