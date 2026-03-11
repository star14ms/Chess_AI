from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union

# Note: These defaults should ideally match the YAML defaults


@dataclass
class NetworkConfig:
    # Defaults matching network.py where possible
    input_channels: int = 26
    dim_piece_type: int = 16  # Standard number of piece types
    board_size: int = 8
    initial_conv_block_out_channels: List[int] = [32, 64, 64, 128, 128, 128, 128]
    num_residual_layers: int = 0
    residual_blocks_out_channels: Optional[List[List[int]]] = [] * 0
    action_space_size: int = 4672  # 1700 or 4672 - controls which action space and board class to use
    num_pieces: int = 32  # Standard number of pieces
    value_head_hidden_size: int = 256
    policy_linear_out_features: Optional[List[int]] = field(default_factory=lambda: [4672])
    conv_bias: bool = False  # Whether to use bias in convolutional layers
    policy_dropout: float = 0.0
    value_dropout: float = 0.0
    conv_dropout: float = 0.0
    freeze_first_n_conv_layers: Optional[int] = 0  # Freeze first N layers of conv body (initial conv + residual). 0 or null = train all.


@dataclass
class MCTSConfig:
    iterations: int = 100
    c_puct: float = 1.41
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    temperature_decay_moves: int = 30
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    batch_size: int = 1  # Number of leaves to evaluate in one batched network call
    pre_init_draws: bool = False  # Pre-initialize draw terminals (stalemate, insufficient material, etc.) with term_val. False = only checkmate pre-init.


@dataclass
class OptimizerConfig:
    type: str = "Adam"  # Adam or SGD
    learning_rate: float = 0.001
    momentum: Optional[float] = 0.9  # Only used for SGD
    weight_decay: float = 1e-4


# New dataclass for environment settings
@dataclass
class EnvConfig:
    type: str = "chess"  # "chess" or "gomoku" - specifies which environment to use
    observation_mode: str = "vector"
    render_mode: Optional[str] = None  # Default to None
    save_video_folder: Optional[str] = None  # Default to None
    history_steps: int = 8  # Number of history snapshots for observation; must be >= 1


@dataclass
class TrainingConfig:
    device: str = "auto"  # auto, cuda, mps, cpu
    amp: bool = False  # Enable CUDA AMP (mixed precision) during training
    replay_buffer_size: int = 50000
    num_training_iterations: int = 1000
    num_training_steps: int = 10  # Renamed from training_steps_per_iteration for clarity
    batch_size: int = 128
    grad_clip_norm: Optional[float] = None  # Max gradient norm; None = no clipping. Use 1.0 for TPU stability.
    tpu_learning_rate_multiplier: float = 0.2  # Scale LR by this on TPU (e.g. 0.2 = 5x lower) to avoid NaN after a few steps
    diagnose_tpu_gradients: bool = False  # If True and on TPU, log grad norms, loss components, batch stats to find gradient deviation cause
    replay_buffer_float32: bool = False  # If True, store replay buffer in float32 (avoids float16 underflow on TPU)
    skip_replay_buffer_load: bool = False  # If True, skip loading buffer from checkpoint (refill from self-play)
    use_multiprocessing: bool = True  # Flag to enable/disable multiprocessing for self-play
    self_play_workers: int = 0  # Number of parallel workers for self-play. 0 means use default heuristic (e.g., half CPU cores).
    games_per_worker: int = 1  # Number of concurrent games per worker. >1 increases parallelism when using inference server.
    use_inference_server: bool = False  # Use a dedicated process/thread for batched GPU/TPU inference
    inference_server_device: Optional[str] = None  # Device for inference server (e.g., "cuda:0", "xla" for TPU)
    self_play_dtype: Optional[str] = None  # "float16", "float32", or null for auto (float16 on cuda/mps, float32 elsewhere)
    inference_server_max_batch_size: Optional[int] = None  # Max batch size for self-play inference; None = use training batch_size
    inference_server_min_stacked_requests: Optional[int] = None  # Min requests before processing; None = max_stacked // 2
    inference_server_max_wait_ms: int = 2  # Max wait before dispatching a batch
    inference_server_logging_enabled: bool = False  # Enable inference server logging (startup, throughput, exceptions)
    inference_queue_maxsize: Optional[int] = None  # Max queued inference requests; None = num_workers
    enable_thermal_pause: bool = False  # Allow brief thermal throttling pauses during self-play
    manual_pause_seconds: Optional[float] = None  # If set, sleep this many seconds before self-play
    self_play_steps_per_epoch: int = 1024  # Number of steps to collect before moving to training phase
    continual_training: bool = True  # If True, continue training from the last checkpoint
    continual_queue_maxsize: int = 64  # Maximum size of the queue for continual training
    max_game_moves: int = 200
    checkpoint_dir: str = "./checkpoints"  # Will be relative to hydra output dir
    checkpoint_load: Optional[str] = None  # Dir or direct file path to load from (defaults to checkpoint_dir/model.pth if None)
    game_history_dir: Optional[str] = "./output/"  # Directory to save game history files (moves in SAN notation). Set to None to disable.
    progress_bar: bool = True  # If True, show bars only when not multiprocessing; if False, never show
    initial_board_fen: Optional[
        Union[
            Dict[str, float | Dict[str, float | str]],
            List[Dict[str, float | str]],
            List[str],
            str,
        ]
    ] = None  # Dict mapping FEN->weight or {"weight","quality","max_game_moves"}; string can be FEN or JSON path. JSON path may be dict (weighted) or array of entries (uniform selection, uses "FEN"/"fen" field). List enables multi-dataset selection with weights and optional per-dataset max_game_moves.
    max_training_time_seconds: Optional[int] = None  # Maximum training time in seconds. At the end of each iteration, predicts total elapsed time after next iteration and stops if it would exceed this limit. Set to None to disable.
    draw_reward: Optional[float] = None  # Fixed reward value for draws. If None, uses draw_reward_table based on termination type and position quality
    draw_reward_table: Optional[Dict[str, Dict[str, float]]] = None  # {termination_type: {position_quality: reward}}. Used when draw_reward is None
    last_n_moves_to_store: Dict[str, int] = field(
        default_factory=dict
    )  # Dict: termination -> int. 0 = exclude from buffer; n>0 = store only last n moves; absent = full game
    draw_sample_ratio: Optional[float] = None  # Target fraction of batch from draws (0.0–1.0). None = uniform sampling. E.g. 0.4 = 40% draws per batch.
    follow_dataset_trajectory: bool = False  # If True and a dataset solution exists, select the dataset move deterministically instead of sampling from MCTS policy. Policy target remains MCTS.


@dataclass
class Config:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    env: EnvConfig = field(default_factory=EnvConfig)  # Add EnvConfig
    # Hydra specific config is usually handled separately in the yaml
