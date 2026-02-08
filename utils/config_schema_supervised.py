from dataclasses import dataclass, field
from typing import Optional, List, Union

from utils.config_schema_rl import Config


@dataclass
class SupervisedLoggingConfig:
    no_progress: bool = False
    theme_log_min_total: int = 100
    theme_ignore: List[str] = field(
        default_factory=lambda: [
            "mate",
            "mateIn1",
            "mateIn2",
            "mateIn3",
            "mateIn4",
            "mateIn5",
            "veryLong",
            "long",
            "short",
            "oneMove",
            "opening",
            "middlegame",
            "endgame",
        ]
    )
    theme_plot_include_missing: bool = False


@dataclass
class SupervisedConfig:
    data_paths: List[str] = field(
        default_factory=lambda: [
            "data/mate_in_1_flipped_expanded.jsonl",
            "data/mate_in_2_flipped_expanded.jsonl",
            "data/mate_in_3_flipped_expanded.jsonl",
            "data/mate_in_4_flipped_expanded.jsonl",
            "data/mate_in_5_flipped_expanded.jsonl",
            "data/endgame_without_mate_flipped.json",
        ]
    )
    epochs: int = 10
    batch_size: Optional[int] = None
    learning_rate: float = 1e-4
    policy_dropout: Optional[float] = 0.25
    value_dropout: Optional[float] = 0.25
    weight_decay: Optional[float] = 1e-2
    checkpoint_dir: str = "${hydra:runtime.output_dir}"
    save_every: int = 1
    num_workers: Union[int, str] = "auto"
    max_rows: Optional[int] = None
    device: str = "auto"
    skip_value_sources: List[str] = field(
        default_factory=lambda: ["endgame_without_mate_flipped"]
    )
    val_split: float = 0.1
    seed: int = 1337
    early_stop_patience: int = 10
    lr_patience: int = 3
    lr_factor: float = 0.5
    resume: Optional[str] = None
    amp: bool = False
    logging: SupervisedLoggingConfig = field(default_factory=SupervisedLoggingConfig)


@dataclass
class SupervisedTrainConfig(Config):
    supervised: SupervisedConfig = field(default_factory=SupervisedConfig)
