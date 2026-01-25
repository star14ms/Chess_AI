import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque
import os
import gc
import time
import logging
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing # Keep for potential parallel execution
import queue
from torch import nn
import pickle
import gzip
import json
import chess
import random
import re

# Assuming files are in the MCTS directory relative to the project root
from mcts_node import MCTSNode
from mcts_algorithm import MCTS
from utils.profile_model import get_optimal_worker_count, profile_model, format_time
from utils.progress import NullProgress
from utils.training_utils import (
    RewardComputer,
    iter_json_array,
    select_fen_from_dict,
    select_random_fen_from_entries,
    select_random_fen_from_json_list,
)

create_network = None
create_environment = None
get_game_result = None
is_first_player_turn = None
get_legal_actions = None
create_board_from_serialized = None
_INITIAL_FEN_CACHE = {}
_DATASET_ENTRIES_CACHE = {}


def _resolve_json_path(json_path: str) -> str:
    if os.path.isabs(json_path):
        return json_path
    train_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(train_file_dir, '..'))
    potential_paths = [
        json_path,
        os.path.join(project_root, json_path),
        os.path.join(project_root, 'config', os.path.basename(json_path)),
        os.path.join(train_file_dir, json_path),
    ]
    if 'config/' in json_path:
        potential_paths.insert(0, os.path.join(project_root, json_path))
    for path in potential_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    return os.path.join(project_root, json_path)


def _select_fen_from_loaded(loaded):
    if isinstance(loaded, dict) and len(loaded) > 0:
        return select_fen_from_dict(loaded)
    if isinstance(loaded, list) and loaded:
        return select_random_fen_from_entries(loaded)
    return None, None


def _select_fen_from_json_path(json_path: str):
    try:
        resolved = _resolve_json_path(json_path)
        cache_key = os.path.abspath(resolved)
        if cache_key in _INITIAL_FEN_CACHE:
            loaded = _INITIAL_FEN_CACHE[cache_key]
        else:
            load_start = time.perf_counter()
            with open(resolved, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            load_elapsed = time.perf_counter() - load_start
            loaded_count = None
            try:
                loaded_count = len(loaded)
            except Exception:
                loaded_count = None
            _INITIAL_FEN_CACHE[cache_key] = loaded
        return _select_fen_from_loaded(loaded)
    except Exception as e:
        print(f"Warning: Failed to load initial_board_fen from JSON file {json_path}: {e}")
        return None, None


def _select_fen_from_source(source):
    if isinstance(source, str) and source.endswith('.json'):
        return _select_fen_from_json_path(source)
    if isinstance(source, dict):
        return select_fen_from_dict(source)
    if isinstance(source, list):
        return select_random_fen_from_entries(source)
    if isinstance(source, str):
        return source, None
    return None, None


def _dataset_cache_key(value) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except Exception:
        return repr(value)


def _extract_fen_from_entry(entry):
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        fen = entry.get("FEN") or entry.get("fen")
        return str(fen).strip() if fen else None
    return None


def _sample_fens_from_json(path: str, max_samples: int) -> tuple[list[str], int]:
    sample: list[str] = []
    seen = 0
    for entry in iter_json_array(path):
        fen = _extract_fen_from_entry(entry)
        if not fen:
            continue
        seen += 1
        if len(sample) < max_samples:
            sample.append(fen)
        else:
            j = random.randint(1, seen)
            if j <= max_samples:
                sample[j - 1] = fen
    return sample, seen


def _build_initial_fen_pool_cfg(initial_board_fen_cfg, max_samples: int | None):
    if max_samples is None:
        return None
    if initial_board_fen_cfg is not None:
        try:
            if OmegaConf.is_config(initial_board_fen_cfg):
                initial_board_fen_cfg = OmegaConf.to_container(initial_board_fen_cfg, resolve=True)
        except Exception:
            pass
    if not isinstance(initial_board_fen_cfg, list):
        return None
    pooled_entries = []
    for entry in initial_board_fen_cfg:
        source, weight = _parse_dataset_entry(entry)
        if source is None:
            continue
        label = None
        max_game_moves = None
        if isinstance(entry, dict):
            label = entry.get("label") or entry.get("name")
            max_game_moves = entry.get("max_game_moves")
        if not label:
            label = _shorten_dataset_label(source)
        if isinstance(source, str) and source.endswith(".json"):
            resolved = _resolve_json_path(source)
            start = time.perf_counter()
            sample, total = _sample_fens_from_json(resolved, max_samples)
            elapsed = time.perf_counter() - start
            pooled_entry = {
                "entries": sample,
                "weight": float(weight) if isinstance(weight, (int, float)) else 1.0,
                "label": label,
            }
            if max_game_moves is not None:
                pooled_entry["max_game_moves"] = max_game_moves
            pooled_entries.append(pooled_entry)
        else:
            pooled_entries.append(entry)
    return pooled_entries if pooled_entries else None


def _get_cached_dataset_entries(initial_board_fen_cfg):
    if initial_board_fen_cfg is not None:
        try:
            if OmegaConf.is_config(initial_board_fen_cfg):
                initial_board_fen_cfg = OmegaConf.to_container(initial_board_fen_cfg, resolve=True)
        except Exception:
            pass
    if not isinstance(initial_board_fen_cfg, list):
        return [], None, []
    cache_key = _dataset_cache_key(initial_board_fen_cfg)
    cached = _DATASET_ENTRIES_CACHE.get(cache_key)
    if cached is not None:
        return cached["entries"], cached["weights"], cached["labels"]
    entries = _collect_dataset_entries(initial_board_fen_cfg)
    if not entries:
        _DATASET_ENTRIES_CACHE[cache_key] = {"entries": [], "weights": None, "labels": []}
        return [], None, []
    dataset_weights = [max(0.0, entry["weight"]) for entry in entries]
    total_weight = sum(dataset_weights)
    normalized = [w / total_weight for w in dataset_weights] if total_weight > 0 else None
    labels = [entry["label"] for entry in entries]
    _DATASET_ENTRIES_CACHE[cache_key] = {
        "entries": entries,
        "weights": normalized,
        "labels": labels,
    }
    return entries, normalized, labels


def _parse_dataset_entry(entry):
    if isinstance(entry, str):
        return entry, 1.0
    if isinstance(entry, dict):
        if "path" in entry or "file" in entry or "dataset" in entry:
            source = entry.get("path") or entry.get("file") or entry.get("dataset")
            weight = float(entry.get("weight", 1.0))
            return source, weight
        if "entries" in entry or "data" in entry or "fens" in entry:
            source = entry.get("entries") or entry.get("data") or entry.get("fens")
            weight = float(entry.get("weight", 1.0))
            return source, weight
        if len(entry) == 1:
            key, value = next(iter(entry.items()))
            if isinstance(key, str) and isinstance(value, (int, float)):
                return key, float(value)
        weight = entry.get("weight", 1.0)
        if not isinstance(weight, (int, float)):
            weight = 1.0
        return entry, float(weight)
    return None, None


def _shorten_dataset_label(source) -> str:
    if isinstance(source, str):
        base = os.path.splitext(os.path.basename(source))[0]
        lower = base.lower()
        mate_match = re.search(r"mate[_-]?in[_-]?(\d+)", lower)
        if mate_match:
            return f"m{mate_match.group(1)}"
        return base[:6] or "data"
    return "data"


def _collect_dataset_entries(initial_board_fen_cfg):
    if not isinstance(initial_board_fen_cfg, list):
        return []
    entries = []
    for entry in initial_board_fen_cfg:
        source, weight = _parse_dataset_entry(entry)
        if source is None:
            continue
        label = None
        max_game_moves = None
        if isinstance(entry, dict):
            label = entry.get("label") or entry.get("name")
            max_game_moves = entry.get("max_game_moves")
        if not label:
            label = _shorten_dataset_label(source)
        entry_weight = float(weight) if isinstance(weight, (int, float)) else 1.0
        entry_info = {"source": source, "weight": entry_weight, "label": label}
        if max_game_moves is not None:
            entry_info["max_game_moves"] = max_game_moves
        entries.append(entry_info)
    if not entries:
        return []
    label_counts: dict[str, int] = {}
    for item in entries:
        label = item["label"]
        count = label_counts.get(label, 0)
        label_counts[label] = count + 1
        if count > 0:
            item["label"] = f"{label}#{count + 1}"
    return entries


def _format_source_accuracy(labels: list[str], correct: list[int], total: list[int], digits: int = 2) -> str:
    if not labels:
        return "-"
    parts: list[str] = []
    for label, c, t in zip(labels, correct, total):
        if t > 0:
            acc = c / t * 100
            parts.append(f"{label}={acc:.{digits}f}%")
        else:
            parts.append(f"{label}=N/A")
    return " | ".join(parts)

def initialize_factories_from_cfg(cfg: OmegaConf) -> None:
    global create_network, create_environment, get_game_result, is_first_player_turn, get_legal_actions, create_board_from_serialized
    if cfg.env.type == "chess":
        from training_modules.chess import (
            create_chess_env as create_environment,
            create_chess_network as create_network,
            get_chess_game_result as get_game_result,
            is_white_turn as is_first_player_turn,
            get_chess_legal_actions as get_legal_actions,
            create_board_from_fen as create_board_from_serialized,
        )
    elif cfg.env.type == "gomoku":
        from training_modules.gomoku import (
            create_gomoku_env as create_environment,
            create_gomoku_network as create_network,
            get_gomoku_game_result as get_game_result,
            is_gomoku_first_player_turn as is_first_player_turn,
            get_gomoku_legal_actions as get_legal_actions,
            create_board_from_state as create_board_from_serialized,
        )
    else:
        raise ValueError(f"Unsupported environment type: {cfg.env.type}")

# Helper function for parallel execution (module scope for pickling)
def self_play_worker(game_id, network_state_dict, cfg: DictConfig, device_str: str):
    """Worker function to run a single self-play game."""
    # Convert plain dict back to OmegaConf if needed (for compatibility)
    if isinstance(cfg, dict) and not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    # Ensure factories are initialized inside spawned workers
    initialize_factories_from_cfg(cfg)
    # Determine device for this worker
    device = torch.device(device_str)
    # Ensure correct CUDA device context in worker
    if device.type == 'cuda':
        try:
            torch.cuda.set_device(device)
        except Exception:
            pass

    # Re-initialize Network in the worker process
    if cfg.mcts.iterations > 0:
        network = create_network(cfg, device)
        network.load_state_dict(network_state_dict)
        network.to(device).eval() # Ensure it's on the correct device and in eval mode
    else:
        network = None

    game_data, game_info = run_self_play_game(
        cfg,
        network,
        env=None,
        progress=None, # Pass None for progress within worker
        device=device
    )
    return (game_data, game_info)

# Wrapper for imap_unordered (module scope for pickling)
def worker_wrapper(args):
    """Unpacks arguments for self_play_worker when using imap."""
    game_id, network_state_dict, cfg, device_str = args
    # Call the original worker function with unpacked args
    return self_play_worker(game_id, network_state_dict, cfg, device_str)

# Persistent continual self-play actor
def continual_self_play_worker(checkpoint_path: str, cfg: DictConfig, device_str: str, out_queue, stop_event):
    initialize_factories_from_cfg(cfg)
    device = torch.device(device_str)
    if device.type == 'cuda':
        try:
            torch.cuda.set_device(device)
        except Exception:
            pass
    network = None
    if cfg.mcts.iterations > 0:
        network = create_network(cfg, device)
        if os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
                network.load_state_dict(ckpt['model_state_dict'])
            except Exception:
                pass
        network.to(device).eval()
    last_mtime = os.path.getmtime(checkpoint_path) if os.path.exists(checkpoint_path) else 0.0
    while not stop_event.is_set():
        # Hot-reload if newer checkpoint exists
        try:
            if os.path.exists(checkpoint_path):
                mtime = os.path.getmtime(checkpoint_path)
                if mtime > last_mtime and network is not None:
                    try:
                        ckpt = torch.load(checkpoint_path, map_location=device)
                        network.load_state_dict(ckpt['model_state_dict'])
                        network.to(device).eval()
                        last_mtime = mtime
                    except Exception:
                        pass
        except Exception:
            pass

        # Play one game and enqueue (include device info so we know which GPU finished)
        try:
            game_data, game_info = run_self_play_game(cfg, network if cfg.mcts.iterations > 0 else None, env=None, progress=None, device=device)
            if game_data:
                # Include device string in game_info so we can track which GPU finished
                game_info_with_device = game_info.copy()
                game_info_with_device['device'] = device_str
                out_queue.put((game_data, game_info_with_device))
        except Exception:
            # Continue on errors to keep the actor alive
            pass

        # Modest cleanup to avoid memory growth
        if device.type == 'cuda':
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
        gc.collect()

# --- Replay Buffer (Keep as is) ---
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def add_game(self, game_data):
        for experience in game_data:
            self.add(experience)

    def sample(self, batch_size, action_space_size=None):
        if len(self.buffer) < batch_size:
            return None
        batch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indices]

        # Pre-allocate states array for better performance
        first_state = batch[0][0]
        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        
        # Normalize policy targets to expected action space size
        # This handles cases where checkpoint data has different action space sizes
        if action_space_size is None:
            # Try to infer from first policy target
            first_policy = batch[0][1]
            if isinstance(first_policy, np.ndarray):
                action_space_size = len(first_policy)
            else:
                action_space_size = 4672  # Default fallback
        
        # Pre-allocate policy array and use vectorized operations where possible
        normalized_policies = np.zeros((batch_size, action_space_size), dtype=np.float32)
        source_ids = np.full((batch_size,), -1, dtype=np.int64)
        for i, exp in enumerate(batch):
            policy = exp[1]
            if isinstance(policy, np.ndarray):
                policy_len = len(policy)
                if policy_len == action_space_size:
                    normalized_policies[i] = policy
                elif policy_len < action_space_size:
                    # Pad with zeros if smaller
                    normalized_policies[i, :policy_len] = policy
                else:
                    # Truncate if larger
                    normalized_policies[i] = policy[:action_space_size]
            # else: already zeros from pre-allocation
            if len(exp) > 4 and exp[4] is not None:
                try:
                    source_ids[i] = int(exp[4])
                except (TypeError, ValueError):
                    source_ids[i] = -1
        
        policy_targets = normalized_policies
        # Ensure FENs are handled as a list of strings, not converted to numpy array directly if they vary in length etc.
        # fens = [exp[2] for exp in batch] 
        boards = [exp[2] for exp in batch] # Now these are board objects
        # Pre-allocate value targets array
        value_targets = np.array([exp[3] for exp in batch], dtype=np.float32).reshape(-1, 1)

        return states, policy_targets, boards, value_targets, source_ids # Return boards

    def __len__(self):
        return len(self.buffer)
    
    def get_state(self, env_type='chess'):
        """Returns the replay buffer state for checkpointing in compressed format.
        
        Args:
            env_type: Type of environment ('chess' or 'gomoku')
        """
        if len(self.buffer) == 0:
            return {
                'buffer': [],
                'maxlen': self.buffer.maxlen,
                'compressed': False
            }
        
        # Convert experiences to a more compact format
        # Store board as serialized strings instead of full board objects
        compact_buffer = []
        for exp in self.buffer:
            if len(exp) == 5:
                state, policy, board, value, source_id = exp
            else:
                state, policy, board, value = exp
                source_id = None
            # Handle None policy - replace with zero array of default action space size
            if policy is None:
                # Default action space size (4672 for legacy, can be overridden)
                default_action_space = 4672
                policy = np.zeros(default_action_space, dtype=np.float32)
            elif not isinstance(policy, np.ndarray):
                # If policy is not an array, create zero array
                default_action_space = 4672
                policy = np.zeros(default_action_space, dtype=np.float32)
            
            # Serialize board based on environment type
            if env_type == 'chess':
                # Convert board to FEN string (much smaller than board object)
                board_str = board.fen()
            elif env_type == 'gomoku':
                # Serialize gomoku board as: size,move_count;row0;row1;...
                size = board.size
                move_count = board.move
                rows = [','.join(str(board.board_state[i][j]) for j in range(size)) for i in range(size)]
                board_str = f"{size},{move_count};{';'.join(rows)}"
            else:
                # Fallback: try FEN if available
                board_str = board.fen() if hasattr(board, 'fen') else str(board)
            
            compact_exp = (
                state.astype(np.float16),  # Use half precision to save space
                policy.astype(np.float16),
                board_str,
                float(value),
                source_id
            )
            compact_buffer.append(compact_exp)
        
        # Compress the buffer data using gzip
        buffer_bytes = pickle.dumps(compact_buffer, protocol=pickle.HIGHEST_PROTOCOL)
        compressed_bytes = gzip.compress(buffer_bytes, compresslevel=6)
        
        return {
            'buffer_compressed': compressed_bytes,
            'maxlen': self.buffer.maxlen,
            'env_type': env_type,
            'compressed': True
        }
    
    def load_state(self, state, board_factory_fn=None):
        """Loads the replay buffer state from a checkpoint.
        
        Args:
            state: The replay buffer state dictionary
            board_factory_fn: Function that takes a serialized board string and returns a board object
                             (only needed for compressed format)
        """
        if state is None:
            return
        
        # Check if this is compressed format
        if state.get('compressed', False):
            # Decompress and reconstruct
            compressed_bytes = state['buffer_compressed']
            buffer_bytes = gzip.decompress(compressed_bytes)
            compact_buffer = pickle.loads(buffer_bytes)
            
            if board_factory_fn is None:
                raise ValueError("board_factory_fn is required for loading compressed replay buffer")
            
            # Convert back to full format with board objects
            full_buffer = []
            for compact_exp in compact_buffer:
                if len(compact_exp) == 5:
                    state_data, policy, board_str, value, source_id = compact_exp
                else:
                    state_data, policy, board_str, value = compact_exp
                    source_id = None
                # Reconstruct board object from serialized string
                board = board_factory_fn(board_str)
                exp = (
                    state_data.astype(np.float32),  # Convert back to float32
                    policy.astype(np.float32),
                    board,
                    value,
                    source_id
                ) if source_id is not None else (
                    state_data.astype(np.float32),  # Convert back to float32
                    policy.astype(np.float32),
                    board,
                    value
                )
                full_buffer.append(exp)
            
            self.buffer = deque(full_buffer, maxlen=state['maxlen'])
        else:
            # Legacy uncompressed format
            self.buffer = deque(state['buffer'], maxlen=state['maxlen'])


# --- Self-Play Function (Update args to use config subsections) ---
def run_self_play_game(cfg: OmegaConf, network: nn.Module | None, env=None,
                       progress: Progress | None = None, device: torch.device | None = None):
    """Plays one game of self-play using MCTS and returns the game data."""
    if env is None:
        env = create_environment(cfg, render=device.type == 'cpu' and not cfg.training.get('use_multiprocessing', False))
    
    # Handle initial_board_fen: supports single dataset or weighted list of datasets
    initial_fen = None
    initial_position_quality = None
    initial_dataset_id = None
    initial_dataset_label = None
    max_game_moves_override = None
    initial_board_fen_cfg = cfg.training.get('initial_board_fen', None)
    
    # Convert OmegaConf DictConfig to plain dict if needed (for pickling/multiprocessing)
    if initial_board_fen_cfg is not None:
        try:
            # Convert OmegaConf to plain dict for compatibility with multiprocessing
            from omegaconf import OmegaConf
            if OmegaConf.is_config(initial_board_fen_cfg):
                initial_board_fen_cfg = OmegaConf.to_container(initial_board_fen_cfg, resolve=True)
        except Exception:
            # If conversion fails, try to use as-is (might already be a plain dict)
            pass

        if isinstance(initial_board_fen_cfg, list):
            dataset_entries, normalized_weights, _labels = _get_cached_dataset_entries(initial_board_fen_cfg)
            if dataset_entries and normalized_weights:
                selected_idx = random.choices(range(len(dataset_entries)), weights=normalized_weights, k=1)[0]
                selected_entry = dataset_entries[selected_idx]
                initial_dataset_id = selected_idx
                initial_dataset_label = selected_entry["label"]
                max_game_moves_override = selected_entry.get("max_game_moves")
                initial_fen, initial_position_quality = _select_fen_from_source(selected_entry["source"])
        else:
            initial_fen, initial_position_quality = _select_fen_from_source(initial_board_fen_cfg)
            if isinstance(initial_board_fen_cfg, str) and initial_board_fen_cfg.endswith(".json"):
                initial_dataset_label = _shorten_dataset_label(initial_board_fen_cfg)
            if isinstance(initial_board_fen_cfg, dict):
                max_game_moves_override = initial_board_fen_cfg.get("max_game_moves")
    
    options = {
        'fen': initial_fen
    } if initial_fen else None
    obs, _ = env.reset(options=options)
    if network is not None:
        network.eval()

    # Initialize reward computer
    reward_computer = RewardComputer(cfg, network, device)

    # If we don't have a hardcoded quality from config, assume side-to-move is winning.
    # Store quality from White's perspective to stay compatible with draw reward logic.
    if initial_position_quality is None:
        try:
            white_to_move = bool(env.board.turn)
        except Exception:
            white_to_move = True
        initial_position_quality = 'winning' if white_to_move else 'losing'

    game_history = []
    move_list_san = []  # Track moves in SAN notation
    move_count = 0
    terminated = False
    truncated = False

    mcts_iterations = cfg.mcts.iterations
    c_puct = cfg.mcts.c_puct
    temp_start = cfg.mcts.temperature_start
    temp_end = cfg.mcts.temperature_end
    temp_decay_moves = cfg.mcts.temperature_decay_moves
    temp_custom_start = cfg.mcts.get('temperature_custom_start', 0.1)
    dirichlet_alpha = cfg.mcts.dirichlet_alpha
    dirichlet_epsilon = cfg.mcts.dirichlet_epsilon
    action_space_size = cfg.network.action_space_size
    max_moves = cfg.training.max_game_moves
    if isinstance(max_game_moves_override, (int, float)) and max_game_moves_override > 0:
        max_moves = int(max_game_moves_override)
    
    # Check if we started from a non-standard position
    # Standard starting position board part: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    initial_board_fen = env.board.fen()
    standard_starting_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    is_custom_start = not initial_board_fen.startswith(standard_starting_position)

    # Track MCTS tree statistics
    tree_stats_list = []  # Store stats for each move
    gpu_cleanup_interval = 15  # Clean GPU memory every N moves (increased from 5 to reduce overhead)

    # Manual position tracking for repetition detection (fallback if board's move_stack is corrupted)
    # This tracks positions independently of the board's move_stack, making it robust even when
    # the board state is reset or copied during MCTS exploration or rendering
    position_counts = {}  # Track position occurrences for repetition detection
    position_tracking_errors = 0  # Track errors in position tracking for debugging
    
    # Track previous turn to detect same-color double moves
    previous_turn = None  # Will be set after first move

    task_id_game = None
    if progress is not None:
        task_id_game = progress.add_task(
            f"Max Moves (0/{max_moves})",
            total=max_moves,
            transient=True,
            games=0,
            steps=0,
        )
        progress.start_task(task_id_game)

    # Cache FEN string to avoid repeated calls to env.board.fen()
    # FEN is only recomputed after board changes (after env.step())
    cached_fen = env.board.fen()

    while not terminated and not truncated and move_count < max_moves:
        # Use custom temperature for non-standard starting positions (endgames, custom positions, etc.)
        if is_custom_start:
            temperature = temp_custom_start
        else:
            # Normal temperature decay for standard starting positions
            temperature = temp_start * ((temp_end / temp_start) ** min(1.0, move_count / temp_decay_moves))
        fen = cached_fen  # Use cached FEN instead of calling env.board.fen()
        
        # Backup position tracking before MCTS (in case board gets modified during exploration)
        # This ensures we have a position snapshot even if the board state changes
        try:
            fen_parts = fen.split()
            if len(fen_parts) >= 4:
                position_key = ' '.join(fen_parts[:4])  # Board position, active color, castling, en passant
                # Initialize count if not present (actual increment happens after move)
                if position_key not in position_counts:
                    position_counts[position_key] = 0
        except Exception as e:
            position_tracking_errors += 1
            # Log but continue - this is just a backup, main tracking happens after env.step()
        
        # Save board state before MCTS to restore it afterwards
        # This ensures board state is correct even if MCTS exploration modifies it
        # Use FEN string for restoration (more efficient than full board copy)
        # The FEN string is immutable and won't be affected by any board modifications
        fen_string_before_mcts = cached_fen  # Use cached FEN instead of calling env.board.fen()
        
        if cfg.mcts.iterations > 0:
            # Use stack=False for MCTS nodes - they only need current position (FEN) for exploration.
            # Repetition detection is handled in the actual game board, not in MCTS tree nodes.
            root_node = MCTSNode(env.board.copy(stack=False))
            mcts_env = env if cfg.env.render_mode == 'human' and not cfg.training.get('use_multiprocessing', False) else None
            draw_reward = cfg.training.get('draw_reward', -0.0)
            # MCTS needs a numeric draw_reward value (use default if None for position-aware)
            draw_reward_for_mcts = draw_reward if draw_reward is not None else -0.0
            mcts_player = MCTS(
                network,
                device=device,
                env=mcts_env,
                C_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                action_space_size=action_space_size,
                history_steps=cfg.env.history_steps,
                draw_reward=draw_reward_for_mcts
            )
            mcts_start = time.perf_counter()
            mcts_player.search(root_node, mcts_iterations, batch_size=cfg.mcts.batch_size, progress=progress)
            mcts_elapsed = time.perf_counter() - mcts_start
            mcts_policy = mcts_player.get_policy_distribution(root_node, temperature=temperature)
            
            # Get value prediction from root node (MCTS value estimate, better than raw network value)
            # This is the value from the current player's perspective
            root_value = root_node.Q() if root_node.N > 0 else None
            
            # Monitor MCTS tree size
            try:
                tree_stats = root_node.get_tree_stats()
                tree_stats_list.append(tree_stats)
            except Exception:
                # If tree stats calculation fails, continue without it
                pass
            
            # AlphaZero-style: sample only from legal actions
            legal_actions = get_legal_actions(env.board)
            if legal_actions:
                legal_indices = [a - 1 for a in legal_actions if 0 <= (a - 1) < len(mcts_policy)]
                if legal_indices:
                    legal_probs = np.array([mcts_policy[i] for i in legal_indices], dtype=np.float64)
                    if legal_probs.sum() > 0:
                        legal_probs = legal_probs / legal_probs.sum()
                        action_to_take = np.random.choice([i + 1 for i in legal_indices], p=legal_probs)
                    else:
                        action_to_take = np.random.choice(legal_actions)
                else:
                    action_to_take = np.random.choice(legal_actions)
            else:
                # Fallback to full policy if no legal actions (should be terminal)
                action_to_take = np.random.choice(len(mcts_policy), p=mcts_policy) + 1  # +1 because actions are 1-indexed
            
            # Save a copy of mcts_policy before cleanup (needed for game_history)
            mcts_policy_copy = mcts_policy.copy()
            
            # Explicit cleanup: Delete MCTS tree and player to free memory immediately
            del root_node
            del mcts_player
            mcts_policy = None  # Help GC
            mcts_policy = mcts_policy_copy  # Restore for game_history
            
            # Periodic GPU memory cleanup
            if device.type == 'cuda' and (move_count + 1) % gpu_cleanup_interval == 0:
                try:
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        else:
            mcts_policy = np.zeros(cfg.network.action_space_size)
            legal_actions = get_legal_actions(env.board)
            for action_id in legal_actions:
                mcts_policy[action_id - 1] = 1
            action_to_take = np.random.choice(legal_actions)
            root_value = None  # No MCTS, so no value available

        current_obs = obs
        # Use stack=False to avoid copying expensive move history stack
        # Board object is still needed for legal_actions generation during training
        board_copy_at_state = env.board.copy(stack=False)
        # Store value prediction if available (from MCTS root node)
        game_history.append((current_obs, mcts_policy, board_copy_at_state, root_value))
        # Clear mcts_policy after appending to help GC
        mcts_policy = None
        
        # Restore board state after MCTS to ensure it's in the correct state
        # MCTS might have modified the board during exploration, so we restore from the saved state
        # This is critical to ensure action_id_to_move() returns legal moves
        try:
            # Restore board from the FEN string saved before MCTS (more reliable than board copy)
            # This ensures we restore to the exact state we saved, even if the board copy was modified
            env.board.set_fen(fen_string_before_mcts)
            
            # Note: We don't restore move_stack because set_fen() already restores the board state correctly
            # The FEN contains all necessary information (position, turn, castling rights, en passant)
            # Repetition detection uses FEN-based position tracking, not move_stack
        except Exception as e:
            # If restoration fails, log warning but continue
            # The board might still be in a usable state
            logging.warning(
                f"Failed to restore board state after MCTS: {e}. "
                f"Current FEN: {env.board.fen()}, Expected FEN: {fen_string_before_mcts}"
            )
        
        # Log only if board restoration failed (position differs, not just move counter) or action is no longer legal
        # Compare only the position part of FEN (first 4 parts), not move counters
        # Recompute FEN after restoration to check if it matches
        fen_after_restore = env.board.fen()
        fen_after_parts = fen_after_restore.split()
        fen_before_parts = fen_string_before_mcts.split()
        position_matches = (len(fen_after_parts) >= 4 and len(fen_before_parts) >= 4 and 
                           fen_after_parts[:4] == fen_before_parts[:4])
        
        if not position_matches or action_to_take not in env.board.legal_actions:
            pass  # Logging handled below if action is not legal
        
        # Record SAN notation BEFORE playing the move, but ensure board is in correct state
        # This prevents issues with pop/push affecting turn alternation
        if action_to_take not in env.board.legal_actions:
            # Enforce legal action selection (AlphaZero-style: only legal moves allowed)
            legal_actions = list(env.board.legal_actions)
            if legal_actions:
                if mcts_policy is not None and len(mcts_policy) > 0:
                    legal_indices = [a - 1 for a in legal_actions if 0 <= (a - 1) < len(mcts_policy)]
                    if legal_indices:
                        legal_probs = np.array([mcts_policy[i] for i in legal_indices], dtype=np.float64)
                        if legal_probs.sum() > 0:
                            legal_probs = legal_probs / legal_probs.sum()
                            action_to_take = np.random.choice([i + 1 for i in legal_indices], p=legal_probs)
                        else:
                            action_to_take = np.random.choice(legal_actions)
                    else:
                        action_to_take = np.random.choice(legal_actions)
                else:
                    action_to_take = np.random.choice(legal_actions)
                logging.warning(
                    f"Illegal action replaced with legal move: action={action_to_take}, "
                    f"root_value={root_value}, FEN={env.board.fen()}, "
                    f"legal_actions_count={len(legal_actions)}"
                )
            else:
                logging.warning(
                    f"No legal actions available after illegal action detection: "
                    f"action={action_to_take}, root_value={root_value}, FEN={env.board.fen()}"
                )
            # Continue with SAN logging using the corrected legal action (if any)

        if action_to_take not in env.board.legal_actions:
            san_move = "ILLEGAL"
        else:
            # Get the move and record SAN BEFORE playing it
            # This ensures we record from the correct board state without affecting turn alternation
            move = env.board.action_id_to_move(action_to_take)
            
            if move is not None:
                # Verify the move is actually legal before trying to get SAN
                # This prevents errors if board state changed or action_id_to_move returned wrong move
                legal_moves_list = list(env.board.legal_moves)
                move_in_legal = move in legal_moves_list
                
                # Log only if move is not in legal_moves (the bug we're tracking)
                if not move_in_legal:
                    # #region agent log
                    try:
                        with open('/Users/minseo/Documents/Github/_star14ms/Chess_AI/.cursor/debug.log', 'a') as f:
                            legal_moves_sample = [env.board.san(m) for m in legal_moves_list[:10]]
                            f.write(json.dumps({
                                'sessionId': 'debug-session',
                                'runId': 'run1',
                                'hypothesisId': 'C',
                                'location': 'train.py:598',
                                'message': 'ILLEGAL MOVE DETECTED',
                                'data': {
                                    'action_id': action_to_take,
                                    'move_uci': move.uci(),
                                    'move_count': move_count,
                                    'fen': env.board.fen(),
                                    'legal_moves_sample': legal_moves_sample,
                                    'turn': 'White' if env.board.turn == chess.WHITE else 'Black'
                                },
                                'timestamp': int(time.time() * 1000)
                            }) + '\n')
                    except: pass
                    # #endregion
                    pass  # Move will be handled as illegal below
                
                if move_in_legal:
                    # Get SAN notation for the move
                    # Note: legal_moves override ensures moves that don't escape check are filtered out
                    try:
                        san_move = env.board.san(move)
                    except Exception as e:
                        # board.san() can fail even for legal moves in some edge cases (python-chess limitation)
                        # Fallback to UCI notation if SAN generation fails
                        san_move = move.uci()
                        logging.warning(
                            f"SAN generation failed for action {action_to_take}: "
                            f"move={move.uci()}, root_value={root_value}, "
                            f"FEN={env.board.fen()}, error={e}"
                        )
                else:
                    # Move is not legal - this shouldn't happen but handle gracefully
                    # Use UCI notation as fallback
                    san_move = move.uci()
                    logging.warning(
                        f"action_id_to_move returned non-legal move for action {action_to_take}: "
                        f"move={move.uci()}, FEN={env.board.fen()}, "
                        f"legal_actions_count={len(env.board.legal_actions)}, "
                        f"move_in_legal_moves={move in env.board.legal_moves}"
                    )
                    # Logging already done above in the "if not move_in_legal" block
            else:
                # Should not happen if action is in legal_actions, but handle gracefully
                san_move = "ILLEGAL"
                logging.warning(
                    f"action_id_to_move returned None for action {action_to_take}: "
                    f"root_value={root_value}, "
                    f"FEN={env.board.fen()}, legal_actions_count={len(env.board.legal_actions)}"
                )
        
        move_list_san.append(san_move)
        
        # Check for same-color double moves before applying the move
        current_turn_before_move = env.board.turn
        current_color_before = 'White' if current_turn_before_move == chess.WHITE else 'Black'
        
        # Also check if the move being attempted matches the current turn
        # This catches cases where a move for the wrong color is being attempted
        move_color_mismatch = False
        if san_move != "ILLEGAL" and action_to_take in env.board.legal_actions:
            try:
                move = env.board.action_id_to_move(action_to_take)
                if move is not None:
                    # Check if the move's from_square has a piece of the correct color
                    piece = env.board.piece_at(move.from_square)
                    if piece is not None:
                        piece_color = 'White' if piece.color == chess.WHITE else 'Black'
                        if piece_color != current_color_before:
                            move_color_mismatch = True
            except Exception:
                pass  # If we can't check, continue
        
        if previous_turn is not None and current_turn_before_move == previous_turn:
            # Same color is trying to move twice in a row!
            logging.error(
                f"CRITICAL BUG: Same color double move detected at move {move_count + 1}! "
                f"Previous turn: {'White' if previous_turn == chess.WHITE else 'Black'}, "
                f"Current turn: {current_color_before}, "
                f"Move: {san_move}, Action: {action_to_take}, "
                f"FEN: {env.board.fen()}"
            )
        
        if move_color_mismatch:
            # Move is for the wrong color
            logging.error(
                f"CRITICAL BUG: Move color mismatch at move {move_count + 1}! "
                f"Current turn: {current_color_before}, "
                f"Move: {san_move}, Action: {action_to_take}, "
                f"FEN: {env.board.fen()}"
            )
        
        # Now play the move - this will switch turns correctly
        obs, _, terminated, truncated, info = env.step(action_to_take)
        
        # Update cached FEN after board change (env.step() modifies the board)
        cached_fen = env.board.fen()
        
        # Update previous_turn for next iteration (after move, turn has changed)
        previous_turn = current_turn_before_move  # Store the turn BEFORE the move (who just moved)
        
        # Manual repetition detection (fallback if board's move_stack is corrupted)
        # Track position using FEN without move counters (first 4 parts: board, active color, castling, en passant)
        # This works independently of the board's move_stack, so it's reliable even when
        # the board state is reset or copied during MCTS exploration or rendering
        try:
            current_fen = cached_fen  # Use cached FEN instead of calling env.board.fen()
            fen_parts = current_fen.split()
            
            # Validate FEN format (python-chess FEN should have at least 4 parts)
            if len(fen_parts) >= 4:
                # Use first 4 parts to uniquely identify position:
                # 1. Board position (piece placement)
                # 2. Active color (w/b)
                # 3. Castling rights
                # 4. En passant target square
                # We exclude move counters (parts 5-6) as they don't affect position identity
                position_key = ' '.join(fen_parts[:4])
                position_counts[position_key] = position_counts.get(position_key, 0) + 1
                repetition_count = position_counts[position_key]
                
                # Fivefold repetition - automatic draw (FIDE rule 9.6.2)
                # This is mandatory and doesn't require a claim
                if repetition_count >= 5:
                    terminated = True
                    if progress is not None:
                        progress.update(task_id_game, description=f"Terminated: FIVEfold repetition (move {move_count+1})", advance=0)
                # Threefold repetition - claimable draw (FIDE rule 9.6.1)
                # Only terminates if claim_draw is enabled (which it is by default)
                elif repetition_count >= 3 and env.claim_draw:
                    terminated = True
                    if progress is not None:
                        progress.update(task_id_game, description=f"Terminated: THREEfold repetition (move {move_count+1})", advance=0)
            else:
                # FEN format unexpected - this shouldn't happen with python-chess
                position_tracking_errors += 1
                # Log warning but continue - rely on board's detection if available
                import warnings
                warnings.warn(
                    f"Unexpected FEN format at move {move_count+1}: "
                    f"expected >=4 parts, got {len(fen_parts)}. "
                    f"FEN: {current_fen[:100]}",
                    RuntimeWarning
                )
        except Exception as e:
            # If position tracking fails, log it but continue (rely on board's detection if available)
            position_tracking_errors += 1
            import warnings
            warnings.warn(
                f"Position tracking failed at move {move_count+1}: {e}. "
                f"Falling back to board's repetition detection.",
                RuntimeWarning
            )

        if progress is not None:
            progress.update(task_id_game, description=f"Max Moves ({move_count+1}/{max_moves}) | temp={temperature:.3f}", advance=1)

        move_count += 1
        
        # Force garbage collection periodically to help with memory cleanup
        if move_count % gpu_cleanup_interval == 0:
            gc.collect()

    # Get draw reward from config (can be None for position-aware rewards)
    draw_reward = cfg.training.get('draw_reward', -0.0)
    # Use a default value for calculate_chess_reward (it needs a numeric value)
    draw_reward_for_calc = draw_reward if draw_reward is not None else -0.0
    
    # Get final game result from the perspective of the player whose turn it is at the final state
    # calculate_chess_reward returns from previous player's perspective (who just moved)
    # Since the move was applied, board.turn changed, so we flip to get current player's perspective
    from MCTS.training_modules.chess import calculate_chess_reward
    final_value_from_prev = calculate_chess_reward(env.board, claim_draw=True, draw_reward=draw_reward_for_calc)
    # For draws, we use a sentinel value to detect them
    draw_sentinel = draw_reward_for_calc
    if abs(final_value_from_prev - draw_sentinel) < 0.01:
        final_value = draw_sentinel  # Draws are detected by matching this sentinel value
    else:
        # Convert from previous player's perspective to current player's perspective (flip)
        final_value = -final_value_from_prev

    # Get game outcome/termination reason
    # Check if we terminated due to manual repetition detection
    manual_repetition = False
    winner = None  # Track actual winner: chess.WHITE, chess.BLACK, or None for draw
    if position_counts:
        max_repetition = max(position_counts.values()) if position_counts.values() else 0
        if max_repetition >= 5:
            termination_reason = "FIVEFOLD_REPETITION"
            manual_repetition = True
            # Force draw result for fivefold repetition (use sentinel value)
            final_value = draw_sentinel
            winner = None  # Draw
        elif max_repetition >= 3:
            termination_reason = "THREEFOLD_REPETITION"
            manual_repetition = True
            # Force draw result for threefold repetition (use sentinel value)
            final_value = draw_sentinel
            winner = None  # Draw
    
    # If we didn't detect manual repetition, check board's outcome
    # (Note: board's detection may have failed if move_stack was corrupted)
    if not manual_repetition:
        outcome = env.board.outcome(claim_draw=True)
        if outcome:
            termination_reason = outcome.termination.name
            winner = outcome.winner  # Store actual winner (chess.WHITE, chess.BLACK, or None)
            # If board detected a draw (threefold/fivefold repetition, stalemate, etc.), ensure result is draw
            if outcome.winner is None:  # Draw
                final_value = draw_sentinel
        elif truncated or move_count >= max_moves:
            termination_reason = "MAX_MOVES"
            # Max moves is typically a draw (use sentinel value)
            final_value = draw_sentinel
            winner = None  # Draw
        else:
            termination_reason = "UNKNOWN"
            winner = None
    
    # Log position tracking statistics for debugging (if there were errors)
    if position_tracking_errors > 0:
        import warnings
        warnings.warn(
            f"Position tracking encountered {position_tracking_errors} error(s) during game. "
            f"Total positions tracked: {len(position_counts)}, "
            f"Max repetition: {max(position_counts.values()) if position_counts else 0}",
            RuntimeWarning
        )

    # AlphaZero-style value targets: final outcome from the current player's perspective
    draw_value = draw_reward if draw_reward is not None else 0.0

    def _value_from_winner(board_at_state, outcome_winner, draw_val):
        if outcome_winner is None:
            return float(draw_val)
        if board_at_state.turn == chess.WHITE:
            return 1.0 if outcome_winner == chess.WHITE else -1.0
        return 1.0 if outcome_winner == chess.BLACK else -1.0

    full_game_data = []
    for i, history_item in enumerate(game_history):
        # New format only: (state_obs, policy_target, board_at_state, mcts_value)
        if len(history_item) != 4:
            raise ValueError(
                f"Invalid game_history item length: expected 4, got {len(history_item)}"
            )
        state_obs, policy_target, board_at_state, _value_target = history_item
        outcome_value = _value_from_winner(board_at_state, winner, draw_value)
        if initial_dataset_id is not None:
            full_game_data.append((state_obs, policy_target, board_at_state, outcome_value, initial_dataset_id))
        else:
            full_game_data.append((state_obs, policy_target, board_at_state, outcome_value))
    
    if progress is not None and task_id_game is not None:
        progress.update(task_id_game, visible=False)
    
    # Calculate average tree statistics
    avg_tree_stats = {}
    if tree_stats_list:
        avg_tree_stats = {
            'avg_nodes': sum(s['node_count'] for s in tree_stats_list) / len(tree_stats_list),
            'max_nodes': max(s['node_count'] for s in tree_stats_list),
            'avg_depth': sum(s['max_depth'] for s in tree_stats_list) / len(tree_stats_list),
            'max_depth': max(s['max_depth'] for s in tree_stats_list),
            'avg_branching': sum(s['avg_branching'] for s in tree_stats_list) / len(tree_stats_list)
        }
    
    # Store the actual computed reward for the first state (for logging/debugging)
    actual_reward_for_logging = None
    if full_game_data:
        first_state_board = full_game_data[0][2] if len(full_game_data[0]) > 2 else None
        if first_state_board is not None:
            actual_reward_for_logging = _value_from_winner(first_state_board, winner, draw_value)
        else:
            actual_reward_for_logging = full_game_data[0][3] if len(full_game_data[0]) > 3 else final_value
    
    # Return game data, move list in SAN, and termination reason
    game_info = {
        'moves_san': ' '.join(move_list_san),
        'termination': termination_reason,
        'move_count': move_count,
        'result': final_value,  # Sentinel value for draw detection
        'actual_reward': actual_reward_for_logging if actual_reward_for_logging is not None else final_value,  # Actual computed reward
        'winner': winner,  # Store actual winner: chess.WHITE, chess.BLACK, or None for draw
        'tree_stats': avg_tree_stats if avg_tree_stats else None,
        'draw_reward': draw_reward,  # Store draw_reward for reference in statistics
        'initial_fen': initial_fen,  # Store initial board FEN for game history
        'initial_position_quality': initial_position_quality,  # Store initial position quality
        'initial_dataset_label': initial_dataset_label,
        'initial_dataset_id': initial_dataset_id,
    }

    return (full_game_data, game_info)


def _parse_optional_seconds(value):
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "none", "null"):
            return None
        return int(float(s))
    if isinstance(value, (int, float)):
        return int(value)
    raise TypeError(
        f"training.max_training_time_seconds must be int/float/str/None, got {type(value)}: {value!r}"
    )


def _select_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _init_progress(cfg: DictConfig) -> tuple[Progress, bool]:
    progress = NullProgress(rich=cfg.training.progress_bar)
    show_progress = bool(cfg.training.get("progress_bar", True))
    if show_progress:
        default_columns = (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        )
        progress = Progress(*default_columns, transient=False)
        progress.start()
    return progress, show_progress


def _init_network_and_optimizer(
    cfg: DictConfig, device: torch.device, progress: Progress
) -> tuple[nn.Module, optim.Optimizer, DictConfig, float]:
    network = create_network(cfg, device)
    network.eval()

    profile_network = create_network(cfg, device)
    profile_network.eval()
    N, C, H, W = cfg.training.batch_size, cfg.network.input_channels, cfg.network.board_size, cfg.network.board_size
    profile_model(profile_network, (torch.randn(N, C, H, W).to(device),))
    del profile_network

    opt_cfg = cfg.optimizer
    actual_learning_rate = opt_cfg.learning_rate
    if opt_cfg.type == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=opt_cfg.learning_rate, weight_decay=opt_cfg.weight_decay)
    elif opt_cfg.type == "SGD":
        momentum = opt_cfg.momentum if opt_cfg.momentum is not None else 0.9
        optimizer = optim.SGD(
            network.parameters(),
            lr=opt_cfg.learning_rate,
            momentum=momentum,
            weight_decay=opt_cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_cfg.type}")
    progress.print(f"Optimizer initialized: {opt_cfg.type}")
    return network, optimizer, opt_cfg, actual_learning_rate


def _init_replay_buffer(cfg: DictConfig, progress: Progress):
    replay_buffer = ReplayBuffer(cfg.training.replay_buffer_size)
    return replay_buffer


def _init_checkpoint_dirs(cfg: DictConfig, progress: Progress):
    checkpoint_dir = cfg.training.checkpoint_dir
    checkpoint_dir_load = cfg.training.get("checkpoint_dir_load", None)
    load_dir = checkpoint_dir_load if checkpoint_dir_load not in (None, "", "null") else checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    progress.print(f"Checkpoints will be saved in: {os.path.abspath(checkpoint_dir)}")
    return checkpoint_dir, load_dir


def _init_game_history_dir(cfg: DictConfig, progress: Progress):
    game_history_dir = cfg.training.get("game_history_dir", None)
    if game_history_dir not in (None, "", "null"):
        os.makedirs(game_history_dir, exist_ok=True)
        progress.print(f"Game histories will be saved in: {os.path.abspath(game_history_dir)}")
    else:
        game_history_dir = None
    return game_history_dir


def _save_game_history(
    game_moves_list,
    game_history_dir: str | None,
    iteration: int,
    avg_policy_loss: float,
    avg_value_loss: float,
):
    if not game_moves_list or not game_history_dir:
        return
    games_file = os.path.join(
        game_history_dir,
        f"games_iter_{iteration+1}_p{avg_policy_loss:.4f}_v{avg_value_loss:.4f}.txt",
    )
    with open(games_file, 'w') as f:
        for i, game_info in enumerate(game_moves_list):
            # Use actual winner to determine result string
            import chess
            winner = game_info.get('winner', None)
            if winner is None:
                result_str = "1/2-1/2"  # Draw
            elif winner == chess.WHITE:
                result_str = "1-0"  # White won
            else:
                result_str = "0-1"  # Black won
            f.write(f"Game {i+1}: {result_str} ({game_info['termination']}, {game_info['move_count']} moves)\n")
            # Write initial FEN and quality if available
            initial_fen = game_info.get('initial_fen', None)
            initial_quality = game_info.get('initial_position_quality', None)
            if initial_fen or initial_quality:
                parts = []
                if initial_fen:
                    parts.append(f"Initial FEN: {initial_fen}")
                if initial_quality:
                    parts.append(f"White's perspective: {initial_quality}")
                f.write(" | ".join(parts) + "\n")
            # Write reward value (use actual_reward if available, otherwise result)
            actual_reward = game_info.get('actual_reward', None)
            result_value = game_info.get('result', None)
            if actual_reward is not None:
                f.write(f"Reward: {actual_reward:.4f}\n")
            elif result_value is not None:
                f.write(f"Reward: {result_value:.4f}\n")
            f.write(f"{game_info['moves_san']}\n\n")


def _init_self_play_infra(cfg: DictConfig):
    manager = multiprocessing.Manager()
    sp_queue = manager.Queue(maxsize=cfg.training.get("continual_queue_maxsize", 64))
    stop_event = manager.Event()
    continual_enabled = bool(cfg.training.get("continual_training", False))
    actors: list[multiprocessing.Process] = []
    actor_device_map: dict[int, str] = {}
    return manager, sp_queue, stop_event, continual_enabled, actors, actor_device_map


def _load_checkpoint(
    cfg: DictConfig,
    checkpoint_path: str,
    network: nn.Module,
    optimizer: optim.Optimizer,
    opt_cfg: DictConfig,
    device: torch.device,
    progress: Progress,
    replay_buffer,
    actual_learning_rate: float,
    history: dict,
) -> tuple[int, int, float, bool, dict]:
    if not os.path.exists(checkpoint_path):
        progress.print("No existing checkpoint found. Starting training from scratch...")
        return 0, 0, actual_learning_rate, False, history

    progress.print(f"\nFound existing checkpoint at {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.to(device)

        if "optimizer_state_dict" in checkpoint:
            try:
                checkpoint_opt_state = checkpoint["optimizer_state_dict"]
                if "param_groups" in checkpoint_opt_state and len(checkpoint_opt_state["param_groups"]) > 0:
                    checkpoint_lr = checkpoint_opt_state["param_groups"][0].get(
                        "lr", opt_cfg.learning_rate
                    )
                    if checkpoint_lr != opt_cfg.learning_rate:
                        actual_learning_rate = checkpoint_lr
                        progress.print(
                            f"Using learning rate from checkpoint: {checkpoint_lr} (config had: {opt_cfg.learning_rate})"
                        )
                    else:
                        actual_learning_rate = checkpoint_lr

                skip_optimizer_load = False
                optimizer_param_names = checkpoint.get("optimizer_param_names")
                restored_by_name = False
                if "param_groups" in checkpoint_opt_state:
                    ckpt_groups = checkpoint_opt_state["param_groups"]
                    opt_groups = optimizer.param_groups
                    if len(ckpt_groups) != len(opt_groups):
                        skip_optimizer_load = True
                    for ckpt_group, opt_group in zip(ckpt_groups, opt_groups):
                        if len(ckpt_group.get("params", [])) != len(opt_group.get("params", [])):
                            skip_optimizer_load = True
                            break

                use_manual_state_restore = False
                if skip_optimizer_load and optimizer_param_names:
                    try:
                        name_to_param = dict(network.named_parameters())
                        new_state = optimizer.state_dict()
                        loaded_states = 0
                        total_candidates = 0
                        state_dict = checkpoint_opt_state.get("state", {})
                        for group, names in zip(
                            checkpoint_opt_state.get("param_groups", []), optimizer_param_names
                        ):
                            for old_param_id, param_name in zip(group.get("params", []), names):
                                if not isinstance(param_name, str) or not param_name:
                                    continue
                                if not isinstance(old_param_id, int):
                                    try:
                                        if isinstance(old_param_id, torch.Tensor):
                                            old_param_id = int(old_param_id.item())
                                        else:
                                            old_param_id = int(old_param_id)
                                    except Exception:
                                        continue
                                total_candidates += 1
                                param = name_to_param.get(param_name)
                                if param is None:
                                    continue
                                if old_param_id in state_dict:
                                    new_state["state"][id(param)] = state_dict[old_param_id]
                                    loaded_states += 1

                        optimizer.state.clear()
                        for group, names in zip(
                            checkpoint_opt_state.get("param_groups", []), optimizer_param_names
                        ):
                            for old_param_id, param_name in zip(group.get("params", []), names):
                                if not isinstance(param_name, str) or not param_name:
                                    continue
                                param = name_to_param.get(param_name)
                                if param is None:
                                    continue
                                if not isinstance(old_param_id, int):
                                    try:
                                        if isinstance(old_param_id, torch.Tensor):
                                            old_param_id = int(old_param_id.item())
                                        else:
                                            old_param_id = int(old_param_id)
                                    except Exception:
                                        continue
                                if old_param_id in state_dict:
                                    optimizer.state[param] = state_dict[old_param_id]
                        use_manual_state_restore = True
                        restored_by_name = loaded_states > 0
                        if restored_by_name:
                            progress.print(
                                f"Loaded optimizer state by name for {loaded_states}/{total_candidates} params"
                            )
                        else:
                            skip_optimizer_load = True
                    except Exception as exc:
                        progress.print(f"Warning: Name-based optimizer restore failed: {exc}")
                        skip_optimizer_load = True

                if skip_optimizer_load and not restored_by_name:
                    progress.print(
                        "Warning: Optimizer param_groups mismatch between checkpoint and current model. "
                        "Skipping optimizer state load."
                    )
                    raise RuntimeError("skip_optimizer_load")

                if not use_manual_state_restore:
                    optimizer.load_state_dict(checkpoint_opt_state)
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)

                restored_lr = optimizer.param_groups[0]["lr"]
                if abs(restored_lr - actual_learning_rate) > 1e-8:
                    progress.print(
                        f"Warning: Learning rate mismatch! Checkpoint had {actual_learning_rate}, restored to {restored_lr}"
                    )
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = actual_learning_rate

                if opt_cfg.type == "Adam" and optimizer.state:
                    first_param_id = optimizer.param_groups[0]["params"][0]
                    step_count = (
                        optimizer.state[first_param_id].get("step", 0)
                        if first_param_id in optimizer.state
                        else 0
                    )
                    progress.print(
                        f"Successfully loaded optimizer state: LR={restored_lr:.2e}, Adam step={step_count}"
                    )
                else:
                    progress.print(f"Successfully loaded optimizer state: LR={restored_lr:.2e}")
            except Exception as e:
                progress.print(
                    "Warning: Optimizer will start fresh and be reinitialized for RL training (this will cause higher initial losses)"
                )
                if opt_cfg.type == "Adam":
                    optimizer = optim.Adam(
                        network.parameters(),
                        lr=opt_cfg.learning_rate,
                        weight_decay=opt_cfg.weight_decay,
                    )
                elif opt_cfg.type == "SGD":
                    momentum = opt_cfg.momentum if opt_cfg.momentum is not None else 0.9
                    optimizer = optim.SGD(
                        network.parameters(),
                        lr=opt_cfg.learning_rate,
                        momentum=momentum,
                        weight_decay=opt_cfg.weight_decay,
                    )
                actual_learning_rate = opt_cfg.learning_rate
                if not isinstance(e, RuntimeError) or str(e) != "skip_optimizer_load":
                    import traceback

                    traceback.print_exc()
        else:
            progress.print("Warning: No optimizer state found in checkpoint")
            progress.print("Optimizer will start fresh (this will cause higher initial losses)")

        start_iter = int(checkpoint.get("iteration", 0))
        total_games_simulated = int(checkpoint.get("total_games_simulated", 0))

        if "history" in checkpoint:
            history = checkpoint["history"]
            if "non_draw_count" not in history:
                history["non_draw_count"] = [0] * len(history["policy_loss"])
            if "repetition_draw_count" not in history:
                history["repetition_draw_count"] = [0] * len(history["policy_loss"])
            if "other_draw_count" not in history:
                history["other_draw_count"] = [0] * len(history["policy_loss"])
            progress.print(
                f"Loaded training history with {len(history['policy_loss'])} recorded iterations"
            )

        buffer_loaded = False
        if "replay_buffer_state" in checkpoint:
            try:
                replay_buffer_state = checkpoint["replay_buffer_state"]
                replay_buffer.load_state(replay_buffer_state, create_board_from_serialized)

                progress.print(
                    f"Loaded replay buffer from checkpoint: {len(replay_buffer)} experiences"
                )
                buffer_loaded = True
            except Exception as e:
                progress.print(f"Warning: Failed to load replay buffer from checkpoint: {e}")

        if buffer_loaded:
            progress.print(
                f"Successfully loaded checkpoint from iteration {start_iter} with {total_games_simulated} games simulated and {len(replay_buffer)} experiences in replay buffer"
            )
        else:
            progress.print(
                f"Successfully loaded checkpoint from iteration {start_iter} with {total_games_simulated} games simulated (no replay buffer found)"
            )
        return start_iter, total_games_simulated, actual_learning_rate, buffer_loaded, history
    except Exception as e:
        progress.print(f"Error loading checkpoint: {e}")
        progress.print("Starting training from scratch...")
        return 0, 0, actual_learning_rate, False, history

# --- Training Loop Function --- 
# Use DictConfig for type hint from Hydra
def run_training_loop(cfg: DictConfig) -> None: 
    """Main function to run the training loop using Hydra config."""
    progress, show_progress = _init_progress(cfg)

    max_training_time_seconds = _parse_optional_seconds(
        cfg.training.get("max_training_time_seconds", None)
    )

    # --- Setup ---
    # Ensure factories are initialized in the main process
    initialize_factories_from_cfg(cfg)
    # Training device selection: always use fastest available accelerator
    device = _select_training_device()
    log_str_trining_device = f"{device} for training"

    # Print action space size
    progress.print(f"Using action space size: {cfg.network.action_space_size}")

    # Determine number of workers
    num_workers_config = cfg.training.get('self_play_workers', 0)
    total_cores = os.cpu_count()

    # Get optimal worker count using the utility function
    num_workers = get_optimal_worker_count(
        total_cores=total_cores,
        num_workers_config=num_workers_config,
        use_multiprocessing=cfg.training.use_multiprocessing
    )

    # Check config flag to decide execution mode
    use_multiprocessing_flag = cfg.training.get('use_multiprocessing', False)

    # Mode selection will occur after defining checkpoint and queue

    network, optimizer, opt_cfg, actual_learning_rate = _init_network_and_optimizer(
        cfg, device, progress
    )
    replay_buffer = _init_replay_buffer(cfg, progress)

    # Build a pooled initial FEN list to avoid per-worker JSON loads
    progress.print("Loading initial FEN datasets (sampling/weighting)...")
    fen_pool_start = time.perf_counter()
    pool_size = cfg.training.get("initial_fen_pool_size", None)
    if pool_size is None and use_multiprocessing_flag:
        pool_size = 50000
    pooled_initial_fen_cfg = _build_initial_fen_pool_cfg(
        cfg.training.get("initial_board_fen", None),
        int(pool_size) if pool_size is not None else None,
    )
    if pooled_initial_fen_cfg is not None:
        cfg.training.initial_board_fen = pooled_initial_fen_cfg

    progress.print(f"Initial FEN pooling completed in {time.perf_counter() - fen_pool_start:.2f}s")

    dataset_entries, _weights, source_labels = _get_cached_dataset_entries(cfg.training.get("initial_board_fen", None))

    # Loss Functions
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # Checkpoint directories (relative to hydra run dir)
    checkpoint_dir, load_dir = _init_checkpoint_dirs(cfg, progress)

    # Game history directory setup (optional)
    game_history_dir = _init_game_history_dir(cfg, progress)

    env = create_environment(cfg, render=not use_multiprocessing_flag)

    # Continual self-play setup (actors + queue)
    (
        manager,
        sp_queue,
        stop_event,
        continual_enabled,
        actors,
        actor_device_map,
    ) = _init_self_play_infra(cfg)

    # --- Mode selection for self-play ---
    # Note: Actor launching for continual mode happens after checkpoint loading
    # to check buffer size and exclude training device if we have enough data
    if continual_enabled:
        # Will launch actors after checkpoint loading
        pass
    elif use_multiprocessing_flag and num_workers > 1:
        device = "cpu"
        # Describe mixed actors plan
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            cpu_workers = max(0, num_workers - min(gpu_count, max(0, num_workers - 1))) if num_workers > 1 else num_workers
            gpu_workers = min(gpu_count, max(0, num_workers - 1)) if num_workers > 1 else 0
            progress.print(f"Using {num_workers} workers for self-play: {gpu_workers} GPU actor(s), {cpu_workers} CPU actor(s)")
        else:
            progress.print(f"Using {num_workers} CPU workers for self-play (no GPUs detected)")
    else:
        use_multiprocessing_flag = False
        progress.print(f"Using {device} for self-play, {log_str_trining_device}")

    # --- Main Training Loop ---
    start_iter = 0 # Initialize start_iter
    progress.print("Starting training loop...")
    total_training_start_time = time.time()
    total_games_simulated = 0
    
    # Initialize metric tracking lists for learning curves
    history = {
        'policy_loss': [],
        'value_loss': [],
        'illegal_move_ratio': [],
        'illegal_move_prob': [],
        'non_draw_count': [],
        'repetition_draw_count': [],
        'other_draw_count': [],
    }

    # Check for existing checkpoint
    checkpoint_path = os.path.join(load_dir, "model.pth")
    (
        start_iter,
        total_games_simulated,
        actual_learning_rate,
        buffer_loaded,
        history,
    ) = _load_checkpoint(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        network=network,
        optimizer=optimizer,
        opt_cfg=opt_cfg,
        device=device,
        progress=progress,
        replay_buffer=replay_buffer,
        actual_learning_rate=actual_learning_rate,
        history=history,
    )

    # --- Launch continual self-play actors (after checkpoint loading) ---
    if continual_enabled:
        # Helper function to get available accelerators (CUDA and MPS)
        def get_available_accelerators():
            accelerators = []
            if torch.cuda.is_available():
                for g in range(torch.cuda.device_count()):
                    accelerators.append(f"cuda:{g}")
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                accelerators.append("mps")
            return accelerators
        
        # Always use all available GPUs for self-play initially
        # Training GPU will be dynamically assigned to whichever GPU finishes first when we have enough data
        available_accelerators = get_available_accelerators()
        desired_processes = max(1, num_workers if num_workers > 0 else 1)
        device_pool = []
        
        # Use all available accelerators for self-play (training GPU will be assigned dynamically)
        accelerator_slots = min(len(available_accelerators), desired_processes)
        for i in range(accelerator_slots):
            device_pool.append(available_accelerators[i])
        
        # Fill remaining slots with CPU workers
        while len(device_pool) < desired_processes:
            device_pool.append('cpu')
        
        checkpoint_path_for_actors = os.path.join(cfg.training.checkpoint_dir, "model.pth")
        for dev in device_pool:
            p = multiprocessing.get_context("spawn").Process(
                target=continual_self_play_worker,
                args=(checkpoint_path_for_actors, cfg, dev, sp_queue, stop_event)
            )
            p.daemon = True
            p.start()
            actors.append(p)
            # Track both CUDA and MPS devices (not just CUDA)
            if dev.startswith('cuda:') or dev == 'mps':
                actor_device_map[p] = dev
        
        has_enough_data = len(replay_buffer) >= cfg.training.batch_size
        if has_enough_data:
            progress.print(f"Buffer has enough data ({len(replay_buffer)} >= {cfg.training.batch_size}), training accelerator will be assigned dynamically to first available accelerator")
        else:
            progress.print(f"Buffer too small ({len(replay_buffer)} < {cfg.training.batch_size}), using all {accelerator_slots} accelerator(s) for self-play")
        progress.print(f"Continual self-play: launched {len(actors)} actor(s): {device_pool}")

    # Track iteration durations for time prediction
    iteration_durations = []  # List to store duration of each completed iteration
    previous_total_elapsed_time = 0  # Track previous total elapsed time to calculate iteration duration
    
    # Use num_training_iterations from cfg
    for iteration in range(start_iter, cfg.training.num_training_iterations):
        iteration_start_time = time.time()
        progress.print(f"\n--- Training Iteration {iteration+1}/{cfg.training.num_training_iterations} ---")

        # --- Self-Play Phase --- 
        self_play_columns = (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[games]} Games (1st: {task.fields[first_wins]}, 2nd: {task.fields[second_wins]}, {task.fields[draw_rate]:.1f}% Draw), {task.fields[steps]} Steps"),
        )
        progress.columns = self_play_columns
        
        # Only move to CPU if using multiple workers
        if use_multiprocessing_flag:
            network.to('cpu').eval()
        else:
            network.to(device).eval()

        games_data_collected = []
        iteration_start_time_selfplay = time.time()
        # Get draw reward from config (can be None for position-aware)
        draw_reward = cfg.training.get('draw_reward', -0.1)
        # Use sentinel value for draw detection (needed for statistics)
        draw_sentinel = draw_reward if draw_reward is not None else -0.1
        # Track game-level outcomes from the first player's perspective
        num_wins = 0
        num_losses = 0
        num_first_wins = 0
        num_second_wins = 0
        num_draws = 0
        num_checkmates = 0
        games_completed_this_iter = 0
        # Track draw reasons and collect game moves
        from collections import defaultdict
        draw_reasons = defaultdict(int)
        game_moves_list = []  # Store all game moves for logging
        # Track pre-generated games (for continual mode)
        games_from_queue = 0
        games_waited_for = 0
        # Track games per device
        device_contributions = defaultdict(int)

        def _first_player_from_game_info(game_info) -> str:
            initial_fen = game_info.get('initial_fen', None)
            if cfg.env.type == 'gomoku':
                return 'b'
            if initial_fen:
                parts = initial_fen.split()
                if len(parts) >= 2 and parts[1].lower() in ('w', 'b'):
                    return parts[1].lower()
            return 'w'

        def _update_outcome_stats(game_info):
            nonlocal num_wins, num_losses, num_first_wins, num_second_wins, num_draws
            import chess
            winner = game_info.get('winner', None)
            if winner is None:
                num_draws += 1
                draw_reasons[game_info['termination']] += 1
                return
            if winner == chess.WHITE:
                num_wins += 1
                winner_color = 'w'
            elif winner == chess.BLACK:
                num_losses += 1
                winner_color = 'b'
            else:
                return
            first_player = _first_player_from_game_info(game_info)
            if winner_color == first_player:
                num_first_wins += 1
            else:
                num_second_wins += 1

        if continual_enabled:
            # Drain queue to collect games for this iteration
            # Strategy: Drain all available games, wait if needed to reach minimum threshold
            # Limit: Stop if new experiences would completely replace the entire replay buffer
            games_data_collected = []
            min_steps = cfg.training.self_play_steps_per_epoch  # Minimum steps threshold
            max_experiences = cfg.training.replay_buffer_size  # Maximum experiences to collect
            collected_games = 0
            queue_drain_start = time.time()
            task_id_selfplay = (
                progress.add_task(
                    "Self-Play",
                    total=min_steps,
                    games=0,
                    steps=0,
                    draw_rate=0.0,
                    first_wins=0,
                    second_wins=0,
                )
                if show_progress
                else None
            )
            
            # Phase 1: Drain all immediately available games from queue (non-blocking, batched)
            # Phase 3 optimization: Batch queue operations for better performance
            batch_collect_size = min(8, sp_queue.maxsize if hasattr(sp_queue, 'maxsize') else 8)
            while True:
                batch_collected = 0
                try:
                    # Try to collect a batch of games
                    for _ in range(batch_collect_size):
                        try:
                            game_result = sp_queue.get_nowait()
                            game_data, game_info = game_result
                            games_from_queue += 1
                            if game_data:
                                games_data_collected.extend(game_data)
                                collected_games += 1
                                game_moves_list.append(game_info)
                                if game_info.get('termination') == "CHECKMATE":
                                    num_checkmates += 1
                                # Track device contribution
                                if 'device' in game_info:
                                    device_contributions[game_info['device']] += 1
                                batch_collected += 1
                                
                                # Update game outcome counters
                                try:
                                    _update_outcome_stats(game_info)
                                except Exception:
                                    pass
                                
                                if task_id_selfplay is not None:
                                    draw_rate = (num_draws / collected_games * 100.0) if collected_games > 0 else 0.0
                                    progress.update(
                                        task_id_selfplay,
                                        advance=len(game_data),
                                        total=min_steps,
                                        games=collected_games,
                                        steps=len(games_data_collected),
                                        draw_rate=draw_rate,
                                        first_wins=num_first_wins,
                                        second_wins=num_second_wins,
                                        refresh=True,
                                    )
                                
                                # Check if we've collected enough steps or hit buffer limit
                                if len(games_data_collected) >= min_steps or len(games_data_collected) >= max_experiences:
                                    if len(games_data_collected) >= max_experiences:
                                        break
                        except queue.Empty:
                            break
                    
                    # If no games collected in this batch, break
                    if batch_collected == 0:
                        break
                    # Check thresholds after batch
                    if len(games_data_collected) >= min_steps or len(games_data_collected) >= max_experiences:
                        break
                except queue.Empty:
                    # No more immediately available games
                    break
            
            # Phase 2: If we haven't reached minimum steps, wait for more games
            # Phase 3 optimization: Reduced timeout for faster response
            timeout_interval = 0.5  # Reduced from 1.0 for faster response
            if len(games_data_collected) < min_steps and len(games_data_collected) < max_experiences:
                if collected_games == 0:
                    # First wait, record when actual waiting started
                    iteration_start_time_selfplay = time.time()
                
                while len(games_data_collected) < min_steps and len(games_data_collected) < max_experiences:
                    try:
                        game_result = sp_queue.get(timeout=timeout_interval)
                        game_data, game_info = game_result
                        games_waited_for += 1
                        if game_data:
                            games_data_collected.extend(game_data)
                            collected_games += 1
                            game_moves_list.append(game_info)
                            if game_info.get('termination') == "CHECKMATE":
                                num_checkmates += 1
                            # Track device contribution
                            if 'device' in game_info:
                                device_contributions[game_info['device']] += 1
                            # Check if we've reached step threshold or hit buffer limit
                            if len(games_data_collected) >= min_steps or len(games_data_collected) >= max_experiences:
                                if task_id_selfplay is not None:
                                    draw_rate = (num_draws / collected_games * 100.0) if collected_games > 0 else 0.0
                                    progress.update(
                                        task_id_selfplay,
                                        advance=len(game_data),
                                        total=min_steps,
                                        games=collected_games,
                                        steps=len(games_data_collected),
                                        draw_rate=draw_rate,
                                        first_wins=num_first_wins,
                                        second_wins=num_second_wins,
                                        refresh=True,
                                    )
                                if len(games_data_collected) >= max_experiences:
                                    break
                            # Update game outcome counters
                            try:
                                _update_outcome_stats(game_info)
                            except Exception:
                                pass
                            if task_id_selfplay is not None:
                                draw_rate = (num_draws / collected_games * 100.0) if collected_games > 0 else 0.0
                                progress.update(
                                    task_id_selfplay,
                                    advance=len(game_data),
                                    total=min_steps,
                                    games=collected_games,
                                    steps=len(games_data_collected),
                                    draw_rate=draw_rate,
                                    first_wins=num_first_wins,
                                    second_wins=num_second_wins,
                                    refresh=True,
                                )
                    except queue.Empty:
                        # Keep waiting for actors to generate more (with reduced timeout)
                        pass
            
            # Phase 3: After reaching minimum, drain any additional available games (batched)
            # Phase 3 optimization: Batch queue operations for better performance
            # This keeps the queue from growing too large and uses fresh games
            # But stop if we'd replace the entire buffer
            if len(games_data_collected) < max_experiences:
                extra_drain_attempts = 0
                max_extra_drain = min(batch_collect_size * 4, sp_queue.maxsize if hasattr(sp_queue, 'maxsize') else 32)
                while extra_drain_attempts < max_extra_drain and len(games_data_collected) < max_experiences:
                    batch_collected = 0
                    try:
                        # Try to collect a batch of games
                        for _ in range(batch_collect_size):
                            try:
                                game_result = sp_queue.get_nowait()
                                game_data, game_info = game_result
                                games_from_queue += 1
                                if game_data:
                                    games_data_collected.extend(game_data)
                                    collected_games += 1
                                    game_moves_list.append(game_info)
                                    if game_info.get('termination') == "CHECKMATE":
                                        num_checkmates += 1
                                    # Track device contribution
                                    if 'device' in game_info:
                                        device_contributions[game_info['device']] += 1
                                    batch_collected += 1
                                    
                                    # Update game outcome counters
                                    try:
                                        _update_outcome_stats(game_info)
                                    except Exception:
                                        pass
                                    
                                    if task_id_selfplay is not None:
                                        draw_rate = (num_draws / collected_games * 100.0) if collected_games > 0 else 0.0
                                        progress.update(
                                            task_id_selfplay,
                                            advance=len(game_data),
                                            total=min_steps,
                                            games=collected_games,
                                            steps=len(games_data_collected),
                                            draw_rate=draw_rate,
                                            first_wins=num_first_wins,
                                            second_wins=num_second_wins,
                                            refresh=True,
                                        )
                                    
                                    # Check if we've hit the buffer limit
                                    if len(games_data_collected) >= max_experiences:
                                        break
                                extra_drain_attempts += 1
                            except queue.Empty:
                                break
                        
                        # If no games collected in this batch, break
                        if batch_collected == 0:
                            break
                        # Check threshold after batch
                        if len(games_data_collected) >= max_experiences:
                            break
                    except queue.Empty:
                        # No more games available
                        break
            
            # Calculate actual self-play time (only time spent waiting, not draining pre-generated games)
            if games_waited_for > 0:
                # We had to wait for games, so time since first wait is the actual self-play time
                pass  # iteration_start_time_selfplay was already set when we started waiting
            else:
                # All games were pre-generated, so self-play time is essentially 0
                # But to avoid confusion, use the queue drain time
                iteration_start_time_selfplay = queue_drain_start
            
            if task_id_selfplay is not None:
                progress.update(task_id_selfplay, visible=False)
            # Record games completed for logging/checkpoint
            games_completed_this_iter = collected_games
            # Quick cleanup
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
            gc.collect()
        elif use_multiprocessing_flag:
            # Prepare arguments for workers
            # Pass network state_dict directly since it's clean
            network_state_dict = network.state_dict()
            # --- Mixed actors device assignment (GPU + CPU) ---
            available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            desired_processes = max(1, num_workers)
            device_pool = []
            # Prefer to keep at least one CPU actor when we have multiple processes
            gpu_slots = min(available_gpus, max(0, desired_processes - 1)) if desired_processes > 1 else 0
            for g in range(gpu_slots):
                device_pool.append(f"cuda:{g}")
            # Fill remaining slots with CPU actors
            while len(device_pool) < desired_processes:
                device_pool.append('cpu')

            if not device_pool:
                device_pool = ['cpu']

            # Collect games until we have enough steps
            min_steps = cfg.training.self_play_steps_per_epoch
            worker_args_packed = []
            game_num = 0
            
            # Convert OmegaConf config to plain dict for multiprocessing (ensures proper serialization)
            # This is important for nested structures like initial_board_fen
            cfg_for_workers = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
            
            # Submit initial batch of games
            # Estimate: assume average ~30-40 moves per game, add buffer for short games
            estimated_games_needed = max(1, (min_steps * 2) // 30)  # Estimate ~30 moves per game, double for safety
            for _ in range(estimated_games_needed):
                device_str = device_pool[game_num % len(device_pool)]
                worker_args_packed.append((game_num, network_state_dict, cfg_for_workers, device_str))
                game_num += 1

            pool = None # Initialize pool to None
            try:
                # Use 'spawn' context for better compatibility across platforms, especially with CUDA/PyTorch
                pool = multiprocessing.get_context("spawn").Pool(processes=len(device_pool))
                # progress.print(f"Submitting {len(worker_args_packed)} games to pool...")

                # Use imap_unordered to get results as they complete
                results_iterator = pool.imap_unordered(worker_wrapper, worker_args_packed)

                # Process results with a progress bar
                task_id_selfplay = progress.add_task(
                    "Self-Play",
                    total=min_steps,
                    games=0,
                    steps=0,
                    draw_rate=0.0,
                )
                # Iterate over results as they become available
                for game_result in results_iterator:
                    if game_result: # Check if worker returned valid data
                        game_data, game_info = game_result
                        games_data_collected.extend(game_data)
                        games_completed_this_iter += 1
                        game_moves_list.append(game_info)
                        if game_info.get('termination') == "CHECKMATE":
                            num_checkmates += 1
                        # Update game outcome counters
                        try:
                            _update_outcome_stats(game_info)
                        except Exception:
                            # If unexpected structure, skip counting for this game
                            pass
                    # Update progress with steps collected
                    steps_in_game = len(game_data) if game_result and game_result[0] else 0
                    draw_rate = (num_draws / games_completed_this_iter * 100.0) if games_completed_this_iter > 0 else 0.0
                    progress.update(
                        task_id_selfplay,
                        advance=steps_in_game,
                        games=games_completed_this_iter,
                        steps=len(games_data_collected),
                        draw_rate=draw_rate,
                        first_wins=num_first_wins,
                        second_wins=num_second_wins,
                        refresh=True,
                    )
                    
                    # Check if we've collected enough steps
                    # Note: We continue processing remaining results in the pool, but we've reached our threshold
                    if len(games_data_collected) >= min_steps:
                        # Mark that we've reached threshold (remaining games will still complete)
                        pass
                progress.update(task_id_selfplay, visible=False)
                # Explicitly close and join the pool after processing results
                pool.close()
                pool.join()

            except Exception as e:
                print(f"Error during parallel self-play: {e}")
                import traceback
                traceback.print_exc()
                # Handle error, maybe skip iteration or exit
                if pool is not None:
                    pool.terminate() # Terminate pool on error
            finally:
                # Ensure pool is closed and joined even if no error occurred in the try block
                if pool is not None and pool._state == multiprocessing.pool.RUN:
                    pool.close()
                    pool.join()

                # Cleanup GPU caches after self-play workers
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                gc.collect()

        else: # Sequential Execution
            min_steps = cfg.training.self_play_steps_per_epoch
            task_id_selfplay = progress.add_task(
                "Self-Play",
                total=min_steps,
                games=0,
                steps=0,
                draw_rate=0.0,
            )
            game_num = 0
            # Collect games until we have enough steps
            while len(games_data_collected) < min_steps:
                game_num += 1
                # Run game in the main process using the main env instance
                game_data, game_info = run_self_play_game(
                    cfg,
                    network if cfg.mcts.iterations > 0 else None,
                    env if cfg.env.render_mode == 'human' else None,  # Use the main env instance
                    progress=progress if (device.type == 'cpu' and show_progress) else None,
                    device=device
                )
                games_data_collected.extend(game_data)
                games_completed_this_iter += 1
                game_moves_list.append(game_info)
                if game_info.get('termination') == "CHECKMATE":
                    num_checkmates += 1
                # Update progress bar with current games and steps
                draw_rate = (num_draws / games_completed_this_iter * 100.0) if games_completed_this_iter > 0 else 0.0
                progress.update(
                    task_id_selfplay,
                    advance=len(game_data),
                    games=games_completed_this_iter,
                    steps=len(games_data_collected),
                    draw_rate=draw_rate,
                    first_wins=num_first_wins,
                    second_wins=num_second_wins,
                    refresh=True,
                )
                # Track device contribution (sequential mode uses training device)
                device_contributions[str(device)] += 1
                # Count outcomes from the actual winner
                if game_data:
                    try:
                        _update_outcome_stats(game_info)
                    except Exception:
                        pass
                # Note: progress already updated above with description and details
            progress.update(task_id_selfplay, visible=False)

            # Cleanup GPU caches after sequential self-play
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
            gc.collect()

        # Add collected data to replay buffer (outside the conditional block)
        for experience in games_data_collected:
            replay_buffer.add(experience)

        self_play_duration = int(time.time() - iteration_start_time_selfplay)
        total_games_simulated += games_completed_this_iter
        
        # Build draw reasons string
        draw_info = ""
        if num_draws > 0 and draw_reasons:
            draw_breakdown = ", ".join([f"{k}: {v}" for k, v in sorted(draw_reasons.items(), key=lambda x: -x[1])])
            draw_info = f" [{draw_breakdown}]"
        
        # Build device contributions string
        device_info = ""
        if device_contributions:
            device_breakdown = ", ".join([f"{dev}: {count}" for dev, count in sorted(device_contributions.items(), key=lambda x: -x[1])])
            device_info = f" | Devices: {device_breakdown}"
        
        # Collect and aggregate MCTS tree statistics
        tree_stats_info = ""
        if game_moves_list:
            tree_stats_all = [g.get('tree_stats') for g in game_moves_list if g.get('tree_stats')]
            if tree_stats_all:
                avg_nodes = sum(s['avg_nodes'] for s in tree_stats_all) / len(tree_stats_all)
                max_nodes = max(s['max_nodes'] for s in tree_stats_all)
                avg_depth = sum(s['avg_depth'] for s in tree_stats_all) / len(tree_stats_all)
                max_depth = max(s['max_depth'] for s in tree_stats_all)
                avg_branching = sum(s['avg_branching'] for s in tree_stats_all) / len(tree_stats_all)
                tree_stats_info = f" | MCTS: nodes={avg_nodes:.0f}/{max_nodes:.0f}, depth={avg_depth:.1f}/{max_depth:.0f}, branch={avg_branching:.2f}"
        
        # Breakdown avg nodes by initial dataset label (e.g., m1..m5)
        tree_stats_by_label_info = ""
        if game_moves_list:
            label_stats = {}
            for g in game_moves_list:
                stats = g.get('tree_stats')
                label = g.get('initial_dataset_label')
                if not stats or not label:
                    continue
                if label not in label_stats:
                    label_stats[label] = {'sum_nodes': 0.0, 'count': 0}
                label_stats[label]['sum_nodes'] += stats.get('avg_nodes', 0.0)
                label_stats[label]['count'] += 1
            if label_stats:
                label_parts = []
                for label in sorted(label_stats.keys()):
                    count = label_stats[label]['count']
                    avg_nodes_label = label_stats[label]['sum_nodes'] / max(1, count)
                    label_parts.append(f"{label}={avg_nodes_label:.0f} ({count})")
                tree_stats_by_label_info = " | MCTS avg nodes by dataset (# games): " + ", ".join(label_parts)
        
        buffer_info = f", buffer={len(replay_buffer)}"
        
        draw_rate_iter = (num_draws / games_completed_this_iter * 100.0) if games_completed_this_iter > 0 else 0.0
        progress.print(
            f"Self-play: {games_completed_this_iter} games, total={total_games_simulated}, "
            f"steps={len(games_data_collected)}{buffer_info} | W Wins: {num_wins}, "
            f"B Wins: {num_losses}, 1st Wins: {num_first_wins}, 2nd Wins: {num_second_wins}, "
            f"Draws: {num_draws} ({draw_rate_iter:.1f}% Draw)"
            f"{draw_info}{device_info}{tree_stats_info}{tree_stats_by_label_info} | {format_time(self_play_duration)}"
        )
        
        # --- Training Phase ---
        # Cleanup and prep GPU before training
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
        gc.collect()
        if len(replay_buffer) < cfg.training.batch_size:
            progress.print("Not enough data in buffer to start training. Skipping phase.")
            avg_policy_loss = float("nan")
            avg_value_loss = float("nan")
            _save_game_history(game_moves_list, game_history_dir, iteration, avg_policy_loss, avg_value_loss)
            continue
        
        # Dynamic accelerator assignment: if we have enough data and using GPU/MPS, wait for first available accelerator
        training_device = device
        if continual_enabled and device.type in ('cuda', 'mps') and len(replay_buffer) >= cfg.training.batch_size:
            # Wait for whichever accelerator finishes its current self-play game first
            # Wait for next game from queue to identify which accelerator just finished
            try:
                game_result = sp_queue.get(timeout=1.0)
                game_data, game_info = game_result
                # Extract which accelerator this game came from
                if 'device' in game_info:
                    finished_device_str = game_info['device']
                    if finished_device_str.startswith('cuda:') or finished_device_str == 'mps':
                        training_device = torch.device(finished_device_str)
                        # Put the game back in queue since we just wanted to know which accelerator finished
                        sp_queue.put((game_data, game_info))
                    else:
                        # CPU finished, use first available accelerator instead
                        available_accelerators = [dev for p, dev in actor_device_map.items() if (dev.startswith('cuda:') or dev == 'mps') and p.is_alive()]
                        if available_accelerators:
                            training_device = torch.device(available_accelerators[0])
                        sp_queue.put((game_data, game_info))
                else:
                    # Old format without device info, use first available accelerator
                    available_accelerators = [dev for p, dev in actor_device_map.items() if (dev.startswith('cuda:') or dev == 'mps') and p.is_alive()]
                    if available_accelerators:
                        training_device = torch.device(available_accelerators[0])
                    sp_queue.put((game_data, game_info))
            except queue.Empty:
                # No game finished yet, use first available accelerator
                available_accelerators = [dev for p, dev in actor_device_map.items() if (dev.startswith('cuda:') or dev == 'mps') and p.is_alive()]
                if available_accelerators:
                    training_device = torch.device(available_accelerators[0])
        
        training_columns = (
            TextColumn("[progress.description]{task.description}"), BarColumn(),
            TaskProgressColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(), TimeElapsedColumn(),
            TextColumn("[Loss P: {task.fields[loss_p]:.4f} V: {task.fields[loss_v]:.4f} Ill: {task.fields[illegal_r]:.2%} P: {task.fields[illegal_p]:.2%}]")
        )
        progress.columns = training_columns

        # progress.print("Starting training phase...")
        network.to(training_device).train()
        
        # CRITICAL: Refresh optimizer parameter references after moving network
        # When network.to(device) creates new parameter tensors, optimizer must be updated
        if use_multiprocessing_flag:
            # Save optimizer state before refreshing
            opt_state = optimizer.state_dict()
            # Extract current learning rate from optimizer state to preserve it
            current_lr = optimizer.param_groups[0]['lr']
            # Recreate optimizer with new parameter references, using preserved learning rate
            if opt_cfg.type == "Adam":
                optimizer = optim.Adam(network.parameters(), lr=current_lr, weight_decay=opt_cfg.weight_decay)
            elif opt_cfg.type == "SGD":
                momentum = opt_cfg.momentum if opt_cfg.momentum is not None else 0.9
                optimizer = optim.SGD(network.parameters(), lr=current_lr, momentum=momentum, weight_decay=opt_cfg.weight_decay)
            # Restore optimizer state (this preserves momentum, adaptive rates, etc.)
            try:
                optimizer.load_state_dict(opt_state)
                # Ensure optimizer state tensors are on correct device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                # Verify learning rate was preserved
                restored_lr = optimizer.param_groups[0]['lr']
                if abs(restored_lr - current_lr) > 1e-8:
                    # Fix learning rate if it wasn't preserved correctly
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
            except Exception as e:
                progress.print(f"Warning: Could not restore optimizer state after device move: {e}")
                # At minimum, preserve the learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                import traceback
                traceback.print_exc()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_illegal_moves_in_iteration = 0
        total_samples_in_iteration = 0
        avg_illegal_prob_mass = 0.0
        per_source_correct = [0 for _ in source_labels]
        per_source_total = [0 for _ in source_labels]
        illegal_metrics_interval = max(1, int(cfg.training.get("illegal_metrics_interval", 1)))
        task_id_train = progress.add_task(
            "Training Epochs",
            total=cfg.training.num_training_steps,
            loss_p=float("nan"),
            loss_v=float("nan"),
            illegal_r=float("nan"),
            illegal_p=float("nan"),
        )
        # Use values from cfg
        for epoch in range(cfg.training.num_training_steps):
            batch = replay_buffer.sample(cfg.training.batch_size, action_space_size=cfg.network.action_space_size)
            if batch is None: continue

            # states_np, policy_targets_np, fens_batch, value_targets_np = batch
            states_np, policy_targets_np, boards_batch, value_targets_np, source_ids_np = batch # Unpack boards
            states_tensor = torch.from_numpy(states_np).to(device)
            policy_targets_tensor = torch.from_numpy(policy_targets_np).to(device)
            value_targets_tensor = torch.from_numpy(value_targets_np).to(device)

            policy_logits, value_preds = network(states_tensor)

            # AlphaZero-style: policy targets are already legal-only, no masking needed.
            policy_loss = policy_loss_fn(policy_logits, policy_targets_tensor)
            # Ensure value shapes match before loss calc
            value_loss = value_loss_fn(value_preds.squeeze(-1), value_targets_tensor.squeeze(-1))

            total_loss = policy_loss + value_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

            if source_labels and source_ids_np is not None:
                source_ids_tensor = torch.from_numpy(source_ids_np).to(device)
                preds = torch.argmax(policy_logits, dim=1)
                target_indices = torch.argmax(policy_targets_tensor, dim=1)
                for source_idx in range(len(source_labels)):
                    mask = source_ids_tensor == source_idx
                    count = mask.sum().item()
                    if count:
                        per_source_total[source_idx] += count
                        per_source_correct[source_idx] += (preds[mask] == target_indices[mask]).sum().item()
            
            # Track illegal move metrics (model behavior monitoring)
            do_illegal_metrics = (
                illegal_metrics_interval == 1
                or epoch % illegal_metrics_interval == 0
                or epoch == cfg.training.num_training_steps - 1
            )
            if do_illegal_metrics:
                with torch.no_grad():
                    policy_probs = torch.softmax(policy_logits, dim=1)
                    batch_illegal_prob_mass = 0.0
                    batch_illegal_moves = 0
                    predicted_action_indices = torch.argmax(policy_logits, dim=1)
                    
                    for i in range(len(boards_batch)):
                        current_board = boards_batch[i]
                        legal_moves = set(get_legal_actions(current_board))
                        if not legal_moves:
                            continue
                        legal_indices = torch.tensor(
                            [move_id - 1 for move_id in legal_moves],
                            device=device,
                            dtype=torch.long,
                        )
                        illegal_prob = 1.0 - policy_probs[i, legal_indices].sum().item()
                        batch_illegal_prob_mass += illegal_prob
                        
                        predicted_action_id = predicted_action_indices[i].item() + 1
                        if predicted_action_id not in legal_moves:
                            batch_illegal_moves += 1
                    
                    avg_illegal_prob_mass = batch_illegal_prob_mass / len(boards_batch)
                
                    total_illegal_moves_in_iteration += batch_illegal_moves
                    total_samples_in_iteration += len(boards_batch)
            
            current_avg_policy_loss = total_policy_loss / (epoch + 1)
            current_avg_value_loss = total_value_loss / (epoch + 1)
            current_illegal_ratio = (
                total_illegal_moves_in_iteration / total_samples_in_iteration
                if total_samples_in_iteration > 0
                else 0.0
            )

            progress.update(
                task_id_train, advance=1,
                loss_p=current_avg_policy_loss, loss_v=current_avg_value_loss,
                illegal_r=current_illegal_ratio,
                illegal_p=avg_illegal_prob_mass
            )
        progress.update(task_id_train, visible=False)

        avg_policy_loss = total_policy_loss / cfg.training.num_training_steps if cfg.training.num_training_steps > 0 else 0
        avg_value_loss = total_value_loss / cfg.training.num_training_steps if cfg.training.num_training_steps > 0 else 0
        avg_illegal_ratio = total_illegal_moves_in_iteration / total_samples_in_iteration if total_samples_in_iteration > 0 else 0.0
        per_source_acc_str = _format_source_accuracy(
            source_labels, per_source_correct, per_source_total, digits=2
        )
        progress.print(
            f"Training finished: Avg Policy Loss: {avg_policy_loss:.4f}, "
            f"Avg Value Loss: {avg_value_loss:.4f}, "
            f"Avg Illegal Move Ratio: {avg_illegal_ratio:.2%}, "
            f"Avg Illegal Move Prob: {avg_illegal_prob_mass:.2%} | "
            f"Policy top-1 by dataset: {per_source_acc_str}"
        )
        _save_game_history(game_moves_list, game_history_dir, iteration, avg_policy_loss, avg_value_loss)
        iteration_duration = int(time.time() - iteration_start_time)
        total_elapsed_time = int(time.time() - total_training_start_time)
        
        # Calculate draw statistics for this iteration
        repetition_draw_types = {'THREEFOLD_REPETITION', 'FIVEFOLD_REPETITION'}
        repetition_draw_count = sum(draw_reasons.get(dt, 0) for dt in repetition_draw_types)
        other_draw_count = num_draws - repetition_draw_count
        non_draw_count = games_completed_this_iter - num_draws
        
        # Record metrics for learning curve visualization
        history['policy_loss'].append(avg_policy_loss)
        history['value_loss'].append(avg_value_loss)
        history['illegal_move_ratio'].append(avg_illegal_ratio)
        history['illegal_move_prob'].append(avg_illegal_prob_mass)
        history['non_draw_count'].append(non_draw_count)
        history['repetition_draw_count'].append(repetition_draw_count)
        history['other_draw_count'].append(other_draw_count)

        # Save checkpoint after each iteration
        checkpoint = {
            'iteration': iteration + 1,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_games_simulated': total_games_simulated,
            'replay_buffer_state': replay_buffer.get_state(env_type=cfg.env.type),
            'history': history,
        }
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"model_iter_{iteration+1}_p{avg_policy_loss:.4f}_v{avg_value_loss:.4f}.pth",
        )
        torch.save(checkpoint, checkpoint_path)
        
        checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        buffer_size_str = str(len(replay_buffer))
        
        progress.print(
            f"Iteration {iteration+1} completed in {format_time(iteration_duration)} "
            f"(total: {format_time(total_elapsed_time)}) | Checkpoint saved with "
            f"{buffer_size_str} experiences in replay buffer ({checkpoint_size_mb:.1f} MB)"
        )
        
        # Check if training would exceed maximum time after next iteration
        # Do this AFTER saving checkpoint so current iteration's work is preserved
        # Recalculate total_elapsed_time to include checkpoint saving time
        if max_training_time_seconds is not None:
            # Recalculate elapsed time after checkpoint saving to include that overhead
            total_elapsed_time_after_checkpoint = int(time.time() - total_training_start_time)
            
            # Calculate actual iteration duration (including checkpoint saving time)
            # Duration is the difference between current and previous total elapsed time
            actual_iteration_duration = total_elapsed_time_after_checkpoint - previous_total_elapsed_time
            previous_total_elapsed_time = total_elapsed_time_after_checkpoint
                    
            # Track iteration duration for time prediction
            iteration_durations.append(actual_iteration_duration)
            
            # Calculate average iteration time from completed iterations
            if len(iteration_durations) > 0:
                avg_iteration_time = sum(iteration_durations) / len(iteration_durations)
                # Predict total time after completing the next iteration
                # Use the updated elapsed time that includes checkpoint saving
                predicted_total_time = total_elapsed_time_after_checkpoint + avg_iteration_time
                
                if predicted_total_time > max_training_time_seconds:
                    progress.print(f"\n⚠️  Maximum training time limit reached!")
                    progress.print(f"   Current elapsed time: {format_time(total_elapsed_time_after_checkpoint)}")
                    progress.print(f"   Average iteration time: {format_time(int(avg_iteration_time))}")
                    progress.print(f"   Predicted total time after next iteration: {format_time(int(predicted_total_time))}")
                    progress.print(f"   Maximum allowed time: {format_time(max_training_time_seconds)}")
                    progress.print(f"   Stopping training at iteration {iteration + 1} to respect time limit.")
                    break

    # Final training summary
    total_training_time = time.time() - total_training_start_time
    progress.print(f"\nTraining completed in {format_time(int(total_training_time))}")
    progress.print(f"Final model saved at: {os.path.abspath(os.path.join(checkpoint_dir, 'model.pth'))}")

    env.close()
    # Stop continual actors if running
    if continual_enabled:
        try:
            stop_event.set()
            for p in actors:
                if p.is_alive():
                    p.join(timeout=2.0)
        except Exception:
            pass
    progress.print("\nTraining loop finished.")


# --- Hydra Entry Point --- 
# Ensure config_path points to the directory containing train_gomoku.yaml
@hydra.main(config_path="../config", config_name="train_mcts", version_base=None)
def main(cfg: DictConfig) -> None:
    # Initialize factories in the main process
    initialize_factories_from_cfg(cfg)
    
    print("Configuration:\n")
    # Use OmegaConf.to_yaml for structured printing
    print(OmegaConf.to_yaml(cfg))
    run_training_loop(cfg)

if __name__ == "__main__":
    main()
