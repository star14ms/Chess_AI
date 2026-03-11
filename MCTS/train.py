import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque
import os
import gc
import time
import logging
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing # Keep for potential parallel execution
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import queue
from torch import nn
import pickle
import gzip
import json
import chess
import random
import re
import shutil
from torch.amp import autocast, GradScaler

# Assuming files are in the MCTS directory relative to the project root
from mcts_node import MCTSNode
from mcts_algorithm import MCTS
from utils.profile_model import get_optimal_worker_count, profile_model, format_time
from utils.progress import LoggingProgressWrapper, NullProgress
from utils.thermal import maybe_pause_for_thermal_throttle
from inference_server import InferenceClient, inference_server_worker, inference_server_worker_tpu, _resolve_self_play_dtype
from utils.training_utils import (
    RewardComputer,
    freeze_first_n_conv_layers,
    repair_fen_en_passant,
    select_fen_from_dict,
    select_random_fen_from_entries,
    select_random_fen_from_file,
    select_random_fen_from_json_list,
)
from utils.dataset_labels import abbreviate_dataset_label, format_dataset_label

create_network = None
create_environment = None
get_game_result = None
is_first_player_turn = None
get_legal_actions = None
create_board_from_serialized = None
_INITIAL_FEN_CACHE = {}
_DATASET_ENTRIES_CACHE = {}
_LINE_OFFSET_CACHE: dict[str, list[int]] = {}  # per-process cache for random file access


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
    return None, None, []


def _select_fen_from_json_path(json_path: str):
    """Select a random FEN from JSON/JSONL file via O(1) random line access (no full-file load).
    Returns (fen, quality, themes, full_entry). full_entry is the parsed dict when available, else None."""
    try:
        resolved = _resolve_json_path(json_path)
        return select_random_fen_from_file(resolved, offset_cache=_LINE_OFFSET_CACHE)
    except Exception as e:
        print(f"Warning: Failed to load initial_board_fen from file {json_path}: {e}")
        return None, None, [], None


def _extract_solution_moves(entry: dict | None) -> list[str] | None:
    """Extract solution moves from a dataset entry. Returns list of move strings (UCI or SAN), or None."""
    if not entry or not isinstance(entry, dict):
        return None
    moves_field = entry.get("Moves") or entry.get("moves")
    if moves_field:
        if isinstance(moves_field, str):
            return [m.strip() for m in moves_field.split() if m.strip()]
        if isinstance(moves_field, list):
            return [str(m).strip() for m in moves_field if str(m).strip()]
    # Expanded format (one position per line): has policy_uci for the next move only
    policy_uci = entry.get("policy_uci")
    if policy_uci and isinstance(policy_uci, str) and policy_uci.strip():
        return [policy_uci.strip()]
    return None


def _select_fen_from_source(source):
    """Return (fen, quality, themes, full_entry). full_entry is the parsed dict when from JSON, else None."""
    fen, quality, themes, full_entry = None, None, [], None
    if isinstance(source, str) and (source.endswith(".json") or source.endswith(".jsonl")):
        fen, quality, themes, full_entry = _select_fen_from_json_path(source)
    elif isinstance(source, dict):
        result = select_fen_from_dict(source)
        fen, quality = (result[0], result[1]) if len(result) >= 2 else (result[0], None)
        themes = result[2] if len(result) >= 3 else []
    elif isinstance(source, list):
        result = select_random_fen_from_entries(source)
        fen, quality = (result[0], result[1]) if len(result) >= 2 else (result[0], None)
        themes = result[2] if len(result) >= 3 else []
    elif isinstance(source, str):
        fen, quality = source, None
    if fen is not None:
        fen = repair_fen_en_passant(fen)
    return (fen, quality, themes, full_entry) if fen is not None else (None, None, [], None)


def _dataset_cache_key(value) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except Exception:
        return repr(value)


def _log_nan_batch_diagnostics_numpy_only(epoch, policy_targets_np, value_targets_np):
    """Log batch stats using only numpy arrays (no TPU tensor access). Safe to run before .item() check."""
    try:
        p = policy_targets_np
        n_zeros = int(np.sum(p == 0))
        n_small = int(np.sum((p > 0) & (p < 6e-5)))
        p_sum = p.sum(axis=1)
        v = value_targets_np
        msg = (
            f"[TPU diag] First batch (epoch {epoch}) - numpy only (no TPU sync): "
            f"policy_targets zeros={n_zeros} small={n_small} row_sums min={p_sum.min():.6f} max={p_sum.max():.6f} | "
            f"value_targets min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}"
        )
        logging.info(msg)
        print(msg, file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[TPU diag] numpy-only diagnostic error: {e}", file=sys.stderr, flush=True)


def _log_early_nan_hypotheses(epoch, states_np, policy_targets_np, value_targets_np, policy_logits, value_preds):
    """Log hypothesis-based checks to find NaN source as early as possible. Order = pipeline order."""
    def _log(msg):
        logging.info(msg)
        print(msg, file=sys.stderr, flush=True)

    _log("[TPU diag] === Early NaN hypothesis check (first batch) ===")
    # H1: Input states corrupted
    s_nan, s_inf = bool(np.isnan(states_np).any()), bool(np.isinf(states_np).any())
    _log(f"  H1 states_np:     has_nan={s_nan} has_inf={s_inf} shape={states_np.shape} "
         f"min={states_np.min():.4f} max={states_np.max():.4f}")
    if s_nan or s_inf:
        _log("  -> LIKELY CAUSE: corrupted observation states in replay buffer")

    # H2: Policy targets corrupted
    p_nan, p_inf = bool(np.isnan(policy_targets_np).any()), bool(np.isinf(policy_targets_np).any())
    p_sum = policy_targets_np.sum(axis=1)
    _log(f"  H2 policy_targets: has_nan={p_nan} has_inf={p_inf} row_sums min={p_sum.min():.6f} max={p_sum.max():.6f}")
    if p_nan or p_inf or (p_sum < 0.99).any() or (p_sum > 1.01).any():
        _log("  -> LIKELY CAUSE: invalid policy targets (NaN/Inf or row_sum != 1)")

    # H3: Value targets corrupted
    v_nan, v_inf = bool(np.isnan(value_targets_np).any()), bool(np.isinf(value_targets_np).any())
    _log(f"  H3 value_targets: has_nan={v_nan} has_inf={v_inf} min={value_targets_np.min():.4f} max={value_targets_np.max():.4f}")
    if v_nan or v_inf:
        _log("  -> LIKELY CAUSE: invalid value targets in replay buffer")

    # H4: Model outputs NaN (policy_logits) - TPU sync may hang if tensor has NaN
    try:
        with torch.no_grad():
            pl = policy_logits.detach().float().cpu().numpy()
        pl_nan, pl_inf = bool(np.isnan(pl).any()), bool(np.isinf(pl).any())
        _log(f"  H4 policy_logits:  has_nan={pl_nan} has_inf={pl_inf} min={pl.min():.4f} max={pl.max():.4f}")
        if pl_nan or pl_inf:
            _log("  -> LIKELY CAUSE: model forward pass outputs NaN/Inf (checkpoint weights or TPU numerics)")
    except Exception as e:
        _log(f"  H4 policy_logits:  TPU sync failed (may hang on NaN): {e}")

    # H5: Model outputs NaN (value_preds)
    try:
        with torch.no_grad():
            vp = value_preds.detach().float().cpu().numpy()
        vp_nan, vp_inf = bool(np.isnan(vp).any()), bool(np.isinf(vp).any())
        _log(f"  H5 value_preds:   has_nan={vp_nan} has_inf={vp_inf} min={vp.min():.4f} max={vp.max():.4f}")
        if vp_nan or vp_inf:
            _log("  -> LIKELY CAUSE: value head outputs NaN/Inf (checkpoint or TPU numerics)")
    except Exception as e:
        _log(f"  H5 value_preds:   TPU sync failed (may hang on NaN): {e}")

    _log("[TPU diag] === End hypothesis check ===")


def _log_nan_batch_diagnostics(progress, epoch, policy_targets_np, value_targets_np, policy_logits, value_preds, policy_loss, value_loss):
    """Log batch stats when NaN loss detected to help find root cause of TPU gradient deviation."""
    def _out(msg):
        logging.info(msg)
        print(msg, file=sys.stderr, flush=True)
        if progress is not None and hasattr(progress, "print"):
            try:
                progress.print(msg)
            except Exception:
                pass

    try:
        _out(f"[TPU diag] NaN/Inf loss at epoch {epoch}. Batch stats:")
        # Numpy arrays first (no TPU sync - safe)
        p = policy_targets_np
        n_zeros = int(np.sum(p == 0))
        n_small = int(np.sum((p > 0) & (p < 6e-5)))
        p_sum = p.sum(axis=1)
        _out(f"  policy_targets: shape={p.shape}, zeros={n_zeros}, small(<6e-5)={n_small}, "
             f"row_sums min={p_sum.min():.6f} max={p_sum.max():.6f} mean={p_sum.mean():.6f}")
        v = value_targets_np
        _out(f"  value_targets: min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")
        # TPU tensors: .cpu()/.item() can hang on NaN - wrap tightly
        try:
            with torch.no_grad():
                pl = policy_logits.detach().float().cpu().numpy()
            _out(f"  policy_logits: min={pl.min():.4f} max={pl.max():.4f} has_nan={bool(np.isnan(pl).any())}")
        except Exception as e:
            _out(f"  policy_logits: (TPU sync failed: {e})")
        try:
            with torch.no_grad():
                vp = value_preds.detach().float().cpu().numpy()
            _out(f"  value_preds: min={vp.min():.4f} max={vp.max():.4f} has_nan={bool(np.isnan(vp).any())}")
        except Exception as e:
            _out(f"  value_preds: (TPU sync failed: {e})")
        try:
            _out(f"  policy_loss={float(policy_loss.item())}, value_loss={float(value_loss.item())}")
        except Exception as e:
            _out(f"  loss values: (TPU sync failed: {e})")
    except Exception as e:
        try:
            _out(f"[TPU diag] Error during diagnostic: {e}")
        except Exception:
            print(f"[TPU diag] Error during diagnostic: {e}", file=sys.stderr, flush=True)


def _compute_grad_norm(model) -> float:
    """Compute total gradient norm across all parameters. Returns float('nan') on error."""
    try:
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_sq += param_norm.item() ** 2
        return total_norm_sq ** 0.5
    except Exception:
        return float("nan")


def _state_dict_has_nan_or_inf(state_dict: dict) -> bool:
    """Check if any tensor in state_dict contains NaN or Inf. Used to reject corrupted checkpoints."""
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            try:
                v_cpu = v.cpu()  # Materialize TPU/XLA tensors before checking
                if torch.isnan(v_cpu).any().item() or torch.isinf(v_cpu).any().item():
                    return True
            except Exception:
                pass  # If we can't check (e.g. TPU sync fails), assume OK
    return False


def _empty_device_cache(device=None):
    """Clear GPU cache for CUDA and/or MPS to reduce memory fragmentation (like diagnostic_mcts)."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
    if device is None or (isinstance(device, torch.device) and device.type == "mps"):
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            if hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass


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
        formatted = format_dataset_label(base)
        return formatted or "data"
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
        source_lower = str(source).lower()
        if "minimal" in source_lower and isinstance(label, str) and re.match(r"^m\d+(#\d+)?$", label):
            label = "mm" + label[1:]
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


def _format_source_accuracy(labels: list[str], correct: list[int], total: list[int], digits: int = 2, abbreviate: bool = True) -> str:
    if not labels:
        return "-"
    parts: list[str] = []
    for label, c, t in zip(labels, correct, total):
        disp = abbreviate_dataset_label(label) if abbreviate else label
        if t > 0:
            acc = c / t * 100
            parts.append(f"{disp}={acc:.{digits}f}%")
        else:
            parts.append(f"{disp}=N/A")
    return " | ".join(parts)




def _ensure_mate_success_history(history: dict, mate_labels: list[str]) -> dict:
    if "mate_success" not in history or not isinstance(history.get("mate_success"), dict):
        history["mate_success"] = {}
    if mate_labels:
        history["mate_success_labels"] = list(mate_labels)
    elif "mate_success_labels" not in history:
        history["mate_success_labels"] = []

    expected_len = len(history.get("policy_loss", []))
    for label in mate_labels:
        series = history["mate_success"].get(label)
        if series is None:
            history["mate_success"][label] = [0.0] * expected_len
        elif len(series) < expected_len:
            history["mate_success"][label].extend([0.0] * (expected_len - len(series)))

    return history

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
def self_play_worker(
    game_id,
    network_state_dict,
    cfg: DictConfig,
    device_str: str,
    inference_request_queue=None,
    inference_reply_queue=None,
    inference_timeout_s: float = 30.0,
):
    """Worker function to run a single self-play game."""
    # Convert plain dict back to OmegaConf if needed (for compatibility)
    if isinstance(cfg, dict) and not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    # Ensure factories are initialized inside spawned workers
    initialize_factories_from_cfg(cfg)
    inference_client = None
    if inference_request_queue is not None and inference_reply_queue is not None:
        inference_client = InferenceClient(inference_request_queue, inference_reply_queue, timeout_s=inference_timeout_s)

    # Determine device for this worker
    if inference_client is not None:
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    # Ensure correct CUDA device context in worker
    if device.type == 'cuda':
        try:
            torch.cuda.set_device(device)
        except Exception:
            pass

    # Re-initialize Network in the worker process
    if cfg.mcts.iterations > 0 and inference_client is None:
        network = create_network(cfg, device)
        network.load_state_dict(network_state_dict)
        network.to(device).eval()
        sp_dtype = _resolve_self_play_dtype(cfg, device_str)
        if sp_dtype == torch.float16:
            network.half()
    else:
        network = None

    game_data, game_info = run_self_play_game(
        cfg,
        network,
        env=None,
        progress=None, # Pass None for progress within worker
        device=device,
        inference_client=inference_client,
    )
    return (game_data, game_info)

# Wrapper for imap_unordered (module scope for pickling)
def worker_wrapper(args):
    """Unpacks arguments for self_play_worker when using imap. Returns (game_id, game_data, game_info) for submission-order stats."""
    (
        game_id,
        network_state_dict,
        cfg,
        device_str,
        inference_request_queue,
        inference_reply_queue,
        inference_timeout_s,
    ) = args
    # Call the original worker function with unpacked args
    game_data, game_info = self_play_worker(
        game_id,
        network_state_dict,
        cfg,
        device_str,
        inference_request_queue=inference_request_queue,
        inference_reply_queue=inference_reply_queue,
        inference_timeout_s=inference_timeout_s,
    )
    return (game_id, game_data, game_info)

# Persistent continual self-play actor
def continual_self_play_worker(
    checkpoint_path: str,
    cfg: DictConfig,
    device_str: str,
    out_queue,
    stop_event,
    inference_request_queue=None,
    inference_reply_queue=None,
    inference_timeout_s: float = 30.0,
    worker_id: int = -1,
    games_per_worker: int = 1,
    pause_for_training=None,
):
    if isinstance(cfg, dict) and not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    initialize_factories_from_cfg(cfg)
    inference_client = None
    if inference_request_queue is not None and inference_reply_queue is not None:
        inference_client = InferenceClient(
            inference_request_queue,
            inference_reply_queue,
            timeout_s=inference_timeout_s,
            worker_id=worker_id if worker_id >= 0 else None,
            pause_for_training=pause_for_training,
        )

    if inference_client is not None:
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    if device.type == 'cuda':
        try:
            torch.cuda.set_device(device)
        except Exception:
            pass
    network = None
    if cfg.mcts.iterations > 0 and inference_client is None:
        network = create_network(cfg, device)
        if os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
                network.load_state_dict(ckpt['model_state_dict'])
            except Exception:
                pass
        network.to(device).eval()
        sp_dtype = _resolve_self_play_dtype(cfg, device_str)
        if sp_dtype == torch.float16:
            network.half()
    last_mtime = os.path.getmtime(checkpoint_path) if os.path.exists(checkpoint_path) else 0.0
    reload_lock = threading.Lock() if (games_per_worker > 1 and network is not None) else None

    def _should_stop():
        """Safely check stop_event; treat connection errors (manager shutdown) as stop."""
        try:
            return stop_event.is_set()
        except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
            return True

    def _run_game_loop():
        nonlocal last_mtime
        while not _should_stop():
            # Hot-reload if newer checkpoint exists
            try:
                if os.path.exists(checkpoint_path):
                    mtime = os.path.getmtime(checkpoint_path)
                    if mtime > last_mtime and network is not None:
                        lock_ctx = reload_lock if reload_lock is not None else nullcontext()
                        with lock_ctx:
                            if mtime > last_mtime and network is not None:
                                try:
                                    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
                                    sd = ckpt.get("model_state_dict")
                                    if sd is not None and not _state_dict_has_nan_or_inf(sd):
                                        network.load_state_dict(sd)
                                        network.to(device).eval()
                                        if sp_dtype == torch.float16:
                                            network.half()
                                        last_mtime = mtime
                                except Exception:
                                    pass
            except Exception:
                pass

            # Play one game and enqueue (include device info so we know which GPU finished)
            try:
                game_data, game_info = run_self_play_game(
                    cfg,
                    network if cfg.mcts.iterations > 0 else None,
                    env=None,
                    progress=None,
                    device=device,
                    inference_client=inference_client,
                )
                if game_data:
                    game_info_with_device = game_info.copy()
                    game_info_with_device['device'] = device_str
                    out_queue.put((game_data, game_info_with_device))
            except Exception:
                pass
            _empty_device_cache(device)
            gc.collect()

    if games_per_worker <= 1:
        _run_game_loop()
    else:
        with ThreadPoolExecutor(max_workers=games_per_worker) as executor:
            futures = [executor.submit(_run_game_loop) for _ in range(games_per_worker)]
            for f in futures:
                try:
                    f.result()
                except Exception:
                    pass

# --- Replay Buffer (Keep as is) ---
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def add_game(self, game_data):
        for experience in game_data:
            self.add(experience)

    def sample(self, batch_size, action_space_size=None, draw_sample_ratio=None):
        """Sample a batch, optionally controlling draw ratio.

        Args:
            batch_size: Number of experiences to sample
            action_space_size: Action space size for policy normalization
            draw_sample_ratio: Target fraction of batch from draw positions (0.0–1.0).
                None = uniform sampling. E.g. 0.4 = 40% draws, 60% wins per batch.
        """
        if len(self.buffer) < batch_size:
            return None

        # Stratified sampling when draw_sample_ratio is set (0.0 = no draws, 1.0 = all draws)
        if draw_sample_ratio is not None and 0 <= draw_sample_ratio <= 1:
            draw_indices = []
            win_indices = []
            for i, exp in enumerate(self.buffer):
                val = float(exp[3])
                is_draw = abs(val - 1.0) > 1e-6 and abs(val + 1.0) > 1e-6
                if is_draw:
                    draw_indices.append(i)
                else:
                    win_indices.append(i)

            n_draw_target = int(batch_size * draw_sample_ratio)
            n_win_target = batch_size - n_draw_target

            n_draw_actual = min(n_draw_target, len(draw_indices))
            n_win_actual = min(n_win_target, len(win_indices))

            # If one pool can't supply enough, fill from the other
            if n_draw_actual + n_win_actual < batch_size:
                if not draw_indices:
                    n_draw_actual = 0
                    n_win_actual = min(batch_size, len(win_indices))
                elif not win_indices:
                    n_win_actual = 0
                    n_draw_actual = min(batch_size, len(draw_indices))
                else:
                    shortfall = batch_size - (n_draw_actual + n_win_actual)
                    n_draw_add = min(shortfall, len(draw_indices) - n_draw_actual)
                    n_draw_actual += n_draw_add
                    shortfall -= n_draw_add
                    n_win_actual += min(shortfall, len(win_indices) - n_win_actual)

            batch_indices_list = []
            if n_draw_actual > 0 and draw_indices:
                draw_arr = np.array(draw_indices)
                batch_indices_list.append(
                    np.random.choice(draw_arr, size=n_draw_actual, replace=False)
                )
            if n_win_actual > 0 and win_indices:
                win_arr = np.array(win_indices)
                batch_indices_list.append(
                    np.random.choice(win_arr, size=n_win_actual, replace=False)
                )

            if batch_indices_list:
                batch_indices = np.concatenate(batch_indices_list)
                np.random.shuffle(batch_indices)
            else:
                batch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        else:
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
    
    def get_state(self, env_type='chess', use_float32=False):
        """Returns the replay buffer state for checkpointing in compressed format.
        
        Args:
            env_type: Type of environment ('chess' or 'gomoku')
            use_float32: If True, store state/policy in float32 (avoids float16 underflow of small probs on TPU)
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
            
            # Store arrays as (bytes, shape, dtype) to avoid NumPy's deprecated pickle path (numpy.core.numeric)
            # use_float32 avoids float16 underflow of small policy probs (can cause TPU NaN)
            state_arr = state.astype(np.float32 if use_float32 else np.float16)
            policy_arr = policy.astype(np.float32 if use_float32 else np.float16)
            compact_exp = (
                state_arr.tobytes(),
                state_arr.shape,
                state_arr.dtype.name,
                policy_arr.tobytes(),
                policy_arr.shape,
                policy_arr.dtype.name,
                board_str,
                float(value),
                source_id
            )
            compact_buffer.append(compact_exp)
        
        # Compress the buffer data using gzip (no numpy arrays in structure -> no deprecated unpickle path)
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
                # New format (v2): (state_bytes, state_shape, state_dtype, policy_bytes, policy_shape, policy_dtype, board_str, value, source_id)
                if len(compact_exp) == 9 and isinstance(compact_exp[0], bytes):
                    (state_bytes, state_shape, state_dtype, policy_bytes, policy_shape, policy_dtype,
                     board_str, value, source_id) = compact_exp
                    state_data = np.frombuffer(state_bytes, dtype=np.dtype(state_dtype)).reshape(state_shape).astype(np.float32)
                    policy = np.frombuffer(policy_bytes, dtype=np.dtype(policy_dtype)).reshape(policy_shape).astype(np.float32)
                else:
                    # Legacy format: numpy arrays pickled directly (may trigger NumPy deprecation warning)
                    if len(compact_exp) == 5:
                        state_data, policy, board_str, value, source_id = compact_exp
                    else:
                        state_data, policy, board_str, value = compact_exp
                        source_id = None
                    state_data = state_data.astype(np.float32)
                    policy = policy.astype(np.float32)
                # Reconstruct board object from serialized string
                board = board_factory_fn(board_str)
                exp = (
                    state_data,
                    policy,
                    board,
                    value,
                    source_id
                ) if source_id is not None else (
                    state_data,
                    policy,
                    board,
                    value
                )
                full_buffer.append(exp)
            
            self.buffer = deque(full_buffer, maxlen=state['maxlen'])
        else:
            # Legacy uncompressed format
            self.buffer = deque(state['buffer'], maxlen=state['maxlen'])


# --- Self-Play Function (Update args to use config subsections) ---
def run_self_play_game(
    cfg: OmegaConf,
    network: nn.Module | None,
    env=None,
    progress: Progress | None = None,
    device: torch.device | None = None,
    inference_client: InferenceClient | None = None,
    self_play_dtype_override: torch.dtype | None = None,
):
    """Plays one game of self-play using MCTS and returns the game data."""
    manual_pause_seconds = cfg.training.get("manual_pause_seconds", None)
    if manual_pause_seconds not in (None, "", "null"):
        try:
            manual_pause_seconds = float(manual_pause_seconds)
        except Exception:
            manual_pause_seconds = None
    if manual_pause_seconds is not None and manual_pause_seconds > 0:
        if progress is not None:
            progress.print(f"Manual pause: sleeping {manual_pause_seconds:.1f}s before self-play.")
        time.sleep(manual_pause_seconds)
    elif cfg.training.get("enable_thermal_pause", False):
        maybe_pause_for_thermal_throttle(cfg, progress=progress, phase="self-play")
    if env is None:
        env = create_environment(cfg, render=device.type == 'cpu' and not cfg.training.get('use_multiprocessing', False))
    
    # Handle initial_board_fen: supports single dataset or weighted list of datasets
    initial_fen = None
    initial_position_quality = None
    initial_themes = []
    initial_dataset_id = None
    initial_dataset_label = None
    initial_dataset_source = None
    initial_solution_moves = None  # Labeled trajectory (Moves from dataset) for ground-truth comparison
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

        # If string path to .yaml file, load it (e.g. training.initial_board_fen=./kaggle_fen_override.yaml)
        if isinstance(initial_board_fen_cfg, str) and (
            initial_board_fen_cfg.endswith(".yaml") or initial_board_fen_cfg.endswith(".yml")
        ):
            try:
                import yaml
                with open(_resolve_json_path(initial_board_fen_cfg), "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                initial_board_fen_cfg = loaded if isinstance(loaded, list) else (loaded.get("training") or {}).get("initial_board_fen", loaded)
            except Exception:
                pass

        if isinstance(initial_board_fen_cfg, list):
            dataset_entries, normalized_weights, _labels = _get_cached_dataset_entries(initial_board_fen_cfg)
            if dataset_entries and normalized_weights:
                selected_idx = random.choices(range(len(dataset_entries)), weights=normalized_weights, k=1)[0]
                selected_entry = dataset_entries[selected_idx]
                initial_dataset_id = selected_idx
                initial_dataset_label = selected_entry["label"]
                max_game_moves_override = selected_entry.get("max_game_moves")
                source_value = selected_entry.get("source")
                if isinstance(source_value, str):
                    if source_value.endswith(".json") or source_value.endswith(".jsonl"):
                        initial_dataset_source = _resolve_json_path(source_value)
                    else:
                        initial_dataset_source = source_value
                initial_fen, initial_position_quality, initial_themes, initial_entry = _select_fen_from_source(selected_entry["source"])
                initial_solution_moves = _extract_solution_moves(initial_entry) if initial_entry else None
        else:
            initial_fen, initial_position_quality, initial_themes, _ = _select_fen_from_source(initial_board_fen_cfg)
            initial_solution_moves = None
            if isinstance(initial_board_fen_cfg, str) and (
                initial_board_fen_cfg.endswith(".json") or initial_board_fen_cfg.endswith(".jsonl")
            ):
                initial_dataset_source = _resolve_json_path(initial_board_fen_cfg)
                initial_dataset_label = _shorten_dataset_label(initial_board_fen_cfg)
            if isinstance(initial_board_fen_cfg, dict):
                max_game_moves_override = initial_board_fen_cfg.get("max_game_moves")
                for key in ("path", "file", "dataset"):
                    source_value = initial_board_fen_cfg.get(key)
                    if isinstance(source_value, str):
                        if source_value.endswith(".json") or source_value.endswith(".jsonl"):
                            initial_dataset_source = _resolve_json_path(source_value)
                        else:
                            initial_dataset_source = source_value
                        break
    
    options = {
        'fen': initial_fen
    } if initial_fen else None
    obs, _ = env.reset(options=options)
    if network is not None:
        network.eval()

    # Initialize reward computer (for position-aware draw rewards during training)
    reward_computer = RewardComputer(cfg, network, device, inference_client=inference_client)

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
    move_list_uci = []  # Track moves in UCI (unambiguous for replay)
    move_list_action_ids = []  # Track action IDs played (for solution-following check)
    policy_details_list = []  # Per-state policy info for game_history file
    move_count = 0
    terminated = False
    truncated = False

    # Pre-compute solution action IDs from labeled trajectory (for ground-truth comparison)
    solution_action_ids = []
    if initial_solution_moves and initial_fen:
        try:
            from MCTS.training_modules.chess import create_board_from_fen
            sol_board = create_board_from_fen(initial_fen)
            for move_str in initial_solution_moves:
                if sol_board.is_game_over():
                    break
                move = None
                try:
                    if len(move_str) >= 4 and move_str[1].isdigit():
                        move = chess.Move.from_uci(move_str)
                    else:
                        move = sol_board.parse_san(move_str)
                except Exception:
                    pass
                if move and move in sol_board.legal_moves:
                    aid = sol_board.move_to_action_id(move)
                    if aid is None:
                        # Cannot convert move to action_id; stop building solution to avoid misalignment
                        break
                    solution_action_ids.append(aid)
                    sol_board.push(move)
                else:
                    break
        except Exception:
            pass

    mcts_iterations = cfg.mcts.iterations
    c_puct = cfg.mcts.c_puct
    temp_start = cfg.mcts.temperature_start
    temp_end = cfg.mcts.temperature_end
    temp_decay_moves = cfg.mcts.temperature_decay_moves
    dirichlet_alpha = cfg.mcts.dirichlet_alpha
    dirichlet_epsilon = cfg.mcts.dirichlet_epsilon
    action_space_size = cfg.network.action_space_size
    max_moves = cfg.training.max_game_moves
    if isinstance(max_game_moves_override, (int, float)) and max_game_moves_override > 0:
        max_moves = int(max_game_moves_override)
    
    # Track MCTS tree statistics
    tree_stats_list = []  # Store stats for each move
    gpu_cleanup_interval = 8  # Clean GPU memory every N moves (8 ensures cleanup within short mate-in-2/3/4 games)

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
        # Temperature decay: same formula for all positions (standard and custom start)
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
            draw_reward = cfg.training.get('draw_reward')
            draw_reward_table = cfg.training.get('draw_reward_table')
            if draw_reward_table is not None and OmegaConf.is_config(draw_reward_table):
                draw_reward_table = OmegaConf.to_container(draw_reward_table, resolve=True)
            sp_dtype = self_play_dtype_override if self_play_dtype_override is not None else _resolve_self_play_dtype(cfg, str(device) if device else "cpu")
            mcts_player = MCTS(
                network,
                device=device,
                env=mcts_env,
                C_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                action_space_size=action_space_size,
                history_steps=cfg.env.history_steps,
                draw_reward=draw_reward,
                pre_init_draws=getattr(cfg.mcts, 'pre_init_draws', False),
                inference_client=inference_client,
                draw_reward_table=draw_reward_table,
                initial_position_quality=initial_position_quality,
                self_play_dtype=sp_dtype,
            )
            mcts_start = time.perf_counter()
            mcts_player.search(root_node, mcts_iterations, batch_size=cfg.mcts.batch_size, progress=progress)
            mcts_elapsed = time.perf_counter() - mcts_start
            # Restore board state before any use of env.board.
            try:
                env.board.set_fen(fen_string_before_mcts)
            except Exception as e:
                logging.warning(f"Failed to restore board before legal_actions: {e}")
            # Use root_node.board for legal_actions: it is the pre-MCTS copy, never modified during
            # search. env.board may be modified when mcts_env is not None (sequential expansion).
            # Use temperature=0 when any root child is a winning terminal (mate in one available)
            # Ensures we always pick the mating move regardless of initial FEN
            move_temp = temperature
            if root_node.children:
                for child in root_node.children.values():
                    if child.is_terminal() and child.N > 0 and child.Q() >= 0.99:
                        move_temp = 0.0
                        break
            mcts_policy = mcts_player.get_policy_distribution(root_node, temperature=move_temp)
            # Raw MCTS policy (visit_counts/total, no temperature): used for explanatory logging
            raw_mcts_policy = mcts_player.get_policy_distribution(root_node, temperature=1.0)
            
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
            
            # AlphaZero-style: sample only from legal actions (or follow dataset trajectory if enabled)
            legal_actions = get_legal_actions(root_node.board)
            selection_probs = {}  # action_id -> actual prob used for np.random.choice (renormalized over legal)
            is_following_solution = (
                solution_action_ids
                and move_list_action_ids == solution_action_ids[: len(move_list_action_ids)]
            )
            ground_truth_action = (
                solution_action_ids[move_count]
                if is_following_solution and move_count < len(solution_action_ids)
                else None
            )
            follow_trajectory = (
                cfg.training.get("follow_dataset_trajectory", False)
                and ground_truth_action is not None
                and ground_truth_action in legal_actions
            )
            if follow_trajectory:
                action_to_take = ground_truth_action
                selection_probs[ground_truth_action] = 1.0
            elif legal_actions:
                legal_indices = [a - 1 for a in legal_actions if 0 <= (a - 1) < len(mcts_policy)]
                if legal_indices:
                    legal_probs = np.array([mcts_policy[i] for i in legal_indices], dtype=np.float64)
                    if legal_probs.sum() > 0:
                        legal_probs = legal_probs / legal_probs.sum()
                        action_to_take = np.random.choice([i + 1 for i in legal_indices], p=legal_probs)
                        for idx, aid in enumerate([i + 1 for i in legal_indices]):
                            selection_probs[aid] = float(legal_probs[idx])
                    else:
                        action_to_take = np.random.choice(legal_actions)
                        for aid in legal_actions:
                            selection_probs[aid] = 1.0 / len(legal_actions)
                else:
                    action_to_take = np.random.choice(legal_actions)
                    for aid in legal_actions:
                        selection_probs[aid] = 1.0 / len(legal_actions)
            else:
                raise RuntimeError(
                    f"No legal actions at move {move_count + 1} (expected terminal). "
                    f"FEN={env.board.fen()}, outcome={env.board.outcome()}"
                )
            
            # Save a copy of mcts_policy before cleanup (needed for game_history)
            mcts_policy_copy = mcts_policy.copy()
            
            # Build policy details for game_history file (before cleanup)
            # Top actions by converted policy (descending)
            sorted_indices = np.argsort(mcts_policy)[::-1]
            top_actions = [
                int(idx + 1) for idx in sorted_indices
                if mcts_policy[idx] > 1e-9
            ][:3]
            # Store only probs for significant actions (not full 4672 arrays)
            sig_actions = {int(action_to_take), ground_truth_action or -1}
            sig_actions.update(top_actions[:3])
            sig_actions.discard(-1)
            raw_probs = {aid: float(raw_mcts_policy[aid - 1]) for aid in sig_actions if 1 <= aid <= len(raw_mcts_policy)}
            conv_probs = {aid: float(mcts_policy_copy[aid - 1]) for aid in sig_actions if 1 <= aid <= len(mcts_policy_copy)}
            sel_probs = {aid: selection_probs.get(aid, conv_probs.get(aid, 0.0)) for aid in sig_actions}
            policy_details_list.append({
                "temperature": temperature,
                "move_temp": move_temp,
                "dirichlet_alpha": dirichlet_alpha,
                "dirichlet_epsilon": dirichlet_epsilon,
                "selected_action": int(action_to_take),
                "top1_action": top_actions[0] if len(top_actions) >= 1 else None,
                "top2_action": top_actions[1] if len(top_actions) >= 2 else None,
                "top3_action": top_actions[2] if len(top_actions) >= 3 else None,
                "ground_truth_action": ground_truth_action,
                "raw_probs": raw_probs,
                "converted_probs": conv_probs,
                "selection_probs": sel_probs,
            })
            
            # Explicit cleanup: Delete MCTS tree and player to free memory immediately
            del root_node
            del mcts_player
            mcts_policy = None  # Help GC
            mcts_policy = mcts_policy_copy  # Restore for game_history
            
            # Periodic GPU memory cleanup (CUDA + MPS)
            if device.type in ('cuda', 'mps') and (move_count + 1) % gpu_cleanup_interval == 0:
                _empty_device_cache(device)
        else:
            mcts_policy = np.zeros(cfg.network.action_space_size)
            legal_actions = get_legal_actions(env.board)
            for action_id in legal_actions:
                mcts_policy[action_id - 1] = 1
            is_following_solution = (
                solution_action_ids
                and move_list_action_ids == solution_action_ids[: len(move_list_action_ids)]
            )
            ground_truth_action = (
                solution_action_ids[move_count]
                if is_following_solution and move_count < len(solution_action_ids)
                else None
            )
            follow_trajectory = (
                cfg.training.get("follow_dataset_trajectory", False)
                and ground_truth_action is not None
                and ground_truth_action in legal_actions
            )
            action_to_take = ground_truth_action if follow_trajectory else np.random.choice(legal_actions)
            root_value = None  # No MCTS, so no value available
            # Policy details for no-MCTS case (uniform policy)
            top_actions = list(legal_actions)[:3] if legal_actions else []
            sig_actions = {int(action_to_take), ground_truth_action or -1}
            sig_actions.update(top_actions[:3])
            sig_actions.discard(-1)
            n_legal = len(legal_actions) or 1
            uniform_sel = 1.0 / n_legal
            raw_probs = {aid: float(mcts_policy[aid - 1]) for aid in sig_actions if 1 <= aid <= len(mcts_policy)}
            sel_probs = (
                {aid: (1.0 if aid == ground_truth_action else 0.0) for aid in sig_actions}
                if follow_trajectory
                else {aid: uniform_sel for aid in sig_actions}
            )
            policy_details_list.append({
                "temperature": temperature,
                "move_temp": temperature,
                "dirichlet_alpha": None,
                "dirichlet_epsilon": None,
                "selected_action": int(action_to_take),
                "top1_action": top_actions[0] if len(top_actions) >= 1 else None,
                "top2_action": top_actions[1] if len(top_actions) >= 2 else None,
                "top3_action": top_actions[2] if len(top_actions) >= 3 else None,
                "ground_truth_action": ground_truth_action,
                "raw_probs": raw_probs,
                "converted_probs": raw_probs.copy(),
                "selection_probs": sel_probs,
            })

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
        
        # No fallback: fail fast with debug info when action is illegal
        legal_actions = list(env.board.legal_actions)
        if action_to_take not in legal_actions:
            raise RuntimeError(
                f"action_to_take {action_to_take} not in legal_actions (move {move_count + 1}). "
                f"FEN={env.board.fen()}, root_value={root_value}, "
                f"position_matches={position_matches}, "
                f"legal_actions={legal_actions[:15]}{'...' if len(legal_actions) > 15 else ''}"
            )

        # Record SAN notation BEFORE playing the move
        move = env.board.action_id_to_move(action_to_take)
        if move is None:
            raise RuntimeError(
                f"action_id_to_move({action_to_take}) returned None (move {move_count + 1}). "
                f"FEN={env.board.fen()}, root_value={root_value}, "
                f"legal_actions_count={len(legal_actions)}"
            )
        if move not in env.board.legal_moves:
            raise RuntimeError(
                f"action_id_to_move({action_to_take}) returned non-legal move {move.uci()} (move {move_count + 1}). "
                f"FEN={env.board.fen()}, root_value={root_value}"
            )

        # Get SAN notation (move already validated above)
        try:
            san_move = env.board.san(move)
        except Exception as e:
            raise RuntimeError(
                f"SAN generation failed for action {action_to_take}, move={move.uci()} (move {move_count + 1}). "
                f"FEN={env.board.fen()}, error={e}"
            ) from e

        move_list_san.append(san_move)
        move_list_uci.append(move.uci())
        move_list_action_ids.append(action_to_take)

        # Check for same-color double moves before applying the move
        current_turn_before_move = env.board.turn
        current_color_before = 'White' if current_turn_before_move == chess.WHITE else 'Black'
        
        # Check if the move being attempted matches the current turn
        move_color_mismatch = False
        piece = env.board.piece_at(move.from_square)
        if piece is not None:
            piece_color = 'White' if piece.color == chess.WHITE else 'Black'
            if piece_color != current_color_before:
                move_color_mismatch = True
        
        if previous_turn is not None and current_turn_before_move == previous_turn:
            raise RuntimeError(
                f"Same color double move at move {move_count + 1}: "
                f"previous_turn=current_turn={current_color_before}, "
                f"move={san_move}, action={action_to_take}, FEN={env.board.fen()}"
            )
        
        if move_color_mismatch:
            raise RuntimeError(
                f"Move color mismatch at move {move_count + 1}: current turn={current_color_before}, "
                f"move={san_move} (action {action_to_take}), FEN={env.board.fen()}"
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
    use_draw_table = draw_reward is None and cfg.training.get("draw_reward_table") and termination_reason and initial_position_quality

    def _value_from_winner(board_at_state, outcome_winner, draw_val):
        if outcome_winner is None:
            return float(draw_val)
        if board_at_state.turn == chess.WHITE:
            return 1.0 if outcome_winner == chess.WHITE else -1.0
        return 1.0 if outcome_winner == chess.BLACK else -1.0

    def _draw_reward_for_position(state_obs, board_at_state, mcts_value, term_reason, init_quality):
        """Use draw_reward_table when available; otherwise fixed draw_reward."""
        if use_draw_table:
            is_white_turn = board_at_state.turn == chess.WHITE
            return reward_computer.compute_draw_reward(
                state_obs, is_white_turn, term_reason, mcts_value, init_quality
            )
        return draw_reward if draw_reward is not None else draw_reward_for_calc

    full_game_data = []
    for i, history_item in enumerate(game_history):
        # New format only: (state_obs, policy_target, board_at_state, mcts_value)
        if len(history_item) != 4:
            raise ValueError(
                f"Invalid game_history item length: expected 4, got {len(history_item)}"
            )
        state_obs, policy_target, board_at_state, _value_target = history_item
        if winner is not None:
            outcome_value = _value_from_winner(board_at_state, winner, 0.0)  # draw_val unused for wins
        else:
            outcome_value = _draw_reward_for_position(state_obs, board_at_state, _value_target, termination_reason, initial_position_quality)
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
    actual_reward_for_logging = full_game_data[0][3] if full_game_data else None
    
    # Return game data, move list in SAN, and termination reason
    game_info = {
        'moves_san': ' '.join(move_list_san),
        'moves_uci': ' '.join(move_list_uci),
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
        'initial_dataset_source': initial_dataset_source,
        'position_themes': initial_themes if initial_themes else None,
        'policy_details': policy_details_list,
    }

    # Final cleanup before return (helps with memory when using large datasets)
    if device is not None and device.type in ('cuda', 'mps'):
        _empty_device_cache(device)
    gc.collect()

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


def _is_xla_device(device: torch.device) -> bool:
    """Return True if device is XLA/TPU. Used to choose map_location when loading checkpoints."""
    if device is None:
        return False
    return str(device.type).lower() in ("xla", "tpu")


def _select_training_device() -> torch.device:
    if os.environ.get("XRT_TPU_CONFIG") or os.environ.get("COORDINATOR_ADDRESS"):
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            pass
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
    cfg: DictConfig, device: torch.device, progress: Progress, log_path: str | Path | None = None
) -> tuple[nn.Module, optim.Optimizer, DictConfig, float]:
    network = create_network(cfg, device)
    network.eval()

    # Freeze first N conv layers if configured (train only layers after)
    freeze_n_raw = cfg.network.get("freeze_first_n_conv_layers", 0)
    freeze_n = int(freeze_n_raw) if freeze_n_raw is not None else 0
    if freeze_n > 0:
        freeze_first_n_conv_layers(network, freeze_n)
        n_trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
        progress.print(f"Frozen first {freeze_n} conv layers. Training {n_trainable:,} parameters.")

    profile_network = create_network(cfg, device)
    profile_network.eval()
    N, C, H, W = cfg.training.batch_size, cfg.network.input_channels, cfg.network.board_size, cfg.network.board_size
    if log_path is not None:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Using action space size: {cfg.network.action_space_size}\n")
                f.flush()
        except Exception:
            pass
    profile_model(profile_network, (torch.randn(N, C, H, W).to(device),), log_path=log_path)
    del profile_network

    opt_cfg = cfg.optimizer
    actual_learning_rate = opt_cfg.learning_rate
    trainable_params = [p for p in network.parameters() if p.requires_grad]
    if opt_cfg.type == "Adam":
        optimizer = optim.Adam(trainable_params, lr=opt_cfg.learning_rate, weight_decay=opt_cfg.weight_decay)
    elif opt_cfg.type == "SGD":
        momentum = opt_cfg.momentum if opt_cfg.momentum is not None else 0.9
        optimizer = optim.SGD(
            trainable_params,
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


def _init_checkpoint_dirs(cfg: DictConfig, progress: Progress, log_path: str | Path | None = None):
    checkpoint_dir = cfg.training.checkpoint_dir
    checkpoint_load = cfg.training.get("checkpoint_load", None)
    load_base = checkpoint_load if checkpoint_load not in (None, "", "null") else checkpoint_dir
    # Allow direct file path (e.g. .pth); otherwise treat as directory
    if load_base.endswith(".pth") or (os.path.exists(load_base) and os.path.isfile(load_base)):
        load_checkpoint_path = load_base
    else:
        load_checkpoint_path = os.path.join(load_base, "model.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    abs_path = os.path.abspath(checkpoint_dir)
    progress.print(f"Checkpoints will be saved in: {abs_path}")
    return checkpoint_dir, load_checkpoint_path


def _init_game_history_dir(cfg: DictConfig, progress: Progress, log_path: str | Path | None = None):
    game_history_dir = cfg.training.get("game_history_dir", None)
    if game_history_dir not in (None, "", "null"):
        os.makedirs(game_history_dir, exist_ok=True)
        abs_path = os.path.abspath(game_history_dir)
        progress.print(f"Game histories will be saved in: {abs_path}")
    else:
        game_history_dir = None
    return game_history_dir


def _save_game_history(
    game_moves_list,
    game_history_dir: str | None,
    iteration: int,
    avg_policy_loss: float,
    avg_value_loss: float,
    *,
    train_iteration: int | None = None,
    positions_label: str | None = None,
    mcts_iterations: int | None = None,
    max_positions: int | None = None,
):
    """Save game history to a text file.

    Training mode: uses iteration, avg_policy_loss, avg_value_loss for filename.
    Diagnostic mode: when train_iteration, positions_label, mcts_iterations, max_positions
    are provided, uses abbreviated format: games_t{train}_p{pos}_i{iter}_n{max}.txt
    """
    if not game_moves_list or not game_history_dir:
        return
    # Diagnostic naming: t=train_iter, p=positions, i=mcts_iter, n=max_pos
    if train_iteration is not None and positions_label is not None and mcts_iterations is not None and max_positions is not None:
        pos_slug = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(positions_label))[:24]
        games_filename = f"games_t{train_iteration}_p{pos_slug}_i{mcts_iterations}_n{max_positions}.txt"
    else:
        games_filename = f"games_iter_{iteration+1}_p{avg_policy_loss:.4f}_v{avg_value_loss:.4f}.txt"
    games_file = os.path.join(game_history_dir, games_filename)
    with open(games_file, 'w') as f:
        # Legend once at top (only if any game has policy details)
        has_any_policy = any(g.get('policy_details') for g in game_moves_list)
        if has_any_policy:
            f.write("Raw=visit/total. Selection=prob used for np.random.choice (renormalized over legal). *SELECTED ✓GROUND_TRUTH\n\n")
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
            term = (game_info.get('termination') or 'unknown').upper()
            f.write(f"Game {i+1}: {result_str} ({term}, {game_info['move_count']} moves)\n")
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
            initial_fen_source = game_info.get('initial_dataset_source', None)
            initial_fen_label = game_info.get('initial_dataset_label', None)
            if initial_fen_source:
                f.write(f"Initial FEN Source: {initial_fen_source}\n")
            if initial_fen_label:
                f.write(f"Initial FEN Label: {initial_fen_label}\n")
            # Write position themes if available (from dataset entry)
            position_themes = game_info.get('position_themes')
            if position_themes and isinstance(position_themes, (list, tuple)):
                themes_str = ', '.join(str(t) for t in position_themes)
                f.write(f"Position Themes: {themes_str}\n")
            # Write reward value (use actual_reward if available, otherwise result)
            actual_reward = game_info.get('actual_reward', None)
            result_value = game_info.get('result', None)
            if actual_reward is not None:
                f.write(f"Reward: {actual_reward:.4f}\n")
            elif result_value is not None:
                f.write(f"Reward: {result_value:.4f}\n")
            f.write(f"{game_info['moves_san']}\n")

            # Policy probability details for each state (one block per move)
            policy_details = game_info.get('policy_details', [])
            moves_san_list = game_info['moves_san'].split() if game_info.get('moves_san') else []
            moves_uci_list = game_info['moves_uci'].split() if game_info.get('moves_uci') else []
            # Prefer UCI for replay (unambiguous); SAN can pick wrong piece when ambiguous
            use_uci_replay = len(moves_uci_list) >= len(moves_san_list) and len(moves_uci_list) > 0
            from MCTS.training_modules.chess import create_board_from_fen
            init_fen = game_info.get('initial_fen') or chess.STARTING_FEN
            for move_idx, pd in enumerate(policy_details):
                move_san = moves_san_list[move_idx] if move_idx < len(moves_san_list) else "?"
                temp = pd.get("temperature")
                d_alpha = pd.get("dirichlet_alpha")
                d_eps = pd.get("dirichlet_epsilon")
                if d_alpha is not None and d_eps is not None:
                    header = f"  --- Move {move_idx + 1} ({move_san}) | temp={temp:.4f} | Dirichlet α={d_alpha} ε={d_eps} ---\n"
                else:
                    header = f"  --- Move {move_idx + 1} ({move_san}) | temp={temp:.4f} | Dirichlet N/A ---\n"
                f.write(f"\n{header}")
                raw_probs = pd.get("raw_probs") or {}
                sel_probs = pd.get("selection_probs") or pd.get("converted_probs") or {}
                sel = pd.get("selected_action")
                gt = pd.get("ground_truth_action")
                aids = {sel, pd.get("top1_action"), pd.get("top2_action"), pd.get("top3_action"), gt} - {None}
                # Reconstruct board at this position for action_id -> move (use UCI when available to avoid SAN ambiguity)
                board_at_pos = create_board_from_fen(init_fen)
                for k in range(move_idx):
                    if use_uci_replay and k < len(moves_uci_list):
                        board_at_pos.push(chess.Move.from_uci(moves_uci_list[k]))
                    elif k < len(moves_san_list):
                        board_at_pos.push(board_at_pos.parse_san(moves_san_list[k]))
                legal_at_pos = list(board_at_pos.legal_actions)
                # Pad to at least top 3 when 3+ legal moves exist
                if len(aids) < 3 and len(legal_at_pos) >= 3:
                    for aid in legal_at_pos:
                        if aid not in aids:
                            aids.add(aid)
                            if len(aids) >= 3:
                                break
                entries = []
                move_san = moves_san_list[move_idx] if move_idx < len(moves_san_list) else "?"
                gt_move_uci = pd.get("ground_truth_move_uci")
                legal_at_pos_set = set(legal_at_pos)
                for aid in aids:
                    raw_prob = raw_probs.get(int(aid), 0.0)
                    sel_prob = sel_probs.get(int(aid), 0.0)
                    move_str = "?"
                    if int(aid) in legal_at_pos_set:
                        mv = board_at_pos.action_id_to_move(int(aid))
                        if mv is None:
                            raise RuntimeError(
                                f"_save_game_history: action_id_to_move({aid}) returned None at move {move_idx + 1}, "
                                f"FEN={board_at_pos.fen()[:60]}, legal_at_pos_sample={list(legal_at_pos)[:10]}"
                            )
                        move_str = board_at_pos.san(mv)
                    elif aid == sel:
                        raise RuntimeError(
                            f"_save_game_history: selected action {aid} not in legal_at_pos at move {move_idx + 1}, "
                            f"FEN={board_at_pos.fen()[:60]}, legal_at_pos_sample={list(legal_at_pos)[:10]}"
                        )
                    tags = ""
                    if aid == sel:
                        tags += "*"  # SELECTED
                    if aid == gt:
                        tags += "✓"  # GROUND_TRUTH(labeled)
                    entries.append((raw_prob, f"{tags}{move_str}: {raw_prob*100:.1f}% ({sel_prob*100:.1f}%)"))
                # When ground truth exists but isn't legal (model deviated), show solution move
                if gt is None and gt_move_uci and gt_move_uci not in {e[1].split(":")[0].lstrip("*✓") for e in entries}:
                    mv = chess.Move.from_uci(gt_move_uci)
                    gt_display = board_at_pos.san(mv) if mv in board_at_pos.legal_moves else gt_move_uci
                    entries.append((0.0, f"✓{gt_display}: 0.0% (0.0%)"))
                entries.sort(key=lambda x: -x[0])  # by raw prob descending (high->low)
                line = " | ".join(e[1] for e in entries)
                f.write(f"  {line}\n")
            f.write("\n")


def _init_self_play_infra(cfg: DictConfig):
    manager = multiprocessing.Manager()
    sp_queue = manager.Queue(maxsize=cfg.training.get("continual_queue_maxsize", 64))
    stop_event = manager.Event()
    continual_enabled = bool(cfg.training.get("continual_training", False))
    actors: list[multiprocessing.Process] = []
    actor_device_map: dict[int, str] = {}
    return manager, sp_queue, stop_event, continual_enabled, actors, actor_device_map


def _start_inference_server(cfg: DictConfig, manager, checkpoint_path: str, network_state_dict: dict, reply_queues_by_worker: dict = None):
    if not cfg.training.get("use_inference_server", False):
        return None, None, None, None, None
    if reply_queues_by_worker is None:
        reply_queues_by_worker = {}

    queue_maxsize = cfg.training.get("inference_queue_maxsize")
    if queue_maxsize in (None, "", "null"):
        nw = len(reply_queues_by_worker) if reply_queues_by_worker else 1
        queue_maxsize = max(32, nw * 8)  # Avoid workers blocking on put() when server is slow
    queue_maxsize = int(queue_maxsize)
    training_batch_size = int(cfg.training.get("batch_size", 64))
    num_workers = len(reply_queues_by_worker) if reply_queues_by_worker else 1
    games_per_worker = max(1, int(cfg.training.get("games_per_worker", 1)))
    max_batch = cfg.training.get("inference_server_max_batch_size")
    if max_batch in (None, "", "null"):
        max_batch = training_batch_size
    else:
        max_batch = int(max_batch)
    max_stacked_requests = min(max_batch, num_workers * games_per_worker)
    min_stacked_requests = cfg.training.get("inference_server_min_stacked_requests")
    if min_stacked_requests in (None, "", "null"):
        min_stacked_requests = max(1, max_stacked_requests // 2)
    min_stacked_requests = int(min_stacked_requests)
    max_wait_ms = int(cfg.training.get("inference_server_max_wait_ms", 2))
    logging_enabled = bool(cfg.training.get("inference_server_logging_enabled", False))
    device_str = cfg.training.get("inference_server_device", None)
    if not device_str or str(device_str).lower() in ("none", "null", "auto"):
        if os.environ.get("XRT_TPU_CONFIG") or os.environ.get("COORDINATOR_ADDRESS"):
            device_str = "xla"
        elif torch.cuda.is_available():
            device_str = "cuda:0"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"

    request_queue = multiprocessing.get_context("spawn").Queue(maxsize=queue_maxsize)
    stop_event = manager.Event()
    ready_event = manager.Event()
    cfg_for_worker = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg

    use_tpu = str(device_str).lower() in ("xla", "tpu")
    tpu_lock = None

    if use_tpu:
        import threading
        tpu_lock = threading.Lock()
        t = threading.Thread(
            target=inference_server_worker_tpu,
            args=(
                checkpoint_path,
                cfg_for_worker,
                device_str,
                request_queue,
                stop_event,
                max_stacked_requests,
                min_stacked_requests,
                max_wait_ms,
                reply_queues_by_worker,
                network_state_dict,
                tpu_lock,
                logging_enabled,
            ),
            daemon=True,
        )
        t.start()
        return request_queue, stop_event, t, device_str, tpu_lock
    else:
        p = multiprocessing.get_context("spawn").Process(
            target=inference_server_worker,
            args=(
                checkpoint_path,
                cfg_for_worker,
                device_str,
                request_queue,
                stop_event,
                max_stacked_requests,
                min_stacked_requests,
                max_wait_ms,
                reply_queues_by_worker,
                network_state_dict,
                logging_enabled,
                ready_event,
            ),
        )
        p.daemon = True
        p.start()
        # Wait for GPU inference server to be ready (avoids silent failure when spawn fails)
        ready_timeout_s = 60
        if not ready_event.wait(timeout=ready_timeout_s):
            print(
                f"Inference server on {device_str} did not signal ready within {ready_timeout_s}s. "
                "Self-play may be stuck. Check stderr for CUDA/GPU errors. "
                "On Kaggle TPU kernel, cuda:0 might not exist - use inference_server_device: xla or null.",
                flush=True,
            )
        return request_queue, stop_event, p, device_str, None


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
        # TPU/XLA: torch.load cannot map directly to xla:0. Load to CPU first, then move to device.
        load_device = "cpu" if _is_xla_device(device) else device
        checkpoint = torch.load(checkpoint_path, map_location=load_device, weights_only=False)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.to(device)

        # Load optimizer only if checkpoint is from RL training (not from supervised learning)
        is_rl_checkpoint = (
            checkpoint.get("training_mode") == "rl"
            or (checkpoint.get("training_mode") is None and checkpoint.get("total_games_simulated", 0) > 0)
        )
        if not is_rl_checkpoint:
            progress.print(
                "Checkpoint appears to be from supervised learning (or unknown). Using fresh optimizer for RL training."
            )

        if is_rl_checkpoint and "optimizer_state_dict" in checkpoint:
            try:
                checkpoint_opt_state = checkpoint["optimizer_state_dict"]
                if "param_groups" in checkpoint_opt_state and len(checkpoint_opt_state["param_groups"]) > 0:
                    checkpoint_lr = checkpoint_opt_state["param_groups"][0].get(
                        "lr", opt_cfg.learning_rate
                    )
                    if checkpoint_lr != opt_cfg.learning_rate:
                        actual_learning_rate = opt_cfg.learning_rate
                        progress.print(
                            f"Using learning rate from config: {opt_cfg.learning_rate} (checkpoint had: {checkpoint_lr})"
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
                        f"Warning: Learning rate mismatch! Desired {actual_learning_rate}, restored to {restored_lr}"
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
                trainable_params = [p for p in network.parameters() if p.requires_grad]
                if opt_cfg.type == "Adam":
                    optimizer = optim.Adam(
                        trainable_params,
                        lr=opt_cfg.learning_rate,
                        weight_decay=opt_cfg.weight_decay,
                    )
                elif opt_cfg.type == "SGD":
                    momentum = opt_cfg.momentum if opt_cfg.momentum is not None else 0.9
                    optimizer = optim.SGD(
                        trainable_params,
                        lr=opt_cfg.learning_rate,
                        momentum=momentum,
                        weight_decay=opt_cfg.weight_decay,
                    )
                actual_learning_rate = opt_cfg.learning_rate
                if not isinstance(e, RuntimeError) or str(e) != "skip_optimizer_load":
                    import traceback

                    traceback.print_exc()
        elif not is_rl_checkpoint:
            progress.print("Skipping optimizer load (checkpoint not from RL). Optimizer starts fresh.")
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

        buffer_loaded = False
        if "replay_buffer_state" in checkpoint and not cfg.training.get("skip_replay_buffer_load", False):
            try:
                replay_buffer_state = checkpoint["replay_buffer_state"]
                replay_buffer.load_state(replay_buffer_state, create_board_from_serialized)
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

    # Log path for writing model summary etc. (excluded from Tee progress filter)
    log_path = None
    try:
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        if hydra_cfg is not None:
            log_path = Path(hydra_cfg.runtime.output_dir) / "train.log"
    except Exception:
        pass

    # Rich's Live display bypasses sys.stdout, so tee cannot capture progress.print().
    # Wrap progress to also write progress.print() output to train.log.
    if log_path is not None:
        progress = LoggingProgressWrapper(progress, log_path)

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
        cfg, device, progress, log_path=log_path
    )
    replay_buffer = _init_replay_buffer(cfg, progress)

    dataset_entries, _weights, source_labels = _get_cached_dataset_entries(cfg.training.get("initial_board_fen", None))
    # Mate tracking per datafile: use dataset entry index, include any entry with mate-related label or source path
    mate_entry_indices: list[int] = []
    mate_entry_labels: list[str] = []
    mate_label_to_entry_id: dict[str, int] = {}
    for idx, entry in enumerate(dataset_entries):
        label = entry.get("label", "")
        source = str(entry.get("source", ""))
        is_mate = bool(
            (isinstance(label, str) and re.match(r"^m{1,2}\d+(#\d+)?$", label))
            or ("mate" in source.lower())
        )
        if is_mate:
            mate_entry_indices.append(idx)
            mate_entry_labels.append(label)
            mate_label_to_entry_id[label] = idx
    mate_id_to_idx = {eid: i for i, eid in enumerate(mate_entry_indices)}

    # Loss Functions
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # Checkpoint directories (relative to hydra run dir)
    checkpoint_dir, load_checkpoint_path = _init_checkpoint_dirs(cfg, progress, log_path=log_path)

    # Game history directory setup (optional)
    game_history_dir = _init_game_history_dir(cfg, progress, log_path=log_path)

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
    inference_request_queue = None
    inference_stop_event = None
    inference_process = None
    inference_device_str = None

    # --- Mode selection for self-play ---
    # Note: Actor launching for continual mode happens after checkpoint loading
    # to check buffer size and exclude training device if we have enough data
    if continual_enabled:
        # Will launch actors after checkpoint loading
        pass
    elif use_multiprocessing_flag and num_workers > 1:
        # When using TPU inference server, keep device as TPU for training
        use_tpu_inference = (
            cfg.training.get("use_inference_server", False)
            and str(cfg.training.get("inference_server_device", "")).lower() in ("xla", "tpu")
        ) or os.environ.get("XRT_TPU_CONFIG") or os.environ.get("COORDINATOR_ADDRESS")
        if not use_tpu_inference:
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
        'mate_success': {label: [] for label in mate_entry_labels},
        'mate_success_labels': mate_entry_labels,
    }

    # Check for existing checkpoint (load_checkpoint_path is already resolved dir/model.pth or direct file)
    (
        start_iter,
        total_games_simulated,
        actual_learning_rate,
        buffer_loaded,
        history,
    ) = _load_checkpoint(
        cfg=cfg,
        checkpoint_path=load_checkpoint_path,
        network=network,
        optimizer=optimizer,
        opt_cfg=opt_cfg,
        device=device,
        progress=progress,
        replay_buffer=replay_buffer,
        actual_learning_rate=actual_learning_rate,
        history=history,
    )

    # Early termination if already reached target iterations (e.g. resumed from checkpoint)
    if start_iter >= cfg.training.num_training_iterations:
        progress.print(
            f"\nTraining already complete: reached {cfg.training.num_training_iterations} iterations. Exiting without saving."
        )
        env.close()
        try:
            from hydra.core.hydra_config import HydraConfig
            hydra_cfg = HydraConfig.get()
            if hydra_cfg is not None:
                output_dir = Path(hydra_cfg.runtime.output_dir)
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                    progress.print(f"Removed empty Hydra output dir: {output_dir}")
        except Exception:
            pass
        progress.print("\nTraining loop finished.")
        return

    history = _ensure_mate_success_history(history, mate_entry_labels)
    if mate_entry_labels:
        progress.print(f"Mate-in success tracking enabled per datafile: {', '.join(mate_entry_labels)} (see second plot in visualize_learning_curves_RL.py)")

    # Save loaded model to checkpoint_dir so workers can load it when not using inference server
    # (workers read checkpoint_dir/model.pth; if we loaded from checkpoint_load, that file may not exist yet)
    if continual_enabled and not cfg.training.get("use_inference_server", False):
        actors_checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        if not os.path.exists(actors_checkpoint_path) or load_checkpoint_path != actors_checkpoint_path:
            try:
                model_state = network.state_dict()
                if not _state_dict_has_nan_or_inf(model_state):
                    torch.save(
                        {"model_state_dict": model_state, "iteration": start_iter, "training_mode": "rl"},
                        actors_checkpoint_path,
                    )
            except Exception:
                pass

    # Optional inference server for batched GPU/TPU inference
    inference_request_queue = None
    inference_reply_queues = None
    tpu_lock = None
    if cfg.training.get("use_inference_server", False):
        checkpoint_path_for_actors = os.path.join(cfg.training.checkpoint_dir, "model.pth")
        # Move state_dict to CPU before passing to spawned process (GPU tensors cannot be pickled)
        network_state_cpu = {k: v.cpu().clone() if hasattr(v, "cpu") else v for k, v in network.state_dict().items()}
        desired_processes = max(1, num_workers if num_workers > 0 else 1)
        inference_reply_queues = [multiprocessing.get_context("spawn").Queue() for _ in range(desired_processes)]
        reply_queues_by_worker = {i: q for i, q in enumerate(inference_reply_queues)}
        inference_request_queue, inference_stop_event, inference_process, inference_device_str, tpu_lock = _start_inference_server(
            cfg,
            manager,
            checkpoint_path_for_actors,
            network_state_cpu,
            reply_queues_by_worker,
        )
        if inference_process is not None:
            training_batch_size = int(cfg.training.get("batch_size", 64))
            max_batch = cfg.training.get("inference_server_max_batch_size")
            if max_batch in (None, "", "null"):
                max_batch = training_batch_size
            else:
                max_batch = int(max_batch)
            nw = len(reply_queues_by_worker) if reply_queues_by_worker else 1
            gpw = max(1, int(cfg.training.get("games_per_worker", 1)))
            max_stacked = min(max_batch, nw * gpw)
            eff_wait = int(cfg.training.get("inference_server_max_wait_ms", 2))
            gpw = max(1, int(cfg.training.get("games_per_worker", 1)))
            if gpw > 1 and eff_wait < 10:
                eff_wait = max(eff_wait, min(25, gpw * 5))
            progress.print(
                f"Inference server enabled on {inference_device_str} "
                f"(max_stacked={max_stacked}, max_wait_ms={eff_wait})"
            )

    # TPU: pause self-play workers during training so inference queue drains and training can acquire TPU
    pause_for_training = None
    if tpu_lock is not None and cfg.training.get("continual_training", False):
        pause_for_training = manager.Event()

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

        if inference_request_queue is not None:
            device_pool = ['cpu'] * desired_processes
        else:
            # Use all available accelerators for self-play (training GPU will be assigned dynamically)
            accelerator_slots = min(len(available_accelerators), desired_processes)
            for i in range(accelerator_slots):
                device_pool.append(available_accelerators[i])
            # Fill remaining slots with CPU workers
            while len(device_pool) < desired_processes:
                device_pool.append('cpu')
        
        checkpoint_path_for_actors = os.path.join(cfg.training.checkpoint_dir, "model.pth")
        cfg_for_workers = OmegaConf.to_container(cfg, resolve=True)
        games_per_worker = max(1, int(cfg.training.get("games_per_worker", 1)))
        for worker_idx, dev in enumerate(device_pool):
            inference_reply_queue = inference_reply_queues[worker_idx] if (inference_reply_queues is not None and worker_idx < len(inference_reply_queues)) else None
            p = multiprocessing.get_context("spawn").Process(
                target=continual_self_play_worker,
                args=(
                    checkpoint_path_for_actors,
                    cfg_for_workers,
                    dev,
                    sp_queue,
                    stop_event,
                    inference_request_queue,
                    inference_reply_queue,
                    30.0,
                    worker_idx,
                    games_per_worker,
                    pause_for_training,
                ),
            )
            p.daemon = True
            p.start()
            actors.append(p)
            # Track both CUDA and MPS devices (not just CUDA)
            if dev.startswith('cuda:') or dev == 'mps':
                actor_device_map[p] = dev
        
        if inference_request_queue is not None:
            accelerator_slots = 0
        else:
            accelerator_slots = min(len(available_accelerators), desired_processes)

        has_enough_data = len(replay_buffer) >= cfg.training.batch_size
        if has_enough_data:
            progress.print(f"Buffer has enough data ({len(replay_buffer)} >= {cfg.training.batch_size}), training accelerator will be assigned dynamically to first available accelerator")
        else:
            if inference_request_queue is not None:
                progress.print(
                    f"Buffer too small ({len(replay_buffer)} < {cfg.training.batch_size}), using CPU workers with inference server"
                )
            else:
                progress.print(
                    f"Buffer too small ({len(replay_buffer)} < {cfg.training.batch_size}), using all {accelerator_slots} accelerator(s) for self-play"
                )
        games_info = f" ({games_per_worker} games/worker)" if games_per_worker > 1 else ""
        progress.print(f"Continual self-play: launched {len(actors)} actor(s): {device_pool}{games_info}")

    # Track iteration durations for time prediction
    iteration_durations = []  # List to store duration of each completed iteration
    previous_total_elapsed_time = 0  # Track previous total elapsed time to calculate iteration duration
    
    # Use num_training_iterations from cfg
    use_tpu = tpu_lock is not None
    amp_enabled_for_training = bool(
        cfg.training.get("amp", False)
        and (
            (isinstance(device, torch.device) and device.type in {"cuda", "mps"})
            or use_tpu
            or (
                device == "cpu"
                and use_multiprocessing_flag
                and (torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
            )
        )
    )
    if amp_enabled_for_training:
        progress.print("AMP enabled for training phase.")

    for iteration in range(start_iter, cfg.training.num_training_iterations):
        iteration_start_time = time.time()
        progress.print(f"\n--- Training Iteration {iteration+1}/{cfg.training.num_training_iterations} ---")

        # --- Self-Play Phase --- 
        self_play_columns = (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            # TimeRemainingColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[steps]} Steps (Games/1st/2nd: {task.fields[games]}/{task.fields[first_wins]}/{task.fields[second_wins]}, {task.fields[draw_rate]:.1f}% Draw)"),
        )
        progress.columns = self_play_columns
        
        # Only move to CPU if using multiple workers
        if use_multiprocessing_flag:
            network.to('cpu').eval()
        else:
            network.to(device).eval()

        games_data_collected = []
        iteration_start_time_selfplay = time.time()
        # Epoch 1: fill buffer to full before training; later epochs: collect self_play_steps_per_epoch
        buffer_size = cfg.training.replay_buffer_size
        if iteration == start_iter and len(replay_buffer) < buffer_size // 2:
            min_steps = buffer_size // 2 - len(replay_buffer)
            progress.print(f"Filling buffer before first training: need {min_steps} steps (buffer has {len(replay_buffer)}/{buffer_size//2})")
        else:
            min_steps = cfg.training.self_play_steps_per_epoch
        max_experiences = buffer_size  # Cap collection so we don't replace entire buffer
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
        val_mate_success = [0] * len(mate_entry_labels)
        val_mate_total = [0] * len(mate_entry_labels)
        val_mate_success_per_source = [0] * len(source_labels)
        val_mate_total_per_source = [0] * len(source_labels)
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

        def _update_mate_success(game_info):
            entry_id = game_info.get('initial_dataset_id')
            if entry_id is None and mate_label_to_entry_id:
                label = game_info.get('initial_dataset_label')
                entry_id = mate_label_to_entry_id.get(label) if label else None
            if entry_id is None:
                return
            first_player = _first_player_from_game_info(game_info)
            winner = game_info.get('winner', None)
            success = False
            if winner is not None and game_info.get('termination') == "CHECKMATE":
                winner_color = 'w' if winner == chess.WHITE else 'b' if winner == chess.BLACK else None
                if winner_color is not None and winner_color == first_player:
                    success = True
            if 0 <= entry_id < len(source_labels):
                val_mate_total_per_source[entry_id] += 1
                if success:
                    val_mate_success_per_source[entry_id] += 1
            if mate_id_to_idx and entry_id in mate_id_to_idx:
                idx = mate_id_to_idx[entry_id]
                val_mate_total[idx] += 1
                if success:
                    val_mate_success[idx] += 1

        def _get_draw_buffer_config(termination):
            """Get buffer config for draw termination: None=full game, 0=exclude, n>0=last n moves."""
            draw_cfg = cfg.training.get('last_n_moves_to_store')
            if isinstance(draw_cfg, dict):
                return draw_cfg.get(termination)
            if isinstance(draw_cfg, (list, tuple)):
                return 0 if termination in draw_cfg else None  # backward compat: list = exclude (0)
            return None

        def _include_in_replay_buffer(game_info) -> bool:
            """Include unless draw has value 0 in last_n_moves_to_store dict."""
            if game_info.get('winner') is not None:
                return True
            val = _get_draw_buffer_config(game_info.get('termination'))
            if val is not None and int(val) == 0:
                return False
            return True

        def _trim_draw_game_data(game_data, game_info):
            """For draws: if term has n>0 in config, return last n moves; else full game."""
            if not game_data:
                return game_data
            if game_info.get('winner') is not None:
                return game_data
            val = _get_draw_buffer_config(game_info.get('termination'))
            if val is not None and int(val) > 0:
                return game_data[-int(val):]
            return game_data

        if continual_enabled:
            # Drain queue to collect games for this iteration
            # Strategy: Drain all available games, wait if needed to reach minimum threshold
            # Limit: Stop if new experiences would completely replace the entire replay buffer
            games_data_collected = []
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
                                include_in_replay = _include_in_replay_buffer(game_info)
                                _update_mate_success(game_info)
                                data_to_add = _trim_draw_game_data(game_data, game_info) if include_in_replay else []
                                if include_in_replay:
                                    games_data_collected.extend(data_to_add)
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
                                        advance=len(data_to_add),
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
                            include_in_replay = _include_in_replay_buffer(game_info)
                            _update_mate_success(game_info)
                            data_to_add = _trim_draw_game_data(game_data, game_info) if include_in_replay else []
                            if include_in_replay:
                                games_data_collected.extend(data_to_add)
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
                                        advance=len(data_to_add),
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
                                    advance=len(data_to_add),
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
                                    include_in_replay = _include_in_replay_buffer(game_info)
                                    _update_mate_success(game_info)
                                    data_to_add = _trim_draw_game_data(game_data, game_info) if include_in_replay else []
                                    if include_in_replay:
                                        games_data_collected.extend(data_to_add)
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
                                            advance=len(data_to_add),
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
            _empty_device_cache()
            gc.collect()
        elif use_multiprocessing_flag:
            # Prepare arguments for workers
            # Pass network state_dict directly since it's clean
            network_state_dict = network.state_dict() if inference_request_queue is None else None
            # --- Mixed actors device assignment (GPU + CPU) ---
            available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            desired_processes = max(1, num_workers)
            device_pool = []
            if inference_request_queue is not None:
                device_pool = ['cpu'] * desired_processes
            else:
                # Prefer to keep at least one CPU actor when we have multiple processes
                gpu_slots = min(available_gpus, max(0, desired_processes - 1)) if desired_processes > 1 else 0
                for g in range(gpu_slots):
                    device_pool.append(f"cuda:{g}")
                # Fill remaining slots with CPU actors
                while len(device_pool) < desired_processes:
                    device_pool.append('cpu')

            if not device_pool:
                device_pool = ['cpu']

            # Collect games until we have enough steps (min_steps set above)
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
                inference_reply_queue = manager.Queue() if inference_request_queue is not None else None
                worker_args_packed.append(
                    (
                        game_num,
                        network_state_dict,
                        cfg_for_workers,
                        device_str,
                        inference_request_queue,
                        inference_reply_queue,
                        30.0,
                    )
                )
                game_num += 1

            pool = None # Initialize pool to None
            try:
                # Use 'spawn' context for better compatibility across platforms, especially with CUDA/PyTorch
                pool = multiprocessing.get_context("spawn").Pool(processes=len(device_pool))
                # progress.print(f"Submitting {len(worker_args_packed)} games to pool...")

                # Use imap_unordered to get results as they complete
                results_iterator = pool.imap_unordered(worker_wrapper, worker_args_packed)

                # Process results with a progress bar (buffer by game_id for submission-order stats)
                total_games_submitted = len(worker_args_packed)
                results_buffer = [None] * total_games_submitted
                next_to_display = 0
                task_id_selfplay = progress.add_task(
                    "Self-Play",
                    total=min_steps,
                    games=0,
                    steps=0,
                    draw_rate=0.0,
                )
                # Iterate over results as they become available
                for game_result in results_iterator:
                    if game_result:  # Check if worker returned valid data
                        game_id, game_data, game_info = game_result
                        if 0 <= game_id < total_games_submitted:
                            results_buffer[game_id] = (game_data, game_info)
                    # Consume in-order results for submission-order draw rate
                    prev_next = next_to_display
                    total_advance = 0
                    while next_to_display < total_games_submitted and results_buffer[next_to_display] is not None:
                        game_data, game_info = results_buffer[next_to_display]
                        include_in_replay = _include_in_replay_buffer(game_info)
                        _update_mate_success(game_info)
                        data_to_add = _trim_draw_game_data(game_data, game_info) if include_in_replay else []
                        if include_in_replay:
                            games_data_collected.extend(data_to_add)
                        games_completed_this_iter += 1
                        game_moves_list.append(game_info)
                        if game_info.get('termination') == "CHECKMATE":
                            num_checkmates += 1
                        try:
                            _update_outcome_stats(game_info)
                        except Exception:
                            pass
                        total_advance += len(data_to_add)
                        next_to_display += 1
                    if next_to_display > prev_next:
                        draw_rate = (num_draws / games_completed_this_iter * 100.0) if games_completed_this_iter > 0 else 0.0
                        progress.update(
                            task_id_selfplay,
                            advance=total_advance,
                            games=games_completed_this_iter,
                            steps=len(games_data_collected),
                            draw_rate=draw_rate,
                            first_wins=num_first_wins,
                            second_wins=num_second_wins,
                            refresh=True,
                        )
                    if len(games_data_collected) >= min_steps:
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
                _empty_device_cache()
                gc.collect()

        else: # Sequential Execution
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
                inference_client = None
                self_play_device = device
                if inference_request_queue is not None:
                    inference_reply_queue = manager.Queue()
                    inference_client = InferenceClient(inference_request_queue, inference_reply_queue, timeout_s=30.0)
                    self_play_device = torch.device("cpu")
                game_data, game_info = run_self_play_game(
                    cfg,
                    (network if (cfg.mcts.iterations > 0 and inference_request_queue is None) else None),
                    env if cfg.env.render_mode == 'human' else None,  # Use the main env instance
                    progress=progress if (device.type == 'cpu' and show_progress) else None,
                    device=self_play_device,
                    inference_client=inference_client,
                    self_play_dtype_override=torch.float32 if (inference_request_queue is None and cfg.mcts.iterations > 0) else None,  # Main network stays float32 for training
                )
                include_in_replay = _include_in_replay_buffer(game_info)
                _update_mate_success(game_info)
                data_to_add = _trim_draw_game_data(game_data, game_info) if include_in_replay else []
                if include_in_replay:
                    games_data_collected.extend(data_to_add)
                games_completed_this_iter += 1
                game_moves_list.append(game_info)
                if game_info.get('termination') == "CHECKMATE":
                    num_checkmates += 1
                # Update progress bar with current games and steps (trimmed count only)
                draw_rate = (num_draws / games_completed_this_iter * 100.0) if games_completed_this_iter > 0 else 0.0
                progress.update(
                    task_id_selfplay,
                    advance=len(data_to_add),
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
            _empty_device_cache()
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
                def _label_sort_key(value: str):
                    lowered = value.lower()
                    is_endgame = "endgame" in lowered
                    return (1 if is_endgame else 0, lowered)
                for label in sorted(label_stats.keys(), key=_label_sort_key):
                    count = label_stats[label]['count']
                    avg_nodes_label = label_stats[label]['sum_nodes'] / max(1, count)
                    label_parts.append(f"{label}={avg_nodes_label:.0f} ({count})")
                tree_stats_by_label_info = "MCTS avg nodes by dataset (# games): " + ", ".join(label_parts)
        
        buffer_info = f", buffer={len(replay_buffer)}"
        
        draw_rate_iter = (num_draws / games_completed_this_iter * 100.0) if games_completed_this_iter > 0 else 0.0
        progress.print(
            f"Self-play: {games_completed_this_iter} games, total={total_games_simulated}, "
            f"steps={len(games_data_collected)}{buffer_info} | W Wins: {num_wins}, "
            f"B Wins: {num_losses}, 1st Wins: {num_first_wins}, 2nd Wins: {num_second_wins}, "
            f"Draws: {num_draws} ({draw_rate_iter:.1f}% Draw)"
            f"\n{draw_info}{device_info}{tree_stats_info}\n{tree_stats_by_label_info} | {format_time(self_play_duration)}"
        )
        
        # --- Training Phase ---
        # Cleanup and prep GPU before training
        _empty_device_cache()
        gc.collect()
        if len(replay_buffer) < cfg.training.batch_size:
            progress.print("Not enough data in buffer to start training. Skipping phase.")
            avg_policy_loss = float("nan")
            avg_value_loss = float("nan")
            _save_game_history(game_moves_list, game_history_dir, iteration, avg_policy_loss, avg_value_loss)
            continue
        
        # Dynamic accelerator assignment: if we have enough data and using GPU/MPS (not TPU), wait for first available accelerator
        training_device = device
        use_tpu = tpu_lock is not None
        if continual_enabled and not use_tpu and device.type in ('cuda', 'mps') and len(replay_buffer) >= cfg.training.batch_size:
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
            TextColumn("[Loss P: {task.fields[loss_p]:.4f} V: {task.fields[loss_v]:.4f} top1_ill: {task.fields[illegal_r]:.2%} prob_on_ill: {task.fields[illegal_p]:.2%}]")
        )
        progress.columns = training_columns

        # progress.print("Starting training phase...")
        # For TPU, pause self-play workers so inference queue drains, then acquire lock
        if pause_for_training is not None:
            pause_for_training.set()
            progress.print("TPU: Paused self-play workers, waiting for inference queue to drain...")
            time.sleep(1.0)  # Give workers time to see pause signal before acquire
        if tpu_lock is not None:
            progress.print("TPU: Acquiring lock (may block until inference finishes current batch)...")
            tpu_lock.acquire()
            progress.print("TPU: Lock acquired, starting training.")
        try:
            network.to(training_device).train()

            # CRITICAL: Refresh optimizer parameter references after moving network
            # When network.to(device) creates new parameter tensors, optimizer must be updated
            if use_multiprocessing_flag:
                # Save optimizer state before refreshing
                opt_state = optimizer.state_dict()
                # Extract current learning rate from optimizer state to preserve it
                current_lr = optimizer.param_groups[0]['lr']
                # Recreate optimizer with new parameter references, using preserved learning rate
                trainable_params = [p for p in network.parameters() if p.requires_grad]
                if opt_cfg.type == "Adam":
                    optimizer = optim.Adam(trainable_params, lr=current_lr, weight_decay=opt_cfg.weight_decay)
                elif opt_cfg.type == "SGD":
                    momentum = opt_cfg.momentum if opt_cfg.momentum is not None else 0.9
                    optimizer = optim.SGD(trainable_params, lr=current_lr, momentum=momentum, weight_decay=opt_cfg.weight_decay)
                # Restore optimizer state (this preserves momentum, adaptive rates, etc.)
                try:
                    optimizer.load_state_dict(opt_state)
                    # Ensure optimizer state tensors are on correct device
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(training_device)
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
            # AMP on TPU can cause NaN loss; disable for TPU (PyTorch/XLA issue #3200)
            amp_enabled = bool(
                cfg.training.get("amp", False)
                and not use_tpu
                and (training_device.type in {"cuda", "mps"})
            )
            amp_device = "xla" if use_tpu else ("cuda" if training_device.type == "cuda" else "mps")
            scaler = GradScaler(amp_device, enabled=cfg.training.get("amp", False) and training_device.type == "cuda" and not use_tpu)

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
            if use_tpu:
                bs = cfg.training.batch_size
                est_min = max(2, min(30, bs // 64))  # Rough: ~2 min for 64, scales with batch
                progress.print(
                    f"TPU: First step compiles XLA graph (batch_size={bs}). "
                    f"May take {est_min}-{est_min*2} min. Subsequent steps are fast."
                )
                if bs > 512:
                    progress.print(
                        f"TPU: Consider batch_size=256 or 512 for faster first-step compilation. "
                        f"Current {bs} may take 15-30+ min."
                    )
                if cfg.training.get("diagnose_tpu_gradients", False):
                    progress.print(
                        "TPU: diagnose_tpu_gradients enabled - will log grad norms, loss components, "
                        "and batch stats on NaN to find gradient deviation cause."
                    )
            # Use values from cfg
            for epoch in range(cfg.training.num_training_steps):
                batch = replay_buffer.sample(
                    cfg.training.batch_size,
                    action_space_size=cfg.network.action_space_size,
                    draw_sample_ratio=cfg.training.get("draw_sample_ratio"),
                )
                if batch is None: continue

                # states_np, policy_targets_np, fens_batch, value_targets_np = batch
                states_np, policy_targets_np, boards_batch, value_targets_np, source_ids_np = batch # Unpack boards
                states_tensor = torch.from_numpy(states_np).to(training_device)
                policy_targets_tensor = torch.from_numpy(policy_targets_np).to(training_device)
                value_targets_tensor = torch.from_numpy(value_targets_np).to(training_device)

                with autocast(amp_device, enabled=amp_enabled):
                    policy_logits, value_preds = network(states_tensor)

                    # Early NaN hypothesis check (first batch only, before clamp/loss)
                    if use_tpu and cfg.training.get("diagnose_tpu_gradients", False) and epoch == 0:
                        _log_early_nan_hypotheses(
                            epoch, states_np, policy_targets_np, value_targets_np,
                            policy_logits, value_preds,
                        )

                    # Clamp logits for numerical stability (TPU/XLA can produce NaN with extreme values)
                    policy_logits = torch.clamp(policy_logits, -50.0, 50.0)
                    value_preds = torch.clamp(value_preds.squeeze(-1), -10.0, 10.0)

                    # AlphaZero-style: policy targets are already legal-only, no masking needed.
                    # Note: F.kl_div has no proper autograd kernel on TPU/XLA - use CrossEntropyLoss
                    policy_loss = policy_loss_fn(policy_logits, policy_targets_tensor)
                    # Ensure value shapes match before loss calc
                    value_loss = value_loss_fn(value_preds, value_targets_tensor.squeeze(-1))

                    total_loss = policy_loss + value_loss

                # H6: Check which loss component is NaN (policy vs value) - run before any .item() on total_loss
                if use_tpu and cfg.training.get("diagnose_tpu_gradients", False) and epoch == 0:
                    pl_ok, vl_ok = None, None
                    try:
                        pl_ok = torch.isfinite(policy_loss).item()
                        print(f"[TPU diag] H6a policy_loss: finite={pl_ok}", file=sys.stderr, flush=True)
                    except Exception as e:
                        print(f"[TPU diag] H6a policy_loss: check failed (may hang on NaN): {e}", file=sys.stderr, flush=True)
                    try:
                        vl_ok = torch.isfinite(value_loss).item()
                        print(f"[TPU diag] H6b value_loss: finite={vl_ok}", file=sys.stderr, flush=True)
                    except Exception as e:
                        print(f"[TPU diag] H6b value_loss: check failed (may hang on NaN): {e}", file=sys.stderr, flush=True)
                    if pl_ok is False:
                        print("[TPU diag] -> LIKELY CAUSE: CrossEntropyLoss produces NaN on TPU (soft targets)", file=sys.stderr, flush=True)
                    if vl_ok is False:
                        print("[TPU diag] -> LIKELY CAUSE: MSELoss produces NaN on TPU", file=sys.stderr, flush=True)

                # Skip batch if loss is NaN/Inf (e.g. from bad replay data or TPU numerical instability)
                # On TPU, .item() can hang when tensor has NaN - log numpy stats FIRST (no TPU sync)
                if use_tpu and cfg.training.get("diagnose_tpu_gradients", False) and epoch == 0:
                    _log_nan_batch_diagnostics_numpy_only(epoch, policy_targets_np, value_targets_np)
                try:
                    loss_finite = torch.isfinite(total_loss).item()
                except Exception:
                    loss_finite = False  # Assume NaN if check fails (e.g. TPU sync)
                if not loss_finite:
                    if use_tpu and cfg.training.get("diagnose_tpu_gradients", False):
                        print(f"[TPU diag] NaN/Inf loss at epoch {epoch} - skipping batch", file=sys.stderr, flush=True)
                        _log_nan_batch_diagnostics(
                            progress, epoch, policy_targets_np, value_targets_np,
                            policy_logits, value_preds, policy_loss, value_loss,
                        )
                    optimizer.zero_grad()
                    # Advance progress bar so training doesn't appear stuck
                    do_refresh = (epoch + 1) % 8 == 0 or epoch == cfg.training.num_training_steps - 1
                    progress.update(
                        task_id_train, advance=1,
                        loss_p=float("nan"), loss_v=float("nan"),
                        illegal_r=0.0, illegal_p=0.0,
                        refresh=do_refresh,
                    )
                    continue

                optimizer.zero_grad()
                grad_clip = cfg.training.get("grad_clip_norm")
                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    if use_tpu and cfg.training.get("diagnose_tpu_gradients", False):
                        grad_norm = _compute_grad_norm(network)
                        log_interval = max(1, cfg.training.num_training_steps // 10)
                        if epoch % log_interval == 0 or grad_norm > 10.0 or not np.isfinite(grad_norm):
                            progress.print(
                                f"[TPU diag] epoch={epoch} grad_norm={grad_norm:.4f} "
                                f"loss_p={policy_loss.item():.4f} loss_v={value_loss.item():.4f}"
                            )
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if use_tpu and cfg.training.get("diagnose_tpu_gradients", False):
                        grad_norm = _compute_grad_norm(network)
                        log_interval = max(1, cfg.training.num_training_steps // 10)
                        if epoch % log_interval == 0 or grad_norm > 10.0 or not np.isfinite(grad_norm):
                            progress.print(
                                f"[TPU diag] epoch={epoch} grad_norm={grad_norm:.4f} "
                                f"loss_p={policy_loss.item():.4f} loss_v={value_loss.item():.4f}"
                            )
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
                    if use_tpu:
                        import torch_xla.core.xla_model as xm
                        xm.optimizer_step(optimizer, barrier=True)
                    else:
                        optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

                if source_labels and source_ids_np is not None:
                    source_ids_tensor = torch.from_numpy(source_ids_np).to(training_device)
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
                                device=training_device,
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

                # Update every 8 epochs (or always on last) to avoid Jupyter IOPub rate limit
                do_refresh = (epoch + 1) % 8 == 0 or epoch == cfg.training.num_training_steps - 1
                progress.update(
                    task_id_train, advance=1,
                    loss_p=current_avg_policy_loss, loss_v=current_avg_value_loss,
                    illegal_r=current_illegal_ratio,
                    illegal_p=avg_illegal_prob_mass,
                    refresh=do_refresh,
                )
            progress.update(task_id_train, visible=False)

            avg_policy_loss = total_policy_loss / cfg.training.num_training_steps if cfg.training.num_training_steps > 0 else 0
            avg_value_loss = total_value_loss / cfg.training.num_training_steps if cfg.training.num_training_steps > 0 else 0
            avg_illegal_ratio = total_illegal_moves_in_iteration / total_samples_in_iteration if total_samples_in_iteration > 0 else 0.0
            per_source_acc_str = _format_source_accuracy(
                source_labels, per_source_correct, per_source_total, digits=2
            )
            if not per_source_acc_str:
                per_source_acc_str = "-"
            mate_success_str = ""
            if source_labels:
                mate_acc = _format_source_accuracy(
                    source_labels, val_mate_success_per_source, val_mate_total_per_source, digits=2
                )
                mate_success_str = f" | Mate success: {mate_acc}"
            progress.print(
                f"Training finished: Avg Policy Loss: {avg_policy_loss:.4f}, "
                f"Avg Value Loss: {avg_value_loss:.4f}, "
                f"top1_illegal: {avg_illegal_ratio:.2%} | prob_on_illegal: {avg_illegal_prob_mass:.2%}"
            )
            progress.print(
                f"  Policy top-1 per dataset: {per_source_acc_str}"
                f"{mate_success_str}"
            )
            _save_game_history(game_moves_list, game_history_dir, iteration, avg_policy_loss, avg_value_loss)
        finally:
            if tpu_lock is not None:
                tpu_lock.release()
            if pause_for_training is not None:
                pause_for_training.clear()
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
        if mate_entry_labels:
            for idx, label in enumerate(mate_entry_labels):
                total = val_mate_total[idx]
                ratio = (val_mate_success[idx] / total) if total > 0 else 0.0
                history.setdefault('mate_success', {}).setdefault(label, []).append(ratio)
            history["mate_success_labels"] = list(mate_entry_labels)

        # Save checkpoint after each iteration (skip if model has NaN/Inf to avoid corrupting inference)
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        model_state = network.state_dict()
        if _state_dict_has_nan_or_inf(model_state):
            progress.print(
                "WARNING: Skipping checkpoint save - model contains NaN/Inf. "
                "Inference will continue with previous checkpoint. Fix training (e.g. lower LR, disable AMP on TPU)."
            )
        else:
            checkpoint = {
                'iteration': iteration + 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'training_mode': 'rl',
                'total_games_simulated': total_games_simulated,
                'replay_buffer_state': replay_buffer.get_state(
                    env_type=cfg.env.type,
                    use_float32=bool(cfg.training.get("replay_buffer_float32", False)),
                ),
                'history': history,
            }
            torch.save(checkpoint, checkpoint_path)
        
        checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024) if os.path.exists(checkpoint_path) else 0.0
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
                    p.join(timeout=10.0)
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=2.0)
        except Exception:
            pass
    if inference_process is not None:
        try:
            if inference_stop_event is not None:
                inference_stop_event.set()
            if inference_process.is_alive():
                inference_process.join(timeout=2.0)
        except Exception:
            pass
    progress.print("\nTraining loop finished.")


# --- Hydra Entry Point --- 
# Ensure config_path points to the directory containing train_gomoku.yaml
@hydra.main(config_path="../config", config_name="train_mcts", version_base=None)
def main(cfg: DictConfig) -> None:
    # Tee stdout/stderr to terminal and train.log; exclude progress bar and related clutter from log
    try:
        import re
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        if hydra_cfg is not None:
            output_dir = Path(hydra_cfg.runtime.output_dir)
            log_path = output_dir / "train.log"
            _log_file = open(log_path, "w", encoding="utf-8")
            _ansi_pattern = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]?")

            def _is_progress_bar_output(data: str) -> bool:
                """True if data is progress bar / Rich live display output to skip in log.
                progress.print() outputs real messages (with letters) and should be logged."""
                if not data:
                    return True
                if data.startswith("\r"):
                    return True  # Carriage return = overwrite line = progress bar
                clean = _ansi_pattern.sub("", data).strip()
                if not clean:
                    return True
                # progress.print() outputs lines with letters (words). Progress bar is only
                # bar chars (━╺╸█▓▒░●), digits, %, spaces. If it has letters, log it.
                if any(c.isalpha() for c in clean):
                    return False  # Has letters -> progress.print or regular print
                return True  # No letters -> progress bar visual only

            class _Tee:
                """Write to terminal always; write to train.log excluding progress bar output."""
                def __init__(self, stream, file):
                    self._stream = stream
                    self._file = file
                def write(self, data):
                    self._stream.write(data)
                    self._stream.flush()
                    if not _is_progress_bar_output(data):
                        clean = _ansi_pattern.sub("", data)
                        if clean:
                            self._file.write(clean)
                            self._file.flush()
                def flush(self):
                    self._stream.flush()
                    self._file.flush()
                def fileno(self):
                    return self._stream.fileno()
                def isatty(self):
                    return self._stream.isatty()

            sys.stdout = _Tee(sys.__stdout__, _log_file)
            sys.stderr = _Tee(sys.__stderr__, _log_file)
    except Exception:
        pass  # Fallback: run without tee if Hydra not available

    # Initialize factories in the main process
    initialize_factories_from_cfg(cfg)
    
    print("Configuration:\n")
    # Use OmegaConf.to_yaml for structured printing
    print(OmegaConf.to_yaml(cfg))
    run_training_loop(cfg)

if __name__ == "__main__":
    main()
