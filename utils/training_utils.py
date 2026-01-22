"""Utility functions for training that are not directly part of the training loop."""
import csv
import json
import os
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import IterableDataset


def select_fen_from_dict(fen_dict):
    """Select a FEN string from a dictionary based on weights.
    
    Supports two formats:
    1. Legacy: {"fen": weight} - returns (fen, None)
    2. New: {"fen": {"weight": w, "quality": q}} - returns (fen, quality)
    
    Args:
        fen_dict: Dictionary mapping FEN strings to weights (floats) or dicts with weight/quality
    
    Returns:
        Tuple of (selected FEN string, position quality) or (None, None) if dictionary is empty/invalid
    """
    if not fen_dict or not isinstance(fen_dict, dict):
        return None, None
    
    fens = []
    weights = []
    qualities = []
    
    for fen, value in fen_dict.items():
        fens.append(fen)
        # Handle both legacy format (float) and new format (dict with weight/quality)
        if isinstance(value, dict):
            weights.append(value.get('weight', 0.0))
            qualities.append(value.get('quality', None))
        else:
            # Legacy format: just a float weight
            weights.append(float(value))
            qualities.append(None)
    
    # Normalize weights to ensure they sum to 1
    total_weight = sum(weights)
    if total_weight == 0:
        return None, None
    
    normalized_weights = [w / total_weight for w in weights]
    
    # Use random.choices to select based on weights
    selected_idx = random.choices(range(len(fens)), weights=normalized_weights, k=1)[0]
    selected_fen = fens[selected_idx]
    selected_quality = qualities[selected_idx]
    
    return selected_fen, selected_quality


def iter_json_array(path: str | Path):
    decoder = json.JSONDecoder()
    buffer = ""
    in_array = False
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not in_array:
                    if buffer.startswith("["):
                        buffer = buffer[1:]
                        in_array = True
                    else:
                        idx = buffer.find("[")
                        if idx == -1:
                            buffer = ""
                            break
                        buffer = buffer[idx + 1 :]
                        in_array = True
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    buffer = buffer.lstrip()
                if buffer.startswith("]"):
                    return
                try:
                    obj, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                yield obj
                buffer = buffer[idx:]
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]

        buffer = buffer.strip()
        if in_array and buffer:
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            try:
                obj, _ = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                return
            yield obj


def iter_csv_rows(path: str | Path):
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def validate_json_lines(path: str | Path) -> int:
    path = Path(path)
    invalid_lines = 0
    total_lines = 0
    data_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            total_lines += 1
            stripped = line.strip()
            if not stripped or stripped == "[" or stripped == "]":
                continue
            data_lines += 1
            if stripped.endswith(","):
                stripped = stripped[:-1].rstrip()
            try:
                json.loads(stripped)
            except json.JSONDecodeError as exc:
                invalid_lines += 1
                print(f"Invalid JSON line {line_number}: {exc}")
                print(f"Invalid line content (head): {stripped[:240]}")
                raise SystemExit("Invalid JSON line detected.")
    print(f"Total lines: {total_lines} | Data lines: {data_lines} | Invalid: {invalid_lines}")
    return data_lines


def count_dataset_entries(dataset: IterableDataset) -> int:
    return sum(1 for _ in dataset)


def select_device(device_str: str | None) -> torch.device:
    if device_str and device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def freeze_value_head(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("value_head"):
            param.requires_grad = False


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    best_val_loss: float | None = None,
    patience_counter: int | None = None,
    train_losses: list[float] | None = None,
    train_accs: list[float] | None = None,
    val_losses: list[float] | None = None,
    val_accs: list[float] | None = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg_payload = OmegaConf.to_container(cfg, resolve=False)
    name_by_id = {id(param): name for name, param in model.named_parameters()}
    optimizer_param_names = []
    for group in optimizer.param_groups:
        names = [name_by_id.get(id(param)) for param in group.get("params", [])]
        optimizer_param_names.append(names)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_param_names": optimizer_param_names,
        "config": cfg_payload,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if best_val_loss is not None:
        payload["best_val_loss"] = best_val_loss
    if patience_counter is not None:
        payload["patience_counter"] = patience_counter
    if train_losses is not None:
        payload["train_losses"] = train_losses
    if train_accs is not None:
        payload["train_accs"] = train_accs
    if val_losses is not None:
        payload["val_losses"] = val_losses
    if val_accs is not None:
        payload["val_accs"] = val_accs
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer_loaded = False
    if "optimizer_state_dict" in checkpoint:
        checkpoint_opt_state = checkpoint["optimizer_state_dict"]
        optimizer_param_names = checkpoint.get("optimizer_param_names")
        skip_optimizer_load = False

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
        restored_by_name = False
        if skip_optimizer_load and optimizer_param_names:
            try:
                name_to_param = dict(model.named_parameters())
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
                    print(
                        f"Loaded optimizer state by name for {loaded_states}/{total_candidates} params."
                    )
                else:
                    skip_optimizer_load = True
            except Exception as exc:
                print(f"Warning: Name-based optimizer restore failed: {exc}")
                skip_optimizer_load = True

        if skip_optimizer_load and not restored_by_name:
            print(
                "Warning: Optimizer param_groups mismatch between checkpoint and current model. "
                "Skipping optimizer state load."
            )
        else:
            if not use_manual_state_restore:
                optimizer.load_state_dict(checkpoint_opt_state)
            optimizer_loaded = True

    if optimizer_loaded:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as exc:
            print(f"Warning: Failed to load scheduler state: {exc}")

    if not optimizer_loaded:
        print("Optimizer will start fresh.")

    return checkpoint


def select_random_fen_from_entries(entries) -> tuple[str | None, str | None]:
    if not entries:
        return None, None
    valid = []
    for entry in entries:
        if isinstance(entry, str):
            fen = entry.strip()
        elif isinstance(entry, dict):
            fen = entry.get("FEN") or entry.get("fen")
            fen = str(fen).strip() if fen else None
        else:
            continue
        if fen:
            valid.append(fen)
    if not valid:
        return None, None
    return random.choice(valid), None


def select_random_fen_from_json_list(path: str | Path) -> tuple[str | None, str | None]:
    selected = None
    count = 0
    for entry in iter_json_array(path):
        if isinstance(entry, str):
            fen = entry.strip()
        elif isinstance(entry, dict):
            fen = entry.get("FEN") or entry.get("fen")
            fen = str(fen).strip() if fen else None
        else:
            continue
        if not fen:
            continue
        count += 1
        if random.randint(1, count) == 1:
            selected = fen
    return selected, None


class RewardComputer:
    """Class to compute position-aware draw rewards based on position quality and termination type."""
    
    def __init__(self, cfg: OmegaConf, network: Optional[nn.Module] = None, device: Optional[torch.device] = None):
        """Initialize the reward computer.
        
        Args:
            cfg: Configuration object with reward settings
            network: Optional neural network for position evaluation
            device: Optional device for network inference
        """
        self.cfg = cfg
        self.network = network
        self.device = device
        self.draw_reward_table = cfg.training.get('draw_reward_table', None)
        self.default_draw_reward = cfg.training.get('draw_reward', -0.0)
        
        # Default rewards if table not available
        self.default_rewards = {
            'winning': -0.8,
            'equal': -0.1,
            'losing': 0.2
        }
    
    @staticmethod
    def is_endgame_position(fen: Optional[str]) -> bool:
        """Detect if a FEN position represents an endgame.
        
        Endgame is typically defined as having fewer pieces on the board.
        We count pieces (excluding kings) - if total < 10, it's likely an endgame.
        
        Args:
            fen: FEN string to analyze, or None
        
        Returns:
            bool: True if position appears to be an endgame
        """
        if not fen:
            return False
        
        # Standard starting position - not an endgame
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        if fen.startswith(starting_fen):
            return False
        
        # Count pieces in the board part of FEN (first part before space)
        board_part = fen.split()[0] if ' ' in fen else fen
        
        # Count non-empty squares (pieces)
        # FEN uses: / for rank separator, numbers for empty squares, letters for pieces
        piece_count = 0
        for char in board_part:
            if char.isalpha():  # Piece (K, Q, R, B, N, P or lowercase)
                piece_count += 1
        
        # Endgame typically has fewer pieces
        # Excluding the 2 kings, if we have < 10 other pieces, it's likely an endgame
        # This threshold can be adjusted, but 10 pieces (excluding kings) is a reasonable cutoff
        return piece_count < 12  # 2 kings + 10 other pieces = 12 total
    
    def evaluate_position_quality(self, state_obs, is_first_player: bool, 
                                  precomputed_value: Optional[float] = None) -> str:
        """Evaluate position quality using network or precomputed value.
        
        Args:
            state_obs: Observation vector for the position
            is_first_player: Whether position is from first player's perspective
            precomputed_value: Optional pre-computed value from MCTS
        
        Returns:
            str: Position quality ('winning', 'equal', or 'losing')
        """
        # Use precomputed value if available
        if precomputed_value is not None:
            predicted_value = precomputed_value
        elif self.network is not None:
            # Get network's value prediction
            if self.device is None:
                try:
                    device = next(self.network.parameters()).device
                except:
                    device = torch.device('cpu')
            else:
                device = self.device
            
            obs_tensor = torch.tensor(state_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                self.network.eval()
                _, value_pred = self.network(obs_tensor)
                predicted_value = value_pred.item()  # Range: -1 to +1
        else:
            # No network available - return 'equal' as default
            return 'equal'
        
        # Adjust predicted value to current player's perspective
        if not is_first_player:
            predicted_value = -predicted_value
        
        # Determine position quality category
        if predicted_value > 0.1:
            return 'winning'
        elif predicted_value > -0.1:
            return 'equal'
        else:
            return 'losing'
    
    def compute_draw_reward(self, state_obs, is_first_player: bool,
                           termination_type: Optional[str] = None,
                           precomputed_value: Optional[float] = None,
                           initial_position_quality: Optional[str] = None) -> float:
        """Compute draw reward based on position quality and termination type.
        
        Args:
            state_obs: Observation vector for the position
            is_first_player: Whether position is from first player's perspective
            termination_type: Optional termination type (e.g., "THREEFOLD_REPETITION", "STALEMATE")
            precomputed_value: Optional pre-computed value from MCTS
            initial_position_quality: Optional initial position quality (from white's perspective) - used for endgames
        
        Returns:
            float: Position-aware draw reward
        """
        # Adjust quality for current player's perspective
        # initial_position_quality is from White's perspective, so we need to flip it for Black
        if initial_position_quality is not None:
            if is_first_player:
                # White's turn: use quality as-is (already from White's perspective)
                quality = initial_position_quality
            else:
                # Black's turn: flip quality from White's perspective to Black's perspective
                if initial_position_quality == "winning":
                    quality = "losing"
                elif initial_position_quality == "losing":
                    quality = "winning"
                else:  # "equal"
                    quality = "equal"
        else:
            quality = None
        
        # Get reward from table if available
        if self.draw_reward_table and termination_type and quality:
            termination_rewards = self.draw_reward_table.get(termination_type, None)
            if termination_rewards:
                reward = termination_rewards.get(quality, None)
                if reward is not None:
                    return reward
        
        # Fallback: use default rewards
        if quality:
            return self.default_rewards.get(quality, self.default_draw_reward if self.default_draw_reward is not None else -0.1)
        else:
            return self.default_draw_reward if self.default_draw_reward is not None else -0.1
    
    def evaluate_initial_position(self, initial_obs, is_first_player: bool) -> Optional[str]:
        """Evaluate initial board position and return quality.
        
        Args:
            initial_obs: Initial observation vector
            is_first_player: Whether the first player is to move
        
        Returns:
            Optional[str]: Position quality ('winning', 'equal', 'losing') or None if network unavailable
        """
        if self.network is None:
            return None
        
        # Determine device
        device = self.device
        if device is None:
            try:
                device = next(self.network.parameters()).device
            except:
                device = torch.device('cpu')
        
        # Evaluate initial position
        obs_tensor = torch.tensor(initial_obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            self.network.eval()
            _, initial_value_pred = self.network(obs_tensor)
            initial_value = initial_value_pred.item()  # Range: -1 to +1
        
        # Determine initial position quality from first player's perspective
        if not is_first_player:
            initial_value = -initial_value
        
        # Categorize initial position quality
        if initial_value > 0.1:
            return 'winning'
        elif initial_value > -0.1:
            return 'equal'
        else:
            return 'losing'


# Backward compatibility: keep old function names
def is_endgame_position(fen: Optional[str]) -> bool:
    """Detect if a FEN position represents an endgame."""
    return RewardComputer.is_endgame_position(fen)


def compute_position_aware_draw_reward(network: nn.Module, board_state, state_obs, 
                                       is_first_player: bool, cfg: OmegaConf, device: torch.device,
                                       precomputed_value: float | None = None,
                                       termination_type: str | None = None,
                                       initial_position_quality: str | None = None) -> float:
    """Compute draw reward (backward compatibility wrapper)."""
    reward_computer = RewardComputer(cfg, network, device)
    return reward_computer.compute_draw_reward(
        state_obs, is_first_player, termination_type,
        precomputed_value, initial_position_quality
    )

