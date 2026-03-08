"""
Diagnostic test for mating positions (mate-in-1, mate-in-2, etc.):
- Load checkpoint
- Play each position until game termination (self-play with MCTS)
- Mate-in-1: tracks whether first move delivers checkmate
- Mate-in-2+: tracks whether side-to-move eventually wins
- Summary separates game-outcome stats from mate-in-one specific stats
"""

import sys
import os
import json
import gc
from pathlib import Path
import multiprocessing
import torch
import chess
from omegaconf import OmegaConf
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict
import random
import time
import numpy as np
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from MCTS.training_modules.chess import create_chess_network, create_board_from_fen
from MCTS.mcts_algorithm import MCTS
from MCTS.mcts_node import MCTSNode


def _ci95_proportion(wins: int, total: int) -> Tuple[float, float]:
    """Compute 95% confidence interval for a proportion (Wald interval). Returns (lo, hi) as 0-100 percentages."""
    if total <= 0:
        return (0.0, 0.0)
    p = wins / total
    se = (p * (1 - p) / total) ** 0.5
    z = 1.96
    lo = max(0.0, p - z * se)
    hi = min(1.0, p + z * se)
    return (100 * lo, 100 * hi)


def _format_won_with_ci(side_won_count: int, played_count: int) -> str:
    """Format 'Won: X (pct%, 95% CI [lo%, hi%])' with 95% CI."""
    if played_count <= 0:
        return "Won: 0"
    pct = 100 * side_won_count / played_count
    lo, hi = _ci95_proportion(side_won_count, played_count)
    return f"Won: {side_won_count} ({pct:.1f}%, 95% CI [{lo:.1f}%, {hi:.1f}%])"


def _load_positions(data) -> dict:
    """Normalize positions from JSON: dict (FEN -> meta) or list of {FEN: ...} objects."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {item["FEN"]: item for item in data if isinstance(item, dict) and "FEN" in item}
    raise TypeError(f"Positions must be dict or list, got {type(data)}")


def _first_mappable_action_id(board, action_ids: List[int]) -> Optional[int]:
    """Return first action_id that board.action_id_to_move can resolve (round-trip safe)."""
    for aid in action_ids:
        if board.action_id_to_move(aid) is not None:
            return aid
    return None


def find_mating_move(board) -> Optional[chess.Move]:
    """Find the mating move from the current position."""
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()
    return None


def get_top_moves_by_visits(root_node: MCTSNode, top_k: int = 10) -> list[Tuple[int, int, Optional[chess.Move]]]:
    """
    Get top moves by visit count.
    Returns: list of (action_id, visit_count, move) tuples, sorted by visit count descending.
    """
    if not root_node.children:
        return []
    
    # Get all children with their visit counts
    moves_data = []
    for action_id, child_node in root_node.children.items():
        visit_count = child_node.N
        move = root_node.board.action_id_to_move(action_id)
        if move is None:
            raise RuntimeError(
                f"get_top_moves_by_visits: action_id_to_move({action_id}) returned None, "
                f"FEN={root_node.board.fen()}, legal_actions={list(root_node.board.legal_actions)[:10]}"
            )
        moves_data.append((action_id, visit_count, move))
    
    # Sort by visit count descending
    moves_data.sort(key=lambda x: x[1], reverse=True)
    
    return moves_data[:top_k]


def _get_inference_device_str(device_str: Optional[str] = None) -> str:
    """Resolve inference device (same logic as _start_inference_server)."""
    if device_str is None or str(device_str).lower() in ("none", "null", "auto"):
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return str(device_str)


def _start_inference_server(
    cfg: OmegaConf,
    checkpoint_path: str,
    num_workers: int,
    device_str: Optional[str] = None,
):
    """Start inference server process for batched GPU inference. Returns (request_queue, reply_queues, stop_event, process)."""
    from MCTS.inference_server import inference_server_worker, inference_server_worker_tpu

    device_str = _get_inference_device_str(device_str)

    # resolve=False: Hydra interpolations (e.g. ${now:...}) require Hydra context; diagnostic runs standalone
    cfg_for_worker = OmegaConf.to_container(cfg, resolve=False) if OmegaConf.is_config(cfg) else cfg
    reply_queues = [multiprocessing.get_context("spawn").Queue() for _ in range(num_workers)]
    reply_queues_by_worker = {i: q for i, q in enumerate(reply_queues)}
    request_queue = multiprocessing.get_context("spawn").Queue(maxsize=128)
    manager = multiprocessing.Manager()
    stop_event = manager.Event()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network_state_dict = {k: v.cpu().clone() if hasattr(v, "cpu") else v for k, v in checkpoint["model_state_dict"].items()}

    use_tpu = str(device_str).lower() in ("xla", "tpu")
    batch_size = cfg.mcts.batch_size
    training_batch_size = int(cfg.training.get("batch_size", 64))
    nw = num_workers if num_workers else 1
    gpw = max(1, int(cfg.training.get("games_per_worker", 1)))
    max_stacked = min(training_batch_size, nw * gpw)
    min_stacked = cfg.training.get("inference_server_min_stacked_requests")
    if min_stacked in (None, "", "null"):
        min_stacked = max(1, max_stacked // 2)
    min_stacked = int(min_stacked)
    max_wait_ms = cfg.training.get("inference_server_max_wait_ms", 2)
    max_wait_ms = int(max_wait_ms) if max_wait_ms is not None and max_wait_ms != "" else 2

    if use_tpu:
        import threading
        tpu_lock = threading.Lock()
        proc = threading.Thread(
            target=inference_server_worker_tpu,
            args=(
                checkpoint_path,
                cfg_for_worker,
                device_str,
                request_queue,
                stop_event,
                max_stacked,
                min_stacked,
                max_wait_ms,
                reply_queues_by_worker,
                network_state_dict,
                tpu_lock,
            ),
            daemon=True,
        )
        proc.start()
    else:
        proc = multiprocessing.get_context("spawn").Process(
            target=inference_server_worker,
            args=(
                checkpoint_path,
                cfg_for_worker,
                device_str,
                request_queue,
                stop_event,
                max_stacked,
                min_stacked,
                max_wait_ms,
                reply_queues_by_worker,
                network_state_dict,
            ),
        )
        proc.daemon = True
        proc.start()

    return request_queue, reply_queues, stop_event, proc, device_str


def _diagnostic_worker(
    worker_id: int,
    positions_queue,
    results_queue,
    checkpoint_path: str,
    cfg_dict: dict,
    mcts_iterations: int,
    max_game_moves: int,
    inference_request_queue,
    inference_reply_queue,
):
    """Worker that processes positions from queue and puts results. Uses inference client when queues provided."""
    cfg = OmegaConf.create(cfg_dict)
    from MCTS.inference_server import InferenceClient

    inference_client = None
    network = None
    device = torch.device("cpu")

    if inference_request_queue is not None and inference_reply_queue is not None:
        inference_client = InferenceClient(
            inference_request_queue, inference_reply_queue, timeout_s=30.0, worker_id=worker_id
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        network = create_chess_network(cfg, device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.eval()

    positions_processed = 0
    while True:
        try:
            item = positions_queue.get(timeout=1.0)
            if item is None:
                break
        except Exception:
            continue
        index, fen_position, meta, collect_hist = (item[0], item[1], item[2] if len(item) > 2 else {}, item[3] if len(item) > 3 else False)
        result = test_single_position(
            cfg=cfg,
            fen_position=fen_position,
            mcts_iterations=mcts_iterations,
            device=device,
            network=network,
            inference_client=inference_client,
            max_game_moves=max_game_moves,
            collect_game_history=collect_hist,
            position_metadata=meta,
        )
        results_queue.put((index, result))
        positions_processed += 1
        if positions_processed % 25 == 0:
            gc.collect()


def _get_best_action_by_visits(root_node: MCTSNode):
    """Return the action_id with highest visit count, or None if no children."""
    top = get_top_moves_by_visits(root_node, top_k=1)
    return top[0][0] if top else None


def _extract_solution_moves(entry: dict | None) -> list[str] | None:
    """Extract solution moves from a dataset entry. Returns list of move strings (UCI or SAN), or None."""
    if not entry or not isinstance(entry, dict):
        return None
    moves_field = entry.get("Moves") or entry.get("moves")
    if not moves_field:
        return None
    if isinstance(moves_field, str):
        return [m.strip() for m in moves_field.split() if m.strip()]
    if isinstance(moves_field, list):
        return [str(m).strip() for m in moves_field if str(m).strip()]
    return None


def test_single_position(
    cfg: OmegaConf,
    fen_position: str,
    mcts_iterations: int,
    device: torch.device,
    network: Optional[torch.nn.Module] = None,
    inference_client: Optional[Any] = None,
    verbose: bool = False,
    max_game_moves: int = 200,
    collect_game_history: bool = False,
    position_metadata: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Play from the given position until game termination (self-play with MCTS)."""
    
    result = {
        'fen': fen_position,
        'success': False,
        'has_mate_in_one': False,
        'first_move_was_mating': False,
        'termination': None,
        'winner': None,
        'move_count': 0,
        'side_to_move_won': False,
        'mating_move': None,
        'mating_action_id': None,
        'error': None
    }
    
    try:
        board = create_board_from_fen(fen_position)
        side_to_move_init = board.turn
        
        mating_move = find_mating_move(board)
        mating_action_id = None
        if mating_move is not None:
            result['has_mate_in_one'] = True
            result['mating_move'] = mating_move.uci()
            mating_action_id = board.move_to_action_id(mating_move)
            result['mating_action_id'] = mating_action_id
        
        draw_reward = cfg.training.get('draw_reward', 0.0) if hasattr(cfg, 'training') else 0.0
        if draw_reward is not None and OmegaConf.is_config(draw_reward):
            draw_reward = float(OmegaConf.to_container(draw_reward, resolve=True))
        mcts_player = MCTS(
            network=network,
            device=device,
            env=None,
            C_puct=cfg.mcts.c_puct,
            dirichlet_alpha=cfg.mcts.dirichlet_alpha,
            dirichlet_epsilon=0.0,  # Zero exploration for evaluation: deterministic best-move selection
            action_space_size=cfg.network.action_space_size,
            history_steps=cfg.env.history_steps,
            draw_reward=draw_reward,
            pre_init_draws=getattr(cfg.mcts, 'pre_init_draws', False),
            inference_client=inference_client,
        )
        
        move_count = 0
        first_move_was_mating = False
        moves_san_list: List[str] = []
        moves_uci_list: List[str] = []
        policy_details_list: List[Dict[str, Any]] = []
        init_fen = fen_position
        meta = position_metadata or {}
        solution_moves = _extract_solution_moves(meta.get("full_entry"))
        dirichlet_alpha = getattr(cfg.mcts, "dirichlet_alpha", 0.3)
        dirichlet_epsilon = 0.0  # Diagnostic uses deterministic play

        while not board.is_game_over(claim_draw=True) and move_count < max_game_moves:
            root_node = MCTSNode(board.copy(stack=False))
            mcts_player.search(root_node, iterations=mcts_iterations, batch_size=cfg.mcts.batch_size, progress=None)

            # AlphaZero-style: filter to legal, renormalize policy, then select (matches train.py)
            # Use root_node.board for legal_actions: same position as board, ensures consistency with MCTS tree
            mcts_policy = mcts_player.get_policy_distribution(root_node, temperature=0.0)
            legal_actions = list(root_node.board.legal_actions or [])
            legal_children = (
                [a for a in legal_actions if a in root_node.children]
                if root_node.children else []
            )
            candidates = legal_children if legal_children else legal_actions
            selection_probs = {}
            if candidates:
                legal_indices = [a - 1 for a in candidates if 0 <= (a - 1) < len(mcts_policy)]
                if legal_indices:
                    legal_probs = np.array([mcts_policy[i] for i in legal_indices], dtype=np.float64)
                    if legal_probs.sum() > 0:
                        legal_probs = legal_probs / legal_probs.sum()
                        best_idx = int(np.argmax(legal_probs))
                        action_id = legal_indices[best_idx] + 1
                        selection_probs = {
                            legal_indices[i] + 1: float(legal_probs[i])
                            for i in range(len(legal_indices))
                        }
                    else:
                        action_id = _first_mappable_action_id(board, candidates)
                        selection_probs = {a: 1.0 / len(candidates) for a in candidates}
                else:
                    action_id = _first_mappable_action_id(board, candidates)
                    selection_probs = {a: 1.0 / len(candidates) for a in candidates}
            else:
                action_id = None

            if collect_game_history:
                raw_mcts_policy = mcts_player.get_policy_distribution(root_node, temperature=1.0)
                converted_policy = mcts_policy
                # Ground truth: solution move at this index (if we're following solution)
                ground_truth_action = None
                ground_truth_move_uci = None
                if solution_moves and move_count < len(solution_moves):
                    sol_str = solution_moves[move_count]
                    ground_truth_move_uci = sol_str
                    if sol_str:
                        try:
                            mv = board.parse_san(sol_str)
                            if mv and mv in board.legal_moves:
                                ground_truth_action = board.move_to_action_id(mv)
                        except Exception:
                            try:
                                mv = chess.Move.from_uci(sol_str)
                                if mv in board.legal_moves:
                                    ground_truth_action = board.move_to_action_id(mv)
                            except Exception:
                                pass
                sorted_indices = np.argsort(converted_policy)[::-1]
                top_actions = [
                    int(idx + 1) for idx in sorted_indices
                    if converted_policy[idx] > 1e-9
                ][:3]
                legal_actions_list = list(root_node.board.legal_actions or [])
                if action_id is not None:
                    sig_actions = {int(action_id), ground_truth_action or -1}
                    sig_actions.update(top_actions[:3])
                    # Pad to at least 3 when 3+ legal moves exist
                    if len(sig_actions) < 3 and len(legal_actions_list) >= 3:
                        for aid in legal_actions_list:
                            if aid not in sig_actions:
                                sig_actions.add(aid)
                                if len(sig_actions) >= 3:
                                    break
                    sig_actions.discard(-1)
                    raw_probs = {aid: float(raw_mcts_policy[aid - 1]) for aid in sig_actions if 1 <= aid <= len(raw_mcts_policy)}
                    conv_probs = {aid: float(converted_policy[aid - 1]) for aid in sig_actions if 1 <= aid <= len(converted_policy)}
                    sel_probs = {aid: selection_probs.get(aid, 0.0) for aid in sig_actions}
                    policy_details_list.append({
                        "temperature": 0.0,
                        "move_temp": 0.0,
                        "dirichlet_alpha": dirichlet_alpha,
                        "dirichlet_epsilon": dirichlet_epsilon,
                        "selected_action": int(action_id),
                        "top1_action": top_actions[0] if len(top_actions) >= 1 else None,
                        "top2_action": top_actions[1] if len(top_actions) >= 2 else None,
                        "top3_action": top_actions[2] if len(top_actions) >= 3 else None,
                        "ground_truth_action": ground_truth_action,
                        "ground_truth_move_uci": ground_truth_move_uci,
                        "raw_probs": raw_probs,
                        "converted_probs": conv_probs,
                        "selection_probs": sel_probs,
                    })

            # Resolve move using root_node.board (same position as board; ensures round-trip consistency)
            # before freeing root_node
            legal_list = list(root_node.board.legal_actions or [])
            if action_id is None:
                if not legal_list:
                    del root_node
                    break
                result["error"] = f"action_id was None but legal moves exist (move {move_count + 1})"
                del root_node
                break

            if action_id not in legal_list:
                result["error"] = (
                    f"action_id {action_id} not in legal_actions (move {move_count + 1}); "
                    f"legal_actions={legal_list[:10]}{'...' if len(legal_list) > 10 else ''}"
                )
                del root_node
                break

            move = root_node.board.action_id_to_move(action_id)
            if move is None:
                result["error"] = (
                    f"action_id_to_move({action_id}) returned None (move {move_count + 1}); "
                    f"FEN={root_node.board.fen()}; legal_actions={legal_list[:10]}{'...' if len(legal_list) > 10 else ''}"
                )
                del root_node
                break
            del root_node  # Free MCTS tree immediately (can be large)

            if move_count == 0 and mating_action_id is not None and action_id == mating_action_id:
                first_move_was_mating = True

            if collect_game_history:
                moves_san_list.append(board.san(move))
                moves_uci_list.append(move.uci())
            board.push(move)
            move_count += 1
            # Periodic GC to free MCTS tree (circular refs delay collection)
            if move_count % 10 == 0:
                gc.collect()

        result['move_count'] = move_count
        result['first_move_was_mating'] = first_move_was_mating
        
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            result['winner'] = 'white' if outcome.winner == chess.WHITE else 'black' if outcome.winner == chess.BLACK else None
            result['termination'] = getattr(outcome.termination, 'name', str(outcome.termination)).lower() if outcome.termination else "draw"
            result['side_to_move_won'] = (outcome.winner == side_to_move_init) if outcome.winner is not None else False
        else:
            result['termination'] = "max_moves"
            result['side_to_move_won'] = False
        
        result['success'] = first_move_was_mating if result['has_mate_in_one'] else result['side_to_move_won']

        if collect_game_history:
            winner = result.get('winner')
            result['game_info'] = {
                'winner': chess.WHITE if winner == 'white' else (chess.BLACK if winner == 'black' else None),
                'termination': result.get('termination', 'unknown'),
                'move_count': result['move_count'],
                'moves_san': ' '.join(moves_san_list),
                'moves_uci': ' '.join(moves_uci_list),
                'initial_fen': init_fen,
                'initial_position_quality': meta.get('quality'),
                'initial_dataset_source': meta.get('source'),
                'initial_dataset_label': meta.get('label'),
                'position_themes': meta.get('themes'),
                'actual_reward': 1.0 if result.get('side_to_move_won') else (0.0 if result.get('termination') == 'max_moves' else 0.5),
                'policy_details': policy_details_list,
            }
        
    except Exception as e:
        result['error'] = str(e)
    finally:
        gc.collect()  # Free MCTS tree and player from this position
    return result


def run_batch_diagnostic_test(
    checkpoint_path: str,
    mate_positions: Dict,
    mcts_iterations: int = 1000,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    max_positions: Optional[int] = None,
    max_game_moves: int = 200,
    use_inference_server: bool = False,
    num_workers: int = 4,
    inference_device: Optional[str] = None,
    positions_file: Optional[str] = None,
    game_history_dir: Optional[str] = None,
):
    """Run diagnostic test on all positions in batch."""
    
    # Load config
    cfg = OmegaConf.load(os.path.join(project_root, "config/train_mcts.yaml"))
    
    # Set device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    # Use file-based sampling (same as train.py): read random lines via offset index, no full JSON parse
    from utils.training_utils import select_random_fen_from_file, build_line_offset_index

    positions_label = "inmem"  # default when using in-memory positions
    if positions_file and os.path.exists(positions_file):
        path = Path(positions_file)
        resolved = str(path.resolve())
        is_json_array = resolved.lower().endswith(".json") and not resolved.lower().endswith(".jsonl")
        offset_cache = {}
        offsets = build_line_offset_index(path, is_json_array=is_json_array)
        total_in_file = len(offsets)
        n_sample = min(max_positions or total_in_file, total_in_file)
        if n_sample <= 0:
            raise ValueError(f"No positions in file: {positions_file}")
        seen = set()
        mate_positions_meta = {}  # {fen: {themes, quality, full_entry, source, label}}
        resolved_path = str(path.resolve())
        dataset_label = os.path.splitext(os.path.basename(resolved_path))[0] or "data"
        positions_label = dataset_label
        max_attempts = n_sample * 4
        attempts = 0
        while len(seen) < n_sample and attempts < max_attempts:
            fen, quality, themes, full_entry = select_random_fen_from_file(positions_file, offset_cache=offset_cache)
            if fen:
                seen.add(fen)
                mate_positions_meta[fen] = {
                    "Themes": themes or [],
                    "themes": themes or [],
                    "quality": quality,
                    "full_entry": full_entry,
                    "source": resolved_path,
                    "label": dataset_label,
                }
            attempts += 1
        positions_to_test = list(seen)[:n_sample]
        random.shuffle(positions_to_test)
        mate_positions = mate_positions_meta
    else:
        positions_to_test = list(mate_positions.keys())
        if max_positions:
            positions_to_test = random.sample(
                positions_to_test,
                min(max_positions, len(positions_to_test))
            )
        else:
            random.shuffle(positions_to_test)
        total_in_file = len(mate_positions)
    total_positions = len(positions_to_test)

    collect_game_history = bool(game_history_dir)
    if collect_game_history:
        # Create date/time folder structure like mcts_train: base/YYYY-MM-DD/HH-MM-SS/game_history
        from datetime import datetime
        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        time_folder = now.strftime("%H-%M-%S")
        game_history_dir = os.path.join(game_history_dir, date_folder, time_folder, "game_history")
        os.makedirs(game_history_dir, exist_ok=True)
        print(f"Game history will be saved to: {os.path.abspath(game_history_dir)}")

    print("=" * 80)
    print("BATCH DIAGNOSTIC TEST: Mating Positions (Play to Termination)")
    print("=" * 80)
    if use_inference_server:
        inf_dev = _get_inference_device_str(inference_device)
        print(f"Device: inference server on {inf_dev} (workers on CPU)")
    else:
        print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Positions file: {positions_file or '(in-memory)'}")
    print(f"Max game moves: {max_game_moves}")
    print(f"MCTS iterations per position: {mcts_iterations}")
    print(f"Total positions in file: {total_in_file}")
    if max_positions:
        print(f"Testing {total_positions} random positions (limited)")
    if use_inference_server:
        print(f"Inference server: enabled on {_get_inference_device_str(inference_device)} ({num_workers} workers)")
    print("=" * 80)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint to get train iteration for file naming
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_iteration = int(checkpoint.get("iteration", 0))
    del checkpoint
    gc.collect()

    results = []
    start_time = time.time()
    # resolve=False: Hydra interpolations (e.g. ${now:...}) require Hydra context; diagnostic runs standalone
    cfg_for_worker = OmegaConf.to_container(cfg, resolve=False) if OmegaConf.is_config(cfg) else cfg

    if use_inference_server:
        # Multiprocessing with inference server
        num_workers = max(1, num_workers)
        request_queue, reply_queues, stop_event, inference_proc, inf_device_str = _start_inference_server(
            cfg, checkpoint_path, num_workers, inference_device
        )
        print(f"\nInference server started on {inf_device_str} ({num_workers} workers)")
        time.sleep(0.5)  # Let server initialize

        positions_queue = multiprocessing.get_context("spawn").Queue()
        results_queue = multiprocessing.get_context("spawn").Queue()
        for i, fen in enumerate(positions_to_test):
            meta = mate_positions.get(fen, {}) if isinstance(mate_positions.get(fen), dict) else {}
            positions_queue.put((i, fen, meta, collect_game_history))
        for _ in range(num_workers):
            positions_queue.put(None)

        workers = []
        for w in range(num_workers):
            p = multiprocessing.get_context("spawn").Process(
                target=_diagnostic_worker,
                args=(
                    w,
                    positions_queue,
                    results_queue,
                    checkpoint_path,
                    cfg_for_worker,
                    mcts_iterations,
                    max_game_moves,
                    request_queue,
                    reply_queues[w],
                ),
            )
            p.daemon = True
            p.start()
            workers.append(p)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=None,
            transient=False,
        ) as progress:
            task = progress.add_task("[cyan]Testing positions...", total=total_positions)
            side_won_count = 0
            played_count = 0
            results_ordered = [None] * total_positions
            next_to_display = 0
            for _ in range(total_positions):
                index, result = results_queue.get()
                results_ordered[index] = result
                while next_to_display < total_positions and results_ordered[next_to_display] is not None:
                    r = results_ordered[next_to_display]
                    results.append(r)
                    if r["error"] is None:
                        played_count += 1
                        if r.get("side_to_move_won", False):
                            side_won_count += 1
                    parts = [f"Played: {played_count}"]
                    if played_count > 0:
                        parts.append(_format_won_with_ci(side_won_count, played_count))
                    progress.update(task, advance=1, description="[cyan]" + " | ".join(parts))
                    next_to_display += 1

        for p in workers:
            p.join(timeout=5.0)
        stop_event.set()
    else:
        # Sequential with local network
        print("\nLoading model...")
        network = create_chess_network(cfg, device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.eval()
        print(f"Model loaded! Iteration: {checkpoint.get('iteration', 'unknown')}")
        print()

        side_won_count = 0
        played_count = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=None,
            transient=False,
        ) as progress:
            task = progress.add_task("[cyan]Testing positions...", total=total_positions)
            for idx, fen_position in enumerate(positions_to_test, 1):
                if verbose:
                    print(f"\n[{idx}/{total_positions}] Testing position...")
                    print(f"FEN: {fen_position}")
                meta = mate_positions.get(fen_position, {}) if isinstance(mate_positions.get(fen_position), dict) else {}
                result = test_single_position(
                    cfg=cfg,
                    fen_position=fen_position,
                    mcts_iterations=mcts_iterations,
                    device=device,
                    network=network,
                    verbose=verbose,
                    max_game_moves=max_game_moves,
                    collect_game_history=collect_game_history,
                    position_metadata=meta,
                )
                results.append(result)
                if result["error"] is None:
                    played_count += 1
                    if result.get("side_to_move_won", False):
                        side_won_count += 1
                parts = [f"Played: {played_count}"]
                if played_count > 0:
                    parts.append(_format_won_with_ci(side_won_count, played_count))
                progress.update(task, advance=1, description="[cyan]" + " | ".join(parts))
                if idx % 25 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass

    elapsed_time = time.time() - start_time
    
    # Calculate statistics
    no_error = [r for r in results if r['error'] is None]
    errors = [r for r in results if r['error'] is not None]
    side_won = [r for r in no_error if r.get('side_to_move_won', False)]
    
    # Termination breakdown (by type and winner)
    termination_counts = defaultdict(int)
    termination_by_outcome = defaultdict(lambda: {"won": 0, "lost": 0, "draw": 0})
    for r in no_error:
        t = r.get('termination') or 'unknown'
        termination_counts[t] += 1
        won = r.get('side_to_move_won', False)
        if t == 'checkmate':
            if won:
                termination_by_outcome[t]["won"] += 1
            else:
                termination_by_outcome[t]["lost"] += 1
        else:
            termination_by_outcome[t]["draw"] += 1

    # Per-piece-type accuracy (KQ, KR, KBB, KBN)
    PIECE_TYPES = ("K_vs_KQ", "K_vs_KR", "K_vs_KBB", "K_vs_KBN")

    def _get_piece_type(fen: str) -> Optional[str]:
        """Extract piece type from mate_positions Themes (e.g. K_vs_KBB)."""
        entry = mate_positions.get(fen) if isinstance(mate_positions.get(fen), dict) else None
        if not entry:
            return None
        themes = entry.get("Themes") or entry.get("themes") or []
        for pt in PIECE_TYPES:
            if pt in themes:
                return pt
        return None

    by_piece = defaultdict(lambda: {"won": 0, "total": 0})
    for r in no_error:
        pt = _get_piece_type(r.get("fen", ""))
        if pt:
            by_piece[pt]["total"] += 1
            if r.get("side_to_move_won", False):
                by_piece[pt]["won"] += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (Play to Termination)")
    print("=" * 80)
    print(f"Total positions tested: {total_positions}")
    print(f"Games played to termination: {len(no_error)}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/total_positions:.2f} sec/position)")
    if no_error:
        avg_moves = sum(r['move_count'] for r in no_error) / len(no_error)
        print(f"Average moves per game: {avg_moves:.1f}")
    print()
    
    # Game-outcome stats (all played games)
    print("--- Game outcomes (all positions) ---")
    if no_error:
        n_won = len(side_won)
        n_played = len(no_error)
        pct = 100 * n_won / n_played
        lo, hi = _ci95_proportion(n_won, n_played)
        print(f"✓ Side-to-move won: {n_won} / {n_played} ({pct:.1f}%, 95% CI [{lo:.1f}%, {hi:.1f}%])")
    else:
        print("✓ Side-to-move won: N/A")
    print(f"\nTermination breakdown:")
    for term, count in sorted(termination_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_positions
        ob = termination_by_outcome.get(term, {})
        if term == "checkmate" and (ob.get("won", 0) or ob.get("lost", 0)):
            won_n = ob.get("won", 0)
            lost_n = ob.get("lost", 0)
            print(f"   {term}: {count} ({pct:.1f}%)")
            print(f"      won (delivered mate): {won_n} ({100*won_n/total_positions:.1f}%)")
            print(f"      lost (got mated): {lost_n} ({100*lost_n/total_positions:.1f}%)")
        elif ob.get("draw", 0):
            print(f"   {term}: {count} ({pct:.1f}%) [draw/timeout]")
        else:
            print(f"   {term}: {count} ({pct:.1f}%)")

    # Per-piece-type accuracy
    if by_piece:
        print(f"\n--- Accuracy by piece type (K+pieces vs K) ---")
        for pt in PIECE_TYPES:
            if pt in by_piece:
                d = by_piece[pt]
                pct = 100 * d["won"] / d["total"] if d["total"] > 0 else 0
                print(f"   {pt}: {d['won']}/{d['total']} ({pct:.1f}%)")
    
    print(f"\n✗ Errors: {len(errors)} ({100*len(errors)/total_positions:.1f}%)")
    if errors:
        print("Errors encountered:")
        error_types = defaultdict(int)
        for r in errors:
            error_types[r['error']] += 1
        for error_msg, count in error_types.items():
            print(f"   {error_msg}: {count}")
    
    print("=" * 80)

    if game_history_dir and no_error:
        game_moves_list = [r["game_info"] for r in no_error if r.get("game_info")]
        if game_moves_list:
            mcts_dir = os.path.join(project_root, "MCTS")
            if mcts_dir not in sys.path:
                sys.path.insert(0, mcts_dir)
            from MCTS.train import _save_game_history
            n_pos = max_positions if max_positions is not None else total_positions
            _save_game_history(
                game_moves_list,
                game_history_dir,
                iteration=0,
                avg_policy_loss=0.0,
                avg_value_loss=0.0,
                train_iteration=train_iteration,
                positions_label=positions_label,
                mcts_iterations=mcts_iterations,
                max_positions=n_pos,
            )
            print(f"\nGame history saved to {game_history_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    # Load config early for inference-server defaults (matches train_mcts.yaml)
    cfg = OmegaConf.load(os.path.join(project_root, "config/train_mcts.yaml"))
    default_use_inference = cfg.training.get("use_inference_server", True)
    default_inference_device = cfg.training.get("inference_server_device")
    default_workers = cfg.training.get("self_play_workers")
    if default_workers is None or default_workers <= 0:
        try:
            from utils.profile_model import get_optimal_worker_count
            total_cores = os.cpu_count() or 4
            default_workers = get_optimal_worker_count(
                total_cores=total_cores,
                num_workers_config=None,
                use_multiprocessing=cfg.training.get("use_multiprocessing", True),
            )
        except Exception:
            default_workers = 4
    default_workers = max(1, int(default_workers)-1)

    parser = argparse.ArgumentParser(description="Diagnostic test for mate-in-one positions")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (default: checkpoints/model.pth)")
    parser.add_argument("--iterations", type=int, default=512, help="MCTS iterations per position (default: 512)")
    parser.add_argument("--max-positions", type=int, default=500, help="Maximum number of positions to test (default: 500)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output for each position")
    parser.add_argument("--positions-file", type=str, default=None, help="Path to positions JSON file (default: data/mate_in_one_positions.json)")
    parser.add_argument("--max-game-moves", type=int, default=7, help="Maximum moves per game before truncation")
    parser.add_argument("--use-inference-server", action="store_true", help="Use inference server (default: from train_mcts.yaml)")
    parser.add_argument("--no-inference-server", action="store_true", help="Disable inference server")
    parser.add_argument("--workers", type=int, default=None, help=f"Parallel workers when using inference server (default: {default_workers} from config)")
    parser.add_argument("--inference-device", type=str, default=None, help="Device for inference server (default: from train_mcts.yaml)")
    parser.add_argument("--game-history-dir", type=str, default="./outputs/mcts_test/", help="Directory to save game history files (same format as training)")
    
    args = parser.parse_args()
    use_inference_server = not args.no_inference_server and (args.use_inference_server or default_use_inference)
    inference_device = args.inference_device if args.inference_device is not None else default_inference_device
    num_workers = args.workers if args.workers is not None else default_workers
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_dir = os.path.join(project_root, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load mate-in-one positions
    if args.positions_file:
        mate_positions_file = args.positions_file
    else:
        mate_positions_file = os.path.join(project_root, "data/mate_in_one_positions.json")
    
    if not os.path.exists(mate_positions_file):
        print(f"ERROR: Positions file not found: {mate_positions_file}")
        sys.exit(1)

    # Use file-based sampling (read random lines via offset index, no full JSON parse).
    # Same method as train.py - avoids loading 230MB+ files into memory.
    mate_positions = {}  # Built from file in run_batch_diagnostic_test

    # Run batch test
    results = run_batch_diagnostic_test(
        checkpoint_path=checkpoint_path,
        mate_positions=mate_positions,
        mcts_iterations=args.iterations,
        verbose=args.verbose,
        max_positions=args.max_positions,
        max_game_moves=args.max_game_moves,
        use_inference_server=use_inference_server,
        num_workers=num_workers,
        inference_device=inference_device,
        positions_file=mate_positions_file,
        game_history_dir=args.game_history_dir,
    )

