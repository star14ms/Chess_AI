import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque
import os
import time
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing # Keep for potential parallel execution
import torch.nn.functional as F
from torch import nn

# Import environment-specific components
from chess_training import (
    create_chess_network,
    create_chess_env,
    get_chess_game_result,
    is_white_turn,
    get_chess_legal_actions,
)

# Assuming files are in the MCTS directory relative to the project root
from mcts_node import MCTSNode
from mcts_algorithm import MCTS
from utils.profile import get_optimal_worker_count, profile_model, format_time

# Environment factory
def create_environment(cfg, render=False):
    """Create the appropriate environment based on config."""
    if cfg.env.type == "chess":
        return create_chess_env(cfg, render=render)
    elif cfg.env.type == "gomoku":
        # TODO: Implement gomoku environment creation
        raise NotImplementedError("Gomoku environment not yet implemented")
    else:
        raise ValueError(f"Unsupported environment type: {cfg.env.type}")

# Network factory
def create_network(cfg, device):
    """Create the appropriate network based on config."""
    if cfg.env.type == "chess":
        return create_chess_network(cfg, device)
    elif cfg.env.type == "gomoku":
        # TODO: Implement gomoku network creation
        raise NotImplementedError("Gomoku network not yet implemented")
    else:
        raise ValueError(f"Unsupported environment type: {cfg.env.type}")

# Game result factory
def get_game_result(env_type, board):
    """Get the game result based on environment type."""
    if env_type == "chess":
        return get_chess_game_result(board)
    elif env_type == "gomoku":
        # TODO: Implement gomoku game result
        raise NotImplementedError("Gomoku game result not yet implemented")
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")

# Turn check factory
def is_first_player_turn(env_type, board):
    """Check if it's the first player's turn based on environment type."""
    if env_type == "chess":
        return is_white_turn(board)
    elif env_type == "gomoku":
        # TODO: Implement gomoku turn check
        raise NotImplementedError("Gomoku turn check not yet implemented")
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")

# Legal moves factory
def get_legal_actions(env_type, board):
    """Get legal moves based on environment type."""
    if env_type == "chess":
        return get_chess_legal_actions(board)
    elif env_type == "gomoku":
        # TODO: Implement gomoku legal moves
        raise NotImplementedError("Gomoku legal moves not yet implemented")
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")

# Helper function for parallel execution
# NOTE: Passing large network objects might be slow or problematic.
# Consider loading the network state_dict in the worker if issues arise.
def self_play_worker(game_id, network_state_dict, cfg: DictConfig, device_str: str):
    """Worker function to run a single self-play game."""
    # Determine device for this worker
    device = torch.device(device_str)

    # Re-initialize Network in the worker process
    if cfg.mcts.iterations > 0:
        network = create_network(cfg, device)
        network.load_state_dict(network_state_dict)
        network.eval() # Ensure it's in eval mode
    else:
        network = None

    game_data = run_self_play_game(
        cfg,
        network,
        env=None,
        progress=None, # Pass None for progress within worker
        device=device
    )
    return game_data

# Wrapper for imap_unordered
def worker_wrapper(args):
    """Unpacks arguments for self_play_worker when using imap."""
    game_id, network_state_dict, cfg, device_str = args
    # Call the original worker function with unpacked args
    return self_play_worker(game_id, network_state_dict, cfg, device_str)

# --- Replay Buffer (Keep as is) ---
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def add_game(self, game_data):
        for experience in game_data:
            self.add(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indices]

        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        policy_targets = np.array([exp[1] for exp in batch], dtype=np.float32)
        # Ensure FENs are handled as a list of strings, not converted to numpy array directly if they vary in length etc.
        # fens = [exp[2] for exp in batch] 
        boards = [exp[2] for exp in batch] # Now these are board objects
        value_targets = np.array([exp[3] for exp in batch], dtype=np.float32).reshape(-1, 1)

        return states, policy_targets, boards, value_targets # Return boards

    def __len__(self):
        return len(self.buffer)


# --- Self-Play Function (Update args to use config subsections) ---
def run_self_play_game(cfg: OmegaConf, network: nn.Module | None, env=None,
                       progress: Progress | None = None, device: torch.device | None = None):
    """Plays one game of self-play using MCTS and returns the game data."""
    if env is None:
        env = create_environment(cfg, render=device.type == 'cpu' and not cfg.training.get('use_multiprocessing', False))
    options = {
        'fen': cfg.training.initial_board_fen
    } if cfg.training.initial_board_fen else None
    obs, _ = env.reset(options=options)
    if network is not None:
        network.eval()

    game_history = []
    move_count = 0
    terminated = False
    truncated = False

    mcts_iterations = cfg.mcts.iterations
    c_puct = cfg.mcts.c_puct
    temp_start = cfg.mcts.temperature_start
    temp_end = cfg.mcts.temperature_end
    temp_decay_moves = cfg.mcts.temperature_decay_moves
    max_moves = cfg.training.max_game_moves

    task_id_game = None
    if progress is not None:
        task_id_game = progress.add_task(f"Max Moves (0/{max_moves})", total=max_moves, transient=True)
        progress.start_task(task_id_game)

    while not terminated and not truncated and move_count < max_moves:
        temperature = temp_start * ((temp_end / temp_start) ** min(1.0, move_count / temp_decay_moves))
        
        if cfg.mcts.iterations > 0:
            root_node = MCTSNode(env.board.copy())
            mcts_env = env if not cfg.training.get('use_multiprocessing', False) else None
            mcts_player = MCTS(
                network,
                device=device,
                env=mcts_env,
                C_puct=c_puct,
                action_space_mode=cfg.network.action_space_mode
            )
            mcts_player.search(root_node, mcts_iterations, progress=progress)
            mcts_policy = mcts_player.get_policy_distribution(root_node, temperature=temperature)
            action_to_take = mcts_player.get_best_move(root_node, temperature=temperature)
        else:
            mcts_policy = np.zeros(cfg.network.action_space_size)
            legal_actions = get_legal_actions(cfg.env.type, env.board)
            for action_id in legal_actions:
                mcts_policy[action_id - 1] = 1
            action_to_take = np.random.choice(legal_actions)

        current_obs = obs
        board_copy_at_state = env.board.copy()
        game_history.append((current_obs, mcts_policy, board_copy_at_state))
        
        obs, reward, terminated, truncated, info = env.step(action_to_take)

        if progress is not None:
            progress.update(task_id_game, description=f"Max Moves ({move_count+1}/{max_moves})", advance=1)

        move_count += 1

    final_value = get_game_result(cfg.env.type, env.board)

    full_game_data = []
    for i, (state_obs, policy_target, board_at_state) in enumerate(game_history):
        is_first_player = is_first_player_turn(cfg.env.type, board_at_state)
        value_target = final_value if is_first_player else -final_value
        full_game_data.append((state_obs, policy_target, board_at_state, value_target))
    
    if progress is not None and task_id_game is not None:
        progress.update(task_id_game, visible=False)

    return full_game_data

# --- Training Loop Function --- 
# Use DictConfig for type hint from Hydra
def run_training_loop(cfg: DictConfig) -> None: 
    """Main function to run the training loop using Hydra config."""
    # Progress bar setup (as before)
    progress = Progress(transient=False)
    progress.start()

    # --- Setup ---
    if cfg.training.device == "auto":
        if not cfg.training.use_multiprocessing:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.training.device)
    log_str_trining_device = f"{device} for training"

    # Print action space mode
    progress.print(f"Using {cfg.network.action_space_mode} action space mode")

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

    if use_multiprocessing_flag and num_workers > 1:
        device = "cpu"
        progress.print(f"Using {num_workers} workers for self-play (out of {total_cores} CPU cores),", log_str_trining_device)
    else:
        use_multiprocessing_flag = False
        progress.print(f"Using {device} for self-play, {log_str_trining_device}")
    
    # Initialize Network using network factory
    network = create_network(cfg, device)
    network.eval()

    # Create a separate network instance for profiling
    profile_network = create_network(cfg, device)
    profile_network.eval()

    # Profile the network
    N, C, H, W = cfg.training.batch_size, cfg.network.input_channels, cfg.network.board_size, cfg.network.board_size
    profile_model(profile_network, (torch.randn(N, C, H, W).to(device),))
    del profile_network  # Clean up the profiling network

    # Initialize Optimizer using optimizer config
    opt_cfg = cfg.optimizer
    if opt_cfg.type == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=opt_cfg.learning_rate, weight_decay=opt_cfg.weight_decay)
    elif opt_cfg.type == "SGD":
         # Ensure momentum is present if using SGD
         momentum = opt_cfg.momentum if opt_cfg.momentum is not None else 0.9
         optimizer = optim.SGD(network.parameters(), lr=opt_cfg.learning_rate, momentum=momentum, weight_decay=opt_cfg.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_cfg.type}")
    progress.print(f"Optimizer initialized: {opt_cfg.type}")

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(cfg.training.replay_buffer_size)

    # Loss Functions
    if cfg.mcts.iterations > 0:
        policy_loss_fn = nn.CrossEntropyLoss()
    else:
        policy_loss_fn = nn.BCEWithLogitsLoss()  # BCE with logits combines sigmoid and BCE loss
    value_loss_fn = nn.MSELoss()

    # Checkpoint directory (relative to hydra run dir)
    checkpoint_dir = cfg.training.checkpoint_dir 
    os.makedirs(checkpoint_dir, exist_ok=True)
    progress.print(f"Checkpoints will be saved in: {os.path.abspath(checkpoint_dir)}")

    env = create_environment(cfg, render=not use_multiprocessing_flag)

    # --- Main Training Loop ---
    start_iter = 0 # Initialize start_iter
    progress.print("Starting training loop...")
    total_training_start_time = time.time()

    # Check for existing checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    if os.path.exists(checkpoint_path):
        progress.print(f"\nFound existing checkpoint at {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            network.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iter = checkpoint['iteration']
            progress.print(f"Successfully loaded checkpoint from iteration {start_iter}")
        except Exception as e:
            progress.print(f"Error loading checkpoint: {e}")
            progress.print("Starting training from scratch...")
            start_iter = 0
    else:
        progress.print("No existing checkpoint found. Starting training from scratch...")

    # Use num_training_iterations from cfg
    for iteration in range(start_iter, cfg.training.num_training_iterations):
        iteration_start_time = time.time()
        progress.print(f"\n--- Training Iteration {iteration+1}/{cfg.training.num_training_iterations} ---")

        # --- Self-Play Phase --- 
        self_play_columns = (
            TextColumn("[progress.description]{task.description}"), BarColumn(),
            TaskProgressColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(), TimeElapsedColumn(),
        )
        progress.columns = self_play_columns
        
        # Only move to CPU if using multiple workers
        if use_multiprocessing_flag:
            network.to('cpu').eval()
        else:
            network.to(device).eval()

        games_data_collected = []
        iteration_start_time_selfplay = time.time()

        if use_multiprocessing_flag:
            # Prepare arguments for workers
            # Pass network state_dict directly since it's clean
            network_state_dict = network.state_dict()
            device_str = 'cpu' # Pass device as string
            # Pack arguments into tuples for the wrapper function
            worker_args_packed = [
                (game_num, network_state_dict, cfg, device_str)
                for game_num in range(cfg.training.self_play_games_per_epoch)
            ]

            pool = None # Initialize pool to None
            try:
                # Use 'spawn' context for better compatibility across platforms, especially with CUDA/PyTorch
                pool = multiprocessing.get_context("spawn").Pool(processes=num_workers)
                # progress.print(f"Submitting {len(worker_args_packed)} games to pool...")

                # Use imap_unordered to get results as they complete
                results_iterator = pool.imap_unordered(worker_wrapper, worker_args_packed)

                # Process results with a progress bar
                task_id_selfplay = progress.add_task("Self-Play", total=len(worker_args_packed))
                # Iterate over results as they become available
                for game_data in results_iterator:
                    if game_data: # Check if worker returned valid data
                        games_data_collected.extend(game_data)
                    # Update progress for each completed game
                    progress.update(task_id_selfplay, advance=1)
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

        else: # Sequential Execution
            task_id_selfplay = progress.add_task("Self-Play", total=cfg.training.self_play_games_per_epoch)
            # Use values from cfg
            for game_num in range(cfg.training.self_play_games_per_epoch):
                progress.update(task_id_selfplay, description=f"Self-Play Game ({game_num+1}/{cfg.training.self_play_games_per_epoch})")
                # Run game in the main process using the main env instance
                game_data = run_self_play_game(
                    cfg,
                    network if cfg.mcts.iterations > 0 else None,
                    env if device.type == 'cpu' else None,  # Use the main env instance
                    progress=progress if device.type == 'cpu' else None,
                    device=device
                )
                games_data_collected.extend(game_data)
                progress.update(task_id_selfplay, advance=1)
            progress.update(task_id_selfplay, visible=False)

        # Add collected data to replay buffer (outside the conditional block)
        for experience in games_data_collected:
            replay_buffer.add(experience)

        self_play_duration = int(time.time() - iteration_start_time_selfplay)
        progress.print(f"Self-play finished ({len(games_data_collected)} steps collected). Duration: {format_time(self_play_duration)}. Buffer size: {len(replay_buffer)}")

        # --- Training Phase ---
        if len(replay_buffer) < cfg.training.batch_size:
            progress.print("Not enough data in buffer to start training. Skipping phase.")
            continue
        
        training_columns = (
            TextColumn("[progress.description]{task.description}"), BarColumn(),
            TaskProgressColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(), TimeElapsedColumn(),
            TextColumn("[Loss P: {task.fields[loss_p]:.4f} V: {task.fields[loss_v]:.4f} Ill: {task.fields[illegal_r]:.2%} P: {task.fields[illegal_p]:.2%}]")
        )
        progress.columns = training_columns

        # progress.print("Starting training phase...")
        network.to(device).train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_illegal_moves_in_iteration = 0
        total_samples_in_iteration = 0

        task_id_train = progress.add_task(
            "Training Epochs", total=cfg.training.training_epochs,
            loss_p=float('nan'), loss_v=float('nan'), illegal_r=float('nan'), illegal_p=float('nan')
        )
        # Use values from cfg
        for epoch in range(cfg.training.training_epochs):
            batch = replay_buffer.sample(cfg.training.batch_size)
            if batch is None: continue

            # states_np, policy_targets_np, fens_batch, value_targets_np = batch
            states_np, policy_targets_np, boards_batch, value_targets_np = batch # Unpack boards
            states_tensor = torch.from_numpy(states_np).to(device)
            policy_targets_tensor = torch.from_numpy(policy_targets_np).to(device)
            value_targets_tensor = torch.from_numpy(value_targets_np).to(device)

            policy_logits, value_preds = network(states_tensor)

            policy_loss = policy_loss_fn(policy_logits, policy_targets_tensor)
            # Ensure value shapes match before loss calc
            value_loss = value_loss_fn(value_preds.squeeze(-1), value_targets_tensor.squeeze(-1))

            total_loss = policy_loss + value_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            # Calculate both metrics for this batch
            with torch.no_grad():
                # 1. Calculate illegal move probability mass
                policy_probs = F.softmax(policy_logits, dim=1)
                batch_illegal_prob_mass = 0.0
                
                # 2. Calculate argmax legality ratio
                batch_illegal_moves = 0
                predicted_action_indices = torch.argmax(policy_logits, dim=1)  # Shape: (batch_size)
                
                for i in range(len(boards_batch)):
                    current_board = boards_batch[i]
                    
                    # Calculate illegal probability mass
                    legal_moves = set(get_legal_actions(cfg.env.type, current_board))
                    legal_indices = torch.tensor([move_id - 1 for move_id in legal_moves], device=device)
                    illegal_prob = 1.0 - policy_probs[i, legal_indices].sum().item()
                    batch_illegal_prob_mass += illegal_prob
                    
                    # Check if argmax move is legal
                    predicted_action_id = predicted_action_indices[i].item() + 1
                    if predicted_action_id not in legal_moves:
                        batch_illegal_moves += 1
                
                avg_illegal_prob_mass = batch_illegal_prob_mass / len(boards_batch)
            
            total_illegal_moves_in_iteration += batch_illegal_moves
            total_samples_in_iteration += len(boards_batch)
            
            current_avg_policy_loss = total_policy_loss / (epoch + 1)
            current_avg_value_loss = total_value_loss / (epoch + 1)
            current_illegal_ratio = total_illegal_moves_in_iteration / total_samples_in_iteration if total_samples_in_iteration > 0 else 0.0

            progress.update(
                task_id_train, advance=1,
                loss_p=current_avg_policy_loss, loss_v=current_avg_value_loss,
                illegal_r=current_illegal_ratio,
                illegal_p=avg_illegal_prob_mass
            )
        progress.update(task_id_train, visible=False)

        avg_policy_loss = total_policy_loss / cfg.training.training_epochs if cfg.training.training_epochs > 0 else 0
        avg_value_loss = total_value_loss / cfg.training.training_epochs if cfg.training.training_epochs > 0 else 0
        avg_illegal_ratio = total_illegal_moves_in_iteration / total_samples_in_iteration if total_samples_in_iteration > 0 else 0.0
        progress.print(f"Training finished: Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f}, Avg Illegal Move Ratio: {avg_illegal_ratio:.2%}, Avg Illegal Move Prob Mass: {avg_illegal_prob_mass:.2%}")
        iteration_duration = int(time.time() - iteration_start_time)
        total_elapsed_time = int(time.time() - total_training_start_time)
        progress.print(f"Iteration {iteration+1} completed in {format_time(iteration_duration)} (total: {format_time(total_elapsed_time)})")

        # Save checkpoint after each iteration
        checkpoint = {
            'iteration': iteration + 1,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, "model.pth"))
        progress.print(f"Checkpoint saved at iteration {iteration + 1}")

    # Final training summary
    total_training_time = time.time() - total_training_start_time
    progress.print(f"\nTraining completed in {format_time(int(total_training_time))}")
    progress.print(f"Final model saved at: {os.path.abspath(os.path.join(checkpoint_dir, 'model.pth'))}")

    env.close()
    progress.print("\nTraining loop finished.")


# --- Hydra Entry Point --- 
# Ensure config_path points to the directory containing train_mcts.yaml
@hydra.main(config_path="../config", config_name="train_mcts", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Configuration:\n")
    # Use OmegaConf.to_yaml for structured printing
    print(OmegaConf.to_yaml(cfg))
    run_training_loop(cfg)

if __name__ == "__main__":
    main()
