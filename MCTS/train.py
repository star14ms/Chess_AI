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
from hydra.utils import get_class # Use instantiate later if needed
import sys
import multiprocessing # Keep for potential parallel execution

# Assuming files are in the MCTS directory relative to the project root
from network import ChessNetwork
from mcts_node import MCTSNode
from mcts_algorithm import MCTS
from utils.policy_human import sample_action
from chess_gym.envs import ChessEnv

sys.path.append('.')
# Dynamically import board class later using get_class

# Helper function for parallel execution
# NOTE: Passing large network objects might be slow or problematic.
# Consider loading the network state_dict in the worker if issues arise.
def self_play_worker(game_id, network_state_dict, cfg: DictConfig, device_str: str):
    """Worker function to run a single self-play game."""
    # Determine device for this worker
    device = torch.device(device_str)

    # Re-initialize Network in the worker process
    # (Ensure parameters match your network's __init__)
    network = ChessNetwork(
        input_channels=cfg.network.input_channels,
        board_size=cfg.network.board_size,
        num_conv_layers=cfg.network.num_conv_layers,
        num_filters=cfg.network.num_filters,
        action_space_size=cfg.network.action_space_size,
        num_attention_heads=cfg.network.num_attention_heads,
        decoder_ff_dim_mult=cfg.network.decoder_ff_dim_mult,
        # Add other required params based on network.py definition
    ).to(device)
    network.load_state_dict(network_state_dict)
    network.eval() # Ensure it's in eval mode

    # Re-initialize Environment in the worker process
    # Assumes env config is part of the main cfg
    env = ChessEnv(
        observation_mode=cfg.env.observation_mode,
        render_mode=None, # Avoid rendering in workers
        save_video_folder=None
    )

    # Progress is tricky with multiprocessing pools directly updating Rich progress.
    # Usually, you track completion externally or use shared state (more complex).
    # For simplicity, we'll omit the fine-grained progress updates from within the worker.
    game_data = run_self_play_game(
        network,
        cfg.mcts,
        cfg.training,
        env,
        progress=None, # Pass None for progress within worker
        device=device
    )
    env.close() # Close env instance in worker
    return game_data

# Wrapper for imap_unordered
def worker_wrapper(args):
    """Unpacks arguments for self_play_worker when using imap."""
    game_id, network_state_dict, cfg, device_str = args
    # Call the original worker function with unpacked args
    return self_play_worker(game_id, network_state_dict, cfg, device_str)

# --- Removed CONFIG dictionary --- 

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
        value_targets = np.array([exp[2] for exp in batch], dtype=np.float32).reshape(-1, 1)

        return states, policy_targets, value_targets

    def __len__(self):
        return len(self.buffer)


# --- Self-Play Function (Update args to use config subsections) ---
def run_self_play_game(network, mcts_cfg: OmegaConf, train_cfg: OmegaConf, env: ChessEnv,
                       progress: Progress | None = None, device: torch.device | None = None):
    """Plays one game of self-play using MCTS and returns the game data."""
    # If running sequentially, ensure env is reset at the start of each game
    # If running in parallel, the worker resets its own env copy
    # This reset is safe for both cases
    obs, _ = env.reset()
    network.eval()

    game_history = []
    move_count = 0
    terminated = False
    truncated = False

    mcts_iterations = mcts_cfg.iterations
    c_puct = mcts_cfg.c_puct
    temp_start = mcts_cfg.temperature_start
    temp_end = mcts_cfg.temperature_end
    temp_decay_moves = mcts_cfg.temperature_decay_moves
    max_moves = train_cfg.max_game_moves

    task_id_game = None
    if progress is not None:
        # Create task only if progress object is provided
        task_id_game = progress.add_task(f"Max Moves (0/{max_moves})", total=max_moves, transient=True)
        progress.start_task(task_id_game)

    while not terminated and not truncated and move_count < max_moves:
        # --- Create MCTS root node with FEN and board_cls --- 
        # Need board_cls from the env instance
        root_node = MCTSNode(env.board)

        # --- Create MCTS instance --- 
        # Pass env only if running sequentially
        mcts_env = env if not train_cfg.get('use_multiprocessing', False) else None # Determine env based on flag
        mcts_player = MCTS(
            network,
            device=device,
            env=mcts_env,  # Pass env (or None)
            C_puct=c_puct
        )

        # --- Run MCTS Search --- 
        mcts_player.search(root_node, mcts_iterations, progress=progress)

        # --- Get Policy and Store History --- 
        temperature = temp_start * ((temp_end / temp_start) ** min(1.0, move_count / temp_decay_moves))
        mcts_policy = mcts_player.get_policy_distribution(root_node, temperature=1.0)
        current_obs = obs # Observation *before* the move
        game_history.append((current_obs, mcts_policy))
        
        # --- Select and Make Move --- 
        chosen_move = mcts_player.get_best_move(root_node, temperature=temperature)

        if chosen_move is None:
            print(f"Warning: MCTS returned None move. Ending game.")
            if progress is not None and task_id_game is not None:
                progress.update(task_id_game, visible=False)
            break

        action_to_take = env.board.move_to_action_id(chosen_move)
        if action_to_take is None:
            print(f"Warning: Could not convert move {chosen_move.uci()} to action ID.")
            if progress is not None and task_id_game is not None:
                progress.update(task_id_game, visible=False)
            break

        try:
            # Step the *main* environment (or worker's copy)
            obs, reward, terminated, truncated, info = env.step(action_to_take)
            # Step opponent agent
            env.step(sample_action(env.board, return_id=True))
        except Exception as e:
            raise RuntimeError(f"Error during env.step: {e}. Ending game.")

        if progress is not None:
            progress.update(task_id_game, description=f"Max Moves ({move_count+1}/{max_moves})", advance=1)

        move_count += 1

    # --- Determine final game outcome --- 
    final_value = 0.0
    if terminated:
        result_str = env.board.result(claim_draw=True)
        if result_str == "1-0": final_value = 1.0
        elif result_str == "0-1": final_value = -1.0

    # Assign value targets
    full_game_data = []
    for i, (state_obs, policy_target) in enumerate(game_history):
        # Determine whose turn it was at the state where the MCTS policy was calculated
        # If state i was White's turn to play, the value target is final_value
        # If state i was Black's turn to play, the value target is -final_value
        # Assume White plays at even indices (0, 2, ...)
        is_white_turn_at_state = (i % 2 == 0)
        value_target = final_value if is_white_turn_at_state else -final_value
        full_game_data.append((state_obs, policy_target, value_target))
    
    if progress is not None and task_id_game is not None:
        progress.update(task_id_game, visible=False)

    return full_game_data

# --- Training Loop Function --- 
# Use DictConfig for type hint from Hydra
def run_training_loop(cfg: DictConfig) -> None: 
    """Main function to run the training loop using Hydra config."""

    # --- Setup ---
    if cfg.training.device == "auto":
        device = torch.device("cpu")
        # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(cfg.training.device)
    print(f"Using device: {device}")

    # Get Board Class dynamically
    try:
        board_cls = get_class(cfg.training.board_cls_str)
        print(f"Using Board Class: {board_cls.__name__}")
    except Exception as e:
        print(f"Error importing board class '{cfg.training.board_cls_str}': {e}")
        sys.exit(1)

    # Initialize Network using network config
    # Make sure network's __init__ signature matches the parameters passed
    network = ChessNetwork(
        input_channels=cfg.network.input_channels,
        board_size=cfg.network.board_size,
        num_conv_layers=cfg.network.num_conv_layers,
        num_filters=cfg.network.num_filters,
        action_space_size=cfg.network.action_space_size, # Get from training cfg
        num_attention_heads=cfg.network.num_attention_heads,
        decoder_ff_dim_mult=cfg.network.decoder_ff_dim_mult,
        # Add other required params based on current network.py
        # e.g., num_pieces=cfg.network.num_pieces if defined
    ).to(device)
    network.train()
    print("Network initialized.")

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
    print(f"Optimizer initialized: {opt_cfg.type}")

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(cfg.training.replay_buffer_size)

    # Loss Functions
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # Checkpoint directory (relative to hydra run dir)
    checkpoint_dir = cfg.training.checkpoint_dir 
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {os.path.abspath(checkpoint_dir)}")

    # --- Main Training Loop ---
    start_iter = 0 # TODO: Implement checkpoint loading to resume
    print("Starting training loop...")

    # Progress bar setup (as before)
    progress_columns_train = [
        TextColumn("[progress.description]{task.description}"), BarColumn(),
        TaskProgressColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(), TimeElapsedColumn(),
        TextColumn("[Loss P: {task.fields[loss_p]:.4f} V: {task.fields[loss_v]:.4f}]"),
    ]
    progress_columns_selfplay = [
        TextColumn("[progress.description]{task.description}"), BarColumn(),
        TaskProgressColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(), TimeElapsedColumn(),
    ]

    env = ChessEnv(
        observation_mode=cfg.env.observation_mode,
        render_mode=cfg.env.render_mode,
        save_video_folder=cfg.env.save_video_folder
    )

    # Use num_training_iterations from cfg
    for iteration in range(start_iter, cfg.training.num_training_iterations):
        iteration_start_time = time.time()
        print(f"\n--- Training Iteration {iteration+1}/{cfg.training.num_training_iterations} ---")

        # --- Self-Play Phase --- 
        network.eval() 
        games_data_collected = []
        iteration_start_time_selfplay = time.time()

        # Check config flag to decide execution mode
        # >>>>>>>>>>>>>> ADDED CONDITIONAL EXECUTION BLOCK <<<<<<<<<<<<<<<<
        # Default to False if the flag is missing
        use_multiprocessing_flag = cfg.training.get('use_multiprocessing', False)

        if use_multiprocessing_flag:
            print("Starting self-play phase (parallel)...")
            # Determine number of workers
            # Ensure cfg.training.self_play_workers exists in your config!
            try:
                num_workers_config = cfg.training.self_play_workers
            except Exception:
                print("Warning: cfg.training.self_play_workers not found. Defaulting workers.")
                num_workers_config = 0 # Indicate to use default

            if num_workers_config > 0:
                num_workers = min(num_workers_config, os.cpu_count())
            else:
                # Default heuristic if 0 or not specified: use half the CPUs, minimum 1
                num_workers = max(1, os.cpu_count() // 2)
            print(f"Using {num_workers} workers for self-play.")

            # Prepare arguments for workers
            # Pass network state_dict instead of the full object for better pickling
            network_state_dict = network.state_dict()
            device_str = str(device) # Pass device as string
            # Pack arguments into tuples for the wrapper function
            worker_args_packed = [
                (game_num, network_state_dict, cfg, device_str)
                for game_num in range(cfg.training.self_play_games_per_epoch)
            ]

            results = []
            pool = None # Initialize pool to None
            try:
                # Use 'spawn' context for better compatibility across platforms, especially with CUDA/PyTorch
                pool = multiprocessing.get_context("spawn").Pool(processes=num_workers)
                print(f"Submitting {len(worker_args_packed)} games to pool...")

                # Use imap_unordered to get results as they complete
                results_iterator = pool.imap_unordered(worker_wrapper, worker_args_packed)

                # Process results with a progress bar
                with Progress(*progress_columns_selfplay, transient=False) as progress:
                    task_id_selfplay = progress.add_task(
                        "Collecting Self-Play Results (Parallel)",
                        total=len(worker_args_packed)
                    )
                    # Iterate over results as they become available
                    for game_data in results_iterator:
                        if game_data: # Check if worker returned valid data
                            games_data_collected.extend(game_data)
                        # Update progress for each completed game
                        progress.update(task_id_selfplay, advance=1)
                # Explicitly close and join the pool after processing results
                pool.close()
                pool.join()

            except Exception as e:
                print(f"Error during parallel self-play: {e}")
                # Handle error, maybe skip iteration or exit
                if pool is not None:
                    pool.terminate() # Terminate pool on error
            finally:
                # Ensure pool is closed and joined even if no error occurred in the try block
                if pool is not None and pool._state == multiprocessing.pool.RUN:
                    pool.close()
                    pool.join()

        else: # Sequential Execution
            print("Starting self-play phase (sequential)...")
            with Progress(*progress_columns_selfplay, transient=False) as progress:
                task_id_selfplay = progress.add_task(
                    "Self-Play Games (Sequential)", 
                    total=cfg.training.self_play_games_per_epoch
                )
                # Use values from cfg
                for game_num in range(cfg.training.self_play_games_per_epoch):
                    progress.update(task_id_selfplay, description=f"Self-Play Game ({game_num+1}/{cfg.training.self_play_games_per_epoch})")
                    # Run game in the main process using the main env instance
                    game_data = run_self_play_game(
                        network,
                        cfg.mcts,       # Pass MCTS config section
                        cfg.training,   # Pass Training config section
                        env,            # Use the main env instance
                        progress=progress,
                        device=device
                    )
                    games_data_collected.extend(game_data)
                    progress.update(task_id_selfplay, advance=1)
        # >>>>>>>>>>>>>> END OF CONDITIONAL EXECUTION BLOCK <<<<<<<<<<<<<<<<

        # Add collected data to replay buffer (outside the conditional block)
        for experience in games_data_collected:
            replay_buffer.add(experience)

        self_play_duration = time.time() - iteration_start_time_selfplay
        print(f"Self-play finished ({len(games_data_collected)} steps collected). Duration: {self_play_duration:.2f}s. Buffer size: {len(replay_buffer)}")

        # --- Training Phase ---
        if len(replay_buffer) < cfg.training.batch_size:
            print("Not enough data in buffer to start training. Skipping phase.")
            continue

        print("Starting training phase...")
        network.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        training_start_time = time.time()

        with Progress(*progress_columns_train, transient=False) as progress:
            task_id_train = progress.add_task(
                "Training Epochs", total=cfg.training.training_epochs,
                loss_p=float('nan'), loss_v=float('nan')
            )
            # Use values from cfg
            for epoch in range(cfg.training.training_epochs):
                batch = replay_buffer.sample(cfg.training.batch_size)
                if batch is None: continue

                states_np, policy_targets_np, value_targets_np = batch
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

                current_avg_policy_loss = total_policy_loss / (epoch + 1)
                current_avg_value_loss = total_value_loss / (epoch + 1)

                progress.update(
                    task_id_train, advance=1,
                    loss_p=current_avg_policy_loss, loss_v=current_avg_value_loss
                )

        training_duration = time.time() - training_start_time
        avg_policy_loss = total_policy_loss / cfg.training.training_epochs if cfg.training.training_epochs > 0 else 0
        avg_value_loss = total_value_loss / cfg.training.training_epochs if cfg.training.training_epochs > 0 else 0
        print(f"Training finished. Duration: {training_duration:.2f}s")
        print(f"Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f}")

        # --- Save Checkpoint ---
        # Use values from cfg
        if (iteration + 1) % cfg.training.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"iteration_{iteration+1}.pth")
            print(f"Saving checkpoint to {checkpoint_path}...")
            torch.save({
                'iteration': iteration + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

        iteration_duration = time.time() - iteration_start_time
        print(f"Iteration {iteration+1} completed in {iteration_duration:.2f}s")
    
    env.close()
    print("\nTraining loop finished.")


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
