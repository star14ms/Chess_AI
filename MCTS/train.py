import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
import time
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
import copy

# Assuming files are in the MCTS directory relative to the project root
from network import ChessNetwork
from mcts_node import MCTSNode
from mcts_algorithm import MCTS

import sys
import os
sys.path.append('.')
# Assuming chess_gym provides the custom board needed by MCTS/Network
from chess_gym.chess_custom import FullyTrackedBoard
from chess_gym.envs import ChessEnv # Make sure ChessEnv is imported

# --- Hyperparameters ---
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    "learning_rate": 0.001,
    "optimizer": "Adam", # Or "SGD"
    "momentum": 0.9, # Only used if optimizer is SGD
    "weight_decay": 1e-4,
    "replay_buffer_size": 50000, # Max number of game steps stored
    "batch_size": 128,
    "mcts_iterations": 100, # Number of simulations per move in self-play
    "self_play_games_per_epoch": 50,
    "training_epochs": 10, # Number of optimization steps per training iteration
    "num_training_iterations": 1000, # Total number of train/self-play cycles
    "checkpoint_dir": "checkpoints", # Directory to save models
    "save_interval": 10, # Save model every N training iterations
    "c_puct": 1.41, # Exploration constant for MCTS UCT
    "self_play_temperature_start": 1.0, # Temperature for move selection in early game
    "self_play_temperature_end": 0.1, # Temperature for move selection in later game
    "temperature_decay_moves": 30, # Number of moves over which temperature decays
    "max_game_moves": 200, # Limit game length to prevent infinite loops
    "board_cls": FullyTrackedBoard # The board class to use
}

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """Adds a single experience tuple (state, policy_target, value_target)."""
        self.buffer.append(experience)

    def add_game(self, game_data):
        """Adds all experiences from a completed game."""
        for experience in game_data:
            self.add(experience)

    def sample(self, batch_size):
        """Samples a batch of experiences."""
        if len(self.buffer) < batch_size:
            return None # Not enough data yet
        batch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indices]

        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        policy_targets = np.array([exp[1] for exp in batch], dtype=np.float32)
        value_targets = np.array([exp[2] for exp in batch], dtype=np.float32).reshape(-1, 1) # Ensure shape (batch_size, 1)

        return states, policy_targets, value_targets

    def __len__(self):
        return len(self.buffer)

# --- Self-Play Function ---
def run_self_play_game(network, mcts_iterations, c_puct, temp_start, temp_end, temp_decay_moves, max_moves, board_cls,
                       progress: Progress | None = None, game_task_id = None, device: torch.device | None = None):
    """Plays one game of self-play using MCTS and returns the game data.

    Args:
        # ... other args ...
        progress: Optional rich Progress object for MCTS sub-task display.
        game_task_id: Optional task ID for the current game within the Progress object.
        device: The torch device to use for MCTS tensor operations.
    """

    # Initialize the environment
    env = ChessEnv(observation_mode='vector', render_mode='human') # Ensure vector observation mode
    obs, info = env.reset() # Get initial observation

    network.eval() # Set network to evaluation mode

    game_history = [] # Stores (observation, mcts_policy) tuples
    move_count = 0
    terminated = False
    truncated = False

    while not terminated and not truncated and move_count < max_moves:
        # Create MCTS root node with a deep copy of the *current* environment state
        # This ensures MCTS search doesn't affect the main game loop's env instance
        try:
            env_copy = copy.deepcopy(env)
        except Exception as e:
             print(f"Error deepcopying environment: {e}. Ending game.")
             break # Cannot proceed without copying
        root_node = MCTSNode(env_copy) 
        
        # Create MCTS instance (using the network and device)
        # Pass the device from the function arguments
        mcts_player = MCTS(network, device=device, player_color=env.board.turn, C_puct=c_puct)

        # Run MCTS search
        # Pass the progress object and task ID for sub-task display
        mcts_player.search(root_node, mcts_iterations, progress=progress, parent_task_id=game_task_id)

        # Calculate temperature for move selection
        temperature = temp_start * ((temp_end / temp_start) ** min(1.0, move_count / temp_decay_moves))

        # Get MCTS policy distribution (pi) from the search results
        mcts_policy = mcts_player.get_policy_distribution(root_node, temperature=1.0)

        # Store the *current* observation and the calculated policy target
        current_obs = obs # Observation before the move is made
        game_history.append((current_obs, mcts_policy))

        # Select the actual move to play in the game based on MCTS results
        chosen_move = mcts_player.get_best_move(root_node, temperature=temperature)

        if chosen_move is None:
            print(f"Warning: MCTS returned None move in game state: {env.board.fen()}. Ending game.")
            break

        # --- Step the *main* environment --- 
        # Convert chess.Move to the action format expected by env.step()
        # Assuming env.action_space handles this or we use move_to_action_id
        # Let's try using the action ID directly if available
        action_to_take = env.board.move_to_action_id(chosen_move)
        if action_to_take is None:
             print(f"Warning: Could not convert chosen move {chosen_move.uci()} to action ID. Trying legacy format.")
             # Fallback to legacy format if ID fails
             action_to_take = env.action_space._move_to_action(chosen_move, return_id=False)

        try:
            obs, reward, terminated, truncated, info = env.step(action_to_take)
        except Exception as e:
            print(f"Error during env.step for action {action_to_take} (move {chosen_move.uci()}): {e}. Ending game.")
            print(f"Board state: {env.board.fen()}")
            terminated = True # Treat step errors as termination

        move_count += 1

    # --- Determine final game outcome --- 
    # Use the environment's board state after the loop finishes
    # The reward from the last step isn't necessarily the final outcome value
    final_value = 0.0
    if terminated: # Only consider result if game terminated naturally
        result_str = env.board.result(claim_draw=True)
        if result_str == "1-0": # White wins
            final_value = 1.0
        elif result_str == "0-1": # Black wins
            final_value = -1.0
        # Otherwise, it's a draw, final_value remains 0.0
    # If truncated, the game didn't finish, assign 0.0 value?
    # Or should we use the network's evaluation of the final state? Let's stick to 0.0 for truncated/unfinished.

    # Assign value targets
    full_game_data = []
    num_moves = len(game_history)
    for i, (state_obs, policy_target) in enumerate(game_history):
        # Whose turn was it when this observation was made?
        # White plays on moves 0, 2, 4... Black plays on 1, 3, 5...
        is_white_turn_at_state = (i % 2 == 0)
        value_target = final_value if is_white_turn_at_state else -final_value
        full_game_data.append((state_obs, policy_target, value_target))

    env.close() # Close the environment instance
    return full_game_data

# --- Training Function ---
def train(config):
    device = torch.device(config["device"])
    print(f"Using device: {device}")

    # Initialize Network
    # Assuming network defaults match observation/action space
    network = ChessNetwork().to(device)
    network.train() # Set to training mode

    # Initialize Optimizer
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "SGD":
         optimizer = optim.SGD(network.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(config["replay_buffer_size"])

    # Loss Functions
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # Ensure checkpoint directory exists
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # --- Main Training Loop ---
    start_iter = 0 # TODO: Load from checkpoint if resuming
    print("Starting training loop...")

    # Define progress bar columns (can be customized)
    progress_columns_train = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn("[progress.percentage]{task.percentage:>3.1f}%"), # Shows completed/total
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        TextColumn("[Loss P: {task.fields[loss_p]:.4f} V: {task.fields[loss_v]:.4f}]"), # Custom loss column
    ]
    # Define separate columns for self-play (without loss)
    progress_columns_selfplay = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ]

    # Outer loop for training iterations (not using rich progress here, just printing)
    for iteration in range(start_iter, config["num_training_iterations"]):
        iteration_start_time = time.time()
        print(f"\n--- Training Iteration {iteration+1}/{config['num_training_iterations']} --- ")

        # --- Self-Play Phase --- Use Rich Progress ---
        print("Starting self-play phase...")
        network.eval()
        games_data = []
        # Use self-play columns
        with Progress(*progress_columns_selfplay, transient=False) as progress:
            task_id_selfplay = progress.add_task("Self-Play Games", total=config["self_play_games_per_epoch"])
            for game_num in range(config["self_play_games_per_epoch"]):
                # Update description for the current game
                progress.update(task_id_selfplay, description=f"Self-Play Game {game_num+1}/{config['self_play_games_per_epoch']}")
                # Pass device to the game function
                game_data = run_self_play_game(
                    network,
                    config["mcts_iterations"],
                    config["c_puct"],
                    config["self_play_temperature_start"],
                    config["self_play_temperature_end"],
                    config["temperature_decay_moves"],
                    config["max_game_moves"],
                    config["board_cls"],
                    progress=progress,
                    game_task_id=task_id_selfplay,
                    device=device # Pass the device
                )
                games_data.extend(game_data)
                progress.update(task_id_selfplay, advance=1)

        # Add collected game data to replay buffer
        for experience in games_data:
            replay_buffer.add(experience)
        self_play_duration = time.time() - iteration_start_time # Recalculate start time for this phase
        print(f"Self-play finished ({len(games_data)} steps collected). Duration: {self_play_duration:.2f}s. Buffer size: {len(replay_buffer)}")

        # --- Training Phase --- Use Rich Progress ---
        if len(replay_buffer) < config["batch_size"]:
            print("Not enough data in buffer to start training yet. Skipping training phase.")
            continue

        print("Starting training phase...")
        network.train() # Network in train mode
        total_policy_loss = 0.0
        total_value_loss = 0.0
        training_start_time = time.time()

        # Use training columns
        with Progress(*progress_columns_train, transient=False) as progress:
            # Initialize custom fields for the loss column
            task_id_train = progress.add_task(
                "Training Epochs",
                total=config["training_epochs"],
                loss_p=float('nan'), # Initialize with NaN or 0.0
                loss_v=float('nan')
            )
            for epoch in range(config["training_epochs"]):
                batch = replay_buffer.sample(config["batch_size"])
                if batch is None: # Should not happen if check above passed, but safety
                    print("Warning: Failed to sample batch.")
                    continue

                states_np, policy_targets_np, value_targets_np = batch

                # Convert to tensors and move to device
                states_tensor = torch.from_numpy(states_np).to(device)
                policy_targets_tensor = torch.from_numpy(policy_targets_np).to(device)
                value_targets_tensor = torch.from_numpy(value_targets_np).to(device)

                # Forward pass
                policy_logits, value_preds = network(states_tensor)

                # Calculate Loss
                # Policy loss: CrossEntropy expects logits and class indices.
                # Our target is a distribution (pi). CrossEntropyLoss works with target distributions too.
                policy_loss = policy_loss_fn(policy_logits, policy_targets_tensor)
                value_loss = value_loss_fn(value_preds, value_targets_tensor.squeeze(1)) # Ensure value target shape matches prediction (batch_size)

                total_loss = policy_loss + value_loss # Simple sum, can be weighted

                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                
                # Calculate running average losses
                current_avg_policy_loss = total_policy_loss / (epoch + 1)
                current_avg_value_loss = total_value_loss / (epoch + 1)

                # Update progress bar, including the custom loss fields
                progress.update(
                    task_id_train,
                    advance=1,
                    loss_p=current_avg_policy_loss,
                    loss_v=current_avg_value_loss
                )

        training_duration = time.time() - training_start_time
        avg_policy_loss = total_policy_loss / config["training_epochs"] if config["training_epochs"] > 0 else 0
        avg_value_loss = total_value_loss / config["training_epochs"] if config["training_epochs"] > 0 else 0
        print(f"Training finished. Duration: {training_duration:.2f}s")
        print(f"Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f}")

        # --- Save Checkpoint ---
        if (iteration + 1) % config["save_interval"] == 0:
            checkpoint_path = os.path.join(config["checkpoint_dir"], f"iteration_{iteration+1}.pth")
            print(f"Saving checkpoint to {checkpoint_path}...")
            torch.save({
                'iteration': iteration + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # Could also save replay buffer, learning rate schedule state, etc.
            }, checkpoint_path)

        iteration_duration = time.time() - iteration_start_time
        print(f"Iteration {iteration+1} completed in {iteration_duration:.2f}s")

    print("\nTraining loop finished.")

# --- Main Execution ---
if __name__ == "__main__":
    train(CONFIG)
