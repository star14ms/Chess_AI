import torch
import torch.optim as optim
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import multiprocessing
import os
import time
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from mcts_node import MCTSNode
from mcts_algorithm import MCTS
from training_modules.breakout import (
    register_envs,
    create_breakout_env,
    create_breakout_network,
)
from collections import deque
from utils.profile import profile_model, get_optimal_worker_count


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def extend(self, experiences):
        self.buffer.extend(experiences)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indices]
        return batch

    def __len__(self):
        return len(self.buffer)


def run_self_play_game(cfg, network, device):
    env = create_breakout_env(cfg)
    obs, _ = env.reset()
    done = False
    game_history = []
    move_count = 0
    max_moves = cfg.training.max_game_moves
    temp_start = cfg.mcts.get('temperature_start', 1.0)
    temp_end = cfg.mcts.get('temperature_end', 0.1)
    temp_decay_moves = cfg.mcts.get('temperature_decay_moves', 30)

    while not done and move_count < max_moves:
        # Compute temperature for this move
        temperature = temp_start * ((temp_end / temp_start) ** min(1.0, move_count / temp_decay_moves))

        root_node = MCTSNode(obs)
        mcts = MCTS(network, device, env)
        mcts.search(root_node, cfg.mcts.iterations)
        policy = mcts.get_policy_distribution(root_node, temperature=temperature)
        action = mcts.get_best_action(root_node, temperature=temperature)
        game_history.append((obs, policy))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        move_count += 1

    value = reward
    return [(obs, policy, value) for obs, policy in game_history]


def self_play_worker(args):
    cfg, network_state_dict, device_str = args
    device = torch.device(device_str)
    network = create_breakout_network(cfg, device)
    network.load_state_dict(network_state_dict)
    network.eval()
    return run_self_play_game(cfg, network, device)


def train(cfg: DictConfig):
    if cfg.env.type == "breakout":
        register_envs()

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
    
    network = create_breakout_network(cfg, device)

    # Profile the network
    N = cfg.training.batch_size
    C = cfg.network.input_channels
    H = cfg.network.get('board_height', 164)
    W = cfg.network.get('board_width', 144)
    profile_network = create_breakout_network(cfg, device)
    profile_network.eval()
    profile_model(profile_network, (torch.randn(N, C, H, W).to(device),))
    del profile_network

    optimizer = optim.Adam(
        network.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.get('weight_decay', 0.0)
    )

    replay_buffer = ReplayBuffer(cfg.training.replay_buffer_size)
    use_multiprocessing = cfg.training.get('use_multiprocessing', False)
    checkpoint_dir = cfg.training.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_interval = cfg.training.save_interval
    total_training_start_time = time.time()

    for iteration in range(cfg.training.num_training_iterations):
        progress.print(f"\n--- Training Iteration {iteration+1}/{cfg.training.num_training_iterations} ---")
        iteration_start_time_selfplay = time.time()

        # --- Self-Play Phase ---
        self_play_columns = (
            TextColumn("[progress.description]{task.description}"), BarColumn(),
            TaskProgressColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(), TimeElapsedColumn(),
        )
        progress.columns = self_play_columns
        games_data_collected = []
        task_id_selfplay = progress.add_task(f"Self-Play", total=cfg.training.self_play_games_per_epoch)
        
        if use_multiprocessing and num_workers > 1:
            network_state_dict = network.state_dict()
            device_str = 'cpu'
            worker_args = [(cfg, network_state_dict, device_str) for _ in range(cfg.training.self_play_games_per_epoch)]
            with multiprocessing.get_context("spawn").Pool(processes=num_workers) as pool:
                for i, game_data in enumerate(pool.imap_unordered(self_play_worker, worker_args)):
                    games_data_collected.extend(game_data)
                    progress.update(task_id_selfplay, advance=1)
        else:
            for i in range(cfg.training.self_play_games_per_epoch):
                game_data = run_self_play_game(cfg, network, device)
                games_data_collected.extend(game_data)
                progress.update(task_id_selfplay, advance=1)

        progress.update(task_id_selfplay, visible=False)
        replay_buffer.extend(games_data_collected)
        self_play_duration = int(time.time() - iteration_start_time_selfplay)
        progress.print(f"Self-play finished ({len(games_data_collected)} steps collected). Duration: {self_play_duration//60}m {self_play_duration%60}s. Buffer size: {len(replay_buffer)}")

        # --- Training Phase ---
        training_columns = (
            TextColumn("[progress.description]{task.description}"), BarColumn(),
            TaskProgressColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(), TimeElapsedColumn(),
            TextColumn("[Loss P: {task.fields[loss_p]:.4f} V: {task.fields[loss_v]:.4f}]")
        )
        progress.columns = training_columns
        total_policy_loss = 0.0
        total_value_loss = 0.0
        task_id_train = progress.add_task(
            f"Training", total=cfg.training.training_epochs,
            loss_p=float('nan'), loss_v=float('nan')
        )

        for epoch in range(cfg.training.training_epochs):
            batch = replay_buffer.sample(cfg.training.batch_size)
            if batch is None:
                continue
            obs_batch = torch.tensor(np.stack([x[0] for x in batch]), dtype=torch.float32, device=device)
            policy_batch = torch.tensor(np.stack([x[1] for x in batch]), dtype=torch.float32, device=device)
            value_batch = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32, device=device)
            policy_logits, value_pred = network(obs_batch)
            policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_batch)
            value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(), value_batch)
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            current_avg_policy_loss = total_policy_loss / (epoch + 1)
            current_avg_value_loss = total_value_loss / (epoch + 1)
            progress.update(
                task_id_train, advance=1,
                loss_p=current_avg_policy_loss, loss_v=current_avg_value_loss
            )

        progress.update(task_id_train, visible=False)
        progress.print(f"Iteration {iteration+1} finished. Buffer size: {len(replay_buffer)} | Avg Policy Loss: {current_avg_policy_loss:.4f} | Avg Value Loss: {current_avg_value_loss:.4f}")

        # --- Checkpointing ---
        if (iteration + 1) % save_interval == 0 or (iteration + 1) == cfg.training.num_training_iterations:
            checkpoint = {
                'iteration': iteration + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"model_{iteration+1}.pth"))
            progress.print(f"Checkpoint saved at iteration {iteration + 1}")

        # --- Iteration Duration Logging ---
        iteration_duration = int(time.time() - iteration_start_time_selfplay)
        total_elapsed_time = int(time.time() - total_training_start_time)
        progress.print(f"Iteration {iteration+1} completed in {iteration_duration//60}m {iteration_duration%60}s (total: {total_elapsed_time//60}m {total_elapsed_time%60}s)")

    progress.stop()


@hydra.main(config_path="../config", config_name="train_breakout", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Configuration:\n")
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main() 