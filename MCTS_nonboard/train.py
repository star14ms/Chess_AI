import torch
import torch.optim as optim
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import multiprocessing
from mcts_node import MCTSNode
from mcts_algorithm import MCTS
from training_modules.breakout import (
    create_breakout_env,
    create_breakout_network,
)

def run_self_play_game(cfg, network, device):
    env = create_breakout_env(cfg)
    obs, _ = env.reset()
    done = False
    game_history = []
    while not done:
        root_node = MCTSNode(obs)
        mcts = MCTS(network, device, env)
        mcts.search(root_node, cfg.mcts.iterations)
        policy = mcts.get_policy_distribution(root_node)
        action = mcts.get_best_action(root_node)
        game_history.append((obs, policy))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
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
    device = torch.device(cfg.training.device)
    network = create_breakout_network(cfg, device)
    optimizer = optim.Adam(network.parameters(), lr=cfg.optimizer.learning_rate)
    replay_buffer = []
    use_multiprocessing = cfg.training.get('use_multiprocessing', False)
    num_workers = cfg.training.get('self_play_workers', 1) or 1
    for iteration in range(cfg.training.num_training_iterations):
        # --- Self-Play Phase ---
        games_data_collected = []
        if use_multiprocessing and num_workers > 1:
            network_state_dict = network.state_dict()
            device_str = 'cpu'
            worker_args = [(cfg, network_state_dict, device_str) for _ in range(cfg.training.self_play_games_per_epoch)]
            with multiprocessing.get_context("spawn").Pool(processes=num_workers) as pool:
                for game_data in pool.imap_unordered(self_play_worker, worker_args):
                    games_data_collected.extend(game_data)
        else:
            for _ in range(cfg.training.self_play_games_per_epoch):
                game_data = run_self_play_game(cfg, network, device)
                games_data_collected.extend(game_data)
        replay_buffer.extend(games_data_collected)
        # --- Training Phase ---
        for epoch in range(cfg.training.training_epochs):
            if len(replay_buffer) < cfg.training.batch_size:
                continue
            batch = replay_buffer[-cfg.training.batch_size:]
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
        print(f"Iteration {iteration+1} finished. Buffer size: {len(replay_buffer)}")

@hydra.main(config_path="../config", config_name="train_breakout", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Configuration:\n")
    print(OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == "__main__":
    main() 