import torch.nn as nn
import ale_py
import gymnasium as gym
import sys
import os
import numpy as np

# Ensure project root is in sys.path for model imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from MCTS_nonboard.models.network_breakout import BreakoutNetwork

# CroppedBreakoutEnv wrapper for cropping observations
class CroppedBreakoutEnv(gym.Wrapper):
    crop_left = 8
    crop_right = 8
    crop_top = 32
    crop_bottom = 14
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_obs = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = observation[self.crop_top:-self.crop_bottom, self.crop_left:-self.crop_right]
        if observation.ndim == 2:
            observation = observation[np.newaxis, :, :]
        if self.prev_obs is None:
            stacked_obs = np.concatenate([observation, observation], axis=0)
        else:
            stacked_obs = np.concatenate([self.prev_obs, observation], axis=0)
        self.prev_obs = observation
        return stacked_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = observation[self.crop_top:-self.crop_bottom, self.crop_left:-self.crop_right]
        if observation.ndim == 2:
            observation = observation[np.newaxis, :, :]
        self.prev_obs = observation
        stacked_obs = np.concatenate([observation, observation], axis=0)
        return stacked_obs, info

    def clone_state(self):
        return self.env.unwrapped.clone_state()

    def restore_state(self, state):
        return self.env.unwrapped.restore_state(state)

def create_breakout_network(cfg, device) -> nn.Module:
    """Create and initialize the Breakout network based on config."""
    network = BreakoutNetwork(
        input_channels=cfg.network.input_channels,
        num_residual_layers=cfg.network.num_residual_layers,
        num_filters=cfg.network.num_filters,
        conv_blocks_channel_lists=cfg.network.conv_blocks_channel_lists,
        action_space_size=cfg.network.action_space_size,
        policy_hidden_size=cfg.network.value_head_hidden_size
    ).to(device)
    return network

def create_breakout_env(cfg, render=False) -> gym.Env:
    """Create and initialize the Breakout environment."""
    gym.register_envs(ale_py)
    env = gym.make(
        "ALE/Breakout-v5",
        render_mode=cfg.env.render_mode if render else None,
        obs_type=cfg.env.observation_mode
    )
    env = CroppedBreakoutEnv(env)
    return env

# def get_breakout_game_result(env) -> float:
#     """Get the game result from the Breakout environment."""
#     # In Breakout, we can use the final reward as the game result
#     # Normalize to [-1, 1] range for consistency with other environments
#     return min(max(env.unwrapped.ale.lives() / 5.0, -1.0), 1.0)

# def get_breakout_legal_actions(env) -> list[int]:
#     """Get legal actions from the Breakout environment."""
#     # In Breakout, all actions are always legal
#     return list(range(env.action_space.n))
