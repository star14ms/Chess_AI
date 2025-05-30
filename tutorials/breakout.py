import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MCTS_nonboard.training_modules.breakout import CroppedBreakoutEnv

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode=None, obs_type="rgb")
# env = gym.make("ALE/Breakout-v5", render_mode='human', obs_type="rgb")

# Create wrapped environment
env = CroppedBreakoutEnv(env)
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

    # print(observation.shape)

    # plt.imshow(observation, cmap='gray')
    # plt.xticks(range(0, observation.shape[1], 20))
    # plt.yticks(range(0, observation.shape[0], 20))
    # plt.axis('on')
    # plt.show()

env.close()
