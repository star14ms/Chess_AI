import gymnasium as gym
import ale_py # Import ale_py to register Atari environments

gym.register_envs(ale_py)

# Create the Breakout environment
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array") # Use render_mode="human" to visualize
# env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array") # Use render_mode="human" to visualize
env.reset()

# Let's see some basic info about the environment
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

# Example of taking random actions
num_steps = 100
for _ in range(num_steps):
    action = env.action_space.sample() # sample random action
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()