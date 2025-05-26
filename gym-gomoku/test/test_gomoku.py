import gymnasium as gym
import gym_gomoku
import time
env = gym.make('Gomoku15x15-v0')

env.reset()

done = False
while not done:
    action = env.action_space.sample() # sample without replacement
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    time.sleep(0.2)
    if done:
        print ("Game is Over")
        break

breakpoint()