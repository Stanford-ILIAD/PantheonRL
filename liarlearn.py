import os
import gym
import liargym

from stable_baselines3 import PPO

env = gym.make("LiarsDice-v0")
env.add_partner_policy(liargym.DefaultLiarAgent())

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
# model.save("ppo_liar")
