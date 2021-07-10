import os
import gym
import liargym
import common

from stable_baselines3 import PPO

env = gym.make("LiarsDice-v0")
# env.add_partner_policy(liargym.DefaultLiarAgent())
env.add_partner_policy(common.OnPolicyAgent(PPO("MlpPolicy", env)))

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100000)
