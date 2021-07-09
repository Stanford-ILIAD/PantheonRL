import os
import gym
import rpsgym

from stable_baselines3 import PPO

env = gym.make("RPS-v0")
env.add_partner_policy(rpsgym.WeightedAgent(0,1,1))

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
# model.save("ppo_rps")
