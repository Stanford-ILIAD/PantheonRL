import os
import gym
import rpsgym
import common
from stable_baselines3 import PPO

env = gym.make("RPS-v0")
# env.add_partner_policy(rpsgym.WeightedAgent(0,1,1))
env.add_partner_policy(common.OnPolicyAgent(PPO("MlpPolicy", env)))

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
# model.save("ppo_rps")
