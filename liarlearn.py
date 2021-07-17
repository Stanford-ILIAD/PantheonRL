import os
import gym
from stable_baselines3 import PPO

from multiagentworld.envs.liargym.liar import LiarDefaultAgent
from multiagentworld.common.agents import OnPolicyAgent, OffPolicyAgent

env = gym.make("LiarsDice-v0")
env.add_partner_policy(LiarDefaultAgent())
env.add_partner_policy(OnPolicyAgent(PPO("MlpPolicy", env)))

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100000)
