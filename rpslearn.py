import os
import gym
from stable_baselines3 import PPO, DQN

from multiagentworld.envs.rpsgym.rps import RPSWeightedAgent
from multiagentworld.common.agents import OnPolicyAgent, OffPolicyAgent

env = gym.make("RPS-v0")
env.add_partner_policy(RPSWeightedAgent(0,1,1))
# env.add_partner_policy(OnPolicyAgent(PPO("MlpPolicy", env)))
env.add_partner_policy(OffPolicyAgent(DQN("MlpPolicy", env)))

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
# model.save("ppo_rps")
