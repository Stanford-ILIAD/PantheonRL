import os
import gym
from stable_baselines3 import PPO

from multiagentworld.envs.liargym.liar import LiarDefaultAgent
from multiagentworld.common.agents import OnPolicyAgent, OffPolicyAgent
from multiagentworld.common.wrappers import TurnBasedFrameStack

# env = TurnBasedFrameStack(gym.make("LiarsDice-v0"), numframes=3)
env = gym.make("LiarsDice-v0")
# env.add_partner_agent(LiarDefaultAgent())
env.add_partner_agent(OnPolicyAgent(PPO("MlpPolicy", env)))

model = PPO("MlpPolicy", env, verbose=1)
wrappedmodel = OnPolicyAgent(model)
wrappedmodel.learn(total_timesteps=100000)
