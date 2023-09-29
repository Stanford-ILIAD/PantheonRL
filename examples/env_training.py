"""
This is a simple example training script for PantheonRL.
"""

import gymnasium as gym
from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent

# Current environment options are: RPS-v0, LiarsDice-v0, BlockEnv-v0, BlockEnv-v1
env = 'BlockEnv-v1'

# Since pantheonrl's MultiAgentEnv is a subclass of the gym Env, you can
# register an environment and construct it using gym.make.
env = gym.make(env)

# Before training your ego agent, you first need to add your partner agents
# to the environment. You can create adaptive partner agents using
# OnPolicyAgent (for PPO/A2C) or OffPolicyAgent (for DQN/SAC). If you set
# verbose to true for these agents, you can also see their learning progress
partner = OnPolicyAgent(PPO('MlpPolicy', env.unwrapped.getDummyEnv(1), verbose=1))
env.unwrapped.add_partner_agent(partner)

# Finally, you can construct an ego agent and train it in the environment
ego = PPO('MlpPolicy', env, verbose=1)
ego.learn(total_timesteps=1000)
