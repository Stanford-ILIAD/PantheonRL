"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""

import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO

import overcookedgym

from pantheonrl.common.agents import OnPolicyAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST

import torch
import random
import numpy as np
# random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# stable_baselines3.common.utils.set_random_seed(0)


layout = 'simple'
assert layout in LAYOUT_LIST

# Since pantheonrl's MultiAgentEnv is a subclass of the gym Env, you can
# register an environment and construct it using gym.make.
env = gym.make('OvercookedMultiEnv-v0', layout_name=layout)

ego = PPO('MlpPolicy', env, verbose=0)
env.ego_ind = 1
# Before training your ego agent, you first need to add your partner agents
# to the environment. You can create adaptive partner agents using
# OnPolicyAgent (for PPO/A2C) or OffPolicyAgent (for DQN/SAC). If you set
# verbose to true for these agents, you can also see their learning progress
partner = OnPolicyAgent(PPO('MlpPolicy', env, verbose=0))
env.unwrapped.add_partner_agent(partner)

# Finally, you can construct an ego agent and train it in the environment

print('ego start', sum([param.data.mean() for param in ego.policy.parameters()]))
print('partner start', sum([param.data.mean() for param in partner.model.policy.parameters()]))

agents, observations = env.unwrapped.n_reset()
ego.learn(total_timesteps=2048)
partner.get_action(observations[0])
print('ego end', sum([param.data.mean() for param in ego.policy.parameters()]))
print('partner end', sum([param.data.mean() for param in partner.model.policy.parameters()]))
