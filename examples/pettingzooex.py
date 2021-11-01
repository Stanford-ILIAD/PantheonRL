"""
This is a simple example training script for PettingZoo environments. It also
demonstrates how to handle environments where the partners have different
observation and action spaces.

To run this script, remember to first install pettingzoo's classic environments
via `pip install "pettingzoo[classic]"`

You can also swap out tictactoe_v3 with some other classic environment
"""

from pettingzoo.classic import tictactoe_v3 as e

from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent
from pantheonrl.envs.pettingzoo import PettingZooAECWrapper

# We have a simple wrapper class that converts PettingZoo environments to
# work with our framework.
#
# WARNING: PettingZoo environments with complex spaces may not be directly
# compatible with our agents.
env = PettingZooAECWrapper(e.env())


print(env.n_players)
# PettingZoo has many multi-player environments. To ensure that each agent
# understands their specific observation/action space, use the getDummyEnv
# function.
for i in range(env.n_players - 1):
    partner = OnPolicyAgent(PPO('MlpPolicy', env.getDummyEnv(i), verbose=1))

    # The second parameter ensures that the partner is assigned to a certain
    # player number. Forgetting this parameter would mean that all of the
    # partner agents can be picked as `player 2`, but none of them can be
    # picked as `player 3`.
    env.add_partner_agent(partner, player_num=i + 1)

ego = PPO('MlpPolicy', env, verbose=1)
ego.learn(total_timesteps=100000)
