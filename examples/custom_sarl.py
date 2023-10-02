"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""

import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO

import overcookedgym

from overcookedgym.overcooked_utils import LAYOUT_LIST

from threading import Thread


stable_baselines3.common.utils.set_random_seed(0)


layout = 'simple'
assert layout in LAYOUT_LIST

# Since pantheonrl's MultiAgentEnv is a subclass of the gym Env, you can
# register an environment and construct it using gym.make.
env = gym.make('OvercookedMultiEnv-v0', layout_name=layout)

dummy_env = env.unwrapped.construct_single_agent_interface(1)

# Before training your ego agent, you first need to add your partner agents
# to the environment. You can create adaptive partner agents using
# OnPolicyAgent (for PPO/A2C) or OffPolicyAgent (for DQN/SAC). If you set
# verbose to true for these agents, you can also see their learning progress
partner = PPO('MlpPolicy', dummy_env, verbose=1)


def learn_thread():
    try:
        while True:
            partner.learn(total_timesteps=partner.n_steps, reset_num_timesteps=False)
    except Exception:
        pass


t = Thread(target=learn_thread)
t.start()

# Finally, you can construct an ego agent and train it in the environment
ego = PPO('MlpPolicy', env, verbose=1)
ego.learn(total_timesteps=10000)

# You need to step the environment once more if you want to train the agent for
# the final epoch. Note that these final set of actions and observations do not
# matter, they just need to exist so the final call to learn happens
env.step(env.action_space.sample())
with dummy_env.obs_cv:
    dummy_env.dead = True
    dummy_env.obs_cv.notify()
t.join()
