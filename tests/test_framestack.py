"""
Testing whether algorithms work end-to-end.

RPS: Discrete (3) actions, Discrete (1) observation
LiarsDice:
BlockEnv: MultiDiscrete action, MultiDiscrete observation

"""
import pytest

import gymnasium as gym
from stable_baselines3 import PPO

import overcookedgym

from pantheonrl import OnPolicyAgent
from pantheonrl.common.wrappers import frame_wrap


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("env_name", ['RPS-v0', 'LiarsDice-v0', 'BlockEnv-v0', 'BlockEnv-v1'])
def test_framestack(env_name, tmp_path):
    try:
        env = frame_wrap(gym.make(env_name).unwrapped, 3)
        partner = OnPolicyAgent(PPO('MlpPolicy', env.get_dummy_env(1), verbose=0, n_steps=64))
        env.unwrapped.add_partner_agent(partner)
        ego = PPO('MlpPolicy', env, verbose=0, n_steps=64)
        ego.learn(total_timesteps=100)
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("env_name", ['OvercookedMultiEnv-v0'])
def test_overcooked(env_name, tmp_path):
    try:
        env = frame_wrap(gym.make(env_name, layout_name='simple').unwrapped, 2)
        partner = OnPolicyAgent(PPO('MlpPolicy', env.get_dummy_env(1), verbose=0, n_steps=64))
        env.unwrapped.add_partner_agent(partner)
        ego = PPO('MlpPolicy', env, verbose=0, n_steps=64)
        ego.learn(total_timesteps=100)
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"
