"""
Testing whether algorithms work end-to-end.

RPS: Discrete (3) actions, Discrete (1) observation
LiarsDice:
BlockEnv: MultiDiscrete action, MultiDiscrete observation

"""
import pytest

import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN

from pantheonrl import OnPolicyAgent, OffPolicyAgent

import overcookedgym


LAYOUTS = ['simple', 'random1', 'random3', 'unident_s', 'random0']


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("env_name", LAYOUTS)
def test_PPO(env_name):
    try:
        env = gym.make('OvercookedMultiEnv-v0', layout_name=env_name)
        partner = OnPolicyAgent(PPO('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=0, n_steps=64))
        env.unwrapped.add_partner_agent(partner)
        ego = PPO('MlpPolicy', env, verbose=0, n_steps=64)
        ego.learn(total_timesteps=1000)
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("env_name", LAYOUTS)
def test_A2C(env_name):
    try:
        env = gym.make('OvercookedMultiEnv-v0', layout_name=env_name)
        partner = OnPolicyAgent(A2C('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=0, n_steps=5))
        env.unwrapped.add_partner_agent(partner)
        ego = A2C('MlpPolicy', env, verbose=0, n_steps=5)
        ego.learn(total_timesteps=1000)
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("env_name", LAYOUTS)
def test_DQN(env_name):
    try:
        env = gym.make('OvercookedMultiEnv-v0', layout_name=env_name)
        partner = OffPolicyAgent(DQN('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=0, train_freq=4))
        env.unwrapped.add_partner_agent(partner)
        ego = DQN('MlpPolicy', env, verbose=0, train_freq=4)
        ego.learn(total_timesteps=1000)
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"
