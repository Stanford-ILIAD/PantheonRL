"""
Testing whether agents work end-to-end.

RPS: Discrete (3) actions, Discrete (1) observation
LiarsDice:
BlockEnv: MultiDiscrete action, MultiDiscrete observation

"""
import pytest

import gymnasium as gym
from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent, StaticPolicyAgent, RecordingAgentWrapper


def generate_PPO_agent(env_name):
    try:
        env = gym.make(env_name)
        partner = OnPolicyAgent(PPO('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=0, n_steps=64))
        env.unwrapped.add_partner_agent(partner)
        ego = PPO('MlpPolicy', env, verbose=0, n_steps=64)
        ego.learn(total_timesteps=1000)
        return partner.model
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("env_name", ['RPS-v0', 'LiarsDice-v0', 'BlockEnv-v0', 'BlockEnv-v1'])
def test_static_policy(env_name):
    try:
        pretrained = generate_PPO_agent(env_name)
        env = gym.make(env_name)
        partner = StaticPolicyAgent(pretrained.policy)
        env.unwrapped.add_partner_agent(partner)
        ego = PPO('MlpPolicy', env, verbose=0, n_steps=64)
        ego.learn(total_timesteps=1000)
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("env_name", ['RPS-v0', 'LiarsDice-v0', 'BlockEnv-v0', 'BlockEnv-v1'])
def test_recording_policy(env_name):
    try:
        env = gym.make(env_name)
        partner = RecordingAgentWrapper(OnPolicyAgent(PPO('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=0, n_steps=64)))
        env.unwrapped.add_partner_agent(partner)
        ego = PPO('MlpPolicy', env, verbose=0, n_steps=64)
        ego.learn(total_timesteps=1000)
        return partner.get_transitions()
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"
