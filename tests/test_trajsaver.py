"""
Testing whether algorithms work end-to-end.

RPS: Discrete (3) actions, Discrete (1) observation
LiarsDice:
BlockEnv: MultiDiscrete action, MultiDiscrete observation

"""
import pytest

import gymnasium as gym
from stable_baselines3 import PPO

import numpy as np

import overcookedgym

from pantheonrl import OnPolicyAgent
from pantheonrl.common.agents import RecordingAgentWrapper
from pantheonrl.common.trajsaver import TurnBasedTransitions, SimultaneousTransitions, TransitionsMinimal
from pantheonrl.common.wrappers import TurnBasedRecorder, SimultaneousRecorder


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("env_name", ['LiarsDice-v0'])
def test_turn_transitions(env_name, tmp_path):
    try:
        env = TurnBasedRecorder(gym.make(env_name).unwrapped)
        partner = OnPolicyAgent(PPO('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=0, n_steps=64))
        env.unwrapped.add_partner_agent(partner)
        ego = PPO('MlpPolicy', env, verbose=0, n_steps=64)
        ego.learn(total_timesteps=100)

        transitions0 = env.get_transitions()
        transitions0.write_transition(tmp_path / env_name)

        transitions1 = TurnBasedTransitions.read_transition(tmp_path / (env_name + ".npy"), env.observation_space, env.action_space)

        assert np.array_equal(transitions0.obs, transitions1.obs), "Ego obs differ"
        assert np.array_equal(transitions0.acts, transitions1.acts), "Ego acts differ"
        assert np.array_equal(transitions0.flags, transitions1.flags), "Flags differ"
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("env_name", ['RPS-v0'])
def test_simultaneous_transitions(env_name, tmp_path):
    try:
        env = SimultaneousRecorder(gym.make(env_name).unwrapped)
        partner = OnPolicyAgent(PPO('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=0, n_steps=64))
        env.unwrapped.add_partner_agent(partner)
        ego = PPO('MlpPolicy', env, verbose=0, n_steps=64)
        ego.learn(total_timesteps=100)

        transitions0 = env.get_transitions()
        transitions0.write_transition(tmp_path / env_name)

        transitions1 = SimultaneousTransitions.read_transition(tmp_path / (env_name + ".npy"), env.observation_space, env.action_space)

        assert np.array_equal(transitions0.egoobs, transitions1.egoobs), "Ego obs differ"
        assert np.array_equal(transitions0.egoacts, transitions1.egoacts), "Ego acts differ"
        assert np.array_equal(transitions0.altobs, transitions1.altobs), "Alt obs differ"
        assert np.array_equal(transitions0.altacts, transitions1.altacts), "Alt acts differ"
        assert np.array_equal(transitions0.flags, transitions1.flags), "Flags differ"
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("env_name", ['OvercookedMultiEnv-v0'])
def test_overcooked_transitions(env_name, tmp_path):
    try:
        env = SimultaneousRecorder(gym.make(env_name, layout_name="simple").unwrapped)
        partner = OnPolicyAgent(PPO('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=0, n_steps=64))
        env.unwrapped.add_partner_agent(partner)
        ego = PPO('MlpPolicy', env, verbose=0, n_steps=64)
        ego.learn(total_timesteps=100)

        transitions0 = env.get_transitions()
        transitions0.write_transition(tmp_path / env_name)

        transitions1 = SimultaneousTransitions.read_transition(tmp_path / (env_name + ".npy"), env.observation_space, env.action_space)

        assert np.array_equal(transitions0.egoobs, transitions1.egoobs), "Ego obs differ"
        assert np.array_equal(transitions0.egoacts, transitions1.egoacts), "Ego acts differ"
        assert np.array_equal(transitions0.altobs, transitions1.altobs), "Alt obs differ"
        assert np.array_equal(transitions0.altacts, transitions1.altacts), "Alt acts differ"
        assert np.array_equal(transitions0.flags, transitions1.flags), "Flags differ"
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("env_name", ['LiarsDice-v0'])
def test_turn_agent_transitions(env_name, tmp_path):
    try:
        env = TurnBasedRecorder(gym.make(env_name).unwrapped)
        partner = RecordingAgentWrapper(OnPolicyAgent(PPO('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=0, n_steps=64)))
        env.unwrapped.add_partner_agent(partner)
        ego = PPO('MlpPolicy', env, verbose=0, n_steps=64)
        ego.learn(total_timesteps=100)

        transitions0 = env.get_transitions().get_alt_transitions()
        transitions0.write_transition(tmp_path / env_name)

        transitions1 = partner.get_transitions()

        assert np.array_equal(transitions0.obs, transitions1.obs), "Ego obs differ"
        assert np.array_equal(transitions0.acts, transitions1.acts), "Ego acts differ"

        transitions2 = TransitionsMinimal.read_transition(tmp_path / (env_name + ".npy"), env.observation_space, env.action_space)
        assert np.array_equal(transitions0.obs, transitions2.obs), "Ego obs differ"
        assert np.array_equal(transitions0.acts, transitions2.acts), "Ego acts differ"
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("env_name", ['RPS-v0'])
def test_simultaneous_agent_transitions(env_name, tmp_path):
    try:
        env = SimultaneousRecorder(gym.make(env_name).unwrapped)
        partner = RecordingAgentWrapper(OnPolicyAgent(PPO('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=0, n_steps=64)))
        env.unwrapped.add_partner_agent(partner)
        ego = PPO('MlpPolicy', env, verbose=0, n_steps=64)
        ego.learn(total_timesteps=100)

        transitions0 = env.get_transitions().get_alt_transitions()
        transitions0.write_transition(tmp_path / env_name)

        transitions1 = partner.get_transitions()

        assert np.array_equal(transitions0.obs, transitions1.obs), "Ego obs differ"
        assert np.array_equal(transitions0.acts, transitions1.acts), "Ego acts differ"

        transitions2 = TransitionsMinimal.read_transition(tmp_path / (env_name + ".npy"), env.observation_space, env.action_space)
        assert np.array_equal(transitions0.obs, transitions2.obs), "Ego obs differ"
        assert np.array_equal(transitions0.acts, transitions2.acts), "Ego acts differ"
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("env_name", ['OvercookedMultiEnv-v0'])
def test_overcooked_agent_transitions(env_name, tmp_path):
    try:
        env = SimultaneousRecorder(gym.make(env_name, layout_name="simple").unwrapped)
        partner = RecordingAgentWrapper(OnPolicyAgent(PPO('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=0, n_steps=64)))
        env.unwrapped.add_partner_agent(partner)
        ego = PPO('MlpPolicy', env, verbose=0, n_steps=64)
        ego.learn(total_timesteps=100)

        transitions0 = env.get_transitions().get_alt_transitions()
        transitions0.write_transition(tmp_path / env_name)

        transitions1 = partner.get_transitions()

        assert np.array_equal(transitions0.obs, transitions1.obs), "Ego obs differ"
        assert np.array_equal(transitions0.acts, transitions1.acts), "Ego acts differ"

        transitions2 = TransitionsMinimal.read_transition(tmp_path / (env_name + ".npy"), env.observation_space, env.action_space)
        assert np.array_equal(transitions0.obs, transitions2.obs), "Ego obs differ"
        assert np.array_equal(transitions0.acts, transitions2.acts), "Ego acts differ"
    except Exception as e:
        assert False, f"Exception raised on {env_name}: {e}"
