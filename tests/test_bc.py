import pytest

import gymnasium as gym
from stable_baselines3 import PPO

import overcookedgym

from pantheonrl import OnPolicyAgent, StaticPolicyAgent
from pantheonrl.common.agents import RecordingAgentWrapper
from pantheonrl.algos.bc import BC


def make_env(option):
    if option == 0:
        env = gym.make('OvercookedMultiEnv-v0', layout_name='simple')
    elif option == 1:
        env = gym.make('RPS-v0')
    elif option == 2:
        env = gym.make('LiarsDice-v0')
    elif option == 3:
        env = gym.make('BlockEnv-v0')
    elif option == 4:
        env = gym.make('BlockEnv-v1')
    env.np_random, _ = gym.utils.seeding.np_random(0)
    return env


def run_standard(ALGO, timesteps, option, n_steps):
    env = make_env(option)
    ego = ALGO('MlpPolicy', env, n_steps=n_steps, verbose=0)
    partner = RecordingAgentWrapper(OnPolicyAgent(PPO('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=0, n_steps=64)))
    env.unwrapped.ego_ind = 0
    env.unwrapped.add_partner_agent(partner)

    ego.learn(total_timesteps=timesteps)
    return ego.policy, partner.get_transitions()


def do_bc(option, data):
    full_env = make_env(option)
    env = full_env.unwrapped.get_dummy_env(1)
    clone = BC(observation_space=env.observation_space,
               action_space=env.action_space,
               expert_data=data,
               l2_weight=0.2)

    clone.train(n_epochs=10)
    return clone


def do_test_standard(ALGO, timesteps, option, n_steps, clone):
    env = make_env(option)
    ego = ALGO('MlpPolicy', env, n_steps=n_steps, verbose=0)
    partner = StaticPolicyAgent(clone.policy)
    env.unwrapped.ego_ind = 0
    env.unwrapped.add_partner_agent(partner)

    ego.learn(total_timesteps=timesteps)


@pytest.mark.timeout(60)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("ALGO", [PPO])
@pytest.mark.parametrize("epochs", [1])
@pytest.mark.parametrize("option", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n_steps", [400])
def test_onpolicy(ALGO, epochs, option, n_steps):
    model1, rb1 = run_standard(ALGO, n_steps * epochs, option, n_steps)
    clone = do_bc(option, rb1)
    do_test_standard(ALGO, n_steps * epochs, option, n_steps, clone)
