import pytest

from stable_baselines3 import PPO

import gymnasium as gym

import overcookedgym

from pantheonrl.algos.adap.adap_learn import ADAP
from pantheonrl.algos.adap.policies import AdapPolicy, AdapPolicyMult
from pantheonrl.algos.adap.agent import AdapAgent


def make_env(option):
    if option == 0:
        env = gym.make('OvercookedMultiEnv-v0', layout_name='simple')
    elif option == 1:
        env = gym.make('RPS-v0')
    elif option == 2:
        env = gym.make('LiarsDice-v0')
    env.np_random, _ = gym.utils.seeding.np_random(0)
    return env


def run_standard(ALGO, timesteps, option, n_steps):
    env = make_env(option)
    ego = ALGO(AdapPolicy, env, n_steps=n_steps, verbose=0)
    env.unwrapped.ego_ind = 0
    partner = AdapAgent(ALGO(AdapPolicy, env, n_steps=n_steps, verbose=0), latent_syncer=ego)
    env.unwrapped.add_partner_agent(partner)

    ego.learn(total_timesteps=timesteps)

@pytest.mark.timeout(60)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("ALGO", [ADAP])
@pytest.mark.parametrize("epochs", [20])
@pytest.mark.parametrize("option", [0])
@pytest.mark.parametrize("n_steps", [40])
def test_onpolicy(ALGO, epochs, option, n_steps):
    run_standard(ALGO, n_steps * epochs, option, n_steps)

