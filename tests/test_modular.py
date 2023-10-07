import pytest

from stable_baselines3 import PPO

import gymnasium as gym

import overcookedgym

from pantheonrl.algos.modular.learn import ModularAlgorithm
from pantheonrl.algos.modular.policies import ModularPolicy
from pantheonrl import Agent, OnPolicyAgent
from pantheonrl.algos.bc import ConstantLRSchedule

class CounterAgent(Agent):
    def __init__(self, agent, idx):
        self.agent = agent
        self.steps = 0
        self.idx = idx

    def get_action(self, obs):
        self.steps += 1
        toreturn = self.agent.get_action(obs)
        # print("ACTION is", toreturn)
        return toreturn

    def update(self, reward, done):
        self.agent.update(reward, done)

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
    env.unwrapped.set_resample_policy("null")
    return env


def run_standard(ALGO, timesteps, option, n_steps):
    env = make_env(option)
    pkwargs = {"num_partners":8}
    ego = ModularAlgorithm(ModularPolicy, env, n_steps=n_steps, verbose=0, policy_kwargs=pkwargs)
    env.unwrapped.ego_ind = 0
    for i in range(12):
        partner = CounterAgent(OnPolicyAgent(ALGO('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=0, n_steps=64)), i)
        env.unwrapped.add_partner_agent(partner)

    ego.learn(total_timesteps=timesteps)

    print([env.unwrapped.partners[0][i].steps for i in range(12)])
    for i in range(12):
        if i < 8:
            assert env.unwrapped.partners[0][i].steps > 0
        else:
            assert env.unwrapped.partners[0][i].steps == 0

@pytest.mark.timeout(60)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("ALGO", [PPO])
@pytest.mark.parametrize("epochs", [20])
@pytest.mark.parametrize("option", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n_steps", [40])
def test_onpolicy(ALGO, epochs, option, n_steps):
    run_standard(ALGO, n_steps * epochs, option, n_steps)

