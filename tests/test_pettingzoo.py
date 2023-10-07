"""
Testing whether algorithms work end-to-end.

RPS: Discrete (3) actions, Discrete (1) observation
LiarsDice:
BlockEnv: MultiDiscrete action, MultiDiscrete observation

"""
import pytest

import gymnasium as gym
from stable_baselines3 import PPO

from pantheonrl import OnPolicyAgent
from pantheonrl.envs.pettingzoo import PettingZooAECWrapper

import pettingzoo.classic
import pettingzoo.butterfly

def make_env(option):
    rgb_input = False
    if option == 0:
        env = pettingzoo.classic.connect_four_v3.env()
    elif option == 1:
        env = pettingzoo.classic.gin_rummy_v4.env()
    elif option == 2:
        env = pettingzoo.classic.go_v5.env()
    elif option == 3:
        env = pettingzoo.classic.leduc_holdem_v4.env()
    elif option == 4:
        env = pettingzoo.classic.rps_v2.env()
    elif option == 5:
        env = pettingzoo.classic.texas_holdem_no_limit_v6.env()
    elif option == 6:
        env = pettingzoo.classic.texas_holdem_v4.env()
    elif option == 7:
        env = pettingzoo.classic.tictactoe_v3.env()

    elif option == 8:
        env = pettingzoo.butterfly.knights_archers_zombies_v10.env()

    elif option == 9:
        env = pettingzoo.butterfly.cooperative_pong_v5.env()
    elif option == 10:
        env = pettingzoo.butterfly.pistonball_v6.env()
    return PettingZooAECWrapper(env)


@pytest.mark.timeout(60)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("option", list(range(9)))
def test_PPO(option):
    try:
        env = make_env(option)
        for i in range(env.n_players - 1):
            partner = OnPolicyAgent(PPO('MlpPolicy', env.get_dummy_env(i+1), verbose=0, n_steps=64))

            env.unwrapped.add_partner_agent(partner, player_num=i+1)
        ego = PPO('MlpPolicy', env, verbose=0, n_steps=64)
        ego.learn(total_timesteps=128)
    except Exception as e:
        assert False, f"Exception raised on {option}: {e}"

