"""
Definition of the Rock-paper-scissors environment.
"""
import gymnasium as gym
import numpy as np

from pantheonrl.common.agents import Agent
from pantheonrl.common.multiagentenv import SimultaneousEnv

ACTION_NAMES = ["ROCK", "PAPER", "SCISSORS"]
ACTION_SPACE = gym.spaces.Discrete(3)
OBS_SPACE = gym.spaces.Discrete(1)

NULL_OBS = 0


class RPSWeightedAgent(Agent):
    """
    Random RPS agent based on weights of each action.
    """

    def __init__(self, r=1, p=1, s=1, np_random=np.random):
        weight = r + p + s
        if weight == 0:
            self.c0 = 1.0 / 3
            self.c1 = 2.0 / 3
        else:
            self.c0 = r / weight
            self.c1 = (r + p) / weight
        self.np_random = np_random

    def get_action(self, obs):
        roll = self.np_random.rand()
        return 0 if roll < self.c0 else 1 if roll < self.c1 else 2

    def update(self, reward, done):
        pass


class RPSEnv(SimultaneousEnv):
    """
    Definition of the RPS environment.

    The observation is always 0, and the valid actions are 0, 1, and 2.
    """

    def __init__(self):
        super().__init__([OBS_SPACE] * 2, [ACTION_SPACE] * 2)
        self.history = []

    def multi_step(self, ego_action, alt_action):
        outcome = (ego_action - alt_action + 3) % 3
        outcome = -1 if outcome == 2 else outcome

        return (NULL_OBS, NULL_OBS), (outcome, -outcome), True, {}

    def multi_reset(self):
        return (NULL_OBS, NULL_OBS)