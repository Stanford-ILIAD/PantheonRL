"""
Definition of the Liar's dice environment.
"""

import gymnasium as gym
import numpy as np

from pantheonrl.common.agents import Agent
from pantheonrl.common.multiagentenv import TurnBasedEnv

N = 6  # Num sides per dice
M = 6  # Num dice per player

WIN = (1, -1)
LOSE = (-1, 1)

MAX_MOVES = 2 * M

BLUFF = [N, 2 * M - 1]
DEFAULT = [N, 0]

ACTION_SPACE = gym.spaces.MultiDiscrete([N + 1, 2 * M])
OBS_SPACE = gym.spaces.MultiDiscrete([M + 1] * N + [N + 1, 2 * M] * MAX_MOVES)


def _rand_roll():
    dice = []
    for i in range(M):
        dice.append(np.random.randint(N))
    return [dice.count(i) for i in range(N)]


class LiarDefaultAgent(Agent):
    """The default liar's dice agent"""

    def get_action(self, obs):
        obs = obs.obs
        obs = obs.tolist()
        hand = obs[:N]
        maxval = max(hand)
        count = hand.index(maxval)
        if obs[N] != N and obs[N + 1] > maxval:
            return np.array(BLUFF)
        return np.array([count, maxval])

    def update(self, reward, done):
        pass


class LiarEnv(TurnBasedEnv):
    """
    Definition of the Liar's dice environment.

    The observation is the current hand. The valid actions are to bluff or to
    state a belief in the total number of dice of a certain number.
    """

    def __init__(self, probegostart=0.5):
        super().__init__(
            [OBS_SPACE] * 2, [ACTION_SPACE] * 2, probegostart=probegostart
        )
        self.history = []
        self.egohand = None
        self.althand = None

    def _get_obs(self, isego):
        prevmove = self.history + DEFAULT * (
            MAX_MOVES - len(self.history) // 2
        )
        return np.array((self.egohand if isego else self.althand) + prevmove)

    def _sanitize_action(self, action):

        if len(self.history) != 0 and (
            action[1] <= self.history[1] or action[0] == N
        ):
            return BLUFF

        if len(self.history) == 0 and action[0] == N:
            return [0, 0]

        return action.tolist()

    def _eval_bluff(self):
        if len(self.history) == 0:
            return False

        side = self.history[0]
        trueans = self.egohand[side] + self.althand[side] - 1
        return self.history[1] > trueans

    def _player_step(self, action, isego):
        action = self._sanitize_action(action)
        if action == BLUFF:
            didwin = self._eval_bluff() == isego
            return self._get_obs(not isego), WIN if didwin else LOSE, True, {}
        self.history = action + self.history
        return self._get_obs(not isego), (0, 0), False, {}

    def ego_step(self, action):
        return self._player_step(action, True)

    def alt_step(self, action):
        return self._player_step(action, False)

    def multi_reset(self, egofirst):
        self.history = []
        self.egohand = _rand_roll()
        self.althand = _rand_roll()

        return self._get_obs(egofirst)
