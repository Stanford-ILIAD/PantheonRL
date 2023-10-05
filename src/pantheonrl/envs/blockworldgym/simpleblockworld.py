"""
Simplified version of blockworld.
"""

# create a gym for a simplified version of blockworld
# which i can use as a starting point to create the normal
# blockworld gym environment

import gymnasium as gym
import numpy as np

from pantheonrl.common.agents import Agent
from pantheonrl.common.multiagentenv import TurnBasedEnv

GRIDLEN = 7  # block world in a 7 x 7 grid
NUM_BLOCKS = (
    5  # the number of blocks will be variable in the non-simplified version,
)
# but allows for a constant sized action space here
NUM_COLORS = 2
NO_COLOR = 0
BLUE = 1
RED = 2  # useful for if we add graphics later

NUM_TOKENS = 16  # number of tokens the planner has

PLANNER_ACTION_SPACE = gym.spaces.Discrete(
    NUM_TOKENS
)  # tokens that represent words
CONSTRUCTOR_ACTION_SPACE = gym.spaces.MultiDiscrete(
    [NUM_BLOCKS, NUM_COLORS + 1]
)
# in the simplified version, the constructor's action space is
# just coloring each block

# for each block, store h/v, coordinate, and color
blocklistformat = [2, GRIDLEN, GRIDLEN, NUM_COLORS + 1] * NUM_BLOCKS
# in the simplified version, constructor can see blocks
CONSTRUCTOR_OBS_SPACE = gym.spaces.MultiDiscrete(
    [NUM_TOKENS] + blocklistformat
)
# constructor's obs space and true colorings
PLANNER_OBS_SPACE = gym.spaces.MultiDiscrete(blocklistformat + blocklistformat)


def generate_grid_world(np_random):
    """
    Generates a random GRIDLEN x GRIDLEN world with NUM_BLOCKS blocks
    Will be replaced in the true version with their generate gridworld
    function, which has gravity/var blocks/etc
    """
    world = np.zeros((GRIDLEN, GRIDLEN))
    blocks_so_far = 0
    grid_world = []
    while blocks_so_far < NUM_BLOCKS:
        new_block = random_block(np_random)
        y = new_block[1]
        x = new_block[2]
        if new_block[0] == 0:
            # horizontal
            if world[y][x] == 1 or world[y][x + 1] == 1:
                continue
            world[y][x] = 1
            world[y][x + 1] = 1
        else:
            # vertical
            if world[y][x] == 1 or world[y + 1][x] == 1:
                continue
            world[y][x] = 1
            world[y + 1][x] = 1
        grid_world.append(new_block)
        blocks_so_far += 1
    return grid_world


def random_block(np_random):
    """
    Places random block in the grid
    """
    block = []
    if np_random.integers(0, 2) == 0:
        # horizontal
        block.append(0)
        x = np_random.integers(0, GRIDLEN - 1)
        y = np_random.integers(0, GRIDLEN)
    else:
        block.append(1)
        x = np_random.integers(0, GRIDLEN)
        y = np_random.integers(0, GRIDLEN - 1)
    block.append(y)
    block.append(x)
    block.append(np_random.integers(0, NUM_COLORS) + 1)
    return block


class SimpleBlockEnv(TurnBasedEnv):
    """ Simple blockworld environment. """

    def __init__(self):
        super().__init__(
            [PLANNER_OBS_SPACE, CONSTRUCTOR_OBS_SPACE],
            [PLANNER_ACTION_SPACE, CONSTRUCTOR_ACTION_SPACE],
            probegostart=1,
        )
        self.viewer = None
        self.gridworld = None
        self.constructor_obs = None
        self.last_token = None

    def multi_reset(self, egofirst):
        self.gridworld = generate_grid_world(self.np_random)
        self.constructor_obs = [
            [block[0], block[1], block[2], 0] for block in self.gridworld
        ]
        self.last_token = 0
        self.viewer = None
        return self._get_obs(egofirst)

    def _get_obs(self, isego):
        if isego:
            return np.array([self.gridworld, self.constructor_obs]).flatten()
        observations = [
            elem for block in self.constructor_obs for elem in block
        ]
        output = np.array(([self.last_token] + observations))
        return output

    def ego_step(self, action):
        self.last_token = action
        # the planner decides when done by taking action NUM_TOKENS - 1
        done = action == NUM_TOKENS - 1
        reward = [0, 0]
        if done:
            reward = self._get_reward()
        return self._get_obs(False), reward, done, {}

    def alt_step(self, action):
        self.constructor_obs[action[0]][3] = action[1]
        return self._get_obs(True), [0, 0], False, {}

    def _get_reward(self):
        # for simplified version, 100 * # colored correctly / total blocks
        # (in the actual one, use F1 score)
        correct_blocks = 0
        for i in range(NUM_BLOCKS):
            if self.gridworld[i][3] == self.constructor_obs[i][3]:
                correct_blocks += 1
        reward = 100 * correct_blocks / NUM_BLOCKS
        return [reward, reward]  # since they both get the same reward


class SBWEasyPartner(Agent):
    """ Easy partner in the simple blockworld """

    def get_action(self, obs):
        obs = obs.obs
        token = obs[0]
        if token > 10:
            token = token // 2
        # tokens 1 - 5 mean color the block at that index red
        if 1 <= token <= 5:
            return [token - 1, RED]
        # tokens 6 - 10 mean color the block at that index blue
        if 6 <= token <= 10:
            return [token - 8, BLUE]
        return [0, obs[4]]

    def update(self, reward, done):
        pass


class SBWDefaultAgent(Agent):
    """ Default partner in the simple blockworld """

    def get_action(self, obs):
        obs = obs.obs
        token = obs[0]
        if token == 0:  # do nothing
            return [0, obs[4]]

        blocks = np.reshape(obs[1:], (NUM_BLOCKS, 4))
        grid = self._gridfromobs(blocks)
        # tokens 1 - 7 mean find the first uncolored one in that row
        # and color it red
        if token <= 7:
            index = self._findfirstuncolored(grid, token - 1, blocks)
            if index != -1:
                return [index, RED]
        # tokens 8 - 14 mean find the first uncolored one in that row
        # and color it blue
        if token <= 14:
            index = self._findfirstuncolored(grid, token - 8, blocks)
            if index != -1:
                return [index, BLUE]
        # otherwise do nothing
        return [0, obs[4]]

    def _findfirstuncolored(self, grid, row, blocks):
        for space in grid[row]:
            if space != -1:
                if blocks[space][3] == 0:
                    return space
        return -1

    def _gridfromobs(self, blocks):
        grid = np.full((GRIDLEN, GRIDLEN), -1)
        for i, block in enumerate(blocks):
            y = block[1]
            x = block[2]
            grid[y][x] = i
            if block[0] == 0:  # horizontal
                grid[y][x + 1] = i
            else:
                grid[y + 1][x] = i
        return grid

    def update(self, reward, done):
        pass
