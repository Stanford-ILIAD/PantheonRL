# create a gym for a simplified version of blockworld
# which i can use as a starting point to create the normal blockworld gym environment

import gym
import numpy as np

from pantheonrl.common.agents import Agent
from pantheonrl.common.multiagentenv import TurnBasedEnv, DummyEnv

GRIDLEN = 7  # block world in a 7 x 7 grid
NUM_BLOCKS = 5  # the number of blocks will be variable in the non-simplified version,
# but allows for a constant sized action space here
NUM_COLORS = 2
NO_COLOR = 0
BLUE = 1
RED = 2  # useful for if we add graphics later

NUM_TOKENS = 16  # number of tokens the planner has

PLANNER_ACTION_SPACE = gym.spaces.Discrete(
    NUM_TOKENS)  # tokens that represent words
CONSTRUCTOR_ACTION_SPACE = gym.spaces.MultiDiscrete(
    [NUM_BLOCKS, NUM_COLORS + 1])
# in the simplified version, the constructor's action space is just coloring each block

# for each block, store h/v, coordinate, and color
blocklistformat = [2, GRIDLEN, GRIDLEN, NUM_COLORS + 1]*NUM_BLOCKS
# in the simplified version, constructor can see blocks
CONSTRUCTOR_OBS_SPACE = gym.spaces.MultiDiscrete([NUM_TOKENS]+blocklistformat)
# constructor's obs space and true colorings
PLANNER_OBS_SPACE = gym.spaces.MultiDiscrete(blocklistformat + blocklistformat)

PartnerEnv = DummyEnv(CONSTRUCTOR_OBS_SPACE, CONSTRUCTOR_ACTION_SPACE)


def generate_grid_world():
    # generates a random GRIDLEN x GRIDLEN world with NUM_BLOCKS blocks
    # will be replaced in the true version with their generate gridworld function, which has gravity/var blocks/etc
    world = np.zeros((GRIDLEN, GRIDLEN))
    blocks_so_far = 0
    grid_world = []
    while blocks_so_far < NUM_BLOCKS:
        new_block = random_block()
        y = new_block[1]
        x = new_block[2]
        if new_block[0] == 0:
            # horizontal
            if world[y][x] == 1 or world[y][x+1] == 1:
                continue
            world[y][x] = 1
            world[y][x+1] = 1
        else:
            # vertical
            if world[y][x] == 1 or world[y+1][x] == 1:
                continue
            world[y][x] = 1
            world[y+1][x] = 1
        grid_world.append(new_block)
        blocks_so_far += 1
    return grid_world


def random_block():
    block = []
    if np.random.randint(2) == 0:
        # horizontal
        block.append(0)
        x = np.random.randint(GRIDLEN - 1)
        y = np.random.randint(GRIDLEN)
    else:
        block.append(1)
        x = np.random.randint(GRIDLEN)
        y = np.random.randint(GRIDLEN - 1)
    block.append(y)
    block.append(x)
    block.append(np.random.randint(NUM_COLORS) + 1)
    return block


class SimpleBlockEnv(TurnBasedEnv):
    def __init__(self):
        super().__init__(probegostart=1)
        self.observation_space = PLANNER_OBS_SPACE
        self.partner_observation_space = CONSTRUCTOR_OBS_SPACE
        self.action_space = PLANNER_ACTION_SPACE
        self.partner_action_space = CONSTRUCTOR_ACTION_SPACE
        self.partner_env = PartnerEnv
        self.viewer = None

    def getDummyEnv(self, player_ind: int):
        return PartnerEnv if player_ind else self

    def multi_reset(self, egofirst):
        self.gridworld = generate_grid_world()
        self.constructor_obs = [[block[0], block[1], block[2], 0]
                                for block in self.gridworld]
        self.last_token = 0
        self.viewer = None
        return self.get_obs(egofirst)

    def get_obs(self, isego):
        if isego:
            return np.array([self.gridworld, self.constructor_obs]).flatten()
        else:
            observations = [
                elem for block in self.constructor_obs for elem in block]
            output = np.array(([self.last_token]+observations))
            return output

    def ego_step(self, action):
        self.last_token = action
        # the planner gets to decide when they are done by taking action NUM_TOKENS - 1
        done = action == NUM_TOKENS - 1
        reward = [0, 0]
        if done:
            reward = self.get_reward()
        return self.get_obs(False), reward, done, {}

    def alt_step(self, action):
        self.constructor_obs[action[0]][3] = action[1]
        return self.get_obs(True), [0, 0], False, {}

    def get_reward(self):
        # for simplified version, 100 * num blocks colored correctly / total blocks
        # (in the actual one, use F1 score)
        correct_blocks = 0
        for i in range(NUM_BLOCKS):
            if self.gridworld[i][3] == self.constructor_obs[i][3]:
                correct_blocks += 1
        reward = 100 * correct_blocks / NUM_BLOCKS
        return [reward, reward]  # since they both get the same reward

    # def render(self, mode="human"):
    #     screen_width = 700
    #     scale = screen_width/GRIDLEN
    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #         self.viewer = rendering.Viewer(screen_width, screen_width)
    #         self.block_renders = []
    #         for blockdata in self.constructor_obs:
    #             y, x = blockdata[1], blockdata[2]
    #             if blockdata[0] == 0: # horizontal
    #                 left, right, top, bottom = x*scale, (x+2)*scale, (GRIDLEN - y)*scale, (GRIDLEN - (y+1))*scale
    #             else: # vertical
    #                 left, right, top, bottom = x*scale, (x+1)*scale, (GRIDLEN - y)*scale, (GRIDLEN - (y+2))*scale
    #             newblock = rendering.FilledPolygon([(left, bottom), (left, top), (right, top), (right, bottom)])
    #             newblock.set_color(0.5, 0.5, 0.5)
    #             self.viewer.add_geom(newblock)
    #             self.block_renders.append(newblock)
    #         for blockdata in self.gridworld:
    #             y, x = blockdata[1], blockdata[2]
    #             if blockdata[0] == 0:  # horizontal
    #                 left, right, top, bottom = x * scale, (x + 2) * scale, (GRIDLEN - y) * scale, (
    #                             GRIDLEN - (y + 1)) * scale
    #             else:  # vertical
    #                 left, right, top, bottom = x * scale, (x + 1) * scale, (GRIDLEN - y) * scale, (
    #                             GRIDLEN - (y + 2)) * scale
    #             newblock = rendering.PolyLine([(left, bottom), (left, top), (right, top), (right, bottom)], close=True)
    #             newblock.set_linewidth(10)
    #             if blockdata[3] == RED:
    #                 newblock.set_color(0.98, 0.02, 0.02)
    #             if blockdata[3] == BLUE:
    #                 newblock.set_color(0.02, 0.02, 0.98)
    #             self.viewer.add_geom(newblock)
    #     for i in range(len(self.block_renders)):
    #         if self.constructor_obs[i][3] == RED:
    #             self.block_renders[i].set_color(0.98, 0.02, 0.02)
    #         if self.constructor_obs[i][3] == BLUE:
    #             self.block_renders[i].set_color(0.02, 0.02, 0.98)
    #     return self.viewer.render(return_rgb_array=mode == "rgb_array")


class SBWEasyPartner(Agent):
    def get_action(self, obs, recording=True):
        token = obs[0]
        if token > 10:
            token = token//2
        # tokens 1 - 5 mean color the block at that index red
        if 1 <= token <= 5:
            return [token - 1, RED]
        # tokens 6 - 10 mean color the block at that index blue
        if 6 <= token <= 10:
            return [token - 8, BLUE]
        else:
            return [0, obs[4]]

    def update(self, reward, done):
        pass


class SBWDefaultAgent(Agent):
    def get_action(self, obs, recording=True):
        token = obs[0]
        if token == 0:  # do nothing
            return [0, obs[4]]
        else:
            blocks = np.reshape(obs[1:], (NUM_BLOCKS, 4))
            grid = self.gridfromobs(blocks)
            # tokens 1 - 7 mean find the first uncolored one in that row and color it red
            if token <= 7:
                index = self.findfirstuncolored(grid, token-1, blocks)
                if index != -1:
                    return [index, RED]
            # tokens 8 - 14 mean find the first uncolored one in that row and color it blue
            if token <= 14:
                index = self.findfirstuncolored(grid, token-8, blocks)
                if index != -1:
                    return [index, BLUE]
            # otherwise do nothing
            return [0, obs[4]]

    def findfirstuncolored(self, grid, row, blocks):
        for space in grid[row]:
            if space != -1:
                if blocks[space][3] == 0:
                    return space
        return -1

    def gridfromobs(self, blocks):
        grid = np.full((GRIDLEN, GRIDLEN), -1)
        for i in range(len(blocks)):
            y = blocks[i][1]
            x = blocks[i][2]
            grid[y][x] = i
            if blocks[i][0] == 0:  # horizontal
                grid[y][x+1] = i
            else:
                grid[y+1][x] = i
        return grid

    def update(self, reward, done):
        pass
