"""
The more complex version of blockworld, where the constructor doesn't see the blocks beforehand
"""
import gymnasium as gym
import numpy as np

from pantheonrl.common.agents import Agent
from pantheonrl.common.multiagentenv import TurnBasedEnv
from pantheonrl.envs.blockworldgym.gridutils import (
    HORIZONTAL,
    VERTICAL,
    generate_random_world,
    gravity,
    place,
    matches,
)

from pantheonrl.envs.blockworldgym import rendering

# import pantheonrl.envs.blockworldgym.rendering as rendering

GRIDLEN = 7  # block world in a 7 x 7 grid
NUM_BLOCKS = (
    5  # the number of blocks will be variable in the non-simplified version,
)
# but allows for a constant sized action space here

# make sure color and action space/resulting grid are consistent
NUM_COLORS = 2
BLUE = 1
RED = 2  # useful for if we add graphics later

NUM_TOKENS = 30  # number of tokens the planner has

PLANNER_ACTION_SPACE = gym.spaces.Discrete(
    NUM_TOKENS
)  # tokens that represent words
# it can drop any block from the top, set h/v and color
CONSTRUCTOR_ACTION_SPACE = gym.spaces.MultiDiscrete([GRIDLEN, 2, NUM_COLORS])
# plus an extra option to do nothing

gridformat = [NUM_COLORS + 1] * GRIDLEN * GRIDLEN
# can see what the planner said and the "real world" grid
CONSTRUCTOR_OBS_SPACE = gym.spaces.MultiDiscrete([NUM_TOKENS] + gridformat)
# can see the planned grid and the "real world" grid
PLANNER_OBS_SPACE = gym.spaces.MultiDiscrete(gridformat + gridformat)


class BlockEnv(TurnBasedEnv):
    """ Full blockworld environment. """

    def __init__(self):
        super().__init__(
            [PLANNER_OBS_SPACE, CONSTRUCTOR_OBS_SPACE],
            [PLANNER_ACTION_SPACE, CONSTRUCTOR_ACTION_SPACE],
            probegostart=1,
        )

        # using same structure as SimpleBlockEnv
        self.constructor_obs = None
        self.gridworld = None
        self.last_token = None
        self.viewer = None

    def multi_reset(self, egofirst):
        self.gridworld = generate_random_world(
            GRIDLEN, NUM_BLOCKS, NUM_COLORS, self.np_random
        )
        self.constructor_obs = np.zeros((GRIDLEN, GRIDLEN))
        self.last_token = 0
        self.viewer = None
        return self._get_obs(egofirst)

    def _get_obs(self, isego):
        if isego:
            return np.concatenate(
                (self.gridworld, self.constructor_obs), axis=None
            )
        observations = list(self.constructor_obs.flatten())
        return np.array(([self.last_token] + observations))

    def ego_step(self, action):
        self.last_token = action
        done = action == NUM_TOKENS - 1
        reward = 0
        if done:
            reward = self._get_reward()
        return self._get_obs(False), [reward, reward], done, {}

    def alt_step(self, action):
        x, orientation, color = action[0], action[1], action[2] + 1
        if not (orientation == HORIZONTAL and x == GRIDLEN - 1):
            y = gravity(self.constructor_obs, orientation, x)
            if y != -1:
                place(self.constructor_obs, x, y, color, orientation)
        return self._get_obs(True), [0, 0], False, {}

    def _get_reward(self):
        # we use F1 score which is 2 * precision * recall / (precision + recall)
        # also = 2 * truepos / (selected + relevant)
        truepos = matches(self.constructor_obs, self.gridworld)
        selected = np.count_nonzero(self.constructor_obs)
        relevant = np.count_nonzero(self.gridworld)
        return 2 * truepos / (selected + relevant)

    def render(self, mode="human"):
        screen_width = 700
        scale = screen_width / GRIDLEN
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_width)
            for i, row in enumerate(self.gridworld):
                for j, grid_block in enumerate(row):
                    left, right, top, bottom = (
                        j * scale,
                        (j + 1) * scale,
                        (GRIDLEN - i) * scale,
                        (GRIDLEN - (i + 1)) * scale,
                    )
                    newblock = rendering.PolyLine(
                        [
                            (left, bottom),
                            (left, top),
                            (right, top),
                            (right, bottom),
                        ],
                        close=True,
                    )
                    newblock.set_linewidth(10)
                    self.viewer.add_geom(newblock)
                    if grid_block == RED:
                        newblock.set_color(0.98, 0.02, 0.02)
                    elif grid_block == BLUE:
                        newblock.set_color(0.02, 0.02, 0.98)
        for i, row in enumerate(self.constructor_obs):
            for j, cons_block in enumerate(row):
                if not self.constructor_obs[i][j] == 0:
                    left, right, top, bottom = (
                        j * scale,
                        (j + 1) * scale,
                        (GRIDLEN - i) * scale,
                        (GRIDLEN - (i + 1)) * scale,
                    )
                    newblock = rendering.FilledPolygon(
                        [
                            (left, bottom),
                            (left, top),
                            (right, top),
                            (right, bottom),
                        ]
                    )
                    newblock.set_color(0.5, 0.5, 0.5)
                    self.viewer.add_geom(newblock)
                    if cons_block == RED:
                        newblock.set_color(0.98, 0.02, 0.02)
                    elif cons_block == BLUE:
                        newblock.set_color(0.02, 0.02, 0.98)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")


class DefaultConstructorAgent(Agent):
    """ The default Constructor partner agent. """

    def get_action(self, obs):
        obs = obs.obs
        token = int(obs[0])
        if token in (0, 29):
            return [GRIDLEN - 1, VERTICAL, 0]
        token -= 1
        color = token % 2
        token = token // 2
        orientation = token % 2
        token = token // 2
        x = token
        return [x, orientation, color]

    def update(self, reward, done):
        pass
