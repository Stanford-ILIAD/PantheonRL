# the more complex version of blockworld, where the constructor doesn't see the blocks beforehand
import gym
import numpy as np

from multiagentworld.common.agents import Agent
from multiagentworld.common.multiagentenv import TurnBasedEnv
from multiagentworld.envs.blockworldgym.gridutils import HORIZONTAL, generate_random_world, gravity, place, matches

GRIDLEN = 7 # block world in a 7 x 7 grid
NUM_BLOCKS = 5 # the number of blocks will be variable in the non-simplified version, 
               # but allows for a constant sized action space here
NUM_COLORS = 2 
NO_COLOR = 0
BLUE = 1 
RED = 2 # useful for if we add graphics later

NUM_TOKENS = 16 # number of tokens the planner has

PLANNER_ACTION_SPACE = gym.spaces.Discrete(NUM_TOKENS) # tokens that represent words
CONSTRUCTOR_ACTION_SPACE = gym.spaces.MultiDiscrete([GRIDLEN, 2, NUM_COLORS]) # it can drop any block from the top, set h/v and color
# plus an extra option to do nothing

gridformat = [NUM_COLORS+1]*GRIDLEN*GRIDLEN
CONSTRUCTOR_OBS_SPACE = gym.spaces.MultiDiscrete([NUM_TOKENS]+gridformat)  # can see what the planner said and the "real world" grid
PLANNER_OBS_SPACE = gym.spaces.MultiDiscrete(gridformat + gridformat) # can see the planned grid and the "real world" grid

class BlockEnv(TurnBasedEnv):
    def __init__(self):
        super().__init__(probegostart=1)
        self.observation_space = PLANNER_OBS_SPACE
        self.partner_observation_space = CONSTRUCTOR_OBS_SPACE
        self.action_space = PLANNER_ACTION_SPACE
        self.partner_action_space = CONSTRUCTOR_ACTION_SPACE
        self.partner_env = PartnerEnv()
        self.viewer = None
    
    def multi_reset(self, egofirst):
        self.gridworld = generate_random_world(GRIDLEN, NUM_BLOCKS, NUM_COLORS)
        self.constructor_obs = np.zeros((GRIDLEN, GRIDLEN))
        self.last_token = 0
        return self.get_obs(egofirst)
    
    def get_obs(self, isego):
        if isego:
            return np.concatenate((self.gridworld, self.constructor_obs), axis=None)
        else:
            observations = list(np.flatten(self.constructor_obs))
            return [self.last_token] + observations

    def ego_step(self, action):
        self.last_token = action
        done = action==NUM_TOKENS-1
        reward = [0, 0]
        if done:
            reward = self.get_reward()
        return self.get_obs(False), reward, done, {}
    
    def alt_step(self, action):
        x, orientation, color = action[0], action[1], action[2]
        if not(orientation == HORIZONTAL and x == GRIDLEN-1):
            y = gravity(self.constructor_obs, orientation, x)
            if y != -1:
                place(self.constructor_obs, x, y, color, orientation)
        return self.get_obs(True), [0,0], False, {}
    
    def get_reward(self):
        # we use F1 score which is 2 * precision * recall / (precision + recall)
        # also = 2 * truepos / (selected + relevant)
        truepos = matches(self.constructor_obs, self.gridworld)
        selected = np.count_nonzero(self.constructor_obs)
        relevant = np.count_nonzero(self.gridworld)
        return 2 * truepos / (selected + relevant)

class PartnerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = CONSTRUCTOR_OBS_SPACE
        self.action_space = PLANNER_OBS_SPACE