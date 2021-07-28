# create a gym for a simplified version of blockworld
# which i can use as a starting point to create the normal blockworld gym environment

import gym
import numpy as np

# from multiagentworld.common.agents import Agent
from multiagentworld.common.multiagentenv import TurnBasedEnv

GRIDLEN = 7 # block world in a 7 x 7 grid
NUM_BLOCKS = 5 # the number of blocks will be variable in the non-simplified version, 
               # but allows for a constant sized action space here
NUM_COLORS = 2 
NO_COLOR = 0
BLUE = 1 
RED = 2 # useful for if we add graphics later

NUM_TOKENS = 20 # number of tokens the planner has

PLANNER_ACTION_SPACE = gym.spaces.Discrete(NUM_TOKENS) # tokens that represent words
CONSTRUCTOR_ACTION_SPACE = gym.spaces.MultiDiscrete([NUM_BLOCKS, NUM_COLORS]) 
# in the simplified version, the constructor's action space is just coloring each block

CONSTRUCTOR_OBS_SPACE # in the simplified version, constructor can see blocks
                                                                  # for each block, store coordinate, H/V, and color
PLANNER_OBS_SPACE  # constructor's obs space and true colorings

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
        if new_block[0] == 'h':
            if world[y][x] == 1 or world[y][x+1] == 1:
                continue
            world[y][x] = 1
            world[y][x+1] = 1
        else:
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
        block.append('h')
        x = np.random.randint(GRIDLEN - 1)
        y = np.random.randint(GRIDLEN)
    else:
        block.append('v')
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
    
    def multi_reset(self, egofirst):
        self.gridworld = generate_grid_world()
        self.constructor_obs = [[block[0], block[1], block[2], 0] for block in self.gridworld]
        self.lastToken = None
        return self.getObs(egofirst)
    
    def get_obs(self, isego):
        # TODO: check format of observations
        if isego:
            return [self.gridworld, self.constructor_obs]
        return [self.lastToken, self.constructor_obs]
    
    def ego_step(self, action):
        self.lastToken = action
        # the planner gets to decide when they are done by taking action 0
        return self.get_obs(False), self.get_reward(), action == 0, {}
    
    def alt_step(self, action):
        # TODO: should our action[1] space be {0,1} or {0,1,2}?
        self.constructor_obs[action[0]][3] = action[1] + 1
        return self.get_obs(True), self.get_reward(), False, {}
    
    def get_reward(self):
        # for simplified version, 100 * num blocks colored correctly / total blocks 
        # (in the actual one, use F1 score)
        correct_blocks = 0
        for i in range(NUM_BLOCKS):
            if self.gridworld[i][3] == self.constructor_obs[i][3]:
                correct_blocks += 1
        reward = 100 * correct_blocks / NUM_BLOCKS 
        return [reward, reward] # since they both get the same reward

