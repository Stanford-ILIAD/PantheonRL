import gym
import numpy as np
from gym.envs.registration import register

ACTION_NAMES = ['ROCK', 'PAPER', 'SCISSORS']
ACTION_SPACE = gym.spaces.Discrete(3)
OBS_SPACE = gym.spaces.Discrete(1)

class AgentPolicy:
    def __init__(self, r=1, p=1, s=1, np_random=np.random):
        weight = r + p + s
        self.c0 = r / weight
        self.c1 = (r + p) / weight
        self.np_random = np_random

    def predict(self, obs):
        roll = self.np_random.rand()
        return 0 if roll < self.c0 else 1 if roll < self.c1 else 2

class RPSEnv(gym.Env):
    
    verbose = True
    
    def __init__(self):
        self.history = []
        self.otherPolicy = AgentPolicy()
        self.observation_space = OBS_SPACE
        self.action_space = ACTION_SPACE
    
    def getObs(self):
        return np.array([0])
    
    def step(self, action, otherAction=None):
        if otherAction is None:
            otherAction = self.otherPolicy.predict(self.getObs())
        
        outcome = (action - otherAction + 3) % 3
        outcome = -1 if outcome == 2 else outcome

        return self.getObs(), outcome, True, {}
    
    def reset(self):
        return self.getObs()

register(
    id='RPS-v0',
    entry_point='rpsgym.rps:RPSEnv'
)
