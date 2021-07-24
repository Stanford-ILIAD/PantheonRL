from multiagentworld.common.multiagentenv import TurnBasedEnv, SimultaneousEnv
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete
import numpy as np


def add_obs(history, toadd, numframes):
    if len(history) == numframes:
        history.pop()

    history.insert(0, toadd)
    return np.array([val for obs in history for val in obs])


def calculate_space(space, numframes):
    if isinstance(space, Box):
        low = np.tile(space.low, numframes)
        high = np.tile(space.high, numframes)
        return Box(low, high, dtype=space.dtype)
    elif isinstance(space, Discrete):
        return MultiDiscrete([space.n] * numframes)
    elif isinstance(space, MultiBinary):
        return MultiBinary(space.n * numframes)
    elif isinstance(space, MultiDiscrete):
        return MultiDiscrete(list(space.nvec) * numframes)
    else:
        raise NotImplementedError


def get_default_obs(env):
    space = env.observation_space
    if isinstance(space, Box):
        return space.low
    elif isinstance(space, Discrete):
        return [0]
    elif isinstance(space, MultiBinary):
        return [0] * space.n
    elif isinstance(space, MultiDiscrete):
        return [0] * len(space.nvec)
    else:
        raise NotImplementedError


class TurnBasedFrameStack(TurnBasedEnv):

    def __init__(self, env, numframes, defaultobs=None):
        super(TurnBasedFrameStack, self).__init__(
            probegostart=env.probegostart, partners=env.partners)
        self.env = env
        self.numframes = numframes

        self.action_space = env.action_space
        self.observation_space = calculate_space(
            env.observation_space, numframes)

        self.defaultobs = get_default_obs(
            env) if defaultobs is None else list(defaultobs)

        self.egohistory = [self.defaultobs] * self.numframes
        self.althistory = [self.defaultobs] * self.numframes

    def ego_step(self, action):
        altobs, rews, done, info = self.env.ego_step(action)
        return add_obs(self.althistory, altobs, self.numframes), \
            rews, done, info

    def alt_step(self, action):
        egoobs, rews, done, info = self.env.alt_step(action)
        return add_obs(self.egohistory, egoobs, self.numframes), \
            rews, done, info

    def multi_reset(self, egofirst):
        newobs = self.env.multi_reset(egofirst)
        self.egohistory = [self.defaultobs] * self.numframes
        self.althistory = [self.defaultobs] * self.numframes
        return add_obs(self.egohistory if egofirst else self.althistory,
                       newobs, self.numframes)


class SimultaneousFrameStack(SimultaneousEnv):

    def __init__(self, env, numframes, defaultobs=None):
        super(SimultaneousFrameStack, self).__init__(partners=env.partners)
        self.env = env
        self.numframes = numframes

        self.action_space = env.action_space
        self.observation_space = calculate_space(
            env.observation_space, numframes)

        self.defaultobs = get_default_obs(
            env) if defaultobs is None else list(defaultobs)

        self.egohistory = [self.defaultobs] * numframes
        self.althistory = [self.defaultobs] * numframes

    def multi_step(self, ego_action, alt_action):
        obs, rews, done, info = self.env.multi_step(ego_action, alt_action)
        return (add_obs(self.egohistory, obs[0], self.numframes),
                add_obs(self.althistory, obs[1], self.numframes)), \
            rews, done, info

    def multi_reset(self):
        obs = self.env.multi_reset()
        self.egohistory = [self.defaultobs] * self.numframes
        self.althistory = [self.defaultobs] * self.numframes
        return (add_obs(self.egohistory, obs[0], self.numframes),
                add_obs(self.althistory, obs[1], self.numframes))
