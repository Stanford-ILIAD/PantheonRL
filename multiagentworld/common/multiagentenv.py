import gym
import numpy as np


class MultiAgentEnv(gym.Env):
    def __init__(self, partners=[]):
        self.partners = partners
        self.partnerid = 0

    def add_partner_policy(self, policy):
        self.partners.append(policy)

    def reset(self):
        self.partnerid = np.random.randint(len(self.partners))


class TurnBasedEnv(MultiAgentEnv):

    def __init__(self, probegostart=0.5, partners=[]):
        super(TurnBasedEnv, self).__init__(partners=partners)
        self.probegostart = probegostart

    def step(self, action):
        altobs, rewA, done, info = self.ego_step(action)

        self.partners[self.partnerid].update(rewA[1], done)

        if done:
            return None, rewA[0], done, info

        altaction = self.partners[self.partnerid].get_action(altobs)
        obs, rewB, done, info = self.alt_step(altaction)
        self.partners[self.partnerid].update(rewB[1], done)

        return obs, rewA[0] + rewB[0], done, info

    def reset(self):
        super(TurnBasedEnv, self).reset()
        egostart = np.random.rand() < self.probegostart
        obs = self.multi_reset(egostart)

        if not egostart:
            altaction = self.partners[self.partnerid].get_action(obs)
            obs, rewB, done, info = self.alt_step(altaction)
            self.partners[self.partnerid].update(rewB[1], done)

        return obs

    def ego_step(self, action):
        """
        Return partner's obs, both rewards, is done, and info
        """
        raise NotImplementedError

    def alt_step(self, action):
        """
        Return ego's obs, both rewards, is done, and info
        """
        raise NotImplementedError

    def multi_reset(self, egofirst):
        raise NotImplementedError


class SimultaneousEnv(MultiAgentEnv):

    def step(self, action):
        altaction = self.partners[self.partnerid].get_action(self.altobs)
        fullobs, fullreward, done, info = self.multi_step(action, altaction)

        self.altobs = fullobs[1]
        self.partners[self.partnerid].update(fullreward[1], done)
        return fullobs[0], fullreward[0], done, info

    def reset(self):
        super(SimultaneousEnv, self).reset()
        fullobs = self.multi_reset()
        self.altobs = fullobs[1]
        return fullobs[0]

    def multi_reset(self):
        raise NotImplementedError

    def multi_step(self, ego_action, alt_action):
        raise NotImplementedError
