from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

import gym
import numpy as np

from .agents import Agent


class MultiAgentEnv(gym.Env, ABC):
    """
    Base class for all Multi-agent (2-player) environments.

    :param partners: List of policies to choose from for the partner agent
    """

    def __init__(self, partners: List[Agent] = []):
        self.partners = partners
        self.partnerid = 0

    def add_partner_agent(self, agent: Agent) -> None:
        """
        Add agent to the list of potential partner agents

        :param agent: Agent to add
        """
        self.partners.append(agent)

    def set_partnerid(self, partnerid: int) -> None:
        """
        Set the current partner agent to use

        :param partnerid: Partnerid to use as current partner
        """
        assert(partnerid >= 0 and partnerid < len(self.partners))
        self.partnerid = partnerid
        
    def resample_partner(self) -> None:
        """ Resample the partner agent used """
        self.partnerid = np.random.randint(len(self.partners))

class TurnBasedEnv(MultiAgentEnv, ABC):
    """
    Base class for all 2-player turn-based games.

    In turn-based games, players take turns receiving observations and making
    actions.

    :param probegostart: Probability that the ego agent gets the first turn
    :param partners: List of policies to choose from for the partner agent
    """

    def __init__(self, probegostart: float = 0.5, partners: List[Agent] = []):
        super(TurnBasedEnv, self).__init__(partners=partners)
        self.probegostart = probegostart
        self.altmoved = False

    def step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], float, bool, Dict]:
        """
        Run one timestep from the perspective of the ego-agent. This involves
        calling the ego_step function and the alt_step function to get to the
        next observation of the ego agent.

        Accepts the ego-agent's action and returns a tuple of (observation,
        reward, done, info) from the perspective of the ego agent.

        :param action: An action provided by the ego-agent.

        :returns:
            observation: Ego-agent's next observation
            reward: Amount of reward returned after previous action
            done: Whether the episode has ended (need to call reset() if True)
            info: Extra information about the environment
        """
        altobs, rewA, done, info = self.ego_step(action)
        info['_partnerid'] = self.partnerid

        # if partner made an action, update with new reward
        if self.altmoved:
            self.partners[self.partnerid].update(rewA[1], done)

        if done:
            return None, rewA[0], done, info

        # perform partner's actions
        altaction = self.partners[self.partnerid].get_action(altobs)
        obs, rewB, done, info = self.alt_step(altaction)
        self.partners[self.partnerid].update(rewB[1], done)
        self.altmoved = True

        return obs, rewA[0] + rewB[0], done, info

    def reset(self) -> np.ndarray:
        """
        Reset environment to an initial state and return the first observation
        for the ego agent.

        Depending on the value of probegostart, the partner agent may make the
        first move, in which case reset gives the ego's observation after this
        action. The partner policy is also resampled.

        :returns: Ego-agent's first observation
        """
        self.resample_partner()
        egostart = (np.random.rand() < self.probegostart)
        obs = self.multi_reset(egostart)
        self.altmoved = not egostart

        if egostart:
            return obs

        # perform partner's action if ego is not starting
        altaction = self.partners[self.partnerid].get_action(obs)
        newobs, rewB, done, info = self.alt_step(altaction)
        self.partners[self.partnerid].update(rewB[1], done)

        assert newobs is not None

        return newobs

    @abstractmethod
    def ego_step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], Tuple[float, float], bool, Dict]:
        """
        Perform the ego-agent's action and return a tuple of (partner's
        observation, both rewards, done, info).

        This function is called by the `step` function along with alt-step.

        :param action: An action provided by the ego-agent.

        :returns:
            partner observation: Partner's next observation
            rewards: Tuple representing the rewards of both agents (ego, alt)
            done: Whether the episode has ended
            info: Extra information about the environment
        """

    @abstractmethod
    def alt_step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], Tuple[float, float], bool, Dict]:
        """
        Perform the partner's action and return a tuple of (ego's observation,
        both rewards, done, info).

        This function is called by the `step` function along with ego-step.

        :param action: An action provided by the partner.

        :returns:
            ego observation: Ego-agent's next observation
            rewards: Tuple representing the rewards of both agents (ego, alt)
            done: Whether the episode has ended
            info: Extra information about the environment
        """

    @abstractmethod
    def multi_reset(self, egofirst: bool) -> np.ndarray:
        """
        Reset the environment and give the observation of the starting agent
        (based on the value of `egofirst`).

        This function is called by the `reset` function.

        :param egofirst: True if the ego has the first turn, False otherwise
        :returns: The observation for the starting agent (ego if `egofirst` is
            True, and the partner's observation otherwise)
        """


class SimultaneousEnv(MultiAgentEnv, ABC):
    """
    Base class for all 2-player simultaneous games.

    :param partners: List of policies to choose from for the partner agent
    """

    def __init__(self, partners: List[Agent] = []):
        super(SimultaneousEnv, self).__init__(partners)
        self.altobs: Optional[np.ndarray] = None

    def step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], float, bool, Dict]:
        """
        Run one timestep from the perspective of the ego-agent. This involves
        calling the multi_step function.

        Accepts the ego-agent's action and returns a tuple of (observation,
        reward, done, info) from the perspective of the ego agent.

        :param action: An action provided by the ego-agent.

        :returns:
            observation: Ego-agent's next observation
            reward: Amount of reward returned after previous action
            done: Whether the episode has ended (need to call reset() if True)
            info: Extra information about the environment
        """
        altaction = self.partners[self.partnerid].get_action(self.altobs)
        fullobs, fullreward, done, info = self.multi_step(action, altaction)
        info['_partnerid'] = self.partnerid

        self.altobs = fullobs[1]
        self.partners[self.partnerid].update(fullreward[1], done)
        return fullobs[0], fullreward[0], done, info

    def reset(self) -> np.ndarray:
        """
        Reset environment to an initial state and return the first observation
        for the ego agent.

        :returns: Ego-agent's first observation
        """
        self.resample_partner()
        fullobs = self.multi_reset()
        self.altobs = fullobs[1]
        return fullobs[0]

    @abstractmethod
    def multi_step(
                    self,
                    ego_action: np.ndarray,
                    alt_action: np.ndarray
                ) -> Tuple[Tuple[Optional[np.ndarray], Optional[np.ndarray]],
                           Tuple[float, float], bool, Dict]:
        """
        Perform the ego-agent's and partner's actions. This function returns a
        tuple of (observations, both rewards, done, info).

        This function is called by the `step` function.

        :param ego_action: An action provided by the ego-agent.
        :param alt_action: An action provided by the partner.

        :returns:
            observations: Tuple representing the next observations (ego, alt)
            rewards: Tuple representing the rewards of both agents (ego, alt)
            done: Whether the episode has ended
            info: Extra information about the environment
        """

    @abstractmethod
    def multi_reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset the environment and give the observation of both agents.

        This function is called by the `reset` function.

        :returns: The observations both agents
        """
