import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple

import gym

from .multiagentenv import TurnBasedEnv, SimultaneousEnv, MultiAgentEnv
from .trajsaver import (TurnBasedTransitions, SimultaneousTransitions,
                        MultiTransitions)
from .util import (calculate_space, get_default_obs)

# Flags for the TurnBasedRecorder wrapper
EGO_NOT_DONE = 0
ALT_NOT_DONE = 1
EGO_DONE = 2
ALT_DONE = 3

# Flags for the SimultaneousRecorder wrapper
NOT_DONE = 0
DONE = 1


def frame_wrap(env: MultiAgentEnv, numframes: int):
    if isinstance(env, TurnBasedEnv):
        return TurnBasedFrameStack(env, numframes)
    else:
        return SimultaneousFrameStack(env, numframes)


def recorder_wrap(env: MultiAgentEnv):
    if isinstance(env, TurnBasedEnv):
        return TurnBasedRecorder(env)
    else:
        return SimultaneousRecorder(env)


class HistoryQueue:
    """
    Ring buffer representing the saved history for the FrameStack wrappers.

    :param defaultelem: The default element for an empty buffer
    :param size: The length of the queue
    """

    def __init__(self, defaultelem: np.ndarray, size: int):
        self.defaultelem = defaultelem
        self.size = size
        self.pos = 0

        self.history: List[np.ndarray] = [defaultelem] * size

    def add(self, toadd: np.ndarray) -> np.ndarray:
        """
        Add the given value to the queue and return the new representation

        :param toadd: The new value to add. This overrides the oldest value
        :return: The new queue representation, where the first element is the
            most recently added element and the last element is the oldest
        """
        self.history[self.pos] = toadd
        ans = np.array([val for ind in range(self.size)
                        for val in self.history[self.pos - ind]])
        self.pos = (self.pos + 1) % self.size
        return ans

    def reset(self) -> None:
        """
        Reset the queue. This fills the buffer with the defaultelement.
        """
        self.history = [self.defaultelem] * self.size
        self.pos = 0


class MultiRecorder(ABC):
    """ Base Class for all Recorder Wrappers"""

    @abstractmethod
    def get_transitions(self) -> MultiTransitions:
        """ Get the transitions that have been recorded """


class TurnBasedRecorder(TurnBasedEnv, MultiRecorder):
    """
    Recorder for all turn-based environments

    :param env: The environment to record
    """

    def __init__(self, env: gym.Env):
        super(TurnBasedRecorder, self).__init__(
            probegostart=env.probegostart, partners=env.partners[0])
        self.env = env

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.allobs: List[np.ndarray] = []
        self.allacts: List[np.ndarray] = []
        self.flags: List[int] = []
        self.incomplete = False

    def ego_step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], Tuple[float, float], bool, Dict]:
        """
        This function calls the embedded environment's ego_step and records the
        action and new observation.
        """
        altobs, rews, done, info = self.env.ego_step(action)
        self.allacts.append(action)
        if not done:
            self.allobs.append(altobs)
            self.flags.append(EGO_NOT_DONE)
        else:
            self.flags.append(EGO_DONE)
            self.incomplete = False
        return altobs, rews, done, info

    def alt_step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], Tuple[float, float], bool, Dict]:
        """
        This function calls the embedded environment's alt_step and records the
        action and new observation.
        """
        egoobs, rews, done, info = self.env.alt_step(action)
        self.allacts.append(action)
        if not done:
            self.allobs.append(egoobs)
            self.flags.append(ALT_NOT_DONE)
        else:
            self.flags.append(ALT_DONE)
            self.incomplete = False
        return egoobs, rews, done, info

    def multi_reset(self, egofirst: bool) -> np.ndarray:
        """
        This function calls the embedded environment's multi_reset and records
        the new observation.
        """
        newobs = self.env.multi_reset(egofirst)
        if self.incomplete:
            self.allobs[-1] = newobs
        else:
            self.allobs.append(newobs)
        self.incomplete = True
        return newobs

    def get_transitions(self) -> TurnBasedTransitions:
        """ Return the recorded transitions """
        obsarray = np.array(self.allobs)
        if self.incomplete:
            obsarray = obsarray[:-1]
        return TurnBasedTransitions(
                    obsarray,
                    np.array(self.allacts),
                    np.array(self.flags)
                )


class SimultaneousRecorder(SimultaneousEnv, MultiRecorder):
    """
    Recorder for all turn-based environments

    :param env: The environment to record
    """

    def __init__(self, env):
        super(SimultaneousRecorder, self).__init__(partners=env.partners[0])
        self.env = env

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.allegoobs = []
        self.allegoacts = []
        self.allaltobs = []
        self.allaltacts = []
        self.allflags = []
        self.incomplete = False

    def multi_step(
                    self,
                    ego_action: np.ndarray,
                    alt_action: np.ndarray
                ) -> Tuple[Tuple[Optional[np.ndarray], Optional[np.ndarray]],
                           Tuple[float, float], bool, Dict]:
        """
        This function calls the embedded environment's multi_step and records
        the new actions and observations.
        """
        obs, rews, done, info = self.env.multi_step(ego_action, alt_action)
        self.allegoacts.append(ego_action)
        self.allaltacts.append(alt_action)
        if not done:
            self.allegoobs.append(obs[0])
            self.allaltobs.append(obs[1])
            self.allflags.append(NOT_DONE)
        else:
            self.allflags.append(DONE)
            self.incomplete = False
        return obs, rews, done, info

    def multi_reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function calls the embedded environment's multi_reset and records
        the new observations.
        """
        obs = self.env.multi_reset()
        self.allegoobs.append(obs[0])
        self.allaltobs.append(obs[1])
        self.incomplete = True
        return obs

    def get_transitions(self) -> SimultaneousTransitions:
        """ Return the recorded transitions """
        egoobsarr = np.array(self.allegoobs)
        altobsarr = np.array(self.allaltobs)
        if self.incomplete:
            egoobsarr = egoobsarr[:-1]
            altobsarr = altobsarr[:-1]
        return SimultaneousTransitions(
                    egoobsarr,
                    np.array(self.allegoacts),
                    altobsarr,
                    np.array(self.allaltacts),
                    np.array(self.allflags)
                )


class TurnBasedFrameStack(TurnBasedEnv):
    """
    Wrapper that stacks the observations of a turn-based environment.

    :param env: The environment to wrap
    :param numframes: The number of frames to stack for each observation
    :param defaultobs: The default observation that fills old segments of the
        frame stacks.
    :param altenv: The optional dummy environment representing the spaces of
        the partner agent.
    :param defaultaltobs: The default observation that fills old segments of
        the frame stacks for the partner agent.
    """

    def __init__(
                self,
                env: gym.Env,
                numframes: int,
                defaultobs: Optional[np.ndarray] = None,
                altenv: Optional[gym.Env] = None,
                defaultaltobs: Optional[np.ndarray] = None
            ):
        super(TurnBasedFrameStack, self).__init__(
            probegostart=env.probegostart, partners=env.partners[0])
        self.env = env
        self.numframes = numframes

        self.action_space = env.action_space
        self.observation_space = calculate_space(
            env.observation_space, numframes)

        if defaultobs is not None:
            defobs = defaultobs
        else:
            defobs = get_default_obs(env)

        if altenv is None:
            altenv = env

        if defaultaltobs is not None:
            defaltobs = defaultaltobs
        else:
            defaltobs = get_default_obs(altenv)

        self.egohistory = HistoryQueue(defobs, numframes)
        self.althistory = HistoryQueue(defaltobs, numframes)

    def ego_step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], Tuple[float, float], bool, Dict]:
        altobs, rews, done, info = self.env.ego_step(action)
        return self.althistory.add(altobs), rews, done, info

    def alt_step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], Tuple[float, float], bool, Dict]:
        egoobs, rews, done, info = self.env.alt_step(action)
        return self.egohistory.add(egoobs), rews, done, info

    def multi_reset(self, egofirst: bool) -> np.ndarray:
        newobs = self.env.multi_reset(egofirst)
        self.egohistory.reset()
        self.althistory.reset()

        if egofirst:
            return self.egohistory.add(newobs)
        else:
            return self.althistory.add(newobs)


class SimultaneousFrameStack(SimultaneousEnv):
    """
    Wrapper that stacks the observations of a simultaneous environment.

    :param env: The environment to wrap
    :param numframes: The number of frames to stack for each observation
    :param defaultobs: The default observation that fills old segments of the
        frame stacks.
    """

    def __init__(
                self,
                env: gym.Env,
                numframes: int,
                defaultobs: Optional[np.ndarray] = None
            ):
        super(SimultaneousFrameStack, self).__init__(partners=env.partners[0])
        self.env = env
        self.numframes = numframes

        self.action_space = env.action_space
        self.observation_space = calculate_space(
            env.observation_space, numframes)

        self.defaultobs = get_default_obs(
            env) if defaultobs is None else list(defaultobs)

        self.egohistory = HistoryQueue(self.defaultobs, self.numframes)
        self.althistory = HistoryQueue(self.defaultobs, self.numframes)

    def multi_step(
                    self,
                    ego_action: np.ndarray,
                    alt_action: np.ndarray
                ) -> Tuple[Tuple[Optional[np.ndarray], Optional[np.ndarray]],
                           Tuple[float, float], bool, Dict]:
        obs, rews, done, info = self.env.multi_step(ego_action, alt_action)
        return (self.egohistory.add(obs[0]),
                self.althistory.add(obs[1])), rews, done, info

    def multi_reset(self) -> Tuple[np.ndarray, np.ndarray]:
        obs = self.env.multi_reset()
        self.egohistory.reset()
        self.althistory.reset()
        return (self.egohistory.add(obs[0]), self.althistory.add(obs[1]))
