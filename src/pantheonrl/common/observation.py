"""
Definition of the Observation type and related functions.
"""
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class Observation:
    """
    Representation of a single observation provided by an environment.

    :param obs: the (partial) observation that an agent receives
    :param state: the full state information
    :param action_mask: a mask specifying what actions are legal.
      If it is None, all actions are permitted
    """

    obs: np.ndarray
    """ The (partial) observation to choose actions """
    state: np.ndarray
    """ The full state information, typically used for value functions """
    action_mask: Optional[np.ndarray] = None
    """ The mask of legal actions """

    def __init__(
        self,
        obs: np.ndarray,
        state: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
    ):
        self.obs = obs
        self.state = state if state is not None else obs
        self.action_mask = action_mask


def extract_obs(observation: Observation) -> np.ndarray:
    """
    Extract only the (partial) observation as a numpy array.
    Used for SB3 agents.

    :param observation: the full observation
    :returns: only the (partial) observation
    """
    return observation.obs


def extract_partial_obs(
    observation: Observation,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract the (partial) observation and the action mask.
    Used for MAPPO agents.

    :param observation: the full observation
    :returns: tuple of (partial) observation and action mask
    """
    return (observation.obs, observation.action_mask)
