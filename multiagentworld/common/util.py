from typing import Tuple, Union

import numpy as np
import torch as th

import gym
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common import policies
from stable_baselines3.common.utils import obs_as_tensor


class SpaceException(Exception):
    """ Raise when an illegal Space is used """


def get_space_size(space: Space) -> int:
    """ Return the length of a space """
    if isinstance(space, Box):
        return len(space.low)
    elif isinstance(space, Discrete):
        return 1
    elif isinstance(space, MultiBinary):
        return space.n
    elif isinstance(space, MultiDiscrete):
        return len(space.nvec)
    else:
        raise SpaceException


def calculate_space(space: Space, numframes: int) -> Space:
    """ Return the new space if the observation is framestacked """
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
        raise SpaceException


def get_default_obs(env: gym.Env):
    """ Return the default observation for a given environment """
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
        raise SpaceException


def action_from_policy(
    obs: np.ndarray,
    policy: policies.ActorCriticPolicy
) -> Tuple[np.ndarray, th.Tensor, th.Tensor]:
    """
    Return the action, values, and log_probs given an observation and policy

    : param obs: Numpy array representing the observation
    : param policy: The actor-critic policy

    : returns: The action, values, and log_probs from the policy
    """
    obs = obs.reshape((-1,) + policy.observation_space.shape)
    with th.no_grad():
        # Convert to pytorch tensor or to TensorDict
        obs_tensor = obs_as_tensor(obs, policy.device)
        actions, values, log_probs = policy.forward(obs_tensor)

    return actions.cpu().numpy(), values, log_probs


def clip_actions(
    actions: np.ndarray,
    policy: Union[policies.ActorCriticPolicy, BaseAlgorithm]
) -> np.ndarray:
    """
    Return the actions clipped by the action space of the policy

    : param actions: Raw action(could be outside of the action space)
    : param policy: The policy to use to clip the actions

    : returns: The actions clipped by the policy
    """
    if isinstance(policy.action_space, gym.spaces.Box):
        actions = np.clip(actions, policy.action_space.low,
                          policy.action_space.high)
    return actions


def resample_noise(model: BaseAlgorithm, n_steps: int) -> None:
    """
    Resample the model's noise given the number of active steps.

    : param model: The model, which might use sde
    : param n_steps: The number of timesteps the model has been active
    """
    if model.use_sde and model.sde_sample_freq > 0 and \
            n_steps % model.sde_sample_freq == 0:
        model.policy.reset_noise(model.env.num_envs)


class FeedForward32Policy(policies.ActorCriticPolicy):
    """A feed forward policy network with two hidden layers of 32 units.
    This matches the IRL policies in the original AIRL paper.
    Note: This differs from stable_baselines3 ActorCriticPolicy in two ways: by
    having 32 rather than 64 units, and by having policy and value networks
    share weights except at the final layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[32, 32])
