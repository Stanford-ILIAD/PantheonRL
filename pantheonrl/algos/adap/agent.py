from typing import Optional

import numpy as np
import torch as th

from pantheonrl.common.util import (action_from_policy, clip_actions,
                                         resample_noise)

from stable_baselines3.common.utils import configure_logger

from pantheonrl.common.agents import Agent
from .adap_learn import ADAP
from .util import SAMPLERS
from .policies import AdapPolicy


class AdapAgent(Agent):
    """
    Agent representing an on-policy learning algorithm (ex: A2C/PPO).

    The `get_action` and `update` functions are based on the `learn` function
    from ``OnPolicyAlgorithm``.

    :param model: Model representing the agent's learning algorithm
    """

    def __init__(self, model: ADAP, latent_syncer: Optional[AdapPolicy]):
        self.model = model
        self._last_episode_starts = [True]
        self.n_steps = 0
        self.values: th.Tensor = th.empty(0)

        self.latent_syncer = latent_syncer

        buf = self.model.rollout_buffer
        self.model.full_obs_shape = (
            buf.obs_shape[0] + self.model.context_size,)
        buf.obs_shape = self.model.full_obs_shape
        buf.reset()
        self.model.set_logger(configure_logger())

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        """
        Return an action given an observation.

        When `record` is True, the agent saves the last transition into its
        buffer. It also updates the model if the buffer is full.

        :param obs: The observation to use
        :param record: Whether to record the obs, action (True when training)
        :returns: The action to take
        """
        buf = self.model.rollout_buffer
        if self.latent_syncer is not None:
            self.model.policy.set_context(self.latent_syncer.get_context())

        # train the model if the buffer is full
        if record and self.n_steps >= self.model.n_steps:
            buf.compute_returns_and_advantage(
                last_values=self.values,
                dones=self._last_episode_starts[0]
            )
            self.model.train()
            buf.obs_shape = self.model.full_obs_shape
            buf.reset()
            self.n_steps = 0

        resample_noise(self.model, self.n_steps)

        actions, values, log_probs = action_from_policy(obs, self.model.policy)

        # modify the rollout buffer with newest info
        if record:
            buf.add(
                np.concatenate(
                                (np.reshape(obs, (1, -1)),
                                 self.model.policy.get_context()),
                                axis=1),
                np.reshape(actions, (1, -1)),
                [0],
                self._last_episode_starts,
                values,
                log_probs
            )

        self.n_steps += 1
        self.values = values
        return clip_actions(actions, self.model)[0]

    def update(self, reward: float, done: bool) -> None:
        """
        Add new rewards and done information.

        The rewards are added to buffer entry corresponding to the most recent
        recorded action.

        :param reward: The reward receieved from the previous action step
        :param done: Whether the game is done
        """
        buf = self.model.rollout_buffer
        self._last_episode_starts = [done]
        buf.rewards[buf.pos - 1][0] += reward

        if done and self.latent_syncer is None:
            sampled_context = SAMPLERS[self.model.context_sampler](
                ctx_size=self.model.context_size, num=1, torch=True)
            self.model.policy.set_context(sampled_context)

    def learn(self, **kwargs) -> None:
        """ Call the model's learn function with the given parameters """
        self.model._custom_logger = False
        self.model.learn(**kwargs)
