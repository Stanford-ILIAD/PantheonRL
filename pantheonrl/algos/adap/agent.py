from typing import Optional

from collections import deque

import numpy as np
import torch as th

from pantheonrl.common.util import (action_from_policy, clip_actions,
                                    resample_noise)

from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.utils import safe_mean

from pantheonrl.common.agents import OnPolicyAgent
from .adap_learn import ADAP
from .util import SAMPLERS
from .policies import AdapPolicy


class AdapAgent(OnPolicyAgent):
    """
    Agent representing an ADAP learning algorithm.

    The `get_action` and `update` functions are based on the `learn` function
    from ``OnPolicyAlgorithm``.

    :param model: Model representing the agent's learning algorithm
    """

    def __init__(self,
                 model: ADAP,
                 log_interval=None,
                 tensorboard_log=None,
                 tb_log_name="AdapAgent",
                 latent_syncer: Optional[AdapPolicy] = None):
        self.model = model
        self._last_episode_starts = [True]
        self.n_steps = 0
        self.values: th.Tensor = th.empty(0)

        self.model.set_logger(configure_logger(
            self.model.verbose, tensorboard_log, tb_log_name))

        self.name = tb_log_name
        self.num_timesteps = 0
        self.log_interval = log_interval or (1 if model.verbose else None)
        self.iteration = 0
        self.model.ep_info_buffer = deque([{"r": 0, "l": 0}], maxlen=100)

        self.latent_syncer = latent_syncer

        buf = self.model.rollout_buffer
        self.model.full_obs_shape = (
            buf.obs_shape[0] + self.model.context_size,)
        buf.obs_shape = self.model.full_obs_shape
        buf.reset()

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        """
        Return an action given an observation.

        When `record` is True, the agent saves the last transition into its
        buffer. It also updates the model if the buffer is full.

        :param obs: The observation to use
        :param record: Whether to record the obs, action (True when training)
        :returns: The action to take
        """
        if self.latent_syncer is not None:
            self.model.policy.set_context(self.latent_syncer.get_context())

        buf = self.model.rollout_buffer

        # train the model if the buffer is full
        if record and self.n_steps >= self.model.n_steps:
            buf.compute_returns_and_advantage(
                last_values=self.values,
                dones=self._last_episode_starts[0]
            )

            if self.log_interval is not None and \
                    self.iteration % self.log_interval == 0:
                self.model.logger.record(
                    "name", self.name, exclude="tensorboard")
                self.model.logger.record(
                    "time/iterations", self.iteration, exclude="tensorboard")

                if len(self.model.ep_info_buffer) > 0 and \
                        len(self.model.ep_info_buffer[0]) > 0:
                    last_exclude = self.model.ep_info_buffer.pop()
                    rews = [ep["r"] for ep in self.model.ep_info_buffer]
                    lens = [ep["l"] for ep in self.model.ep_info_buffer]
                    self.model.logger.record(
                        "rollout/ep_rew_mean", safe_mean(rews))
                    self.model.logger.record(
                        "rollout/ep_len_mean", safe_mean(lens))
                    self.model.ep_info_buffer.append(last_exclude)

                self.model.logger.record(
                    "time/total_timesteps", self.num_timesteps,
                    exclude="tensorboard")
                self.model.logger.dump(step=self.num_timesteps)

            self.model.train()
            self.iteration += 1
            buf.reset()
            self.n_steps = 0

        resample_noise(self.model, self.n_steps)

        actions, values, log_probs = action_from_policy(obs, self.model.policy)

        # modify the rollout buffer with newest info
        obs = np.concatenate((np.reshape(obs, (1, -1)),
                              self.model.policy.get_context()),
                             axis=1)
        if record:
            obs_shape = self.model.policy.observation_space.shape
            act_shape = self.model.policy.action_space.shape
            buf.add(
                np.reshape(obs, (1,) + obs_shape),
                np.reshape(actions, (1,) + act_shape),
                [0],
                self._last_episode_starts,
                values,
                log_probs
            )

        self.n_steps += 1
        self.num_timesteps += 1
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
        super(AdapAgent, self).update(reward, done)

        if done and self.latent_syncer is None:
            sampled_context = SAMPLERS[self.model.context_sampler](
                ctx_size=self.model.context_size, num=1, torch=True)
            self.model.policy.set_context(sampled_context)
