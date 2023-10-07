"""
Module defining the ADAP partner agent.
"""
from typing import Optional

import time

import copy
import sys

import torch

from gymnasium import spaces

import numpy as np

from pantheonrl.common.agents import OnPolicyAgent
from pantheonrl.common.observation import Observation

from stable_baselines3.common.utils import (
    safe_mean,
    obs_as_tensor,
)


from .adap_learn import ADAP
from .util import SAMPLERS
from .policies import AdapPolicy


class AdapAgent(OnPolicyAgent):
    """
    Agent representing an ADAP learning algorithm.

    The `get_action` and `update` functions are based on the `learn` function
    from ``OnPolicyAlgorithm``.

    :param model: Model representing the agent's learning algorithm
    :param log_interval: Optional log interval for policy logging
    :param working_timesteps: Estimate for number of timesteps to train for.
    :param callback: Optional callback fed into the OnPolicyAlgorithm
    :param tb_log_name: Name for tensorboard log
    """

    def __init__(
        self,
        model: ADAP,
        log_interval=None,
        working_timesteps=1000,
        callback=None,
        tb_log_name="AdapAgent",
        latent_syncer: Optional[AdapPolicy] = None,
    ):
        super().__init__(
            model, log_interval, working_timesteps, callback, tb_log_name
        )

        self.latent_syncer = latent_syncer

    def get_action(self, obs: Observation) -> np.ndarray:
        """
        Return an action given an observation.

        The agent saves the last transition into its buffer. It also updates
        the model if the buffer is full.

        :param obs: The observation to use
        :returns: The action to take
        """
        obs = obs.obs
        if not isinstance(obs, np.ndarray):
            obs = np.array([obs])
        callback = self.callback
        rollout_buffer = self.model.rollout_buffer
        if self.model.full_obs_shape is None:
            self.model.full_obs_shape = (
                rollout_buffer.obs_shape[0] + self.model.context_size,
            )

            rollout_buffer.obs_shape = tuple(self.model.full_obs_shape)
            rollout_buffer.reset()

        n_rollout_steps = self.model.n_steps

        if self.model.num_timesteps >= self.total_timesteps:
            self.callback.on_training_end()
            self.iteration = 0
            self.total_timesteps, self.callback = self.model._setup_learn(
                self.working_timesteps,
                self.original_callback,
                False,
                self.tb_log_name,
                False,
            )

            self.callback.on_training_start(locals(), globals())

        if self.n_steps >= n_rollout_steps:
            with torch.no_grad():
                values = self.model.policy.predict_values(
                    obs_as_tensor(obs, self.model.device).unsqueeze(0)
                )
            rollout_buffer.compute_returns_and_advantage(
                last_values=values, dones=self.model._last_episode_starts
            )
            self.old_buffer = copy.deepcopy(rollout_buffer)
            callback.update_locals(locals())
            callback.on_rollout_end()

            self.iteration += 1
            self.model._update_current_progress_remaining(
                self.model.num_timesteps, self.working_timesteps
            )

            if (
                self.log_interval is not None
                and self.iteration % self.log_interval == 0
            ):
                assert self.model.ep_info_buffer is not None
                time_elapsed = max(
                    (time.time_ns() - self.model.start_time) / 1e9,
                    sys.float_info.epsilon,
                )
                fps = int(
                    (
                        self.model.num_timesteps
                        - self.model._num_timesteps_at_start
                    )
                    / time_elapsed
                )
                self.model.logger.record(
                    "time/iterations", self.iteration, exclude="tensorboard"
                )
                if (
                    len(self.model.ep_info_buffer) > 0
                    and len(self.model.ep_info_buffer[0]) > 0
                ):
                    self.model.logger.record(
                        "rollout/ep_rew_mean",
                        safe_mean(
                            [
                                ep_info["r"]
                                for ep_info in self.model.ep_info_buffer
                            ]
                        ),
                    )
                    self.model.logger.record(
                        "rollout/ep_len_mean",
                        safe_mean(
                            [
                                ep_info["l"]
                                for ep_info in self.model.ep_info_buffer
                            ]
                        ),
                    )
                self.model.logger.record("time/fps", fps)
                self.model.logger.record(
                    "time/time_elapsed",
                    int(time_elapsed),
                    exclude="tensorboard",
                )
                self.model.logger.record(
                    "time/total_timesteps",
                    self.model.num_timesteps,
                    exclude="tensorboard",
                )
                self.model.logger.dump(step=self.model.num_timesteps)
            self.model.train()

            # Restarting
            self.model.policy.set_training_mode(False)
            self.n_steps = 0
            rollout_buffer.reset()
            if self.model.use_sde:
                self.model.policy.reset_noise(1)
            self.callback.on_rollout_start()

        if (
            self.model.use_sde
            and self.model.sde_sample_freq > 0
            and self.n_steps % self.model.sde_sample_freq == 0
        ):
            self.model.policy.reset_noise(1)

        with torch.no_grad():
            obs_tensor = obs_as_tensor(obs, self.model.device)
            actions, values, log_probs = self.model.policy(
                obs_tensor.unsqueeze(0)
            )
        actions = actions.cpu().numpy()
        clipped_actions = actions

        if isinstance(self.model.action_space, spaces.Box):
            clipped_actions = np.clip(
                actions,
                self.model.action_space.low,
                self.model.action_space.high,
            )

        self.in_progress_info["l"] += 1
        self.model.num_timesteps += 1
        self.n_steps += 1
        if isinstance(self.model.action_space, spaces.Discrete):
            actions = actions.reshape(-1, 1)
        print(obs.shape)
        obs = np.concatenate((np.reshape(obs, (1, -1)),
                              self.model.policy.get_context()),
                             axis=1)

        rollout_buffer.add(
            obs,
            actions,
            [0],
            self.model._last_episode_starts,
            values,
            log_probs,
        )
        return clipped_actions[0]

    def update(self, reward: float, done: bool) -> None:
        super().update(reward, done)

        if done and self.latent_syncer is None:
            sampled_context = SAMPLERS[self.model.context_sampler](
                ctx_size=self.model.context_size, num=1, use_torch=True
            )
            self.model.policy.set_context(sampled_context)
