"""
Module defining the ADAP partner agent.
"""
from typing import Optional

import numpy as np

from pantheonrl.common.agents import OnPolicyAgent
from pantheonrl.common.observation import Observation

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

        buf = self.model.rollout_buffer
        self.model.full_obs_shape = (
            buf.obs_shape[0] + self.model.context_size,
        )
        buf.obs_shape = self.model.full_obs_shape
        buf.reset()

    def get_action(self, obs: Observation) -> np.ndarray:
        """
        Return an action given an observation.

        The agent saves the last transition into its buffer. It also updates
        the model if the buffer is full.

        :param obs: The observation to use
        :returns: The action to take
        """
        if self.latent_syncer is not None:
            self.model.policy.set_context(
                self.latent_syncer.policy.get_context()
            )
        if not isinstance(obs.obs, np.ndarray):
            obs.obs = np.array([obs.obs])
        obs.obs = np.concatenate(
            (np.reshape(obs.obs, (1, -1)), self.model.policy.get_context()),
            axis=1,
        )
        return super().get_action(obs)

    def update(self, reward: float, done: bool) -> None:
        super().update(reward, done)

        if done and self.latent_syncer is None:
            sampled_context = SAMPLERS[self.model.context_sampler](
                ctx_size=self.model.context_size, num=1, use_torch=True
            )
            self.model.policy.set_context(sampled_context)
