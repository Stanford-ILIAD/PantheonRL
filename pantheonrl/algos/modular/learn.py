import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

class ModularAlgorithm(OnPolicyAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        
        # my additional arguments
        marginal_reg_coef : float = 0.0,
    ):

        super(ModularAlgorithm, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        
        self.marginal_reg_coef = marginal_reg_coef

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
            batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        
        # OnPolicyAlgorithm's _setup_model
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer

        self.rollout_buffer = [buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        ) for _ in range(self.policy.num_partners)]
        
        # PPO's _setup_model
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
        
    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int, partner_idx: int
    ) -> bool:
        """
        Collect rollouts using the current policy and fill a `RolloutBuffer`.

        :param env: (VecEnv) The training environment
        :param callback: (BaseCallback) Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: (RolloutBuffer) Buffer to fill with rollouts
        :param n_steps: (int) Number of experiences to collect per environment
        :return: (bool) True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        self._last_dones = None
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                #actions, values, log_probs = self.policy.forward(obs_tensor)
                actions, values, log_probs = self.policy.forward(obs_tensor, partner_idx=partner_idx)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            env.envs[0].set_partnerid(partner_idx)
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1
            self.num_timesteps += env.num_envs

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        rollout_buffer.compute_returns_and_advantage(values, dones=dones)

        callback.on_rollout_end()

        return True
    
    
    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []

        for partner_idx in range(self.policy.num_partners):
            # train for gradient_steps epochs
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                # Do a complete pass on the rollout buffer
                # for rollout_data in self.rollout_buffer.get(self.batch_size):
                for rollout_data in self.rollout_buffer[partner_idx].get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    # TODO: investigate why there is no issue with the gradient
                    # if that line is commented (as in SAC)
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    #values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions, partner_idx=partner_idx) 
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = log_prob.mean()
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    ###########
                    # Marginal Regularization
                    ###########
                    # each action_dist is a Distribution object containing self.batch_size observations
                    # dist.distribution.probs returns a tensor of shape (self.batch_size, self.action_space)
                    # dist.sample() returns a tensor of shape (self.batch_size)

                    # careful: must extract torch distribution object from stable_baseline Distribution object, otherwise old references get overwritten
                    main_logits, partner_logits = zip( *[self.policy.get_action_logits_from_obs(rollout_data.observations, partner_idx=idx) for idx in range(self.policy.num_partners)] )
                    main_logits = th.stack([logits for logits in main_logits]) # (num_partners, self.batch_size, self.action_space)
                    partner_logits = th.stack([logits for logits in partner_logits]) # (num_partners, self.batch_size, self.action_space)
                    composed_logits = main_logits + partner_logits

                    # Regularize main prob to be the marginals
                    # Wasserstein metric with unitary distances (for categorical actions)
                    main_probs = th.mean( th.exp(main_logits - main_logits.logsumexp(dim=-1, keepdim=True)), dim=0 )
                    composed_probs = th.mean( th.exp(composed_logits - composed_logits.logsumexp(dim=-1, keepdim=True)), dim=0 )
                    marginal_regularization_loss = th.mean(th.sum( th.abs(main_probs - composed_probs), dim=1))
                    ###########

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.marginal_reg_coef * marginal_regularization_loss

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

                all_kl_divs.append(np.mean(approx_kl_divs))

                if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                    print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                    break

        self._n_updates += self.n_epochs
        # explained_var = explained_variance(self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())
        
        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        # self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        # self.logger.record("train/loss", loss.item())
        # self.logger.record("train/explained_variance", explained_var)
        # if hasattr(self.policy, "log_std"):
        #     self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/clip_range", clip_range)
        # if self.clip_range_vf is not None:
        #     self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            for partner_idx in range(self.policy.num_partners):
                try:    self.env.envs[0].set_partnerid(partner_idx)
                except: 
                    print("unable to switch")
                    pass
                continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer[partner_idx], n_rollout_steps=self.n_steps, partner_idx=partner_idx)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self