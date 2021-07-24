import numpy as np
import gym
import torch as th

from multiagentworld.common.trajsaver import TransitionsMinimal

from stable_baselines3.common.utils import (obs_as_tensor, configure_logger,
                                            should_collect_more_steps)


class Agent:
    def get_action(self, obs, recording=True):
        raise NotImplementedError

    def update(self, reward, done):
        """
        Add new rewards and information to buffer
        """
        raise NotImplementedError


class StaticPolicyAgent(Agent):
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env

    def get_action(self, obs, recording=True):
        obs = obs.reshape((-1,) + self.policy.observation_space.shape)
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(obs, self.policy.device)
            actions, values, log_probs = self.policy.forward(obs_tensor)
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.env.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.env.action_space.low,
                                      self.env.action_space.high)

        return clipped_actions[0]

    def update(self, reward, done):
        """
        Add new rewards and information to buffer
        """
        pass


class OnPolicyAgent(Agent):
    def __init__(self, model):
        self.model = model
        self._last_episode_starts = [True]
        self.n_steps = 0

        self.model.set_logger(configure_logger())

    def get_action(self, obs, recording=True):
        obs = obs.reshape((-1,) + self.model.policy.observation_space.shape)
        if self.model.use_sde and self.model.sde_sample_freq > 0 and \
                self.n_steps % self.model.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.model.policy.reset_noise(self.model.env.num_envs)
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(obs, self.model.device)
            actions, values, log_probs = self.model.policy.forward(obs_tensor)
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.model.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.model.action_space.low,
                                      self.model.action_space.high)

        if recording:
            obs = np.reshape(obs, (1, -1))
            actions = np.reshape(actions, (1, -1))
            self.model.rollout_buffer.add(obs, actions, [0],
                                          self._last_episode_starts, values,
                                          log_probs)

        self.n_steps += 1
        self.values = values
        return clipped_actions[0]

    def update(self, reward, done):
        """
        Add new rewards and information to buffer
        """
        buf = self.model.rollout_buffer
        self._last_episode_starts = [done]
        buf.rewards[buf.pos - 1][0] += reward

        if self.n_steps == self.model.n_steps:
            buf.compute_returns_and_advantage(
                last_values=self.values, dones=done)
            self.model.train()
            buf.reset()
            self.n_steps = 0


class OffPolicyAgent(Agent):
    def __init__(self, model):
        self.model = model
        self.episode_rewards, self.total_timesteps = [], []
        self.num_collected_steps, self.num_collected_episodes = 0, 0
        self.old_reward = 0
        self.old_done = False
        self.old_info = {}

        self.episode_reward, self.episode_timesteps = 0.0, 0
        self.old_buffer_action = None
        self.model.set_logger(configure_logger())

    def get_action(self, obs, recording=True):
        if recording:
            if self.old_buffer_action is not None:
                buf = self.model.replay_buffer
                buf.observations[buf.pos] = np.array(obs).copy()
                self.model._store_transition(
                    buf, self.old_buffer_action, obs, self.old_reward,
                    self.old_done, [self.old_info])

            if self.old_done:
                self.num_collected_episodes += 1
                self.model._episode_num += 1
                self.episode_rewards.append(self.episode_reward)
                self.total_timesteps.append(self.episode_timesteps)

                if self.model.action_noise is not None:
                    self.model.action_noise.reset()
                self.episode_reward, self.episode_timesteps = 0.0, 0

        obs = obs.reshape((-1,) + self.model.policy.observation_space.shape)
        if self.model.use_sde and self.model.sde_sample_freq > 0 and \
                self.n_steps % self.model.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.model.policy.reset_noise(self.model.env.num_envs)

        action, buffer_action = self.model._sample_action(
            self.model.learning_starts, self.model.action_noise)

        self.model.num_timesteps += 1
        self.episode_timesteps += 1
        self.num_collected_steps += 1

        # Rescale and perform action
        clipped_actions = action
        # Clip the actions to avoid out of bound error
        if isinstance(self.model.action_space, gym.spaces.Box):
            clipped_actions = np.clip(action, self.model.action_space.low,
                                      self.model.action_space.high)

        self.old_buffer_action = buffer_action
        self.old_reward = 0
        self.model._last_obs = obs
        return clipped_actions[0]

    def update(self, reward, done):
        """
        Add new rewards and information to buffer
        """
        self.episode_reward += reward

        self.old_done = done
        self.old_reward += reward

        if not should_collect_more_steps(self.model.train_freq,
                                         self.num_collected_steps,
                                         self.num_collected_episodes):
            gradient_steps = self.model.gradient_steps
            if gradient_steps <= 0:
                gradient_steps = self.num_collected_steps

            self.model.train(batch_size=self.model.batch_size,
                             gradient_steps=gradient_steps)

            self.episode_rewards, self.total_timesteps = [], []
            self.num_collected_steps, self.num_collected_episodes = 0, 0


class RecordingAgentWrapper(Agent):
    def __init__(self, realagent):
        self.realagent = realagent
        self.allobs = []
        self.allacts = []

    def get_action(self, obs, recording=True):
        action = self.realagent.get_action(obs, recording)
        self.allobs.append(obs)
        self.allacts.append(action)
        return action

    def update(self, reward, done):
        self.realagent.update(reward, done)

    def get_transitions(self) -> TransitionsMinimal:
        obs = np.array(self.allobs)
        acts = np.array(self.allacts)
        return TransitionsMinimal(obs, acts)
