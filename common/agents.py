from stable_baselines3 import PPO
import numpy as np
import gym
import torch as th
from stable_baselines3.common.utils import obs_as_tensor, configure_logger

class Agent:
    def get_action(self, obs, recording=True):
        raise NotImplementedError

    def update(self, reward, done):
        """
        Add new rewards and information to buffer
        """
        raise NotImplementedError


class OnPolicyAgent(Agent):
    def __init__(self, model):
        self.model = model
        self._last_episode_starts = [True]
        self.n_steps = 0

        self.model.set_logger(configure_logger())

    def get_action(self, obs, recording=True):
        obs = obs.reshape((-1,) + self.model.policy.observation_space.shape)
        if self.model.use_sde and self.model.sde_sample_freq > 0 and self.n_steps % self.model.sde_sample_freq == 0:
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
            clipped_actions = np.clip(actions, self.model.action_space.low, self.model.action_space.high)

        if recording:
            obs = np.reshape(obs, (1, -1))
            actions = np.reshape(actions, (1, -1))
            self.model.rollout_buffer.add(obs, actions, [0], self._last_episode_starts, values, log_probs)

        self.n_steps += 1
        self.values = values
        return clipped_actions[0]

    def update(self, reward, done):
        """
        Add new rewards and information to buffer
        """
        self._last_episode_starts = [done]
        self.model.rollout_buffer.rewards[self.model.rollout_buffer.pos - 1][0] += reward

        if self.n_steps == self.model.n_steps:
            self.model.rollout_buffer.compute_returns_and_advantage(last_values=self.values, dones=done)
            self.model.train()
            self.model.rollout_buffer.reset()
            self.n_steps = 0
            print("UPDATE")
