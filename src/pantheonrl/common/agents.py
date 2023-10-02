from abc import ABC, abstractmethod
from typing import List, Dict

from collections import deque
import time

import numpy as np
import torch as th

from .util import action_from_policy, clip_actions, resample_noise
from .trajsaver import TransitionsMinimal
from .observation import Observation

from stable_baselines3.common.utils import (
    configure_logger,
    should_collect_more_steps,
    obs_as_tensor
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import safe_mean

from threading import Condition

from gymnasium import spaces

import copy
import sys


class Agent(ABC):
    """
    Base class for all agents in multi-agent environments
    """

    @abstractmethod
    def get_action(self, obs: Observation, record: bool = True) -> np.ndarray:
        """
        Return an action given an observation.

        :param obs: The observation to use
        :param record: Whether to record the obs, action pair (for training)
        :returns: The action to take
        """

    @abstractmethod
    def update(self, reward: float, done: bool) -> None:
        """
        Add new rewards and done information if the agent can learn.

        Each update corresponds to the most recent `get_action` (where
        `record` is True). If there are multiple calls to `update` that
        correspond to the same `get_action`, their rewards are summed up and
        the last done flag will be used.

        :param reward: The reward receieved from the previous action step
        :param done: Whether the game is done
        """


class DummyAgent(Agent):
    """
    Agent wrapper for standard SARL algorithms assuming a gym interface
    """

    def __init__(self, dummy_env):
        # print("Constructing Dummy Agent")
        self.rew = 0
        self.done = False
        self.dummy_env = dummy_env

        self._action = None
        self.action_cv = Condition()

        self.dummy_env.associated_agent = self

    def get_action(self, obs: Observation, record: bool = True) -> np.ndarray:
        # print("Dummy Agent: got new observation")
        with self.dummy_env.obs_cv:
            self.dummy_env._obs = obs
            self.dummy_env._rew = self.rew
            self.dummy_env._done = self.done
            self.rew = 0
            self.done = False
            # print("Dummy Agent: sent observation notification")
            self.dummy_env.obs_cv.notify()

        with self.action_cv:
            # print("Dummy Agent: waiting for action")
            while self._action is None:
                self.action_cv.wait()
            to_return = self._action
            self._action = None
            # print("Dummy Agent: got action")
        return to_return

    def update(self, reward: float, done: bool) -> None:
        self.rew += reward
        self.done = self.done or done


class StaticPolicyAgent(Agent):
    """
    Agent representing a static (not learning) policy.

    :param policy: Policy representing the agent's responses to observations
    """

    def __init__(self, policy: ActorCriticPolicy):
        self.policy = policy

    def get_action(self, obs: Observation, record: bool = True) -> np.ndarray:
        """
        Return an action given an observation.

        :param obs: The observation to use
        :param record: Whether to record the obs, action (unused)
        :returns: The action to take
        """
        actions, _, _ = action_from_policy(obs.obs, self.policy)
        return clip_actions(actions, self.policy)[0]

    def update(self, reward: float, done: bool) -> None:
        """
        Update does nothing since the agent does not learn.
        """
        pass


class OnPolicyAgent(Agent):
    """
    Agent representing an on-policy learning algorithm (ex: A2C/PPO).

    The `get_action` and `update` functions are based on the `learn` function
    from ``OnPolicyAlgorithm``.

    :param model: Model representing the agent's learning algorithm
    """

    def __init__(self,
                 model: OnPolicyAlgorithm,
                 log_interval=None,
                 tensorboard_log=None,
                 working_timesteps=1000,
                 callback=None,
                 tb_log_name="OnPolicyAgent"):
        self.model = model
        self.tb_log_name = tb_log_name
        self.original_callback = callback

        self.model._last_obs = 0
        self.model._last_episode_starts = np.ones((1,), dtype=bool)
        self.working_timesteps = working_timesteps

        self.iteration = 0

        self.total_timesteps, self.callback = self.model._setup_learn(
            working_timesteps,
            callback,
            False,
            tb_log_name,
            False
        )

        self.callback.on_training_start(locals(), globals())

        assert self.model.env is not None

        self.model.policy.set_training_mode(False)

        self.n_steps = 0
        self.model.rollout_buffer.reset()
        if self.model.use_sde:
            self.model.policy.reset_noise(1)
        self.callback.on_rollout_start()

        # self.values: th.Tensor = th.empty(0)

        # self.model.set_logger(configure_logger(
        #     self.model.verbose, tensorboard_log, tb_log_name))

        # self.name = tb_log_name
        # self.num_timesteps = 0
        self.log_interval = log_interval or (1 if model.verbose else None)
        # self.iteration = 0
        self.model.ep_info_buffer = deque([], maxlen=100)
        self.in_progress_info = {"r": 0, "l": 0}

        # self.model.policy.set_training_mode(False)
        self.old_buffer = None

    def get_action(self, obs: Observation, record: bool = True) -> np.ndarray:
        """
        Return an action given an observation.

        When `record` is True, the agent saves the last transition into its
        buffer. It also updates the model if the buffer is full.

        :param obs: The observation to use
        :param record: Whether to record the obs, action (True when training)
        :returns: The action to take
        """
        obs = obs.obs
        if not isinstance(obs, np.ndarray):
            obs = np.array([obs])
        callback = self.callback
        rollout_buffer = self.model.rollout_buffer
        n_rollout_steps = self.model.n_steps

        if self.model.num_timesteps >= self.total_timesteps:
            self.callback.on_training_end()
            self.iteration = 0
            self.total_timesteps, self.callback = self.model._setup_learn(
                self.working_timesteps,
                self.original_callback,
                False,
                self.tb_log_name,
                False
            )

            self.callback.on_training_start(locals(), globals())

        if record and self.n_steps >= n_rollout_steps:
            with th.no_grad():
                values = self.model.policy.predict_values(obs_as_tensor(obs, self.model.device).unsqueeze(0))
            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=self.model._last_episode_starts)
            self.old_buffer = copy.deepcopy(rollout_buffer)
            callback.update_locals(locals())
            callback.on_rollout_end()

            self.iteration += 1
            self.model._update_current_progress_remaining(self.model.num_timesteps, self.working_timesteps)

            if self.log_interval is not None and self.iteration % self.log_interval == 0:
                assert self.model.ep_info_buffer is not None
                # TODO, Logging
                time_elapsed = max((time.time_ns() - self.model.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.model.num_timesteps - self.model._num_timesteps_at_start) / time_elapsed)
                self.model.logger.record("time/iterations", self.iteration, exclude="tensorboard")
                if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
                    self.model.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
                    self.model.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]))
                self.model.logger.record("time/fps", fps)
                self.model.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.model.logger.record("time/total_timesteps", self.model.num_timesteps, exclude="tensorboard")
                self.model.logger.dump(step=self.model.num_timesteps)
            self.model.train()

            # Restarting
            self.model.policy.set_training_mode(False)
            self.n_steps = 0
            rollout_buffer.reset()
            if self.model.use_sde:
                self.model.policy.reset_noise(1)
            self.callback.on_rollout_start()

        if self.model.use_sde and self.model.sde_sample_freq > 0 and self.n_steps % self.model.sde_sample_freq == 0:
            self.model.policy.reset_noise(1)

        with th.no_grad():
            obs_tensor = obs_as_tensor(obs, self.model.device)
            actions, values, log_probs = self.model.policy(obs_tensor.unsqueeze(0))
        actions = actions.cpu().numpy()
        clipped_actions = actions

        if isinstance(self.model.action_space, spaces.Box):
            clipped_actions = np.clip(actions, self.model.action_space.low, self.model.action_space.high)

        if record:
            self.in_progress_info["l"] += 1
        self.model.num_timesteps += 1
        self.n_steps += 1
        if isinstance(self.model.action_space, spaces.Discrete):
            actions = actions.reshape(-1, 1)

        rollout_buffer.add(
            obs,
            actions,
            [0],
            self.model._last_episode_starts,
            values,
            log_probs
        )
        return clipped_actions[0]

    def update(self, reward: float, done: bool) -> None:
        """
        Add new rewards and done information.

        The rewards are added to buffer entry corresponding to the most recent
        recorded action.

        :param reward: The reward receieved from the previous action step
        :param done: Whether the game is done
        """
        buf = self.model.rollout_buffer
        self.model._last_episode_starts[0] = done
        buf.rewards[buf.pos - 1][0] += reward
        self.in_progress_info["r"] += reward
        if done:
            self.model.ep_info_buffer.append(self.in_progress_info)
            self.in_progress_info = {"r": 0, "l": 0}

    def learn(self, **kwargs) -> None:
        """ Call the model's learn function with the given parameters """
        self.model._custom_logger = False
        self.model.learn(**kwargs)


class OffPolicyAgent(Agent):
    """
    Agent representing an off-policy learning algorithm (ex: DQN/SAC).

    The `get_action` and `update` functions are based on the `learn` function
    from ``OffPolicyAlgorithm``.

    :param model: Model representing the agent's learning algorithm
    """

    def __init__(self,
                 model: OffPolicyAlgorithm,
                 log_interval=None,
                 tensorboard_log=None,
                 working_timesteps=1000,
                 callback=None,
                 tb_log_name="OffPolicyAgent"):
        self.model = model
        self.tb_log_name = tb_log_name
        self.original_callback = callback
        self.log_interval = log_interval

        self.model._last_obs = 0
        self.model._last_episode_starts = np.ones((1,), dtype=bool)
        self.working_timesteps = working_timesteps

        self.iteration = 0

        self.total_timesteps, self.callback = self.model._setup_learn(
            working_timesteps,
            callback,
            False,
            tb_log_name,
            False
        )

        self.callback.on_training_start(locals(), globals())

        self.model.policy.set_training_mode(False)

        self.num_collected_steps = 0
        self.num_collected_episodes = 0
        if self.model.use_sde:
            self.model.policy.reset_noise(1)
        self.callback.on_rollout_start()

        self.buffer_actions = None
        self.rewards = [0.] #np.zeros((1,))
        self.dones = [False] #np.zeros((1,), dtype=bool)
        self.infos = [{}]

        self.log_interval = log_interval or (4 if model.verbose else None)
        self.cur_ep_info = {'r': 0.0, 'l': 0}

    def get_action(self, obs: Observation, record: bool = True) -> np.ndarray:
        """
        Return an action given an observation.

        When `record` is True, the agent saves the last transition into its
        buffer.

        :param obs: The observation to use
        :param record: Whether to record the obs, action (True when training)
        :returns: The action to take
        """
        obs = obs.obs
        if not isinstance(obs, np.ndarray):
            obs = np.array([obs])
        else:
            obs = np.expand_dims(obs, 0)
        callback = self.callback
        train_freq = self.model.train_freq
        replay_buffer = self.model.replay_buffer
        action_noise = self.model.action_noise
        learning_starts = self.model.learning_starts
        log_interval = self.log_interval

        if self.buffer_actions is None:
            self.model._last_obs = obs
        else:
            self.new_obs = obs
            self.model._update_info_buffer(self.infos, self.dones)
            self.model._store_transition(replay_buffer, self.buffer_actions, self.new_obs, self.rewards, self.dones, self.infos)
            self.model._update_current_progress_remaining(self.model.num_timesteps, self.model._total_timesteps)
            self.model._on_step()
            for idx, done in enumerate(self.dones):
                if done:
                    self.num_collected_episodes += 1
                    self.model._episode_num += 1
                    if action_noise is not None:
                        action_noise.reset()
                    if log_interval is not None and self.model._episode_num % log_interval == 0:
                        self.model._dump_logs()

        if not should_collect_more_steps(train_freq, self.num_collected_steps, self.num_collected_episodes):
            callback.on_rollout_end()
            if self.model.num_timesteps > 0 and self.model.num_timesteps > self.model.learning_starts:
                gradient_steps = self.model.gradient_steps if self.model.gradient_steps >= 0 else self.num_collected_steps
                if gradient_steps > 0:
                    self.model.train(batch_size=self.model.batch_size, gradient_steps=gradient_steps)
            self.model.policy.set_training_mode(False)
            self.num_collected_steps = 0
            self.num_collected_episodes = 0
            if self.model.use_sde:
                self.model.policy.reset_noise(1)
            self.callback.on_rollout_start()

        if self.model.use_sde and self.model.sde_sample_freq > 0 and self.num_collected_steps % self.model.sde_sample_freq == 0:
            self.model.actor.reset_noise(1)

        actions, self.buffer_actions = self.model._sample_action(learning_starts, action_noise, 1)
        self.model.num_timesteps += 1
        self.num_collected_steps += 1

        self.rewards[0] = 0
        self.dones[0] = False
        self.infos[0] = {}
        self.cur_ep_info['l'] += 1
        return actions[0]

    def update(self, reward: float, done: bool) -> None:
        """
        Add new rewards and done information.

        The agent trains when the model determines that it has collected enough
        timesteps.

        :param reward: The reward receieved from the previous action step
        :param done: Whether the game is done
        """
        self.rewards[0] += reward
        self.dones[0] = done
        self.infos[0] = {}
        self.cur_ep_info['r'] += reward
        if done:
            self.infos[0]['episode'] = self.cur_ep_info
            self.cur_ep_info = {'r': 0.0, 'l': 0}

    def learn(self, **kwargs) -> None:
        self.model._custom_logger = False
        self.model.learn(**kwargs)


class RecordingAgentWrapper(Agent):
    """
    Wrapper for an agent that records observation-action pairs.

    Users can also use SimultaneousRecorder or TurnBasedRecorder (from
    wrappers.py) to record the transitions in an environment.

    :param realagent: Agent that defines the behaviour of this actor
    """

    def __init__(self, realagent: Agent):
        self.realagent = realagent
        self.allobs: List[np.ndarray] = []
        self.allacts: List[np.ndarray] = []

    def get_action(self, obs: Observation, record: bool = True) -> np.ndarray:
        """
        Return an action given an observation.

        The output is the same as calling `get_action` on the realagent, but
        this wrapper also stores the observation-action pair to a buffer

        :param obs: The observation to use
        :param record: Whether to record the obs, action (True when training)
        :returns: The action to take
        """
        action = self.realagent.get_action(obs, record)
        self.allobs.append(obs.obs)
        self.allacts.append(action)
        return action

    def update(self, reward: float, done: bool) -> None:
        """
        Simply calls the realagent's update function

        :param reward: The reward receieved from the previous action step
        :param done: Whether the game is done
        """
        self.realagent.update(reward, done)

    def get_transitions(self) -> TransitionsMinimal:
        """
        Return the transitions recorded by this agent.

        :returns: A TransitionsMinimal object representing the transitions
        """
        obs = np.array(self.allobs)
        acts = np.array(self.allacts)
        return TransitionsMinimal(obs, acts)
