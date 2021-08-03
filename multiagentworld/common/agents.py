from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
import torch as th

from .util import action_from_policy, clip_actions, resample_noise
from .trajsaver import TransitionsMinimal

from stable_baselines3.common.utils import (
    configure_logger,
    should_collect_more_steps
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


class Agent(ABC):
    """
    Base class for all agents in multi-agent environments
    """

    @abstractmethod
    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
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


class StaticPolicyAgent(Agent):
    """
    Agent representing a static (not learning) policy.

    :param policy: Policy representing the agent's responses to observations
    """

    def __init__(self, policy: ActorCriticPolicy):
        self.policy = policy

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        """
        Return an action given an observation.

        :param obs: The observation to use
        :param record: Whether to record the obs, action (unused)
        :returns: The action to take
        """
        actions, _, _ = action_from_policy(obs, self.policy)
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

    def __init__(self, model: OnPolicyAlgorithm):
        self.model = model
        self._last_episode_starts = [True]
        self.n_steps = 0
        self.values: th.Tensor = th.empty(0)

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

        # train the model if the buffer is full
        if record and self.n_steps >= self.model.n_steps:
            buf.compute_returns_and_advantage(
                last_values=self.values,
                dones=self._last_episode_starts[0]
            )
            self.model.train()
            buf.reset()
            self.n_steps = 0

        resample_noise(self.model, self.n_steps)

        actions, values, log_probs = action_from_policy(obs, self.model.policy)

        # modify the rollout buffer with newest info
        if record:
            buf.add(
                np.reshape(obs, (1, -1)),
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

    def __init__(self, model: OffPolicyAlgorithm):
        self.model = model
        self.episode_rewards: List[float] = []
        self.total_timesteps: List[int] = []
        self.num_collected_steps = 0
        self.num_collected_episodes = 0
        self.old_reward: float = 0.0
        self.old_done = False
        self.old_info: Dict = {}

        self.episode_reward: float = 0.0
        self.episode_timesteps = 0
        self.n_steps = 0
        self.old_buffer_action = None
        self.model.set_logger(configure_logger())

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        """
        Return an action given an observation.

        When `record` is True, the agent saves the last transition into its
        buffer.

        :param obs: The observation to use
        :param record: Whether to record the obs, action (True when training)
        :returns: The action to take
        """
        if record:
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
                self.episode_reward = 0.0
                self.episode_timesteps = 0

        resample_noise(self.model, self.n_steps)

        obs = obs.reshape((-1,) + self.model.policy.observation_space.shape)
        self.model._last_obs = obs

        action, buffer_action = self.model._sample_action(
            self.model.learning_starts, self.model.action_noise)

        self.model.num_timesteps += 1
        self.episode_timesteps += 1
        self.num_collected_steps += 1
        self.n_steps += 1

        self.old_buffer_action = buffer_action
        self.old_reward = 0

        return clip_actions(action, self.model)[0]

    def update(self, reward: float, done: bool) -> None:
        """
        Add new rewards and done information.

        The agent trains when the model determines that it has collected enough
        timesteps.

        :param reward: The reward receieved from the previous action step
        :param done: Whether the game is done
        """
        self.episode_reward += reward

        self.old_done = done
        self.old_reward += reward

        if should_collect_more_steps(self.model.train_freq,
                                     self.num_collected_steps,
                                     self.num_collected_episodes):
            return

        gradient_steps = self.model.gradient_steps
        if gradient_steps <= 0:
            gradient_steps = self.num_collected_steps

        self.model.train(batch_size=self.model.batch_size,
                         gradient_steps=gradient_steps)

        self.episode_rewards = []
        self.total_timesteps = []
        self.num_collected_steps = 0
        self.num_collected_episodes = 0

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

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        """
        Return an action given an observation.

        The output is the same as calling `get_action` on the realagent, but
        this wrapper also stores the observation-action pair to a buffer

        :param obs: The observation to use
        :param record: Whether to record the obs, action (True when training)
        :returns: The action to take
        """
        action = self.realagent.get_action(obs, record)
        self.allobs.append(obs)
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
