from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Callable, Any, Union

import gymnasium as gym
import numpy as np

from .agents import Agent, DummyAgent
from .observation import Observation, extract_obs

import threading
from threading import Condition


class PlayerException(Exception):
    """ Raise when players in the environment are incorrectly set """

class KillEnvException(Exception):
    """ Raise when the DummyEnv is killed """


class DummyEnv(gym.Env):
    """
    Environment representing a partner agent's observation and action space.
    """

    def __init__(self, base_env, agent_ind, extractor=extract_obs):
        super().__init__()
        self.base_env = base_env
        self.agent_ind = agent_ind

        self.observation_space = self.base_env.observation_spaces[agent_ind]
        self.action_space = self.base_env.action_spaces[agent_ind]

        self._obs = None
        self._rew = None
        self._done = True

        self.obs_cv = Condition()
        self.extractor = extractor
        self.associated_agent = None
        self.steps = 0

        self.dead = False

    def step(
            self, action: np.ndarray
    ) -> tuple[Union[Observation, Any], float, bool, bool, dict[str, Any]]:
        assert threading.current_thread() is not threading.main_thread()
        # print("Dummy Env: got new action in step function", self.steps)
        with self.associated_agent.action_cv:
            self.associated_agent._action = action
            # print("Dummy Env: sending action notification")
            self.associated_agent.action_cv.notify()
            self._obs = None

        with self.obs_cv:
            # print("Dummy Env: waiting for observation")
            while self._obs is None:
                self.obs_cv.wait()
                if self.dead:
                    raise KillEnvException("Killing dummy environment")
            to_return = self.extractor(self._obs), self._rew, self._done, False, {}
            if not self._done:
                self._obs = None
            # else:
                # print("DUMMY ENV THINKS DONE")
            # print("Dummy Env: got observation")
        self.steps += 1
        return to_return

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        assert self._done
        assert threading.current_thread() is not threading.main_thread()
        # print("Dummy Env: reset called")
        with self.obs_cv:
            # print("Dummy Env: waiting for observation (reset)")
            while self._obs is None:
                self.obs_cv.wait()
            to_return = self.extractor(self._obs), {}
            self._done = False
            # print("Dummy Env: got observation (reset)")
        # print(to_return)
        return to_return

    def close(self):
        self.associated_agent._action = 0
        self.associated_agent.dummy_env = None
        with self.associated_agent.action_cv:
            self.associated_agent.action_cv.notify()

        import warnings
        warnings.warn('Partner agent\'s dummy environment is dead. Remember to set the the learning time for the partner to be much larger than the program lifetime')


class MultiAgentEnv(gym.Env, ABC):
    """
    Base class for all Multi-agent environments.

    :param ego_ind: The player that the ego represents
    :param n_players: The number of players in the game
    :param resample_policy: The resampling policy to use (see set_resample_policy)
    :param partners: Lists of agents to choose from for the partner players
    :param ego_extractor: Function to extract Observation into the type the
        ego expects
    """

    def __init__(self,
                 observation_spaces: List[gym.spaces.Space],
                 action_spaces: List[gym.spaces.Space],
                 ego_ind: int = 0,
                 n_players: int = 2,
                 resample_policy: str = "default",
                 partners: Optional[List[List[Agent]]] = None,
                 ego_extractor: Callable[[Observation], Any] = extract_obs):
        super().__init__()
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.ego_ind = ego_ind
        self.n_players = n_players

        if partners is not None:
            if len(partners) != n_players - 1:
                raise PlayerException(
                    "The number of partners needs to equal the number \
                    of non-ego players")

            for plist in partners:
                if not isinstance(plist, list) or not plist:
                    raise PlayerException(
                        "Sublist for each partner must be nonempty list")

        self.partners = partners or [[]] * (n_players - 1)
        self.partnerids = [0] * (n_players - 1)

        self._players: Tuple[int, ...] = tuple()
        self._obs: Tuple[Optional[np.ndarray], ...] = tuple()
        self._old_ego_obs: Optional[np.ndarray] = None

        self.should_update = [False] * (self.n_players - 1)
        self.total_rews = [0] * (self.n_players)
        self.ego_moved = False

        self.set_resample_policy(resample_policy)
        self.ego_extractor = ego_extractor

    def getDummyEnv(self, player_num: int):
        """
        Returns a dummy environment with just an observation and action
        space that a partner agent can use to construct their policy network.

        :param player_num: the partner number to query
        """
        return DummyEnv(self, player_num)

    def construct_single_agent_interface(self, player_num: int):
        dummyEnv = self.getDummyEnv(player_num)
        dummyAgent = DummyAgent(dummyEnv)

        partner_num = self._get_partner_num(player_num)
        if len(self.partners[partner_num]) != 0:
            raise PlayerException("Cannot construct multiple single agent \
            interfaces for the same player_num")

        self.add_partner_agent(dummyAgent, player_num)
        return dummyEnv

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.observation_spaces[self.ego_ind]

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.action_spaces[self.ego_ind]

    def set_ego_extractor(self, ego_extractor: Callable[[Observation], Any]):
        self.ego_extractor = ego_extractor

    def _get_partner_num(self, player_num: int) -> int:
        if player_num == self.ego_ind:
            raise PlayerException(
                "Ego agent is not set by the environment")
        elif player_num > self.ego_ind:
            return player_num - 1
        return player_num

    def add_partner_agent(self, agent: Agent, player_num: int = 1) -> None:
        """
        Add agent to the list of potential partner agents. If there are
        multiple agents that can be a specific player number, the environment
        randomly samples from them at the start of every episode.

        :param agent: Agent to add
        :param player_num: the player number that this new agent can be
        """
        self.partners[self._get_partner_num(player_num)].append(agent)

    def set_partnerid(self, agent_id: int, player_num: int = 1) -> None:
        """
        Set the current partner agent to use

        :param agent_id: agent_id to use as current partner
        """
        partner_num = self._get_partner_num(player_num)
        assert(agent_id >= 0 and agent_id < len(self.partners[partner_num]))
        self.partnerids[partner_num] = agent_id

    def resample_random(self) -> None:
        """ Randomly resamples each partner policy """
        self.partnerids = [self.np_random.integers(0, len(plist))
                           for plist in self.partners]

    def resample_round_robin(self) -> None:
        """
        Sets the partner policy to the next option on the list for round-robin
        sampling.

        Note: This function is only valid for 2-player environments
        """
        self.partnerids = [(self.partnerids[0] + 1) % len(self.partners[0])]

    def set_resample_policy(self, resample_policy: str) -> None:
        """
        Set the resample_partner method to round "robin" or "random"

        :param resample_policy: The new resampling policy to use.
        - Valid values are: "default", "robin", "random"
        """
        if resample_policy == "default":
            resample_policy = "robin" if self.n_players == 2 else "random"

        if resample_policy == "robin" and self.n_players != 2:
            raise PlayerException(
                "Cannot do round robin resampling for >2 players")

        if resample_policy == "robin":
            self.resample_partner = self.resample_round_robin
        elif resample_policy == "random":
            self.resample_partner = self.resample_random
        else:
            raise PlayerException(
                f"Invalid resampling policy: {resample_policy}")

    def _get_actions(self, players, obs, ego_act=None):
        actions = []
        for player, ob in zip(players, obs):
            if player == self.ego_ind:
                actions.append(ego_act)
            else:
                p = self._get_partner_num(player)
                agent = self.partners[p][self.partnerids[p]]
                actions.append(agent.get_action(ob))
                if not self.should_update[p]:
                    agent.update(self.total_rews[player], False)
                self.should_update[p] = True
        return np.array(actions)

    def _update_players(self, rews, done):
        for i in range(self.n_players - 1):
            nextrew = rews[i + (0 if i < self.ego_ind else 1)]
            if self.should_update[i]:
                self.partners[i][self.partnerids[i]].update(nextrew, done)

        for i in range(self.n_players):
            self.total_rews[i] += rews[i]

    def step(
            self, action: np.ndarray
    ) -> tuple[Union[Observation, Any], float, bool, bool, dict[str, Any]]:
        """
        Run one timestep from the perspective of the ego-agent. This involves
        calling the ego_step function and the alt_step function to get to the
        next observation of the ego agent.

        Accepts the ego-agent's action and returns a tuple of (observation,
        reward, done, info) from the perspective of the ego agent.

        Note that when the environment is done, the final observation is the
        latest observation provided by the environment, which may be the same
        as the previous observation given to the agent, especially in turn-based
        settings.

        :param action: An action provided by the ego-agent.

        :returns:
            observation: Ego-agent's next observation
            reward: Amount of reward returned after previous action
            terminated: Whether the episode has ended (need to call reset() if True)
            truncated: Whether the episode was truncated (need to call reset() if True)
            info: Extra information about the environment
        """
        ego_rew = 0.0

        while True:
            acts = self._get_actions(self._players, self._obs, action)
            self._players, self._obs, rews, done, info = self.n_step(acts)
            info['_partnerid'] = self.partnerids

            self._update_players(rews, done)

            ego_rew += rews[self.ego_ind] if self.ego_moved \
                else self.total_rews[self.ego_ind]

            self.ego_moved = True

            if self.ego_ind in self._players:
                break

            if done:
                ego_obs = self._old_ego_obs
                return self.ego_extractor(ego_obs), ego_rew, done, False, info

        ego_obs = self._obs[self._players.index(self.ego_ind)]
        self._old_ego_obs = ego_obs
        return self.ego_extractor(ego_obs), ego_rew, done, False, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """
        Reset environment to an initial state and return the first observation
        for the ego agent.

        :returns: Ego-agent's first observation
        """
        super().reset(seed=seed)

        self.resample_partner()
        self._players, self._obs = self.n_reset()
        self.should_update = [False] * (self.n_players - 1)
        self.total_rews = [0] * self.n_players
        self.ego_moved = False

        while self.ego_ind not in self._players:
            acts = self._get_actions(self._players, self._obs)
            self._players, self._obs, rews, done, _ = self.n_step(acts)

            self._update_players(rews, done)
            if done:
                self.resample_partner()
                self._players, self._obs = self.n_reset()
                self.should_update = [False] * (self.n_players - 1)
                self.total_rews = [0] * self.n_players
                self.ego_moved = False

        ego_obs = self._obs[self._players.index(self.ego_ind)]

        assert ego_obs is not None
        self._old_ego_obs = ego_obs
        return self.ego_extractor(ego_obs), {}

    @abstractmethod
    def n_step(
                    self,
                    actions: List[np.ndarray],
                ) -> Tuple[Tuple[int, ...],
                           Tuple[Optional[Observation], ...],
                           Tuple[float, ...],
                           bool,
                           Dict]:
        """
        Perform the actions specified by the agents that will move. This
        function returns a tuple of (next agents, observations, both rewards,
        done, info).

        This function is called by the `step` function.

        :param actions: List of action provided agents that are acting on this
        step.

        :returns:
            agents: Tuple representing the agents to call for the next actions
            observations: Tuple representing the next observations (ego, alt)
            rewards: Tuple representing the rewards of all agents
            done: Whether the episode has ended
            info: Extra information about the environment
        """

    @abstractmethod
    def n_reset(self) -> Tuple[Tuple[int, ...],
                               Tuple[Optional[Observation], ...]]:
        """
        Reset the environment and return which agents will move first along
        with their initial observations.

        This function is called by the `reset` function.

        :returns:
            agents: Tuple representing the agents that will move first
            observations: Tuple representing the observations of both agents
        """


class TurnBasedEnv(MultiAgentEnv, ABC):
    """
    Base class for all 2-player turn-based games.

    In turn-based games, players take turns receiving observations and making
    actions.

    :param probegostart: Probability that the ego agent gets the first turn
    :param partners: List of policies to choose from for the partner agent
    """

    def __init__(self,
                 observation_spaces: List[gym.spaces.Space],
                 action_spaces: List[gym.spaces.Space],
                 probegostart: float = 0.5,
                 partners: Optional[List[Agent]] = None):
        partners = [partners] if partners else None
        super(TurnBasedEnv, self).__init__(
            observation_spaces, action_spaces,
            ego_ind=0, n_players=2, partners=partners)
        self.probegostart = probegostart
        self.ego_next = True

    def n_step(
                    self,
                    actions: List[np.ndarray],
                ) -> Tuple[Tuple[int, ...],
                           Tuple[Optional[Observation], ...],
                           Tuple[float, ...],
                           bool,
                           Dict]:
        agents = (1 if self.ego_next else 0,)
        obs, rews, done, info = self.ego_step(actions[0]) if self.ego_next \
            else self.alt_step(actions[0])

        self.ego_next = not self.ego_next

        return agents, (Observation(obs),), rews, done, info

    def n_reset(self) -> Tuple[Tuple[int, ...],
                               Tuple[Optional[np.ndarray], ...]]:
        self.ego_next = (self.np_random.random() < self.probegostart)
        obs = self.multi_reset(self.ego_next)
        return (0 if self.ego_next else 1,), (Observation(obs),)

    @abstractmethod
    def ego_step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], Tuple[float, float], bool, Dict]:
        """
        Perform the ego-agent's action and return a tuple of (partner's
        observation, both rewards, done, info).

        This function is called by the `step` function along with alt-step.

        :param action: An action provided by the ego-agent.

        :returns:
            partner observation: Partner's next observation
            rewards: Tuple representing the rewards of both agents (ego, alt)
            done: Whether the episode has ended
            info: Extra information about the environment
        """

    @abstractmethod
    def alt_step(
                self,
                action: np.ndarray
            ) -> Tuple[Optional[np.ndarray], Tuple[float, float], bool, Dict]:
        """
        Perform the partner's action and return a tuple of (ego's observation,
        both rewards, done, info).

        This function is called by the `step` function along with ego-step.

        :param action: An action provided by the partner.

        :returns:
            ego observation: Ego-agent's next observation
            rewards: Tuple representing the rewards of both agents (ego, alt)
            done: Whether the episode has ended
            info: Extra information about the environment
        """

    @abstractmethod
    def multi_reset(self, egofirst: bool) -> np.ndarray:
        """
        Reset the environment and give the observation of the starting agent
        (based on the value of `egofirst`).

        This function is called by the `reset` function.

        :param egofirst: True if the ego has the first turn, False otherwise
        :returns: The observation for the starting agent (ego if `egofirst` is
            True, and the partner's observation otherwise)
        """


class SimultaneousEnv(MultiAgentEnv, ABC):
    """
    Base class for all 2-player simultaneous games.

    :param partners: List of policies to choose from for the partner agent
    """

    def __init__(self,
                 observation_spaces: List[gym.spaces.Space],
                 action_spaces: List[gym.spaces.Space],
                 partners: Optional[List[Agent]] = None):
        partners = [partners] if partners else None
        super(SimultaneousEnv, self).__init__(
            observation_spaces, action_spaces,
            ego_ind=0, n_players=2, partners=partners)

    def n_step(
                    self,
                    actions: List[np.ndarray],
                ) -> Tuple[Tuple[int, ...],
                           Tuple[Optional[Observation], ...],
                           Tuple[float, ...],
                           bool,
                           Dict]:
        (obs0, obs1), r, d, i = self.multi_step(actions[0], actions[1])
        return ((0, 1), (Observation(obs0), Observation(obs1)), r, d, i)

    def n_reset(self) -> Tuple[Tuple[int, ...],
                               Tuple[Optional[Observation], ...]]:
        (obs0, obs1) = self.multi_reset()
        return (0, 1), (Observation(obs0), Observation(obs1))

    @abstractmethod
    def multi_step(
                    self,
                    ego_action: np.ndarray,
                    alt_action: np.ndarray
                ) -> Tuple[Tuple[Optional[np.ndarray], Optional[np.ndarray]],
                           Tuple[float, float], bool, Dict]:
        """
        Perform the ego-agent's and partner's actions. This function returns a
        tuple of (observations, both rewards, done, info).

        This function is called by the `step` function.

        :param ego_action: An action provided by the ego-agent.
        :param alt_action: An action provided by the partner.

        :returns:
            observations: Tuple representing the next observations (ego, alt)
            rewards: Tuple representing the rewards of both agents (ego, alt)
            done: Whether the episode has ended
            info: Extra information about the environment
        """

    @abstractmethod
    def multi_reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset the environment and give the observation of both agents.

        This function is called by the `reset` function.

        :returns: The observations of both agents
        """
