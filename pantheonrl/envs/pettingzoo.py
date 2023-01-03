from typing import Tuple, Optional, List, Dict

import numpy as np
import gymnasium
import gym

from pantheonrl.common.multiagentenv import MultiAgentEnv, DummyEnv
from pantheonrl.common.observation import Observation


def gymnasium_to_gym(space: gymnasium.spaces.Space) -> gym.Space:
    """
    Converter from gymnasium spaces to gym spaces for SB3 compatibility
    """
    if isinstance(space, gymnasium.spaces.box.Box):
        return gym.spaces.Box(space.low, space.high, dtype=space.dtype)
    if isinstance(space, gymnasium.spaces.discrete.Discrete):
        return gym.spaces.Discrete(space.n)
    if isinstance(space, gymnasium.spaces.multi_discrete.MultiDiscrete):
        return gym.spaces.MultiDiscrete(space.nvec)
    if isinstance(space, gymnasium.spaces.multi_binary.MultiBinary):
        return gym.spaces.MultiBinary(space.n)

    raise NotImplementedError(f"Space {space} not implemented yet for gymnasium to gym conversion")


class PettingZooAECWrapper(MultiAgentEnv):
    """
    Wrapper for Petting Zoo AEC environments.
    """

    def __init__(self, base_env, ego_ind=0):
        self.base_env = base_env
        super(PettingZooAECWrapper, self).__init__(
            ego_ind, base_env.max_num_agents)
        ego_agent = base_env.possible_agents[ego_ind]
        self.action_space = gymnasium_to_gym(base_env.action_space(ego_agent))

        obs_space = base_env.observation_space(ego_agent)
        if isinstance(obs_space, gymnasium.spaces.dict.Dict):
            obs_space = obs_space.spaces['observation']
        self.observation_space = gymnasium_to_gym(obs_space)
        self._action_mask = None

    def getDummyEnv(self, player_ind: int):
        agent = self.base_env.possible_agents[player_ind]
        ospace = self.base_env.observation_space(agent)
        if isinstance(ospace, gymnasium.spaces.dict.Dict):
            ospace = ospace.spaces['observation']
        ospace = gymnasium_to_gym(ospace)
        aspace = gymnasium_to_gym(self.base_env.action_space(agent))
        return DummyEnv(ospace, aspace)

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
        agent = self.base_env.agent_selection
        act = actions[0]
        if self._action_mask is not None and not self._action_mask[act]:
            act = self._action_mask.tolist().index(1)

        self.base_env.step(act)

        agent = self.base_env.agent_selection
        agent_idx = self.base_env.possible_agents.index(agent)
        obs = self.base_env.observe(agent)

        if isinstance(obs, dict):
            self._action_mask = obs['action_mask']
            obs = obs['observation']

        rewards = [0] * self.n_players
        for key, val in self.base_env.rewards.items():
            rewards[self.base_env.possible_agents.index(key)] = val

        done = all([self.base_env.terminations[x] or self.base_env.truncations[x] for x in self.base_env.possible_agents])
        # print(self.base_env.terminations)
        # done = all(self.base_env.dones.values())
        info = self.base_env.infos[self.base_env.possible_agents[self.ego_ind]]
        obs = Observation(obs=obs, action_mask=self._action_mask)
        return (agent_idx,), (obs,), tuple(rewards), done, info

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
        self.base_env.reset()
        agent = self.base_env.agent_selection
        agent_idx = self.base_env.possible_agents.index(agent)
        obs = self.base_env.observe(agent)

        if isinstance(obs, dict):
            self._action_mask = obs['action_mask']
            obs = obs['observation']

        self.agent_counts = [0] * self.n_players
        obs = Observation(obs=obs, action_mask=self._action_mask)
        return (agent_idx,), (obs,)
