from typing import Tuple, Optional, List, Dict

import numpy as np
from gym import spaces

from pantheonrl.common.multiagentenv import MultiAgentEnv, DummyEnv


class PettingZooAECWrapper(MultiAgentEnv):
    """
    Wrapper for Petting Zoo AEC environments.
    """

    def __init__(self, base_env, ego_ind=0):
        self.base_env = base_env
        super(PettingZooAECWrapper, self).__init__(
            ego_ind, base_env.max_num_agents)
        ego_agent = base_env.possible_agents[ego_ind]
        self.action_space = base_env.action_space(ego_agent)

        obs_space = base_env.observation_space(ego_agent)
        if isinstance(obs_space, spaces.Dict):
            obs_space = obs_space.spaces['observation']
        self.observation_space = obs_space
        self._action_mask = None

    def getDummyEnv(self, player_ind: int):
        agent = self.base_env.possible_agents[player_ind]
        ospace = self.base_env.observation_space(agent)
        if isinstance(ospace, spaces.Dict):
            ospace = ospace.spaces['observation']
        aspace = self.base_env.action_space(agent)
        return DummyEnv(ospace, aspace)

    def n_step(
                    self,
                    actions: List[np.ndarray],
                ) -> Tuple[Tuple[int, ...],
                           Tuple[Optional[np.ndarray], ...],
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

        done = all(self.base_env.dones.values())
        info = self.base_env.infos[self.base_env.possible_agents[self.ego_ind]]
        return (agent_idx,), (obs,), tuple(rewards), done, info

    def n_reset(self) -> Tuple[Tuple[int, ...],
                               Tuple[Optional[np.ndarray], ...]]:
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
        return (agent_idx,), (obs,)
