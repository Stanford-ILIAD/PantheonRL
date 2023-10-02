from typing import Tuple, Optional, List, Dict

import numpy as np
import gymnasium as gym

from pantheonrl.common.multiagentenv import MultiAgentEnv
from pantheonrl.common.observation import Observation


class PettingZooAECWrapper(MultiAgentEnv):
    """
    Wrapper for Petting Zoo AEC environments.
    """

    def __init__(self, base_env, ego_ind=0):
        self.base_env = base_env
        observation_spaces = []
        action_spaces = []
        for player_ind in range(base_env.max_num_agents):
            agent = self.base_env.possible_agents[player_ind]
            ospace = self.base_env.observation_space(agent)
            if isinstance(ospace, gym.spaces.dict.Dict):
                ospace = ospace.spaces['observation']
            aspace = self.base_env.action_space(agent)
            observation_spaces.append(ospace)
            action_spaces.append(aspace)
        super(PettingZooAECWrapper, self).__init__(
            observation_spaces, action_spaces,
            ego_ind, base_env.max_num_agents)
        self._action_mask = None

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

        done = all([
            self.base_env.terminations[x] or self.base_env.truncations[x]
            for x in self.base_env.possible_agents
        ])
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
