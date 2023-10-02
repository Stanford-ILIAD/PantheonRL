import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO

import numpy as np

import overcookedgym

from pantheonrl.common.agents import OnPolicyAgent, Agent
from overcookedgym.overcooked_utils import LAYOUT_LIST

import copy

class FakeAgent(Agent):
    def get_action(self, obs, record: bool = True):
        """
        Return an action given an observation.

        :param obs: The observation to use
        :param record: Whether to record the obs, action (unused)
        :returns: The action to take
        """
        return 0

    def update(self, reward: float, done: bool) -> None:
        """
        Update does nothing since the agent does not learn.
        """
        pass

class FakeEgo:
    def __init__(self, env):
        self.env = env

    def learn(self, total_timesteps):
        self.env.reset()
        for _ in range(total_timesteps):
            _, _, done, _, _ = self.env.step(0)
            if done:
                self.env.reset()

def make_env():
    layout = 'simple'
    assert layout in LAYOUT_LIST

    env = gym.make('OvercookedMultiEnv-v0', layout_name=layout)
    # env = gym.make('RPS-v0')
    return env

def run_standard(timesteps):
    stable_baselines3.common.utils.set_random_seed(0)

    env = make_env()
    ego = PPO('MlpPolicy', env, n_steps=10, verbose=1)
    partner = FakeAgent()
    env.unwrapped.ego_ind = 0
    env.unwrapped.add_partner_agent(partner)

    print('ego start', sum([param.data.mean() for param in ego.policy.parameters()]))
    ego.learn(total_timesteps=timesteps)
    rb = copy.deepcopy(ego.rollout_buffer)
    # print(ego.rollout_buffer.observations, ego.rollout_buffer.actions, ego.rollout_buffer.rewards)
    print('ego end', sum([param.data.mean() for param in ego.policy.parameters()]))
    return ego.policy, rb

def run_reversed(timesteps):
    stable_baselines3.common.utils.set_random_seed(0)

    env = make_env()
    ego = FakeEgo(env)
    partner = OnPolicyAgent(PPO('MlpPolicy', env, n_steps=10, verbose=1))
    env.unwrapped.ego_ind = 1
    env.unwrapped.add_partner_agent(partner, 0)
    print("ADDED PARTNER AGENT")

    print('ego start', sum([param.data.mean() for param in partner.model.policy.parameters()]))
    ego.learn(total_timesteps=timesteps)
    rb = copy.deepcopy(partner.model.rollout_buffer)
    rb.compute_returns_and_advantage(
        last_values=partner.values,
        dones=partner._last_episode_starts[0]
    )
    env.step(0)
    print('ego end', sum([param.data.mean() for param in partner.model.policy.parameters()]))
    return partner.model.policy, rb

def check_equivalent_models(model1, model2):
    # print("MODEL1")
    # for p1 in model1.parameters():
    #     print(p1)
    # print("MODEL2")
    # for p2 in model2.parameters():
    #     print(p2)
    # print(model1.parameters(), model2.parameters())
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        # print(p1.data, p2.data)
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def check_equivalent_buffers(rb1, rb2):
    print("Observations", np.array_equal(rb1.observations.flatten(), rb2.observations.flatten()), rb1.observations, rb2.observations)
    print("actions", np.array_equal(rb1.actions, rb2.actions), rb1.actions, rb2.actions)
    print("rewards", np.array_equal(rb1.rewards, rb2.rewards), rb1.rewards, rb2.rewards)
    print("advantages", np.array_equal(rb1.advantages, rb2.advantages), rb1.advantages, rb2.advantages)
    print("returns", np.array_equal(rb1.returns, rb2.returns), rb1.returns, rb2.returns)
    print("episode_starts", np.array_equal(rb1.episode_starts, rb2.episode_starts), rb1.episode_starts, rb2.episode_starts)
    print("log_probs", np.array_equal(rb1.log_probs, rb2.log_probs), rb1.log_probs, rb2.log_probs)
    print("values", np.array_equal(rb1.values, rb2.values), rb1.values, rb2.values)


model1, rb1 = run_standard(10)
model2, rb2 = run_reversed(10)
print(check_equivalent_models(model1, model2))
print(check_equivalent_buffers(rb1, rb2))

e = make_env()
print(e.n_reset()[1][1].obs)

# Truth:
# [ 1.  0.  0.  0.  0.  0.  0. -1. -1.  1. -2.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  1.  0.  0.  2.  1.  0.  1.  0.  1.  1.  0.  0.  0.  0.  0.  0.
#   1.  0. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0. -2.  2.  0.  0.  0.  2.
#   1.  0.  1.  0.  2. -1.  1.  2.]

# False:
# [ 1.  0.  0.  0.  0.  0.  0.  1.  0. -1. -1.  0.  0.  0.  0.  0.  0.  0.
#   0. -2.  2.  0.  0.  0.  2.  1.  0.  1.  0.  1.  0.  0.  0.  0.  0.  0.
#  -1. -1.  1. -2.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  2.  1.
#   0.  1.  0.  1. -2.  1.  1.  2.]
