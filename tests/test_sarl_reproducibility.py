import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO, A2C, DQN

import numpy as np

import overcookedgym

from pantheonrl.common.agents import Agent
from pantheonrl.common.multiagentenv import KillEnvException

import copy

import threading
from threading import Thread

import pytest


class FakeAgent(Agent):
    def __init__(self, env):
        self.env = env
        self.action_space = copy.deepcopy(self.env.action_space)
        self.action_space.seed(0)

    def get_action(self, obs, record: bool = True):
        """
        Return an action given an observation.

        :param obs: The observation to use
        :param record: Whether to record the obs, action (unused)
        :returns: The action to take
        """
        return self.action_space.sample()

    def update(self, reward: float, done: bool) -> None:
        """
        Update does nothing since the agent does not learn.
        """
        pass


class VerboseEnv(gym.Env):
    def __init__(self, base_env):
        self.base_env = base_env
        self.timestep = 0

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.base_env.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.base_env.action_space

    def step(self, action):
        print("Timestep", self.timestep)
        self.timestep += 1
        print("ACTION is", action)
        returns = self.base_env.step(action)
        print("Step returns", returns)
        return returns

    def reset(self, **kwargs):
        returns = self.base_env.reset(**kwargs)
        print("Reset returns", returns)
        return returns


class ReplayAgent(Agent):
    def __init__(self, rb):
        self.rb = rb
        self.steps = 0

    def get_action(self, obs, record:bool = True):
        if not np.array_equal(obs.obs, self.rb.observations[self.steps]):
            raise Exception(f"Observations mismatch at {self.steps} {obs.obs} {self.rb.observations[self.steps]}")
        self.steps += 1
        return self.rb.actions[self.steps-1][0]

    def update(self, reward, done):
        pass


class FakeEgo:
    def __init__(self, env):
        self.env = env
        self.action_space = copy.deepcopy(self.env.action_space)
        self.action_space.seed(0)

    def learn(self, dummy_env, total_timesteps):
        self.env.reset()
        while dummy_env.steps < total_timesteps:
            _, _, done, _, _ = self.env.step(self.action_space.sample())
            if done:
                self.env.reset()


def make_env(option):
    if option == 0:
        env = gym.make('OvercookedMultiEnv-v0', layout_name='simple')
    elif option == 1:
        env = gym.make('RPS-v0')
    elif option == 2:
        env = gym.make('LiarsDice-v0')
    elif option == 3:
        env = gym.make('BlockEnv-v0')
    elif option == 4:
        env = gym.make('BlockEnv-v1')
    env.np_random, _ = gym.utils.seeding.np_random(0)
    return env


def run_standard(ALGO, timesteps, option, n_steps):
    stable_baselines3.common.utils.set_random_seed(0)

    env = make_env(option)
    ego = ALGO('MlpPolicy', env, n_steps=n_steps, verbose=0)
    partner = FakeAgent(env.unwrapped.getDummyEnv(1))
    env.unwrapped.ego_ind = 0
    env.unwrapped.add_partner_agent(partner)

    # print('ego start', sum([param.data.mean() for param in ego.policy.parameters()]))
    ego.learn(total_timesteps=timesteps)
    rb = copy.deepcopy(ego.rollout_buffer)
    # print(ego.rollout_buffer.observations, ego.rollout_buffer.actions, ego.rollout_buffer.rewards)
    # print('ego end', sum([param.data.mean() for param in ego.policy.parameters()]))
    return ego.policy, rb


def run_reversed(ALGO, timesteps, option, n_steps):
    stable_baselines3.common.utils.set_random_seed(0)

    env = make_env(option)
    env.unwrapped.ego_ind = 1
    ego = FakeEgo(env)
    dummy_env = env.unwrapped.construct_single_agent_interface(0)
    partner = ALGO('MlpPolicy', dummy_env, n_steps=n_steps, verbose=0)
    # print("ADDED PARTNER AGENT")

    dumped_buffer = [None]

    def learn_thread():
        try:
            while True:
                partner.learn(total_timesteps=partner.n_steps, reset_num_timesteps=False)
                dumped_buffer[0] = copy.deepcopy(partner.rollout_buffer)
        except KillEnvException:
            pass
            # dummy_env.unwrapped.close()

    # print('ego start', sum([param.data.mean() for param in partner.policy.parameters()]))
    t = Thread(target=learn_thread, daemon=True)
    t.start()
    ego.learn(dummy_env, total_timesteps=timesteps)
    with dummy_env.obs_cv:
        dummy_env.dead = True
        dummy_env.obs_cv.notify()
    t.join()
    rb = dumped_buffer[0]
    # print('ego end', sum([param.data.mean() for param in partner.policy.parameters()]))
    # print('ego timesteps', dummy_env.steps)
    return partner.policy, rb


def check_equivalent_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        # print(p1.data, p2.data)
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def run_standard_dqn(ALGO, timesteps, option, n_steps):
    stable_baselines3.common.utils.set_random_seed(0)

    env = make_env(option)
    env.unwrapped.ego_ind = 0
    ego = ALGO('MlpPolicy', env, train_freq=n_steps, verbose=0, learning_starts=32, batch_size=32, seed=0)
    partner = FakeAgent(env.unwrapped.getDummyEnv(1))
    env.unwrapped.add_partner_agent(partner)

    print('ego start', sum([param.data.mean() for param in ego.policy.parameters()]))
    ego.learn(total_timesteps=timesteps)
    print('gradient updates', ego._n_updates)
    # print(ego.rollout_buffer.observations, ego.rollout_buffer.actions, ego.rollout_buffer.rewards)
    print('ego end', sum([param.data.mean() for param in ego.policy.parameters()]))
    return ego.policy


def run_reversed_dqn(ALGO, timesteps, option, n_steps):
    stable_baselines3.common.utils.set_random_seed(0)

    env = make_env(option)
    env.unwrapped.ego_ind = 1
    ego = FakeEgo(env)
    dummy_env = env.unwrapped.construct_single_agent_interface(0)
    partner = ALGO('MlpPolicy', dummy_env, train_freq=n_steps, verbose=0, learning_starts=32, batch_size=32, seed=0)
    # print("ADDED PARTNER AGENT")

    def learn_thread():
        try:
            while True:
                partner.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        except KillEnvException:
            return
        except Exception:
            import signal
            import traceback
            # threading.current_thread().interrupt_main()
            traceback.print_exc()
            signal.raise_signal(signal.SIGTERM)
        print("DO NOT RESTART")

    print('ego start', sum([param.data.mean() for param in partner.policy.parameters()]))
    t = Thread(target=learn_thread, daemon=True)
    t.start()
    ego.learn(dummy_env, total_timesteps=timesteps)
    # env.step(env.action_space.sample())
    with dummy_env.obs_cv:
        dummy_env.dead = True
        dummy_env.obs_cv.notify()
    print('gradient updates', partner._n_updates)
    print('ego end', sum([param.data.mean() for param in partner.policy.parameters()]))
    t.join()
    # print('ego timesteps', dummy_env.steps)
    return partner.policy


@pytest.mark.timeout(60)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("ALGO", [DQN])
@pytest.mark.parametrize("epochs", [1, 5])
@pytest.mark.parametrize("option", [0, 1])
@pytest.mark.parametrize("n_steps", [10, 100, 1000])
def test_dqn(ALGO, epochs, option, n_steps):
    model1 = run_standard_dqn(ALGO, n_steps * epochs, option, n_steps)
    model2 = run_reversed_dqn(ALGO, n_steps * epochs, option, n_steps)
    assert check_equivalent_models(model1, model2), "NOT IDENTICAL MODELS"

    assert threading.active_count() == 1, "DID NOT KILL THREADS"


@pytest.mark.timeout(60)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("ALGO", [PPO, A2C])
@pytest.mark.parametrize("epochs", [1, 5])
@pytest.mark.parametrize("option", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n_steps", [10, 100, 1000])
def test_sarl(ALGO, epochs, option, n_steps):
    model1, rb1 = run_standard(ALGO, n_steps * epochs, option, n_steps)
    model2, rb2 = run_reversed(ALGO, n_steps * epochs, option, n_steps)
    assert check_equivalent_models(model1, model2), "NOT IDENTICAL MODELS"

    assert threading.active_count() == 1, "DID NOT KILL THREADS"


# def printifdiff(r1, r2, val):
#     if not np.array_equal(r1.__dict__[val], r2.__dict__[val]):
#         # print(val, False, r1.__dict__[val].flatten(), r2.__dict__[val].flatten())
#         idx = np.where(r1.__dict__[val] != r2.__dict__[val])[0][0]
#         print(val, False, idx, "Standard", r1.__dict__[val][idx], "Reversed", r2.__dict__[val][idx])
#     else:
#         print(val, True)

# def check_equivalent_buffers(rb1, rb2):
#     printifdiff(rb1, rb2, "observations")
#     printifdiff(rb1, rb2, "actions")
#     printifdiff(rb1, rb2, "rewards")
#     printifdiff(rb1, rb2, "advantages")
#     printifdiff(rb1, rb2, "returns")
#     printifdiff(rb1, rb2, "episode_starts")
#     printifdiff(rb1, rb2, "log_probs")
#     printifdiff(rb1, rb2, "values")

# def verify_actions(rb1, timesteps):
#     stable_baselines3.common.utils.set_random_seed(0)

#     env = make_env()
#     partner = FakeAgent(env.unwrapped.getDummyEnv(1))
#     env.unwrapped.ego_ind = 0
#     agents, observations = env.unwrapped.n_reset()
#     t = 0
#     while t < timesteps:
#         actions = []
#         if t == 80 or t == 81:
#             print(t, agents, [o.obs for o in observations], rewards, done)
#         for a, o in zip(agents, observations):
#             if a == 0:
#                 if not np.array_equal(o.obs, rb1.observations[t]):
#                     print("Observations mismatch at", t, o.obs, rb1.observations[t])
#                     return
#                 if t > 0 and rewards[0] != rb1.rewards[t-1]:
#                     print("rewards mismatch at", t, rewards[0], rb1.rewards[t-1])
#                     return
#                 actions.append(rb1.actions[t][0])
#                 t += 1
#             else:
#                 actions.append(partner.get_action(o))
#         agents, observations, rewards, done, _ = env.unwrapped.n_step(actions)
#         if done:
#             agents, observations = env.unwrapped.n_reset()
#     print("ALL GOOD")

# def verify_actions_reversed(rb2, timesteps):
#     print("TEST REVERSED")
#     stable_baselines3.common.utils.set_random_seed(0)

#     env = make_env()
#     env.unwrapped.ego_ind = 1
#     ego = FakeEgo(env)
#     partner = ReplayAgent(rb2)
#     env.unwrapped.add_partner_agent(partner, 0)

#     try:
#         ego.learn(partner, timesteps)
#     except Exception as e:
#         print(e)
#         return
#     print("ALL GOOD")

# check_equivalent_buffers(rb1, rb2)
# verify_actions(rb2, TIME)
# verify_actions_reversed(rb1, TIME)


# print()
# print(rb1.observations[80:85])
# print(rb2.actions[80:85])

# print()
# print(rb2.observations[80:85])
# print(rb2.actions[80:85])


# e = make_env()
# print(e.n_reset()[1][1].obs)

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