import gym

from multiagentworld.envs.liargym.liar import LiarDefaultAgent
from multiagentworld.envs.rpsgym.rps import RPSWeightedAgent
from multiagentworld.common.agents import StaticPolicyAgent
from multiagentworld.common.wrappers import (TurnBasedRecorder,
                                             SimultaneousRecorder)
from multiagentworld.algos.bc import BC
from multiagentworld.common import trajsaver


def run_game(env, numgames, policy, verbose=True):
    rewards = []
    for game in range(numgames):
        obs = env.reset()
        done = False
        while not done:
            action = policy.get_action(obs, False)
            obs, reward, done, _ = env.step(action)
        rewards.append(reward)
    if verbose:
        print(f"numwin: {len([x for x in rewards if x == 1])/numgames}")
        print(f"numlose: {len([x for x in rewards if x == -1])/numgames}")
    return rewards


def liars(write=False, save=False):
    env = gym.make('LiarsDice-v0')
    policy = LiarDefaultAgent()
    env.add_partner_policy(LiarDefaultAgent())

    trainsteps = 100000
    numgames = 100
    recorder = TurnBasedRecorder(env)
    run_game(recorder, numgames, policy)

    transition = recorder.get_transitions()

    if write:
        transition.write_transition("temptransition")
        transition = trajsaver.TurnBasedTransitions.read_transition(
            "temptransition.npy", env.observation_space, env.action_space)

    clone = BC(observation_space=env.observation_space,
               action_space=env.action_space,
               expert_data=transition.get_ego_transitions(),
               l2_weight=0)

    clone.train(n_epochs=trainsteps/numgames)

    if save:
        clone.save_policy("bcliar")

    run_game(env, 1000, StaticPolicyAgent(clone.policy))


def rps(write=False, save=False):
    env = gym.make('RPS-v0')
    policy = RPSWeightedAgent()
    env.add_partner_policy(RPSWeightedAgent())

    trainsteps = 100000
    numgames = 100
    recorder = SimultaneousRecorder(env)
    run_game(recorder, numgames, policy)

    transition = recorder.get_transitions()

    if write:
        transition.write_transition("temptransition")
        transition = trajsaver.SimultaneousTransitions.read_transition(
            "temptransition.npy", env.observation_space, env.action_space)

    clone = BC(observation_space=env.observation_space,
               action_space=env.action_space,
               expert_data=transition.get_ego_transitions(),
               l2_weight=0)

    clone.train(n_epochs=trainsteps/numgames)

    if save:
        clone.save_policy("bcliar")

    run_game(env, 1000, StaticPolicyAgent(clone.policy))


if __name__ == '__main__':
    liars()
