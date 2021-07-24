import gym

from multiagentworld.envs.liargym.liar import LiarDefaultAgent
from multiagentworld.common.agents import (RecordingAgentWrapper,
                                           StaticPolicyAgent)
from multiagentworld.algos.bc import BC
# from multiagentworld.common import trajsaver

env = gym.make('LiarsDice-v0')
policy = RecordingAgentWrapper(LiarDefaultAgent())
env.add_partner_policy(LiarDefaultAgent())

trainsteps = 100000
numgames = 100
rewards = []
for game in range(numgames):
    obs = env.reset()
    done = False
    while not done:
        action = policy.get_action(obs, False)
        obs, reward, done, _ = env.step(action)
    rewards.append(reward)

print(f"numwin: {len([x for x in rewards if x == 1])/numgames}")
print(f"numlose: {len([x for x in rewards if x == -1])/numgames}")

transition = policy.get_transitions()

# trajsaver.write_transition(transition, "temptransition")
# transition = trajsaver.read_transition(
#     "temptransition.npy", env.observation_space, env.action_space)

clone = BC(observation_space=env.observation_space,
           action_space=env.action_space,
           expert_data=transition)

clone.train(n_epochs=trainsteps/numgames)
# clone.save_policy("bcliar")

policy = StaticPolicyAgent(clone.policy, env)
numgames = 10000
rewards = []
for game in range(numgames):
    obs = env.reset()
    done = False
    while not done:
        action = policy.get_action(obs, False)
        obs, reward, done, _ = env.step(action)
    rewards.append(reward)

print(f"numwin: {len([x for x in rewards if x == 1])/numgames}")
print(f"numlose: {len([x for x in rewards if x == -1])/numgames}")
