import gym
import multiagentworld.envs.blockworldgym.simpleblockworld as sbw

print(sbw.PLANNER_OBS_SPACE)
print(sbw.CONSTRUCTOR_OBS_SPACE)
env = gym.make("multiagentworld:SimpleBlockEnv-v0")
print(env.reset())
