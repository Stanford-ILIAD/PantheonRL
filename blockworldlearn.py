import gym
import time
import multiagentworld.envs.blockworldgym.simpleblockworld as sbw
from stable_baselines3 import PPO


from multiagentworld.envs.liargym.liar import LiarDefaultAgent
from multiagentworld.common.agents import OnPolicyAgent, OffPolicyAgent
from multiagentworld.common.wrappers import TurnBasedFrameStack

print(sbw.PLANNER_OBS_SPACE)
print(sbw.CONSTRUCTOR_OBS_SPACE)
env = gym.make("multiagentworld:SimpleBlockEnv-v0")
env.add_partner_policy(sbw.SBWEasyPartner())
env.reset()
env.render()
time.sleep(5)
env.close()

# model = PPO("MlpPolicy", env, verbose=1)
# wrappedmodel = OnPolicyAgent(model)
# wrappedmodel.learn(total_timesteps=100000)