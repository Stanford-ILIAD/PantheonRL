import gym
from stable_baselines3 import PPO

from multiagentworld.envs.liargym.liar import LiarDefaultAgent
from multiagentworld.envs.rpsgym.rps import RPSWeightedAgent
from multiagentworld.common.agents import OnPolicyAgent, OffPolicyAgent
from multiagentworld.common.wrappers import SimultaneousFrameStack

from multiagentworld.algos.modular.learn import ModularAlgorithm
from multiagentworld.algos.modular.policies import ModularPolicy

env = SimultaneousFrameStack(
    gym.make("OvercookedMultiEnv-v0", layout_name="random0"), numframes=3)


env.add_partner_agent(OnPolicyAgent(PPO("MlpPolicy", env)))
env.add_partner_agent(OnPolicyAgent(PPO("MlpPolicy", env)))

policy_kwargs = dict(num_partners=len(env.partners))
model = ModularAlgorithm(ModularPolicy, env, verbose=1, policy_kwargs=policy_kwargs)

model.learn(total_timesteps=100000)
