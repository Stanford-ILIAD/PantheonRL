import gym
from stable_baselines3 import PPO

from multiagentworld.envs.liargym.liar import LiarDefaultAgent
from multiagentworld.envs.rpsgym.rps import RPSWeightedAgent
from multiagentworld.common.agents import OnPolicyAgent, OffPolicyAgent
from multiagentworld.common.wrappers import SimultaneousFrameStack

from multiagentworld.algos.adap.adap_learn import ADAP
from multiagentworld.algos.adap.policies import AdapPolicyMult, AdapPolicy
from multiagentworld.algos.adap.agent import AdapAgent

env = SimultaneousFrameStack(
    gym.make("OvercookedMultiEnv-v0", layout_name="random0"), numframes=3)
model = ADAP(AdapPolicy, env, verbose=1, context_sampler="categorical",
             context_size=3, context_loss_coeff=0.1)
env.and_partner_agent(AdapAgent(ADAP(AdapPolicy, env, context_sampler="categorical",
                                      context_size=3, context_loss_coeff=0),
                                 model.policy))

model.learn(total_timesteps=100000)
