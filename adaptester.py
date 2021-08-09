import gym
from stable_baselines3 import PPO

from multiagentworld.envs.liargym.liar import LiarDefaultAgent
from multiagentworld.envs.rpsgym.rps import RPSWeightedAgent
from multiagentworld.common.agents import OnPolicyAgent, OffPolicyAgent
from multiagentworld.common.wrappers import TurnBasedFrameStack

from multiagentworld.algos.adap.adap_learn import ADAP
from multiagentworld.algos.adap.policies import AdapPolicyMult

env = gym.make("LiarsDice-v0")
env.add_partner_policy(LiarDefaultAgent())

# env = gym.make("RPS-v0")
# env.add_partner_policy(RPSWeightedAgent(0, 1, 1))
# env.add_partner_policy(OnPolicyAgent(PPO("MlpPolicy", env)))

model = ADAP(AdapPolicyMult, env, verbose=1,
             context_size=3, context_loss_coeff=0.1)
# model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
