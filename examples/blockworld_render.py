"""
This is a simple example training script for PantheonRL.
"""

import gymnasium as gym
from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent, StaticPolicyAgent

from pantheonrl.envs.blockworldgym.blockworld import BlockEnv

import time

# Since pantheonrl's MultiAgentEnv is a subclass of the gym Env, you can
# register an environment and construct it using gym.make.
env = gym.make('BlockEnv-v1')

# Before training your ego agent, you first need to add your partner agents
# to the environment. You can create adaptive partner agents using
# OnPolicyAgent (for PPO/A2C) or OffPolicyAgent (for DQN/SAC). If you set
# verbose to true for these agents, you can also see their learning progress
partner = OnPolicyAgent(PPO('MlpPolicy', env.unwrapped.get_dummy_env(1), verbose=1))
env.unwrapped.add_partner_agent(partner)

# Finally, you can construct an ego agent and train it in the environment
ego = PPO('MlpPolicy', env, verbose=1)
ego.learn(total_timesteps=1000)


player1 = StaticPolicyAgent(ego.policy)
player2 = StaticPolicyAgent(partner.model.policy)

test_env = BlockEnv()
test_env.unwrapped.set_ego_extractor(lambda x: x)
test_env.unwrapped.add_partner_agent(player2)

obs, _ = test_env.reset()
done = False
while True:
    time.sleep(1)
    test_env.render()
    obs, rew, terminated, truncated, _ = test_env.step(player1.get_action(obs))

    if terminated or truncated:
        break
