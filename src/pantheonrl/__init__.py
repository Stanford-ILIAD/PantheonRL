"""
`PantheonRL <https://github.com/Stanford-ILIAD/PantheonRL>`_ is a package for training and testing multi-agent reinforcement learning environments. The goal of PantheonRL is to provide a modular and extensible framework for training agent policies, fine-tuning agent policies, ad-hoc pairing of agents, and more.

PantheonRL is built to support Stable-Baselines3 (SB3), allowing direct access to many of SB3's standard RL training algorithms such as PPO. PantheonRL currently follows a decentralized training paradigm -- each agent is equipped with its own replay buffer and update algorithm. The agents objects are designed to be easily manipulable. They can be saved, loaded and plugged into different training procedures such as self-play, ad-hoc / cross-play, round-robin training, or finetuning.
"""
import pantheonrl.envs

from pantheonrl.common.agents import (
    Agent,
    StaticPolicyAgent,
    OnPolicyAgent,
    OffPolicyAgent
)

from pantheonrl.common.multiagentenv import (
    DummyEnv,
    MultiAgentEnv,
    TurnBasedEnv,
    SimultaneousEnv
)

from pantheonrl.common.observation import Observation
