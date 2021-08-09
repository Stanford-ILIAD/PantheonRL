from gym.envs.registration import register

register(
    id='RPS-v0',
    entry_point='multiagentworld.envs.rpsgym.rps:RPSEnv'
)

register(
    id='LiarsDice-v0',
    entry_point='multiagentworld.envs.liargym.liar:LiarEnv'
)

register(
    id='OvercookedMultiEnv-v0',
    entry_point='multiagentworld.envs.overcookedgym.overcooked:'
                'OvercookedMultiEnv'
)

register(
    id='BlockEnv-v0',
    entry_point='multiagentworld.envs.blockworldgym.simpleblockworld:SimpleBlockEnv'
)

register(
    id='BlockEnv-v1',
    entry_point='multiagentworld.envs.blockworldgym.blockworld:BlockEnv'
)

register(
    id='PartnerBlockEnv-v0',
    entry_point='multiagentworld.envs.blockworldgym.simpleblockworld:PartnerEnv'
)