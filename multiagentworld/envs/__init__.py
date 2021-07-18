from gym.envs.registration import registry, register, make, spec

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
    entry_point='multiagentworld.envs.overcookedgym.overcooked:OvercookedMultiEnv'
)
