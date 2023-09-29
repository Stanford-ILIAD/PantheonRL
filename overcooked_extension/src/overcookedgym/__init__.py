from gymnasium.envs.registration import register

register(
    id='OvercookedMultiEnv-v0',
    entry_point='overcookedgym.overcooked:OvercookedMultiEnv'
)
