# lists of options for different settings

# contains environment display names and url names
ENV_LIST = {"block world": "blockworld", "block world (simple)": "simpleblockworld", "overcooked": "overcooked",
            "liar's dice": "liar", "rock-paper-scissors": "rps"}

ADAP_TYPES = ['ADAP', 'ADAP_MULT']
EGO_LIST = ['PPO', 'ModularAlgorithm'] + ADAP_TYPES
PARTNER_LIST = ['PPO', 'DEFAULT', 'FIXED'] + ADAP_TYPES

LAYOUT_LIST = ['corridor', 'five_by_five', 'mdp_test', 'multiplayer_schelling',
               'random0', 'random1', 'random2', 'random3', 'scenario1_s',
               'scenario2', 'scenario2_s', 'scenario3', 'scenario4',
               'schelling', 'schelling_s', 'simple', 'simple_single',
               'simple_tomato', 'small_corridor', 'unident', 'unident_s']