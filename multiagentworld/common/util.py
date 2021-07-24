from stable_baselines3.common import policies


class FeedForward32Policy(policies.ActorCriticPolicy):
    """A feed forward policy network with two hidden layers of 32 units.
    This matches the IRL policies in the original AIRL paper.
    Note: This differs from stable_baselines3 ActorCriticPolicy in two ways: by
    having 32 rather than 64 units, and by having policy and value networks
    share weights except at the final layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[32, 32])
