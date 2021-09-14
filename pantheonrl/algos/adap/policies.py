from typing import Any, Dict, Optional, Type, Union, List, Tuple
from itertools import zip_longest

import torch as th
import gym
from torch import nn

from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor
)


class AdapPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] =
        FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        context_size: int = 3
    ):
        self.context_size = context_size
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def set_context(self, ctxt):
        self.context = ctxt

    def get_context(self):
        return self.context

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim + self.context_size,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _get_latent(self,
                    obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (activations of the last layer of each network)
        for the different networks.
        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        features = th.cat(
            (features, self.context.repeat(features.size()[0], 1)),
            dim=1)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde

    def evaluate_actions(self,
                         obs: th.Tensor,
                         actions: th.Tensor
                         ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs[:, :-self.context_size])
        features = th.cat(
            (features, obs[:, -self.context_size:]),
            dim=1)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()


class MultModel(MlpExtractor):
    def __init__(
                    self,
                    feature_dim,
                    net_arch,
                    activation_fn,
                    device,
                    context_size
                ):
        nn.Module.__init__(self)

        self.obs_space_size = feature_dim + context_size
        self.context_size = context_size

        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        # Layer sizes of the network that only belongs to the policy network
        policy_only_layers = []
        # Layer sizes of the network that only belongs to the value network
        value_only_layers = []
        last_layer_dim_shared = feature_dim

        # Iterate through shared layers and build shared parts of the network
        for layer in net_arch:
            if isinstance(layer, int):  # Check that this is a shared layer
                # TODO: give layer a meaningful name
                # add linear of size layer
                shared_net.append(nn.Linear(last_layer_dim_shared, layer))
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer
            else:
                assert isinstance(layer, dict), \
                    "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), \
                        "Error: net_arch[-1]['pi'] must \
                        contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), \
                        "Error: net_arch[-1]['vf'] must \
                        contain a list of integers."
                    value_only_layers = layer["vf"]
                break

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for pi_layer_size, vf_layer_size in zip_longest(policy_only_layers,
                                                        value_only_layers):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), \
                    "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), \
                    "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If list of layers is empty, the network is an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)

        self.hidden_dim1 = policy_net[0].out_features
        self.agent_branch_1 = nn.Sequential(*policy_net[0:2]).to(device)
        self.agent_scaling = nn.Sequential(
            nn.Linear(self.hidden_dim1, self.hidden_dim1 * self.context_size),
            activation_fn()
        ).to(device)
        self.agent_branch_2 = nn.Sequential(*policy_net[2:]).to(device)

        self.hidden_dim2 = value_net[0].out_features
        self.value_branch_1 = nn.Sequential(*value_net[0:2]).to(device)
        self.value_scaling = nn.Sequential(
            nn.Linear(self.hidden_dim2, self.hidden_dim2 * self.context_size),
            activation_fn()
        ).to(device)
        self.value_branch_2 = nn.Sequential(*value_net[2:]).to(device)

    def get_input_size_excluding_ctx(self):
        return self.obs_space_size - self.context_size

    def get_input_size_inluding_ctx(self):
        return self.obs_space_size

    def policies(self, observations: th.Tensor,
                 contexts: th.Tensor) -> th.Tensor:

        batch_size = observations.shape[0]
        x = self.agent_branch_1(observations)
        x_a = self.agent_scaling(x)
        # reshape to do context multiplication
        x_a = x_a.view((batch_size, self.hidden_dim1, self.context_size))
        x_a_out = th.matmul(x_a, contexts.unsqueeze(-1)).squeeze(-1)
        logits = self.agent_branch_2(x + x_a_out)

        return logits

    def values(self, observations: th.Tensor,
               contexts: th.Tensor) -> th.Tensor:

        batch_size = observations.shape[0]
        x = self.value_branch_1(observations)
        x_a = self.value_scaling(x)
        # reshape to do context multiplication
        x_a = x_a.view((batch_size, self.hidden_dim2, self.context_size))
        x_a_out = th.matmul(x_a, contexts.unsqueeze(-1)).squeeze(-1)
        values = self.value_branch_2(x + x_a_out)
        # values = self.value_branch_2(x_a_out)

        return values

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        features = self.shared_net(features)
        observations = features[:, :-self.context_size]
        contexts = features[:, -self.context_size:]
        return self.policies(observations, contexts), \
            self.values(observations, contexts)


class AdapPolicyMult(AdapPolicy):

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MultModel(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            context_size=self.context_size
        )
