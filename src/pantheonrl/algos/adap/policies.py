"""
Module defining the Policy for ADAP
"""
# pylint: disable=locally-disabled, not-callable

from typing import Any, Dict, Optional, Type, Union, List, Tuple

import torch
import gymnasium as gym
from torch import nn

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)


class AdapPolicy(ActorCriticPolicy):
    """
    Base Policy for the ADAP Actor-critic policy
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[
            BaseFeaturesExtractor
        ] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        context_size: int = 3,
    ):
        self.context_size = context_size
        self.context = None
        self.mlp_extractor = None
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
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def set_context(self, ctxt):
        """Set the context"""
        self.context = ctxt

    def get_context(self):
        """Get the current context"""
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

    def _get_latent(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the latent code (activations of the last layer of each network)
        for the different networks.
        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        features = torch.cat(
            (features, self.context.repeat(features.size()[0], 1)), dim=1
        )
        latent_pi, latent_vf = self.mlp_extractor(features)

        return latent_pi, latent_vf

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        latents = obs[..., -self.context_size :].reshape(
            -1, self.context_size
        )[0]
        obs = obs[..., : -self.context_size].reshape(
            -1, obs.size(dim=-1) - self.context_size
        )
        features = self.extract_features(obs)
        latents = latents.to(features.device, features.dtype)
        features = torch.cat(
            (features, latents.repeat(features.size()[0], 1)), dim=1
        )
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        latents = obs[..., -self.context_size :].reshape(
            -1, self.context_size
        )[0]
        obs = obs[..., : -self.context_size].reshape(
            -1, obs.size(dim=-1) - self.context_size
        )
        features = super(BasePolicy, self).extract_features(
            obs, self.vf_features_extractor
        )
        latents = latents.to(features.device, features.dtype)
        features = torch.cat(
            (features, latents.repeat(features.size()[0], 1)), dim=1
        )
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        latents = obs[..., -self.context_size :].reshape(
            -1, self.context_size
        )[0]
        obs = obs[..., : -self.context_size].reshape(
            -1, obs.size(dim=-1) - self.context_size
        )
        print("NEW OBS", obs)
        features = self.extract_features(obs)
        latents = latents.to(features.device, features.dtype)
        print(features.shape, latents.shape)
        features = torch.cat(
            (features, latents.repeat(features.size()[0], 1)), dim=1
        )
        print(features.shape)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy


class MultModel(nn.Module):
    """Neural Network representing multiplicative layers"""

    def __init__(
        self, feature_dim, net_arch, activation_fn, device, context_size
    ):
        super().__init__()
        self.context_size = context_size
        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get(
                "pi", []
            )  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get(
                "vf", []
            )  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.hidden_dim1 = policy_net[0].out_features
        self.agent_branch_1 = nn.Sequential(*policy_net[0:2]).to(device)
        self.agent_scaling = nn.Sequential(
            nn.Linear(self.hidden_dim1, self.hidden_dim1 * self.context_size),
            activation_fn(),
        ).to(device)
        self.agent_branch_2 = nn.Sequential(*policy_net[2:]).to(device)

        self.hidden_dim2 = value_net[0].out_features
        self.value_branch_1 = nn.Sequential(*value_net[0:2]).to(device)
        self.value_scaling = nn.Sequential(
            nn.Linear(self.hidden_dim2, self.hidden_dim2 * self.context_size),
            activation_fn(),
        ).to(device)
        self.value_branch_2 = nn.Sequential(*value_net[2:]).to(device)

    def policies(
        self, observations: torch.Tensor, contexts: torch.Tensor
    ) -> torch.Tensor:
        """Returns the logits from the policy function"""
        batch_size = observations.shape[0]
        x = self.agent_branch_1(observations)
        x_a = self.agent_scaling(x)
        # reshape to do context multiplication
        x_a = x_a.view((batch_size, self.hidden_dim1, self.context_size))
        x_a_out = torch.matmul(x_a, contexts.unsqueeze(-1)).squeeze(-1)
        logits = self.agent_branch_2(x + x_a_out)

        return logits

    def values(
        self, observations: torch.Tensor, contexts: torch.Tensor
    ) -> torch.Tensor:
        """Returns the response from the value function"""
        batch_size = observations.shape[0]
        x = self.value_branch_1(observations)
        x_a = self.value_scaling(x)
        # reshape to do context multiplication
        x_a = x_a.view((batch_size, self.hidden_dim2, self.context_size))
        x_a_out = torch.matmul(x_a, contexts.unsqueeze(-1)).squeeze(-1)
        values = self.value_branch_2(x + x_a_out)
        # values = self.value_branch_2(x_a_out)

        return values

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the action logits and values"""
        # features = self.shared_net(features)
        observations = features[:, : -self.context_size]
        contexts = features[:, -self.context_size :]
        return self.policies(observations, contexts), self.values(
            observations, contexts
        )

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Returns the action logits and values"""
        # features = self.shared_net(features)
        observations = features[:, : -self.context_size]
        contexts = features[:, -self.context_size :]
        return self.policies(observations, contexts)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Returns the action logits and values"""
        # features = self.shared_net(features)
        observations = features[:, : -self.context_size]
        contexts = features[:, -self.context_size :]
        return self.values(observations, contexts)


class AdapPolicyMult(AdapPolicy):
    """
    Multiplicative Policy for the ADAP Actor-critic policy
    """

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
            context_size=self.context_size,
        )
