from abc import ABC, abstractmethod
import collections
from typing import Union, Type, Dict, List, Tuple, Optional, Any, Callable
from functools import partial

import gym
import torch as th
import torch.nn as nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import preprocess_obs, is_image_space, get_action_dim
from stable_baselines3.common.torch_layers import (FlattenExtractor, BaseFeaturesExtractor, create_mlp,
                                                   NatureCNN, MlpExtractor)
from stable_baselines3.common.utils import get_device, is_vectorized_observation
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.distributions import (make_proba_distribution, Distribution,
                                                    DiagGaussianDistribution, CategoricalDistribution,
                                                    MultiCategoricalDistribution, BernoulliDistribution,
                                                    StateDependentNoiseDistribution)
from stable_baselines3.common.policies import BasePolicy

class ModularPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (Callable) Learning rate schedule (could be constant)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
    :param ortho_init: (bool) Whether to use or not orthogonal initialization
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: (bool) Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: (Type[BaseFeaturesExtractor]) Features extractor to use.
    :param features_extractor_kwargs: (Optional[Dict[str, Any]]) Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: (Type[th.optim.Optimizer]) The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: (Optional[Dict[str, Any]]) Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable[[float], float],
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 device: Union[th.device, str] = 'auto',
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 ortho_init: bool = True,
                 use_sde: bool = False,
                 log_std_init: float = 0.0,
                 full_std: bool = True,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 squash_output: bool = False,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,

                 # my additional arguments
                 num_partners: int = 1,
                 partner_net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None, # net arch for each partner-specific module
                 baseline: bool = False,
                 nomain: bool = False,
                 ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs['eps'] = 1e-5

        super(ModularPolicy, self).__init__(observation_space,
                                                action_space,
                                                features_extractor_class,
                                                features_extractor_kwargs,
                                                optimizer_class=optimizer_class,
                                                optimizer_kwargs=optimizer_kwargs,
                                                squash_output=squash_output)

        self.num_partners = num_partners
        print("CUDA: ", th.cuda.is_available())

        if partner_net_arch is None:
            if features_extractor_class == FlattenExtractor:
                partner_net_arch = [dict(pi=[64, 64], vf=[64, 64])]
            else:
                partner_net_arch = []
        self.partner_net_arch = partner_net_arch
        self.baseline = baseline
        self.nomain = nomain


        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]
            else:
                net_arch = []
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space,
                                                           **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                'full_std': full_std,
                'squash_output': squash_output,
                'use_expln': use_expln,
                'learn_features': sde_net_arch is not None
            }

        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self.lr_schedule = lr_schedule
        self._build(self.lr_schedule)

    # freeze / unfreeze the module networks
    def set_freeze_module(self, module, freeze):
        for param in module.parameters():
            param.requires_grad = not freeze
    def set_freeze_main(self, freeze):
        self.set_freeze_module(self.mlp_extractor, freeze)
        self.set_freeze_module(self.action_net, freeze)
        self.set_freeze_module(self.value_net, freeze)
    def set_freeze_partner(self, freeze):
        for partner_idx in range(self.num_partners):
            self.set_freeze_module(self.partner_mlp_extractor[partner_idx], freeze)
            self.set_freeze_module(self.partner_action_net[partner_idx], freeze)
            self.set_freeze_module(self.partner_value_net[partner_idx], freeze)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(dict(
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            use_sde=self.use_sde,
            log_std_init=self.log_std_init,
            squash_output=default_none_kwargs['squash_output'],
            full_std=default_none_kwargs['full_std'],
            sde_net_arch=default_none_kwargs['sde_net_arch'],
            use_expln=default_none_kwargs['use_expln'],
            lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
            ortho_init=self.ortho_init,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs
        ))
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.
        :param n_envs: (int)
        """
        assert isinstance(self.action_dist,
                          StateDependentNoiseDistribution), 'reset_noise() is only available when using gSDE'
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def make_action_dist_net(self, latent_dim_pi: int, latent_sde_dim: int = 0):
        action_net, log_std = None, None
        if isinstance(self.action_dist, DiagGaussianDistribution):
            action_net, log_std = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi,
                                                                                    log_std_init=self.log_std_init)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            action_net, log_std = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi,
                                                                                    latent_sde_dim=latent_sde_dim,
                                                                                    log_std_init=self.log_std_init)
        elif isinstance(self.action_dist, CategoricalDistribution):
            action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
        return action_net, log_std

    def build_mlp_action_value_net(self, input_dim, net_arch):
        mlp_extractor = MlpExtractor(input_dim, net_arch=net_arch,
                                          activation_fn=self.activation_fn, device=self.device)
        action_net, log_std = self.make_action_dist_net(mlp_extractor.latent_dim_pi)
        value_net = nn.Linear(mlp_extractor.latent_dim_vf, 1)
        return mlp_extractor, action_net, log_std, value_net

    def do_init_weights(self, init_main=False, init_partner=False):
        # Values from stable-baselines.
        # feature_extractor/mlp values are
        # originally from openai/baselines (default gains/init_scales).

        # # Init weights: use orthogonal initialization
        # # with small initial weight for the output
        # if self.ortho_init:
        module_gains = {}
        if init_main:
            module_gains[self.features_extractor] = np.sqrt(2)
            module_gains[self.mlp_extractor] = np.sqrt(2)
            module_gains[self.action_net] = 0.01
            module_gains[self.value_net] = 1
        if init_partner:
            for i in range(self.num_partners):
                module_gains[self.partner_mlp_extractor[i]] = np.sqrt(2)
                module_gains[self.partner_action_net[i]] = 0.01
                module_gains[self.partner_value_net[i]] = 1
        for module, gain in module_gains.items():
            module.apply(partial(self.init_weights, gain=gain))

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        """
        Create the networks and the optimizer.
        :param lr_schedule: (Callable) Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).

        self.mlp_extractor, self.action_net, self.log_std, self.value_net = self.build_mlp_action_value_net(input_dim=self.features_dim, net_arch=self.net_arch)

        partner_builds = [self.build_mlp_action_value_net(input_dim=self.mlp_extractor.latent_dim_pi, net_arch=self.partner_net_arch) for _ in range(self.num_partners)]
        if self.baseline: # use the same partner module for all partners
            print("Baseline architecture: using the same partner module.")
            partner_builds = [partner_builds[0]] * self.num_partners

        self.partner_mlp_extractor, self.partner_action_net, self.partner_log_std, self.partner_value_net = zip(*partner_builds)
        self.partner_mlp_extractor = nn.ModuleList(self.partner_mlp_extractor)
        self.partner_action_net = nn.ModuleList(self.partner_action_net)
        self.partner_value_net = nn.ModuleList(self.partner_value_net)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.do_init_weights(init_main=True, init_partner=True)

    def overwrite_main(self, other):
        self.mlp_extractor, self.action_net, self.log_std, self.value_net = other.mlp_extractor, other.action_net, other.log_std, other.value_net
        self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor,
                partner_idx: int,
                deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: (th.Tensor) Observation
        :param deterministic: (bool) Whether to sample or use deterministic actions
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) action, value and log probability of the action
        """
        latent_pi, latent_vf, _ = self._get_latent(obs=obs)
        partner_latent_pi, partner_latent_vf = self.partner_mlp_extractor[partner_idx](latent_pi)

        distribution = self._get_action_dist_from_latent(latent_pi, partner_latent_pi, partner_idx=partner_idx)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf) + self.partner_value_net[partner_idx](partner_latent_vf)

        return actions, values, log_prob

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.
        :param obs: (th.Tensor) Observation
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)

        return latent_pi, latent_vf, latent_sde

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor,
                                     partner_latent_pi: th.Tensor,
                                     partner_idx: int,
                                     latent_sde: Optional[th.Tensor] = None,
                                     action_mask: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: (th.Tensor) Latent code for the actor
        :param latent_sde: (Optional[th.Tensor]) Latent code for the gSDE exploration function
        :return: (Distribution) Action distribution
        """
        main_logits = self.action_net(latent_pi)
        partner_logits = self.partner_action_net[partner_idx](partner_latent_pi)

        if self.nomain:
            mean_actions = partner_logits
        else:
            mean_actions = main_logits + partner_logits
        
        large_exponent = 30
        if action_mask is not None:
            action_mask = action_mask.to(mean_actions.device)
            mean_actions = mean_actions - large_exponent*(~action_mask)
        th.clamp(mean_actions, min=-1*large_exponent)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            log_std = self.log_std + self.partner_log_std[partner_idx]
            return self.action_dist.proba_distribution(mean_actions, log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            log_std = self.log_std + self.partner_log_std[partner_idx]
            return self.action_dist.proba_distribution(mean_actions, log_std, latent_sde)
        else:
            raise ValueError('Invalid action distribution')

    def _predict(self, observation: th.Tensor, partner_idx: int, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.
        :param observation: (th.Tensor)
        :param deterministic: (bool) Whether to use stochastic or deterministic actions
        :return: (th.Tensor) Taken action according to the policy
        """
        actions, _, _ = self.forward(obs=observation, partner_idx=partner_idx, deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs: th.Tensor,
                         actions: th.Tensor,
                         partner_idx: int,
                         action_mask: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs: (th.Tensor)
        :param actions: (th.Tensor)
        :return: (th.Tensor, th.Tensor, th.Tensor) estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        latent_pi, latent_vf, _ = self._get_latent(obs=obs)
        partner_latent_pi, partner_latent_vf = self.partner_mlp_extractor[partner_idx](latent_pi)

        distribution = self._get_action_dist_from_latent(latent_pi, partner_latent_pi, partner_idx=partner_idx, action_mask=action_mask)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf) + self.partner_value_net[partner_idx](partner_latent_vf)
        return values, log_prob, distribution.entropy()

    def get_action_logits_from_obs(self, obs: th.Tensor, partner_idx: int, action_mask: Optional[th.Tensor] = None) -> th.Tensor:
        latent_pi, _, _ = self._get_latent(obs=obs)
        partner_latent_pi, _ = self.partner_mlp_extractor[partner_idx](latent_pi)

        main_logits = self.action_net(latent_pi)
        partner_logits = self.partner_action_net[partner_idx](partner_latent_pi)

        if action_mask: 
            main_logits = main_logits * action_mask   # set masked out options to 0
            partner_logits = partner_logits * action_mask

        return main_logits, partner_logits