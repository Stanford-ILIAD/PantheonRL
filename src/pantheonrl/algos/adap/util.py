"""
Collection of helper functions for ADAP
"""

import copy
from itertools import combinations

from typing import TYPE_CHECKING

import torch
import numpy as np
from torch.distributions import kl
from stable_baselines3.common import distributions
from stable_baselines3.common.buffers import RolloutBufferSamples


if TYPE_CHECKING:
    from .adap_learn import ADAP
    from .policies import AdapPolicy


def kl_divergence(
    dist_true: distributions.Distribution,
    dist_pred: distributions.Distribution,
) -> torch.Tensor:
    """
    Wrapper for the PyTorch implementation of the full form KL Divergence
    :param dist_true: the p distribution
    :param dist_pred: the q distribution
    :return: KL(dist_true||dist_pred)
    """
    # KL Divergence for different distribution types is out of scope
    assert (
        dist_true.__class__ == dist_pred.__class__
    ), "Error: input distributions should be the same type"

    # MultiCategoricalDistribution is not a PyTorch Distribution subclass
    # so we need to implement it ourselves!
    if isinstance(dist_pred, distributions.MultiCategoricalDistribution):
        return torch.stack(
            [
                kl.kl_divergence(p, q)
                for p, q in zip(dist_true.distribution, dist_pred.distribution)
            ],
            dim=1,
        ).sum(dim=1)

    # Use the PyTorch kl_divergence implementation
    return kl.kl_divergence(dist_true.distribution, dist_pred.distribution)


def get_l2_sphere(ctx_size, num, use_torch=False):
    """ Samples from l2 sphere """
    if use_torch:
        ctxs = torch.rand(num, ctx_size, device="cpu") * 2 - 1
        ctxs = ctxs / (((ctxs) ** 2).sum(dim=-1).reshape(num, 1)) ** (1 / 2)
        ctxs = ctxs.to("cpu")
    else:
        ctxs = np.random.rand(num, ctx_size) * 2 - 1
        ctxs = ctxs / (np.sum((ctxs) ** 2, axis=-1).reshape(num, 1)) ** (1 / 2)
    return ctxs


def get_unit_square(ctx_size, num, use_torch=False):
    """ Samples from unit square centered at 0 """
    if use_torch:
        ctxs = torch.rand(num, ctx_size) * 2 - 1
    else:
        ctxs = np.random.rand(num, ctx_size) * 2 - 1
    return ctxs


def get_positive_square(ctx_size, num, use_torch=False):
    """ Samples from the square with axes between 0 and 1 """
    if use_torch:
        ctxs = torch.rand(num, ctx_size)
    else:
        ctxs = np.random.rand(num, ctx_size)
    return ctxs


def get_categorical(ctx_size, num, use_torch=False):
    """ Samples from categorical distribution """
    if use_torch:
        ctxs = torch.zeros(num, ctx_size)
        ctxs[torch.arange(num), torch.randint(0, ctx_size, size=(num,))] = 1
    else:
        ctxs = np.zeros((num, ctx_size))
        ctxs[np.arange(num), np.random.randint(0, ctx_size, size=(num,))] = 1
    return ctxs


def get_natural_number(ctx_size, num, use_torch=False):
    """
    Returns context vector of shape (num,1) with numbers in range [0, ctx_size]
    """
    if use_torch:
        ctxs = torch.randint(0, ctx_size, size=(num, 1))
    else:
        ctxs = np.random.randint(0, ctx_size, size=(num, 1))
    return ctxs


SAMPLERS = {
    "l2": get_l2_sphere,
    "unit_square": get_unit_square,
    "positive_square": get_positive_square,
    "categorical": get_categorical,
    "natural_numbers": get_natural_number,
}


def get_context_kl_loss(
    policy: "ADAP", model: "AdapPolicy", train_batch: RolloutBufferSamples
):
    """ Gets the KL loss for ADAP """

    original_obs = train_batch.observations[:, : -policy.context_size]

    context_size = policy.context_size
    num_context_samples = policy.num_context_samples
    num_state_samples = policy.num_state_samples

    indices = torch.randperm(original_obs.shape[0])[:num_state_samples]
    sampled_states = original_obs[indices]
    num_state_samples = min(num_state_samples, sampled_states.shape[0])

    all_contexts = set()
    all_action_dists = []
    old_context = model.get_context()
    for _ in range(0, num_context_samples):  # 10 sampled contexts
        sampled_context = SAMPLERS[policy.context_sampler](
            ctx_size=context_size, num=1, use_torch=True
        )

        if sampled_context in all_contexts:
            continue

        all_contexts.add(sampled_context)
        model.set_context(sampled_context)
        latent_pi, _, latent_sde = model._get_latent(sampled_states)
        context_action_dist = model._get_action_dist_from_latent(
            latent_pi, latent_sde
        )
        all_action_dists.append(copy.copy(context_action_dist))

    model.set_context(old_context)
    all_cls = [
        torch.mean(torch.exp(-kl_divergence(a, b)))
        for a, b in combinations(all_action_dists, 2)
    ]
    rawans = sum(all_cls) / len(all_cls)
    return rawans
