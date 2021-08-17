import copy
from itertools import combinations

import torch as th
import numpy as np
from torch.distributions import kl
from stable_baselines3.common import distributions
from stable_baselines3.common.buffers import RolloutBufferSamples

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .adap_learn import ADAP
    from .policies import AdapPolicy


def kl_divergence(dist_true: distributions.Distribution,
                  dist_pred: distributions.Distribution) -> th.Tensor:
    """
    Wrapper for the PyTorch implementation of the full form KL Divergence
    :param dist_true: the p distribution
    :param dist_pred: the q distribution
    :return: KL(dist_true||dist_pred)
    """
    # KL Divergence for different distribution types is out of scope
    assert dist_true.__class__ == dist_pred.__class__, \
        "Error: input distributions should be the same type"

    # MultiCategoricalDistribution is not a PyTorch Distribution subclass
    # so we need to implement it ourselves!
    if isinstance(dist_pred, distributions.MultiCategoricalDistribution):
        return th.stack(
            [kl.kl_divergence(p, q) for p, q in zip(
                dist_true.distribution, dist_pred.distribution)],
            dim=1,
        ).sum(dim=1)

    # Use the PyTorch kl_divergence implementation
    else:
        return kl.kl_divergence(dist_true.distribution, dist_pred.distribution)


def get_L2_sphere(ctx_size, num, torch=False):
    if torch:
        ctxs = th.rand(num, ctx_size, device='cpu') * 2 - 1
        ctxs = ctxs / (th.sum((ctxs)**2, dim=-1).reshape(num, 1))**(1/2)
        ctxs = ctxs.to('cpu')
    else:
        ctxs = np.random.rand(num, ctx_size) * 2 - 1
        ctxs = ctxs / (np.sum((ctxs)**2, axis=-1).reshape(num, 1))**(1/2)
    return ctxs


def get_unit_square(ctx_size, num, torch=False):
    if torch:
        ctxs = th.rand(num, ctx_size) * 2 - 1
    else:
        ctxs = np.random.rand(num, ctx_size) * 2 - 1
    return ctxs


def get_positive_square(ctx_size, num, torch=False):
    if torch:
        ctxs = th.rand(num, ctx_size)
    else:
        ctxs = np.random.rand(num, ctx_size)
    return ctxs


def get_categorical(ctx_size, num, torch=False):
    if torch:
        ctxs = th.zeros(num, ctx_size)
        ctxs[th.arange(num), th.randint(0, ctx_size, size=(num,))] = 1
    else:
        ctxs = np.zeros((num, ctx_size))
        ctxs[np.arange(num), np.random.randint(0, ctx_size, size=(num,))] = 1
    return ctxs


def get_natural_number(ctx_size, num, torch=False):
    '''
    Returns context vector of shape (num,1) with numbers in range [0, ctx_size]
    '''
    if torch:
        ctxs = th.randint(0, ctx_size, size=(num, 1))
    else:
        ctxs = np.random.randint(0, ctx_size, size=(num, 1))
    return ctxs


SAMPLERS = {"l2": get_L2_sphere,
            "unit_square": get_unit_square,
            "positive_square": get_positive_square,
            "categorical": get_categorical,
            "natural_numbers": get_natural_number}


def get_context_kl_loss(policy: 'ADAP', model: 'AdapPolicy',
                        train_batch: RolloutBufferSamples):

    original_obs = train_batch.observations[:, :-policy.context_size]

    context_size = policy.context_size
    num_context_samples = policy.num_context_samples
    num_state_samples = policy.num_state_samples

    indices = th.randperm(original_obs.shape[0])[:num_state_samples]
    sampled_states = original_obs[indices]
    num_state_samples = min(num_state_samples, sampled_states.shape[0])

    all_contexts = set()
    all_action_dists = []
    old_context = model.get_context()
    for i in range(0, num_context_samples):  # 10 sampled contexts
        sampled_context = SAMPLERS[policy.context_sampler](
            ctx_size=context_size, num=1, torch=True)

        if sampled_context in all_contexts:
            continue

        all_contexts.add(sampled_context)
        model.set_context(sampled_context)
        latent_pi, _, latent_sde = model._get_latent(sampled_states)
        context_action_dist = model._get_action_dist_from_latent(
            latent_pi, latent_sde)
        all_action_dists.append(copy.copy(context_action_dist))

    model.set_context(old_context)
    all_CLs = [th.mean(th.exp(-kl_divergence(a, b)))
               for a, b in combinations(all_action_dists, 2)]
    rawans = sum(all_CLs)/len(all_CLs)
    return rawans
