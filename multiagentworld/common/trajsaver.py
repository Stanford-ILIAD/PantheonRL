# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long

import dataclasses
from typing import Dict, Mapping, Sequence, Union, TypeVar, overload

import numpy as np
import torch as th
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from torch.utils import data as th_data
from torch.utils.data.dataloader import default_collate


T = TypeVar("T")


def write_transition(transitions, file, append=False):
    full_list = np.concatenate((transitions.obs, transitions.acts), axis=1)
    np.save(file, full_list)


def get_space_size(space):
    if isinstance(space, Box):
        return len(space.low)
    elif isinstance(space, Discrete):
        return 1
    elif isinstance(space, MultiBinary):
        return space.n
    elif isinstance(space, MultiDiscrete):
        return len(space.nvec)
    else:
        raise NotImplementedError


def read_transition(file, obs_space, act_space):
    full_list = np.load(file)
    obs_size = get_space_size(obs_space)
    obs = full_list[:, :obs_size]
    acts = full_list[:, obs_size:]
    return TransitionsMinimal(obs, acts)


def transitions_collate_fn(
    batch: Sequence[Mapping[str, np.ndarray]],
) -> Dict[str, Union[np.ndarray, th.Tensor]]:
    """
    Custom `torch.utils.data.DataLoader` collate_fn for `TransitionsMinimal`.
    Use this as the `collate_fn` argument to `DataLoader` if using an instance
    of `TransitionsMinimal` as the `dataset` argument.
    """
    batch_no_infos = [
        {k: v for k, v in sample.items()} for sample in batch
    ]

    result = default_collate(batch_no_infos)
    assert isinstance(result, dict)
    return result


def dataclass_quick_asdict(dataclass_instance) -> dict:
    """
    Extract dataclass to items using `dataclasses.fields` + dict comprehension.
    This is a quick alternative to `dataclasses.asdict`, which expensively and
    undocumentedly deep-copies every numpy array value.
    See https://stackoverflow.com/a/52229565/1091722.
    """
    obj = dataclass_instance
    d = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return d


@dataclasses.dataclass(frozen=True)
class TransitionsMinimal(th_data.Dataset):
    """A Torch-compatible `Dataset` of obs-act transitions.
    This class and its subclasses are usually instantiated via
    `imitation.data.rollout.flatten_trajectories`.
    Indexing an instance `trans` of TransitionsMinimal with an integer `i`
    returns the `i`th `Dict[str, np.ndarray]` sample, whose keys are the field
    names of each dataclass field and whose values are the ith elements of each
    field value.
    Slicing returns a possibly empty instance of `TransitionsMinimal` where
    each field has been sliced.
    """

    obs: np.ndarray
    """
    Previous observations. Shape: (batch_size, ) + observation_shape.
    The i'th observation `obs[i]` in this array is the observation seen
    by the agent when choosing action `acts[i]`. `obs[i]` is not required to
    be from the timestep preceding `obs[i+1]`.
    """

    acts: np.ndarray
    """Actions. Shape: (batch_size,) + action_shape."""

    def __len__(self):
        """Returns number of transitions. Always positive."""
        return len(self.obs)

    def __post_init__(self):
        """Performs input validation: check shapes & dtypes match docstring.
        Also make array values read-only.
        """
        for val in vars(self).values():
            if isinstance(val, np.ndarray):
                val.setflags(write=False)

        if len(self.obs) != len(self.acts):
            raise ValueError(
                "obs and acts must have same number of timesteps: "
                f"{len(self.obs)} != {len(self.acts)}"
            )

    @overload
    def __getitem__(self: T, key: slice) -> T:
        pass  # pragma: no cover

    @overload
    def __getitem__(self, key: int) -> Dict[str, np.ndarray]:
        pass  # pragma: no cover

    def __getitem__(self, key):
        """See TransitionsMinimal docstring for indexing and slicing semantics.
        """
        d = dataclass_quick_asdict(self)
        d_item = {k: v[key] for k, v in d.items()}

        if isinstance(key, slice):
            # Return type is the same as this dataclass. Replace field value
            # with slices.
            return dataclasses.replace(self, **d_item)
        else:
            assert isinstance(key, int)
            # Return type is a dictionary. Array values have no batch dimension
            #
            # Dictionary of np.ndarray values is a convenient
            # torch.util.data.Dataset return type, as a
            # torch.util.data.DataLoader taking in this `Dataset` as its first
            # argument knows how to automatically concatenate several
            # dictionaries together to make a single dictionary batch with
            # `torch.Tensor` values.
            return d_item
