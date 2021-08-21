import dataclasses
from abc import ABC, abstractmethod
from typing import Dict, Mapping, Sequence, Union, TypeVar, overload

import numpy as np
import torch as th
from torch.utils import data as th_data
from torch.utils.data._utils.collate import default_collate

from .util import get_space_size


T = TypeVar("T")


def transitions_collate_fn(
    batch: Sequence[Mapping[str, np.ndarray]],
) -> Dict[str, Union[np.ndarray, th.Tensor]]:
    """
    This function is from HumanCompatibleAI's imitation repo:
    https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/
    data/types.py

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
    This function is from HumanCompatibleAI's imitation repo:
    https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/
    data/types.py

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
    """
    This class is modified from HumanCompatibleAI's imitation repo:
    https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/
    data/types.py

    A Torch-compatible `Dataset` of obs-act transitions.
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

    def write_transition(self, file):
        full_list = np.concatenate((self.obs, self.acts), axis=1)
        np.save(file, full_list)

    @classmethod
    def read_transition(cls, file, obs_space, act_space):
        full_list = np.load(file)
        obs_size = get_space_size(obs_space)
        obs = full_list[:, :obs_size]
        acts = full_list[:, obs_size:]
        return TransitionsMinimal(obs, acts)


class MultiTransitions(ABC):
    """ Base class for all classes that store multiple transitions """

    @abstractmethod
    def get_ego_transitions(self) -> TransitionsMinimal:
        """ Returns the ego's transitions """

    @abstractmethod
    def get_alt_transitions(self) -> TransitionsMinimal:
        """ Returns the partner's transitions """


@dataclasses.dataclass(frozen=True)
class TurnBasedTransitions(MultiTransitions):
    obs: np.ndarray
    acts: np.ndarray
    flags: np.ndarray

    def get_ego_transitions(self) -> TransitionsMinimal:
        """ Returns the ego's transitions """
        mask = (self.flags % 2 == 0)
        return TransitionsMinimal(self.obs[mask], self.acts[mask])

    def get_alt_transitions(self) -> TransitionsMinimal:
        """ Returns the partner's transitions """
        mask = (self.flags % 2 == 1)
        return TransitionsMinimal(self.obs[mask], self.acts[mask])

    def write_transition(self, file):
        flags = np.reshape(self.flags, (-1, 1))
        len = flags.shape[0]

        obs = np.reshape(self.obs, (len, -1))
        acts = np.reshape(self.acts, (len, -1))

        full_list = np.concatenate((obs, acts, flags), axis=1)
        np.save(file, full_list)

    @classmethod
    def read_transition(cls, file, obs_space, act_space):
        full_list = np.load(file)
        obs_size = get_space_size(obs_space)
        obs = full_list[:, :obs_size]
        acts = full_list[:, obs_size:-1]
        flags = full_list[:, -1]

        return TurnBasedTransitions(obs, acts, flags)


@dataclasses.dataclass(frozen=True)
class SimultaneousTransitions(MultiTransitions):
    egoobs: np.ndarray
    egoacts: np.ndarray
    altobs: np.ndarray
    altacts: np.ndarray
    flags: np.ndarray

    def get_ego_transitions(self) -> TransitionsMinimal:
        """ Returns the ego's transitions """
        return TransitionsMinimal(self.egoobs, self.egoacts)

    def get_alt_transitions(self) -> TransitionsMinimal:
        """ Returns the partner's transitions """
        return TransitionsMinimal(self.altobs, self.altacts)

    def write_transition(self, file):
        flags = np.reshape(self.flags, (-1, 1))
        len = flags.shape[0]
        egoobs = np.reshape(self.egoobs, (len, -1))
        egoacts = np.reshape(self.egoacts, (len, -1))
        altobs = np.reshape(self.altobs, (len, -1))
        altacts = np.reshape(self.altacts, (len, -1))
        full_list = np.concatenate(
                (egoobs, egoacts, altobs, altacts, flags),
                axis=1
            )
        np.save(file, full_list)

    @classmethod
    def read_transition(cls, file, obs_space, act_space):
        full_list = np.load(file)
        obs_size = get_space_size(obs_space)
        act_size = get_space_size(act_space)
        egoobs = full_list[:, :obs_size]
        egoacts = full_list[:, obs_size:(obs_size + act_size)]
        altobs = full_list[:, (obs_size + act_size):(2 * obs_size + act_size)]
        altacts = full_list[:, (2*obs_size + act_size):-1]
        flags = full_list[:, -1]

        return SimultaneousTransitions(egoobs, egoacts, altobs, altacts, flags)
