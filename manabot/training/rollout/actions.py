"""Lossless legal-action storage between collection and minibatch learning.

PackedActions owns row offsets, original action coordinates and every legal
behavior log probability. Object observations stay on the CPU with
the rollout; only a selected minibatch reconstructs the dense policy ABI. Native
buffers and returned logits retain their configured action axis.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ActionValidator(Protocol):
    """Numerical admission of retained behavior, preserving backend evidence."""

    def packing_mask(
        self, mask: NDArray[np.generic], logs: NDArray[np.generic]
    ) -> NDArray[np.bool_]: ...

    def check_packed(
        self,
        packed: "PackedActions",
        actions: NDArray[np.generic],
        selected_logs: NDArray[np.generic],
    ) -> None: ...


ACTION_FIELDS = frozenset(
    {"actions", "actions_valid", "action_focus", "name_candidates"}
)


@dataclass
class ActionRows:
    observation: dict[str, NDArray[np.generic]]
    log_probabilities: NDArray[np.generic]


@dataclass
class PackedActions:
    offsets: NDArray[np.int64]
    slots: NDArray[np.int32]
    capacity: int
    fields: dict[str, NDArray[np.generic]]
    log_probabilities: NDArray[np.generic]

    @classmethod
    def pack(
        cls,
        observation: dict[str, NDArray[np.generic]],
        log_probabilities: NDArray[np.generic],
        *,
        validator: ActionValidator,
    ) -> "PackedActions":
        """Copy legal rows from [B,A,...]; reject malformed masks or padding mass.

        Distribution/selected-action checks run before learning on every row.
        Padding must be checked here because it is deliberately not retained.
        """
        valid = validator.packing_mask(observation["actions_valid"], log_probabilities)
        counts = valid.sum(-1)
        # Boolean indexing owns its result; no second copy of legal rows is needed.
        return cls(
            np.concatenate(([0], counts.cumsum())).astype(np.int64),
            np.nonzero(valid)[1].astype(np.int32),
            valid.shape[1],
            {
                key: value[valid]
                for key, value in observation.items()
                if key in ACTION_FIELDS and key != "actions_valid"
            },
            log_probabilities[valid],
        )

    def row(self, index: int) -> "PackedActions":
        """One packed row as views of this block, for collection to hold per seat."""
        start, stop = self.offsets[index : index + 2]
        return PackedActions(
            np.array([0, stop - start], dtype=np.int64),
            self.slots[start:stop],
            self.capacity,
            {key: value[start:stop] for key, value in self.fields.items()},
            self.log_probabilities[start:stop],
        )

    @classmethod
    def concatenate(cls, rows: Sequence["PackedActions"]) -> "PackedActions":
        first = rows[0]
        if any(
            row.capacity != first.capacity or row.fields.keys() != first.fields.keys()
            for row in rows
        ):
            raise ValueError("incompatible packed action rows")
        counts = np.concatenate([np.diff(row.offsets) for row in rows])
        return cls(
            np.concatenate(([0], counts.cumsum())),
            np.concatenate([row.slots for row in rows]),
            first.capacity,
            {
                key: np.concatenate([row.fields[key] for row in rows])
                for key in first.fields
            },
            np.concatenate([row.log_probabilities for row in rows]),
        )

    def unpack(self, indexes: NDArray[np.int64]) -> ActionRows:
        """Restore selected rows at their original coordinates, never renumbering."""
        obs = {
            key: np.full(
                (len(indexes), self.capacity, *value.shape[1:]),
                -1 if key == "action_focus" else 0,
                dtype=value.dtype,
            )
            for key, value in self.fields.items()
        }
        obs["actions_valid"] = np.zeros((len(indexes), self.capacity), dtype=np.float32)
        logs = np.full(
            (len(indexes), self.capacity), -np.inf, dtype=self.log_probabilities.dtype
        )
        for dest, row in enumerate(indexes):
            start, stop = self.offsets[row : row + 2]
            slots = self.slots[start:stop]
            for key, value in self.fields.items():
                obs[key][dest, slots] = value[start:stop]
            obs["actions_valid"][dest, slots] = 1
            logs[dest, slots] = self.log_probabilities[start:stop]
        return ActionRows(obs, logs)

    def check(
        self,
        actions: NDArray[np.generic],
        selected_logs: NDArray[np.generic],
        *,
        validator: ActionValidator,
    ) -> None:
        """Admit behavior evidence before learning through the chosen backend."""
        validator.check_packed(self, actions, selected_logs)

    def selected_types(
        self, actions: NDArray[np.generic], type_width: int
    ) -> NDArray[np.generic]:
        rows = np.repeat(np.arange(actions.size), np.diff(self.offsets))
        chosen = self.slots == actions.reshape(-1)[rows]
        if not np.array_equal(rows[chosen], np.arange(actions.size)):
            raise ValueError("selected action absent from packed support")
        result = self.fields["actions"][chosen, :type_width].argmax(-1)
        return result.reshape(actions.shape)

    @property
    def nbytes(self) -> int:
        return sum(value.nbytes for value in self.fields.values()) + sum(
            value.nbytes for value in (self.offsets, self.slots, self.log_probabilities)
        )
