"""Lossless legal-action storage between collection and minibatch learning.

PackedActions stores behavior log probabilities in their original action slots;
the native observation selection owns the legal offers themselves.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from manabot.policy.spans import ObservationSelection

__all__ = [
    "PackedActions",
    "ActionRows",
]


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


@dataclass
class ActionRows:
    # A legal-support mask alone (`{"actions_valid": ...}`) for behavior checks.
    observation: ObservationSelection | Mapping[str, NDArray[np.generic]]
    log_probabilities: NDArray[np.generic]


@dataclass
class PackedActions:
    offsets: NDArray[np.int64]
    slots: NDArray[np.int32]
    capacity: int
    log_probabilities: NDArray[np.generic]
    # References to the same native owners as RecordedObservations, in trajectory order.
    selection: ObservationSelection

    @classmethod
    def pack(
        cls,
        observation: ObservationSelection,
        log_probabilities: NDArray[np.generic],
        *,
        validator: ActionValidator,
    ) -> "PackedActions":
        """Keep legal log probabilities; reject malformed masks or padding mass.

        Distribution/selected-action checks run before learning on every row.
        Padding must be checked here because it is deliberately not retained.
        """
        counts = np.asarray(observation.batch.action_counts, dtype=np.int64)
        mask = np.arange(observation.space.config.max_actions) < counts[:, None]
        valid = validator.packing_mask(mask, log_probabilities)
        return cls(
            np.concatenate(([0], counts.cumsum())).astype(np.int64),
            np.nonzero(valid)[1].astype(np.int32),
            valid.shape[1],
            log_probabilities[valid],
            ObservationSelection(observation.space, observation.coordinates),
        )

    def row(self, index: int) -> "PackedActions":
        """One packed row as views of this block, for collection to hold per seat."""
        start, stop = self.offsets[index : index + 2]
        return PackedActions(
            np.array([0, stop - start], dtype=np.int64),
            self.slots[start:stop],
            self.capacity,
            self.log_probabilities[start:stop],
            self.selection.select(slice(index, index + 1)),
        )

    @classmethod
    def concatenate(cls, rows: Sequence["PackedActions"]) -> "PackedActions":
        if not rows:
            raise ValueError("cannot concatenate empty packed action rows")
        first = rows[0]
        if any(row.capacity != first.capacity for row in rows):
            raise ValueError("incompatible packed action rows")
        counts = np.concatenate([np.diff(row.offsets) for row in rows])
        return cls(
            np.concatenate(([0], counts.cumsum())),
            np.concatenate([row.slots for row in rows]),
            first.capacity,
            np.concatenate([row.log_probabilities for row in rows]),
            ObservationSelection.concatenate([row.selection for row in rows]),
        )

    def unpack_logs(self, indexes: NDArray[np.int64]) -> NDArray[np.generic]:
        logs = np.full(
            (len(indexes), self.capacity), -np.inf, dtype=self.log_probabilities.dtype
        )
        for dest, row in enumerate(indexes):
            start, stop = self.offsets[row : row + 2]
            logs[dest, self.slots[start:stop]] = self.log_probabilities[start:stop]
        return logs

    def behavior(self, indexes: NDArray[np.int64]) -> ActionRows:
        """The original legal support and log probabilities for numerical checks."""
        mask = np.arange(self.capacity) < self.selection.action_counts[indexes, None]
        return ActionRows({"actions_valid": mask}, self.unpack_logs(indexes))

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
        result = self.action_columns(np.flatnonzero(chosen), range(type_width)).argmax(
            -1
        )
        return result.reshape(actions.shape)

    def action_columns(
        self, positions: np.ndarray, columns: Sequence[int]
    ) -> np.ndarray:
        """Project only requested legal offers and columns, in packed behavior order."""
        rows = np.searchsorted(self.offsets, positions, side="right") - 1
        return self.selection.action_columns(rows, self.slots[positions], columns)

    @property
    def nbytes(self) -> int:
        """Owned array bytes. Native offers are counted with their observation batch."""
        return sum(
            value.nbytes for value in (self.offsets, self.slots, self.log_probabilities)
        )
