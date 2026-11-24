"""Lossless legal-action storage between collection and minibatch learning.

PackedActions stores behavior log probabilities in their original action slots.
Native spans own the legal offers. For dense inputs this record also keeps the
legal action fields. Explicit exports restore the configured action axis.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from manabot.policy.spans import ObservationSelection


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
    observation: Mapping[str, NDArray[np.generic]]
    log_probabilities: NDArray[np.generic]


@dataclass
class PackedActions:
    offsets: NDArray[np.int64]
    slots: NDArray[np.int32]
    capacity: int
    fields: dict[str, NDArray[np.generic]]
    log_probabilities: NDArray[np.generic]

    # References to the same native owners as RecordedObservations, in trajectory order.
    native: ObservationSelection | None = None

    @classmethod
    def pack(
        cls,
        observation: Mapping[str, NDArray[np.generic]],
        log_probabilities: NDArray[np.generic],
        *,
        validator: ActionValidator,
    ) -> "PackedActions":
        """Copy legal rows from [B,A,...]; reject malformed masks or padding mass.

        Distribution/selected-action checks run before learning on every row.
        Padding must be checked here because it is deliberately not retained.
        """
        native = None
        if isinstance(observation, ObservationSelection):
            counts = np.asarray(observation.batch.action_counts, dtype=np.int64)
            slots = np.arange(observation.space.config.max_actions)
            mask = slots < counts[:, None]
            native = ObservationSelection(observation.space, observation.coordinates)
            valid = validator.packing_mask(mask, log_probabilities)
        else:
            valid = validator.packing_mask(
                observation["actions_valid"], log_probabilities
            )
            counts = valid.sum(-1)
        # Boolean indexing owns its result; no second copy of legal rows is needed.
        return cls(
            np.concatenate(([0], counts.cumsum())).astype(np.int64),
            np.nonzero(valid)[1].astype(np.int32),
            valid.shape[1],
            {}
            if isinstance(observation, ObservationSelection)
            else {
                key: observation[key][valid]
                for key in ACTION_FIELDS.intersection(observation) - {"actions_valid"}
            },
            log_probabilities[valid],
            native,
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
            self.native.select(slice(index, index + 1))
            if self.native is not None
            else None,
        )

    @classmethod
    def concatenate(cls, rows: Sequence["PackedActions"]) -> "PackedActions":
        first = rows[0]
        if any(
            row.capacity != first.capacity
            or row.fields.keys() != first.fields.keys()
            or (row.native is None) != (first.native is None)
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
            ObservationSelection(
                first.native.space,
                tuple(
                    coord
                    for row in rows
                    if row.native is not None
                    for coord in row.native.coordinates
                ),
            )
            if first.native is not None
            else None,
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
        if self.native is not None:
            mask = np.arange(self.capacity) < self.native.action_counts[indexes, None]
        else:
            mask = np.zeros((len(indexes), self.capacity), dtype=bool)
            for dest, row in enumerate(indexes):
                start, stop = self.offsets[row : row + 2]
                mask[dest, self.slots[start:stop]] = True
        return ActionRows({"actions_valid": mask}, self.unpack_logs(indexes))

    def unpack(self, indexes: NDArray[np.int64]) -> ActionRows:
        """Restore selected rows at their original coordinates, never renumbering."""
        if self.native is not None:
            selected = self.native.select(indexes)
            return ActionRows(
                {
                    key: selected.project(key)
                    for key in ACTION_FIELDS
                    if key in selected.space.shapes
                },
                self.unpack_logs(indexes),
            )
        obs = {
            key: np.full(
                (len(indexes), self.capacity, *value.shape[1:]),
                -1 if key == "action_focus" else 0,
                dtype=value.dtype,
            )
            for key, value in self.fields.items()
        }
        obs["actions_valid"] = np.zeros((len(indexes), self.capacity), dtype=np.float32)
        for dest, row in enumerate(indexes):
            start, stop = self.offsets[row : row + 2]
            slots = self.slots[start:stop]
            for key, value in self.fields.items():
                obs[key][dest, slots] = value[start:stop]
            obs["actions_valid"][dest, slots] = 1
        return ActionRows(obs, self.unpack_logs(indexes))

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
        if self.native is None:
            return self.fields["actions"][np.ix_(positions, columns)]
        rows = np.searchsorted(self.offsets, positions, side="right") - 1
        return self.native.action_columns(rows, self.slots[positions], columns)

    @property
    def nbytes(self) -> int:
        """Owned array bytes. Native offers are counted with their observation batch."""
        return sum(value.nbytes for value in self.fields.values()) + sum(
            value.nbytes for value in (self.offsets, self.slots, self.log_probabilities)
        )
