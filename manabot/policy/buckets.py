"""Bucket selection for bucketed padding: the sizes a batch's variable axes trim to.

Every observation buffer arrives padded to its declared capacity, yet real
batches use little of it. With `ActorCritic(..., bucketed_shapes=True)`, set
by a training stage's `Execution.bucketed_shapes`, the policy trims each
variable axis to a bucket: the next power of two at or above the batch's need,
at least `MIN_BUCKET` and at most the declared capacity. A forward then takes
one of a few shapes per axis. The setting changes no `ModelConfig` field,
parameter, buffer, checkpoint or policy input.

`row_needs` measures what each row needs on every axis; `sizes_of` buckets a
group of rows from those needs (`bucket_sizes` does both for one batch).
A learner measures each rollout row once and regroups the needs for every
epoch's minibatches. Preparation reads original-coordinate bucket views and
uses `object_map` to translate the integer joins it emits.

An axis's need covers every row the default path could read or reject: the
last row with any nonzero value (so poisoned padding is kept and still
rejected) and every row a reference can name (effect CSR spans, frame ids,
action continuation joins, object references). Trimming therefore removes only
all-zero rows nothing refers to, so admission outcomes and messages are
unchanged and outputs differ only by floating-point summation order.

Object axes (cards, permanents and the input v11 stack) move object
coordinates. They are trimmed only when padding objects are inert: not under
`historical_mean`, whose average includes padding, and not with belief rows,
which follow the objects. The deprecated v9/v10 layouts have no stack: its
capacity is zero and its keys are absent.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.memory_store import PrefixObservation
from manabot.env.observation import ObservationBuffers, ObservationSpace, column, width

MIN_BUCKET = 8

# In rollout extent-column order. All buffers on an axis share its capacity.
AXIS_BUFFERS: dict[str, tuple[str, ...]] = {
    "facts": ("facts",),
    "effects": ("effects",),
    "details": ("effect_details",),
    "members": ("effect_members",),
    "relations": ("relations",),
    "actions": ("actions", "name_candidates", "action_focus", "actions_valid"),
    "events": ("events", "events_valid"),
    "cards": (
        "agent_cards",
        "opponent_cards",
        "agent_cards_valid",
        "opponent_cards_valid",
        "semantic_cards",
        "card_turn_state",
    ),
    "known": ("known_hand",),
    "decklists": ("decklists",),
    "permanents": (
        "agent_permanents",
        "opponent_permanents",
        "agent_permanents_valid",
        "opponent_permanents_valid",
    ),
    "stack": ("stack", "stack_valid"),
}
AXES = tuple(AXIS_BUFFERS)
SEAT_MAJOR = frozenset({"semantic_cards", "card_turn_state", "known_hand", "decklists"})
# Column positions come from the native layout by name, so a moved or added
# column cannot silently shift what is remapped. Policy history v1 is the
# event layout whose context columns are object references.
# Object-table references, one-based (0: none) or zero-based (-1: none):
# (buffer, columns, base). `action_focus` is a whole zero-based reference buffer.
OBJECT_REFERENCES = tuple(
    (key, tuple(column(key, name) for name in names), base)
    for key, names, base in (
        ("facts", ("object",), 1),
        ("effects", ("controller", "source"), 1),
        ("effect_members", ("object",), 1),
        ("relations", ("source", "reference"), 1),
        ("agent_permanents", ("attached_to", "blocking"), 1),
        ("opponent_permanents", ("attached_to", "blocking"), 1),
        ("stack", ("source",), 1),
        ("events", ("source_context", "target_context"), 0),
    )
)
_EVENT_WIDTH = width("events")


def bucket(need: int, capacity: int) -> int:
    """The next power of two at or above `need`, within [MIN_BUCKET, capacity]."""
    return min(capacity, max(MIN_BUCKET, 1 << max(need - 1, 0).bit_length()))


@dataclass(frozen=True)
class BucketSizes:
    """Row counts per variable axis for one batch, and the declared capacities."""

    sizes: dict[str, int]
    capacities: dict[str, int]

    @property
    def trims_objects(self) -> bool:
        return any(self.sizes[axis] < self.capacities[axis] for axis in OBJECT_AXES)


@dataclass(frozen=True)
class BucketedObservation:
    """Views in original object coordinates, with the extents used to select them."""

    rows: dict[str, np.ndarray]
    buckets: BucketSizes
    needs: np.ndarray


def has_history_events(obs: Mapping[str, object]) -> bool:
    """Whether `events` holds policy-history rows, whose contexts name objects."""
    events = obs.get("events")
    shape = getattr(events, "shape", None)
    return shape is not None and len(shape) == 3 and shape[-1] == _EVENT_WIDTH


def object_reference_columns(
    obs: ObservationBuffers,
) -> Iterator[tuple[str, tuple[int, ...], int]]:
    """Group live reference columns so trimming copies each buffer once."""
    for key, indices, base in OBJECT_REFERENCES:
        if key in obs and (key != "events" or has_history_events(obs)):
            yield key, indices, base


def object_segments(capacities: Mapping[str, int]) -> list[tuple[str, int, int]]:
    """(axis, start, rows) of each object segment after the two player rows.

    The object table is players, agent cards, opponent cards, agent
    permanents, opponent permanents, then (input v11) the stack, each segment
    at its declared capacity; a v9/v10 stack has none.
    """
    segments: list[tuple[str, int, int]] = []
    start = 2
    for axis in ("cards", "cards", "permanents", "permanents", "stack"):
        rows = capacities.get(axis, 0)
        segments.append((axis, start, rows))
        start += rows
    return segments


# Axes whose rows are object-table segments.
OBJECT_AXES: tuple[str, ...] = ("cards", "permanents", "stack")


def row_needs(obs: ObservationBuffers, space: ObservationSpace) -> np.ndarray | None:
    """Rows each batch row needs on every variable axis, [B, len(AXES)] int64.

    A batch's need on an axis is the maximum over its rows, so needs measured
    once per row serve any later grouping of those rows (`sizes_of`). Object
    axes always count the rows references name, so the same needs serve a
    policy with or without object trimming. None for a batch whose keys or
    shapes differ from the declared layout: it is not trimmed, so the model
    rejects it exactly as it would without buckets.
    """
    # Belief features belong to the policy, outside the native observation.
    native = (
        {key: value for key, value in obs.items() if not key.startswith("belief_")}
        if "belief_card_def_ids" in obs
        else obs
    )
    extents = space.layout.axis_extents(
        native, omit_decision_memory=isinstance(obs, PrefixObservation)
    )
    if extents is None:
        return None
    zeros = np.zeros_like(extents["actions"])
    return np.stack([extents.get(axis, zeros) for axis in AXES], axis=1)


def sizes_of(
    needs: np.ndarray, space: ObservationSpace, *, objects: bool
) -> BucketSizes:
    """The buckets of a batch whose rows need `needs` [B, len(AXES)]."""
    # The first buffer on each axis gives its unbatched row dimension.
    capacities = {}
    for axis, keys in AXIS_BUFFERS.items():
        first = keys[0]
        shape = space.shapes.get(first)
        capacities[axis] = 0 if shape is None else shape[int(first in SEAT_MAJOR)]
    most = needs.max(0, initial=0)
    sizes = {
        axis: bucket(int(most[column]), capacities[axis])
        for column, axis in enumerate(AXES)
    }
    if not objects:
        for axis in OBJECT_AXES:
            sizes[axis] = capacities[axis]
    return BucketSizes(sizes, capacities)


def bucket_sizes(
    obs: ObservationBuffers, space: ObservationSpace, *, objects: bool
) -> BucketSizes | None:
    """Measure one batch's buckets; None leaves it untouched (`row_needs`)."""
    needs = row_needs(obs, space)
    return None if needs is None else sizes_of(needs, space, objects=objects)


def object_columns(buckets: BucketSizes) -> list[slice]:
    """Full-coordinate object slices kept by `buckets`, in object order."""
    return [slice(0, 2)] + [
        slice(start, start + buckets.sizes[axis])
        for axis, start, _ in object_segments(buckets.capacities)
    ]


def bucket_views(
    obs: ObservationBuffers, buckets: BucketSizes
) -> dict[str, np.ndarray]:
    """Slice the measured rows, keeping all references in their original coordinates."""
    trimmed = (
        PrefixObservation(obs, obs.prefixes, obs.memory_capacities)
        if isinstance(obs, PrefixObservation)
        and not any(key in obs for key in MEMORY_FIELDS)
        else dict(obs)
    )
    for axis, keys in AXIS_BUFFERS.items():
        stop = buckets.sizes[axis]
        for key in keys:
            if key in obs:
                rows = obs[key]
                trimmed[key] = (
                    rows[:, :, :stop] if key in SEAT_MAJOR else rows[:, :stop]
                )
    return trimmed


def object_map(buckets: BucketSizes) -> np.ndarray | None:
    """Translate original zero-based object slots into bucket coordinates."""
    if not buckets.trims_objects:
        return None
    segments = object_segments(buckets.capacities)
    _, start, count = segments[-1]
    indexes = np.arange(start + count, dtype=np.int64)
    removed = 0
    for axis, start, count in segments:
        indexes[start : start + count] -= removed
        removed += count - buckets.sizes[axis]
    return indexes


def trim_observation(
    obs: ObservationBuffers, buckets: BucketSizes
) -> dict[str, np.ndarray]:
    """Trim admitted host rows and remap every object join to the new table."""
    trimmed = bucket_views(obs, buckets)
    if not buckets.trims_objects:
        return trimmed

    segments = object_segments(buckets.capacities)

    def remap(values: np.ndarray) -> np.ndarray:
        shift = np.zeros_like(values)
        for axis, start, rows in segments:
            shift += (values >= start + rows) * (rows - buckets.sizes[axis])
        return values - shift

    trimmed["action_focus"] = remap(trimmed["action_focus"])
    for key, indices, base in object_reference_columns(trimmed):
        value = trimmed[key].copy()
        value[..., indices] = remap(value[..., indices] - base) + base
        trimmed[key] = value
    return trimmed
