"""Bucket selection for bucketed padding: the sizes a batch's variable axes trim to.

Every observation buffer arrives padded to its declared capacity, yet real
batches use little of it. With `ActorCritic(..., bucketed_shapes=True)`, set
by a training stage's `Execution.bucketed_shapes`, the policy trims each
variable axis to a bucket: the next power of two at or above the batch's need,
at least `MIN_BUCKET` and at most the declared capacity. A forward then takes
one of a few shapes per axis. The setting changes no `ModelConfig` field,
parameter, buffer, checkpoint or policy input.

This module chooses the sizes, on the host, with NumPy: `row_needs` measures
what each row of a batch needs on every axis and `sizes_of` buckets a group of
rows from those needs (`bucket_sizes` does both for one batch). Admission
measures before any device transfer (`manabot.model.admission`), so a forward
trims device tensors without reading the device; a learner measures each
rollout row once and regroups the needs for every epoch's minibatches. Each backend trims its own arrays to these sizes
(`trim_observation` in `manabot.model.inference` for torch), remapping the
object references listed here.

An axis's need covers every row the default path could read or reject: the
last row with any nonzero value (so poisoned padding is kept and still
rejected) and every row a reference can name (effect CSR spans, frame ids,
action continuation joins, object references). Trimming therefore removes only
all-zero rows nothing refers to, so admission outcomes and messages are
unchanged and outputs differ only by floating-point summation order.

Object axes (cards and permanents) move object coordinates. They are trimmed
only when padding objects are inert: not under `historical_mean`, whose
average includes padding, and not with belief rows, which follow the objects.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from manabot.env.observation import ObservationEncoder
from managym import observation_layout

MIN_BUCKET = 8

# Keys trimmed along axis 1, by axis; seat-major keys ([B, 2, rows, ...])
# are trimmed along axis 2.
ROW_KEYS: dict[str, tuple[str, ...]] = {
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
    ),
    "permanents": (
        "agent_permanents",
        "opponent_permanents",
        "agent_permanents_valid",
        "opponent_permanents_valid",
    ),
}
SEAT_KEYS: dict[str, tuple[str, ...]] = {
    "cards": ("semantic_cards",),
    "known": ("known_hand",),
    "decklists": ("decklists",),
}
# Read only at legal rows, which `actions_valid` bounds; its padding is -1.
_UNMEASURED_KEYS = ("action_focus",)
# Column positions come from the native layout by name, so a moved or added
# column cannot silently shift what is remapped. Policy history v1 is the
# event layout whose context columns are object references.
_LAYOUT = observation_layout(policy_history=True)


def _column(key: str, name: str) -> int:
    return int(_LAYOUT[key]["columns"][name]["index"])


# Object-table references, one-based (0: none) or zero-based (-1: none):
# (key, column). `action_focus` is a whole zero-based reference buffer.
ONE_BASED_REFERENCES: tuple[tuple[str, int], ...] = tuple(
    (key, _column(key, name))
    for key, name in (
        ("facts", "object"),
        ("effects", "controller"),
        ("effects", "source"),
        ("effect_members", "object"),
        ("relations", "source"),
        ("relations", "reference"),
        ("agent_permanents", "attached_to"),
        ("agent_permanents", "blocking"),
        ("opponent_permanents", "attached_to"),
        ("opponent_permanents", "blocking"),
    )
)
ZERO_BASED_REFERENCES: tuple[tuple[str, int], ...] = (
    ("events", _column("events", "source_context")),
    ("events", _column("events", "target_context")),
)
_EVENT_WIDTH = len(_LAYOUT["events"]["columns"])
# Row joins into other variable axes.
_DETAIL_SPAN = (_column("effects", "detail_offset"), _column("effects", "detail_count"))
_MEMBER_SPAN = (_column("effects", "member_offset"), _column("effects", "member_count"))
_RELATION_FRAME = _column("relations", "frame")
_ACTION_CONTINUATION = _column("actions", "continuation")


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
        return any(
            self.sizes[axis] < self.capacities[axis] for axis in ("cards", "permanents")
        )


def _capacities(encoder: ObservationEncoder) -> dict[str, int]:
    shapes = encoder.shapes
    return {
        "facts": shapes["facts"][0],
        "effects": shapes["effects"][0],
        "details": shapes["effect_details"][0],
        "members": shapes["effect_members"][0],
        "relations": shapes["relations"][0],
        "actions": shapes["actions"][0],
        "events": shapes["events"][0],
        "cards": encoder.cards_per_player,
        "known": encoder.cards_per_player,
        "decklists": encoder.cards_per_player,
        "permanents": encoder.perms_per_player,
    }


def _batch_extent(value: np.ndarray, axis: int) -> int:
    """One past the last row along `axis` that any batch row uses (nonzero or NaN).

    One reduction over the batch axis first, with no full-size temporary: a
    learner minibatch is hundreds of megabytes of mostly zero padding, so this
    pass is the measurement's cost.
    """
    flags = np.any(value, axis=0)
    if flags.ndim > 1:
        flags = np.any(flags, axis=tuple(d for d in range(flags.ndim) if d != axis - 1))
    used = np.flatnonzero(flags)
    return int(used[-1]) + 1 if used.size else 0


def _last_rows(reversed_flags: np.ndarray) -> np.ndarray:
    """[B] one past each row's last true entry, from C-contiguous [B,R] flags
    whose rows are reversed (so `argmax` scans memory forward); zero if none."""
    last = reversed_flags.shape[1] - np.argmax(reversed_flags, axis=1)
    return np.where(reversed_flags.any(1), last, 0)


def _extent(value: np.ndarray, axis: int) -> np.ndarray:
    """[B] one past the last row along `axis` each batch row uses.

    The batch's extent bounds every row's, so only that prefix is read per row
    (buckets are usually a small part of the declared capacity). Each row's
    prefix is flattened and its last nonzero element located, then divided
    back to a row; seat-major buffers ([B, seats, rows, ...]) fold their seats
    into the batch.
    """
    count = len(value)
    extent = _batch_extent(value, axis)
    if extent == 0:
        return np.zeros(count, np.int64)
    prefix = value[(slice(None),) * axis + (slice(0, extent),)]
    seats = int(np.prod(prefix.shape[1:axis], dtype=np.int64))
    width = int(np.prod(prefix.shape[axis + 1 :], dtype=np.int64))
    flat = prefix.reshape(count * seats, -1)
    elements = _last_rows(np.not_equal(flat[:, ::-1], 0, order="C"))
    rows = np.where(elements > 0, (elements - 1) // width + 1, 0)
    return rows.reshape(count, seats).max(1, initial=0)


def _float32(values: np.ndarray, invalid: float) -> np.ndarray:
    """`values` as float32 with NaN and infinities replaced by `invalid`."""
    return np.nan_to_num(
        values.astype(np.float32), nan=invalid, posinf=invalid, neginf=invalid
    )


def _reference_extent(values: np.ndarray, capacity: int) -> np.ndarray:
    """[B] largest integer part of each row's finite one-based references,
    clamped to capacity."""
    clipped = np.clip(np.floor(_float32(values, 0.0)), 0, capacity)
    return clipped.reshape(len(values), -1).max(1, initial=0).astype(np.int64)


def has_history_events(obs: Mapping[str, object]) -> bool:
    """Whether `events` holds policy-history rows, whose contexts name objects."""
    events = obs.get("events")
    shape = getattr(events, "shape", None)
    return shape is not None and len(shape) == 3 and shape[-1] == _EVENT_WIDTH


def _object_references(obs: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    """Every object reference in zero-based full object coordinates."""
    references = [obs["action_focus"].astype(np.float32)]
    references.extend(obs[key][..., column] - 1 for key, column in ONE_BASED_REFERENCES)
    if has_history_events(obs):
        references.extend(
            obs[key][..., column] for key, column in ZERO_BASED_REFERENCES
        )
    return references


def object_segments(capacities: Mapping[str, int]) -> list[tuple[str, int, int]]:
    """(axis, start, rows) of each object segment after the two player rows.

    The object table is players, agent cards, opponent cards, agent
    permanents, opponent permanents, each segment at its declared capacity.
    """
    segments: list[tuple[str, int, int]] = []
    start = 2
    for axis in ("cards", "cards", "permanents", "permanents"):
        segments.append((axis, start, capacities[axis]))
        start += capacities[axis]
    return segments


def _object_extents(
    references: list[np.ndarray], capacities: dict[str, int]
) -> dict[str, np.ndarray]:
    """[B] rows of each card and permanent axis that any of a row's references names."""
    segments = object_segments(capacities)
    objects = segments[-1][1] + segments[-1][2]
    count = len(references[0])
    # Which full object coordinates each row's references name: integer parts
    # of finite values inside the table; NaN and infinities name none.
    named = np.zeros((count, objects), dtype=bool)
    flat = named.reshape(-1)
    offsets = np.arange(count, dtype=np.int64)[:, None] * objects
    for values in references:
        index = np.floor(_float32(values, -1.0)).reshape(count, -1)
        inside = (index >= 0) & (index < objects)
        flat[(offsets + index.astype(np.int64))[inside]] = True
    needs = {
        "cards": np.zeros(count, np.int64),
        "permanents": np.zeros(count, np.int64),
    }
    for axis, start, rows in segments:
        # This segment's columns, last first.
        segment = np.ascontiguousarray(named[:, start + rows - 1 : start - 1 : -1])
        needs[axis] = np.maximum(needs[axis], _last_rows(segment))
    return needs


AXES: tuple[str, ...] = (
    "facts",
    "effects",
    "details",
    "members",
    "relations",
    "actions",
    "events",
    "cards",
    "known",
    "decklists",
    "permanents",
)


def row_needs(
    obs: Mapping[str, np.ndarray], encoder: ObservationEncoder, *, objects: bool
) -> np.ndarray | None:
    """Rows each batch row needs on every variable axis, [B, len(AXES)] int64.

    A batch's need on an axis is the maximum over its rows, so needs measured
    once per row serve any later grouping of those rows (`sizes_of`). None for
    a batch whose keys or shapes differ from the declared layout: it is not
    trimmed, so the model rejects it exactly as it would without buckets.
    """
    shapes = encoder.shapes
    if any(
        key not in obs
        or obs[key].ndim != len(shape) + 1
        or tuple(obs[key].shape[1:]) != tuple(shape)
        for key, shape in shapes.items()
    ):
        return None
    capacities = _capacities(encoder)
    count = len(obs["actions_valid"])
    needs = {axis: np.zeros(count, np.int64) for axis in AXES}

    def need(axis: str, rows: np.ndarray) -> None:
        needs[axis] = np.maximum(needs[axis], rows)

    for axis, keys in ROW_KEYS.items():
        for key in keys:
            if key not in _UNMEASURED_KEYS:
                need(axis, _extent(obs[key], 1))
    for axis, keys in SEAT_KEYS.items():
        for key in keys:
            need(axis, _extent(obs[key], 2))
    # Live effect headers name detail and member spans [offset, offset + count).
    effects = obs["effects"]
    for axis, (offset, length) in (
        ("details", _DETAIL_SPAN),
        ("members", _MEMBER_SPAN),
    ):
        need(
            axis,
            _reference_extent(
                effects[..., offset] + effects[..., length], capacities[axis]
            ),
        )
    # Frame ids and action continuation joins index the relation axis.
    for key, column in (
        ("relations", _RELATION_FRAME),
        ("actions", _ACTION_CONTINUATION),
    ):
        need(
            "relations",
            _reference_extent(obs[key][..., column], capacities["relations"]),
        )
    if objects:
        for axis, rows in _object_extents(_object_references(obs), capacities).items():
            need(axis, rows)
    return np.stack([needs[axis] for axis in AXES], axis=1)


def sizes_of(
    needs: np.ndarray, encoder: ObservationEncoder, *, objects: bool
) -> BucketSizes:
    """The buckets of a batch whose rows need `needs` [B, len(AXES)]."""
    capacities = _capacities(encoder)
    most = needs.max(0, initial=0)
    sizes = {
        axis: bucket(int(most[column]), capacities[axis])
        for column, axis in enumerate(AXES)
    }
    if not objects:
        sizes["cards"] = capacities["cards"]
        sizes["permanents"] = capacities["permanents"]
    return BucketSizes(sizes, capacities)


def bucket_sizes(
    obs: Mapping[str, np.ndarray], encoder: ObservationEncoder, *, objects: bool
) -> BucketSizes | None:
    """Measure one batch's buckets; None leaves it untouched (`row_needs`)."""
    needs = row_needs(obs, encoder, objects=objects)
    return None if needs is None else sizes_of(needs, encoder, objects=objects)


def object_columns(buckets: BucketSizes) -> list[slice]:
    """Full-coordinate object slices kept by `buckets`, in object order."""
    return [slice(0, 2)] + [
        slice(start, start + buckets.sizes[axis])
        for axis, start, _ in object_segments(buckets.capacities)
    ]
