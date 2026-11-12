"""Opt-in bucketed padding: trim an admitted batch's variable axes to fixed buckets.

Every observation buffer arrives padded to its declared capacity, yet real
batches use little of it. With `MANABOT_BUCKETED_SHAPES=1`, `ActorCritic`
trims each variable axis to a bucket: the next power of two at or above the
batch's need, at least `MIN_BUCKET` and at most the declared capacity. A forward
then takes one of a few shapes per axis. The setting is runtime only: it changes
no `ModelConfig` field, parameter, buffer, checkpoint or policy input.

Principal entry points:

- `bucketed_shapes_enabled` reads the switch (once, at `ActorCritic` construction).
- `bucket_sizes` measures one batch with a single host read.
- `trim_observation` slices every axis to its bucket and rewrites object
  references into the trimmed object coordinates.

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

from dataclasses import dataclass
import os

import torch

from manabot.env.observation import ObservationEncoder
from managym import observation_layout

BUCKETED_SHAPES_ENV = "MANABOT_BUCKETED_SHAPES"
MIN_BUCKET = 8

# Keys trimmed along axis 1, by axis; seat-major keys ([B, 2, rows, ...])
# are trimmed along axis 2.
_ROW_KEYS: dict[str, tuple[str, ...]] = {
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
_SEAT_KEYS: dict[str, tuple[str, ...]] = {
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


def bucketed_shapes_enabled() -> bool:
    """Read `MANABOT_BUCKETED_SHAPES`: unset, empty or `0` is off, `1` is on."""
    value = os.environ.get(BUCKETED_SHAPES_ENV, "").strip()
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{BUCKETED_SHAPES_ENV} must be 0 or 1, not {value!r}")


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


def _extent(value: torch.Tensor, axis: int) -> torch.Tensor:
    """One past the last row along `axis` holding any nonzero (or NaN) value."""
    rows = value.shape[axis]
    flags = (value != 0).movedim(axis, 1).reshape(value.shape[0], rows, -1).any(-1)
    positions = torch.arange(1, rows + 1, device=value.device)
    return _maximum(flags * positions)


def _maximum(values: torch.Tensor) -> torch.Tensor:
    """Maximum as int64, zero for an empty tensor."""
    flat = values.reshape(-1).long()
    return torch.cat((flat, flat.new_zeros(1))).max()


def _reference_extent(values: torch.Tensor, capacity: int) -> torch.Tensor:
    """Largest integer part of finite one-based references, clamped to capacity."""
    clean = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return _maximum(clean.floor().clamp(0, capacity))


def _object_references(obs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Every object reference in zero-based full object coordinates."""
    references = [obs["action_focus"].float()]
    references.extend(obs[key][..., column] - 1 for key, column in ONE_BASED_REFERENCES)
    if _has_history_events(obs):
        references.extend(
            obs[key][..., column] for key, column in ZERO_BASED_REFERENCES
        )
    return references


def _has_history_events(obs: dict[str, torch.Tensor]) -> bool:
    events = obs.get("events")
    return events is not None and events.ndim == 3 and events.shape[-1] == _EVENT_WIDTH


def _object_extents(
    references: list[torch.Tensor], cards: int, permanents: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of each card and permanent axis that any reference names."""
    card_need: list[torch.Tensor] = []
    permanent_need: list[torch.Tensor] = []
    for values in references:
        index = torch.nan_to_num(values.float(), nan=-1.0, posinf=-1.0, neginf=-1.0)
        index = index.floor()
        for start, size, needs in (
            (2, cards, card_need),
            (2 + cards, cards, card_need),
            (2 + 2 * cards, permanents, permanent_need),
            (2 + 2 * cards + permanents, permanents, permanent_need),
        ):
            offset = index - start
            inside = (offset >= 0) & (offset < size)
            needs.append(_maximum(torch.where(inside, offset + 1, 0)))
    return torch.stack(card_need).max(), torch.stack(permanent_need).max()


def bucket_sizes(
    obs: dict[str, torch.Tensor], encoder: ObservationEncoder, *, objects: bool
) -> BucketSizes | None:
    """Measure one batch's buckets with one host read; None leaves it untouched.

    A batch whose keys or shapes differ from the declared layout is not
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
    needs: dict[str, list[torch.Tensor]] = {axis: [] for axis in capacities}
    for axis, keys in _ROW_KEYS.items():
        needs[axis].extend(
            _extent(obs[key], 1) for key in keys if key not in _UNMEASURED_KEYS
        )
    for axis, keys in _SEAT_KEYS.items():
        needs[axis].extend(_extent(obs[key], 2) for key in keys)
    # Live effect headers name detail and member spans [offset, offset + count).
    effects = obs["effects"]
    needs["details"].append(
        _reference_extent(
            effects[..., _DETAIL_SPAN[0]] + effects[..., _DETAIL_SPAN[1]],
            capacities["details"],
        )
    )
    needs["members"].append(
        _reference_extent(
            effects[..., _MEMBER_SPAN[0]] + effects[..., _MEMBER_SPAN[1]],
            capacities["members"],
        )
    )
    # Frame ids and action continuation joins index the relation axis.
    needs["relations"].append(
        _reference_extent(
            obs["relations"][..., _RELATION_FRAME], capacities["relations"]
        )
    )
    needs["relations"].append(
        _reference_extent(
            obs["actions"][..., _ACTION_CONTINUATION], capacities["relations"]
        )
    )
    if objects:
        card_need, permanent_need = _object_extents(
            _object_references(obs), capacities["cards"], capacities["permanents"]
        )
        needs["cards"].append(card_need)
        needs["permanents"].append(permanent_need)
    axes = list(capacities)
    measured = torch.stack([torch.stack(needs[axis]).max() for axis in axes]).tolist()
    sizes = {
        axis: bucket(int(need), capacities[axis])
        for axis, need in zip(axes, measured, strict=True)
    }
    if not objects:
        sizes["cards"] = capacities["cards"]
        sizes["permanents"] = capacities["permanents"]
    return BucketSizes(sizes, capacities)


def object_remap(values: torch.Tensor, buckets: BucketSizes) -> torch.Tensor:
    """Map zero-based full object coordinates to trimmed ones.

    Each segment shifts by the rows trimmed before it. Integer parts, signs
    and out-of-range values keep their meaning, so reference checks accept and
    reject exactly what they would at full capacity.
    """
    cards, permanents = buckets.capacities["cards"], buckets.capacities["permanents"]
    card_cut = cards - buckets.sizes["cards"]
    permanent_cut = permanents - buckets.sizes["permanents"]
    shift = (
        (values >= 2 + cards).to(values.dtype) * card_cut
        + (values >= 2 + 2 * cards).to(values.dtype) * card_cut
        + (values >= 2 + 2 * cards + permanents).to(values.dtype) * permanent_cut
        + (values >= 2 + 2 * cards + 2 * permanents).to(values.dtype) * permanent_cut
    )
    return values - shift


def trim_observation(
    obs: dict[str, torch.Tensor], buckets: BucketSizes
) -> dict[str, torch.Tensor]:
    """Slice every variable axis to its bucket; other keys pass through."""
    trimmed = dict(obs)
    for axis, keys in _ROW_KEYS.items():
        for key in keys:
            trimmed[key] = obs[key][:, : buckets.sizes[axis]]
    for axis, keys in _SEAT_KEYS.items():
        for key in keys:
            trimmed[key] = obs[key][:, :, : buckets.sizes[axis]]
    if not buckets.trims_objects:
        return trimmed
    trimmed["action_focus"] = object_remap(trimmed["action_focus"], buckets)
    for references, base in ((ONE_BASED_REFERENCES, 1), (ZERO_BASED_REFERENCES, 0)):
        for key in dict.fromkeys(key for key, _ in references):
            if key == "events" and not _has_history_events(trimmed):
                continue
            columns = [column for name, column in references if name == key]
            value = trimmed[key].clone()
            value[..., columns] = (
                object_remap(value[..., columns] - base, buckets) + base
            )
            trimmed[key] = value
    return trimmed


def object_columns(buckets: BucketSizes) -> list[slice]:
    """Full-coordinate object slices kept by `buckets`, in object order."""
    cards, permanents = buckets.capacities["cards"], buckets.capacities["permanents"]
    kept_cards, kept_permanents = buckets.sizes["cards"], buckets.sizes["permanents"]
    base = 2 + 2 * cards
    return [
        slice(0, 2),
        slice(2, 2 + kept_cards),
        slice(2 + cards, 2 + cards + kept_cards),
        slice(base, base + kept_permanents),
        slice(base + permanents, base + permanents + kept_permanents),
    ]
