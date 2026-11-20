"""How the policy reads each numeric input column.

A column's *kind* is what the encoder wrote: the native `ValueLayout`
(`managym/src/agent/observation_layout.rs`), published with its `scale` by
`managym.observation_layout()`. A column's *reading* (`Reading`) is what host
preparation does with it before a linear projection. `Readings` holds one
reading per column of each buffer's saved prefix, so a buffer's width is the
length of its readings.

A fresh model derives its readings from kinds (`fresh_readings`, through
`kind_reading`). A checkpoint records its readings (`input_readings`, beside
`input_columns`) and is read with exactly those, never with the rule, so a
later kind correction or rule change alters fresh models only. A checkpoint
written before readings were recorded gets `legacy_readings`, the treatment
every model had before; it is never extended.

Every float buffer the encoder publishes is accounted for: `prepare` reads it
by its readings (`READ_BY_READINGS`), or it is named in `OWN_PREPARATION`
with the function that decodes it, whose columns all read `own`. A buffer in
neither stops a model from building. The native kinds of own-preparation
buffers are not consulted; several are untrue.

This module imports NumPy, the native layout description and the fact kind
constants only.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from manabot.env.observation import (
    FACT_CONTINUATION_PROGRAM,
    FACT_EFFECT_PROGRAM,
    FACT_TRANSFORMS,
)
import managym

if TYPE_CHECKING:
    from manabot.env.observation import ObservationSpace

Array = np.ndarray


class Reading(StrEnum):
    """What preparation does with one column. Names are stored in checkpoints."""

    AS_IS = "as_is"  # the value as stored
    LOG = "log"  # sign(x) * log1p(|x|) / 10
    RANK = "rank"  # rank / 128, independent of any row capacity
    GATHER = "gather"  # an object join; never a number
    LOOKUP = "lookup"  # an embedding row; never a number
    MASK = "mask"  # the row's validity, which a mask carries; never a number
    # A fact payload by its row's fact kind: program rows 0 (their content is
    # looked up), unscaled transform counts `log`, other payloads as stored.
    FACT_VALUE = "fact_value"
    # A fact's id words: a token row for named kinds; for ordinal kinds (frame
    # or ability indexes) one number, the ordinal / 128.
    FACT_ID = "fact_id"
    # A fact's duration as four flags: until end of turn, indefinite, while the
    # source is controlled, until your next turn.
    DURATION = "duration"
    OWN = "own"  # decoded by its buffer's own preparation (`OWN_PREPARATION`)
    # Stored checkpoints only; the rule never chooses these.
    FACT_VALUE_RAW = "fact_value_raw"  # a fact payload as stored, whatever its kind
    # A fact's duration as the first two flags only; durations 3 and 4 read as
    # absent.
    DURATION_TWO = "duration_two"


type Readings = Mapping[str, tuple[Reading, ...]]

# Every rank divides by this, whatever the capacity of its rows.
RANK_SCALE = 128
# Readings `read_numeric` applies; the others are joins or belong to their
# buffer's own code.
NUMERIC = frozenset(
    {
        Reading.AS_IS,
        Reading.LOG,
        Reading.RANK,
        Reading.FACT_VALUE,
        Reading.FACT_VALUE_RAW,
    }
)
# Fact kinds whose payload `fact_value` reads as zero (their content is a
# looked-up program), and as a count.
PROGRAM_KINDS = (FACT_EFFECT_PROGRAM, FACT_CONTINUATION_PROGRAM)
FACT_COUNT_KINDS = (FACT_TRANSFORMS,)

# Columns read by their place in a structured row rather than their kind, in
# fresh and legacy readings alike: validity carried by a mask, category and id
# words embedded or routed, and the extent of a join.
PLACED_READINGS: Mapping[tuple[str, str], Reading] = {
    ("actions", "valid"): Reading.MASK,
    ("relations", "valid"): Reading.MASK,
    ("facts", "kind"): Reading.LOOKUP,
    ("facts", "id_high"): Reading.FACT_ID,
    ("facts", "id_low"): Reading.FACT_ID,
    ("facts", "detail_high"): Reading.LOOKUP,
    ("facts", "detail_low"): Reading.LOOKUP,
    ("facts", "duration"): Reading.DURATION,
    ("facts", "valid"): Reading.MASK,
    ("effects", "detail_count"): Reading.GATHER,
    ("effects", "member_count"): Reading.GATHER,
    ("effects", "valid"): Reading.MASK,
}

# Frozen overrides for checkpoints written before readings were recorded.
# These keep their original joins, lookups and numeric treatment even if the
# native kind or the fresh reading of a column changes.
LEGACY_PLACED_READINGS: Mapping[tuple[str, str], Reading] = {
    ("actions", "continuation"): Reading.GATHER,
    ("actions", "valid"): Reading.MASK,
    **{
        (f"{seat}_permanents", field): reading
        for seat in ("agent", "opponent")
        for field, reading in (
            ("definition", Reading.LOOKUP),
            ("attached_to", Reading.GATHER),
            ("blocking", Reading.GATHER),
        )
    },
    ("stack", "definition"): Reading.LOOKUP,
    ("stack", "source"): Reading.GATHER,
    ("relations", "source_definition"): Reading.LOOKUP,
    ("relations", "source"): Reading.GATHER,
    ("relations", "reference_definition"): Reading.LOOKUP,
    ("relations", "reference"): Reading.GATHER,
    ("relations", "valid"): Reading.MASK,
    ("facts", "object"): Reading.GATHER,
    ("facts", "kind"): Reading.LOOKUP,
    ("facts", "id_high"): Reading.FACT_ID,
    ("facts", "id_low"): Reading.FACT_ID,
    ("facts", "detail_high"): Reading.LOOKUP,
    ("facts", "detail_low"): Reading.LOOKUP,
    ("facts", "value"): Reading.FACT_VALUE_RAW,
    ("facts", "value_2"): Reading.FACT_VALUE_RAW,
    ("facts", "duration"): Reading.DURATION_TWO,
    ("facts", "timestamp"): Reading.RANK,
    ("facts", "source_definition"): Reading.LOOKUP,
    ("facts", "valid"): Reading.MASK,
    ("effects", "modification_high"): Reading.LOOKUP,
    ("effects", "modification_low"): Reading.LOOKUP,
    ("effects", "timestamp"): Reading.RANK,
    ("effects", "controller"): Reading.GATHER,
    ("effects", "source_definition"): Reading.LOOKUP,
    ("effects", "source"): Reading.GATHER,
    ("effects", "detail_offset"): Reading.GATHER,
    ("effects", "detail_count"): Reading.GATHER,
    ("effects", "member_offset"): Reading.GATHER,
    ("effects", "member_count"): Reading.GATHER,
    ("effects", "valid"): Reading.MASK,
}
# How many flags each duration reading writes, one per duration code from 1.
DURATION_FLAGS: Mapping[Reading, int] = {Reading.DURATION: 4, Reading.DURATION_TWO: 2}

# Buffers `prepare` reads through their readings.
READ_BY_READINGS = frozenset(
    {
        "agent_player",
        "opponent_player",
        "player_turn_state",
        "agent_cards",
        "opponent_cards",
        "agent_permanents",
        "opponent_permanents",
        "stack",
        "actions",
        "relations",
        "facts",
        "effects",
    }
)
_VALIDITY = "prepare: the object validity table and attention mask"
# Float buffers decoded by their own preparation, with the code that does it.
OWN_PREPARATION: Mapping[str, str] = {
    "agent_player_valid": _VALIDITY,
    "opponent_player_valid": _VALIDITY,
    "agent_cards_valid": _VALIDITY,
    "opponent_cards_valid": _VALIDITY,
    "agent_permanents_valid": _VALIDITY,
    "opponent_permanents_valid": _VALIDITY,
    "stack_valid": _VALIDITY,
    "actions_valid": "prepare: the legal-action mask",
    "card_turn_state": "prepare: one flag appended to each card row",
    "semantic_cards": "prepare: catalog rows",
    "known_hand": "prepare: catalog rows; the backend scales a definition "
    "embedding by the count, then divides by 60",
    "known_library": "prepare: catalog rows and 1 / count weights",
    "decklists": "prepare: catalog rows and counts / 60",
    "name_candidates": "prepare: name rows and group bounds",
    "effect_details": "prepare._effects: kinds and token words looked up; "
    "numbers signed-log in the backend",
    "effect_members": "prepare._effects: object routes in row order",
    "events": "prepare._events: kinds and ids looked up, amount / 20; "
    "ages log1p in the backend",
    "events_valid": "prepare._events",
    "decision_records": "decision_memory.prepare_memory",
    "decision_references": "decision_memory.prepare_memory",
    "decision_reference_details": "decision_memory.prepare_memory",
    "decision_members": "decision_memory.prepare_memory",
}


@dataclass(frozen=True)
class ColumnKind:
    """One published column: its name, native kind and encoder divisor."""

    name: str
    kind: str
    scale: int | None


def published_columns(space: "ObservationSpace") -> dict[str, list[ColumnKind]]:
    """Every feature buffer's columns in index order, as the encoder declares them.

    Raises when a float buffer is neither read by readings nor named in
    `OWN_PREPARATION`.
    """
    unread = sorted(
        key
        for key, dtype in space.dtypes.items()
        if np.dtype(dtype).kind == "f"
        and key not in READ_BY_READINGS
        and key not in OWN_PREPARATION
    )
    if unread:
        raise ValueError(f"no reading or own preparation for buffers {unread}")
    layout = managym.observation_layout(
        policy_history=bool(space.config.policy_history_version),
        turn_state=space.turn_state,
        legacy=not space.one_token,
    )
    return {
        key: [
            ColumnKind(name, column["kind"], column.get("scale"))
            for name, column in sorted(
                row["columns"].items(), key=lambda item: item[1]["index"]
            )
        ]
        for key, row in layout.items()
        if row["columns"]
    }


def kind_reading(kind: str, scale: int | None) -> Reading:
    """The reading a fresh model gives a column of this kind.

    Raises for a kind the rule does not know, so a new native kind stops a
    fresh model from building until it is given a reading here.
    """
    match kind:
        case "flag":
            return Reading.AS_IS
        case "count" | "number" if scale is not None:
            return Reading.AS_IS if scale > 1 else Reading.LOG
        case "rank":
            return Reading.RANK
        case "category" | "selector" | "position":
            return Reading.LOG
        case "reference":
            return Reading.GATHER
        case "id":
            return Reading.LOOKUP
        case "tagged":
            return Reading.FACT_VALUE
    raise ValueError(f"no reading for column kind {kind!r} (scale {scale})")


def fresh_readings(space: "ObservationSpace") -> dict[str, tuple[Reading, ...]]:
    """The readings of a model built now, over the runtime's full columns."""
    return assign_readings(
        space,
        lambda column: kind_reading(column.kind, column.scale),
        PLACED_READINGS,
    )


def legacy_readings(space: "ObservationSpace") -> dict[str, tuple[Reading, ...]]:
    """The readings of a checkpoint that records none: what every model did
    before readings were recorded. Never extended."""
    return assign_readings(space, lambda column: Reading.AS_IS, LEGACY_PLACED_READINGS)


def assign_readings(
    space: "ObservationSpace",
    rule: Callable[[ColumnKind], Reading],
    placed: Mapping[tuple[str, str], Reading],
) -> dict[str, tuple[Reading, ...]]:
    """Each published column's `placed` reading, `own` in an own-preparation
    buffer, and otherwise the reading `rule` gives it."""
    return {
        key: tuple(
            placed.get((key, column.name))
            or (Reading.OWN if key not in READ_BY_READINGS else rule(column))
            for column in columns
        )
        for key, columns in published_columns(space).items()
    }


def recorded_readings(
    recorded: object, columns: Mapping[str, Sequence[str]], location: str
) -> dict[str, tuple[Reading, ...]]:
    """Validate a checkpoint's `input_readings` against its `input_columns`.

    Every buffer the columns name needs one known reading per column; an
    unknown name raises with the column and the writer's revision.
    """
    if not isinstance(recorded, dict) or recorded.keys() != columns.keys():
        raise ValueError(
            f"checkpoint input readings differ from its columns; {location}"
        )
    readings: dict[str, tuple[Reading, ...]] = {}
    for key, names in columns.items():
        values = recorded[key]
        if not isinstance(values, list) or len(values) != len(names):
            raise ValueError(
                f"checkpoint {key} readings do not match its {len(names)} columns; {location}"
            )
        parsed = []
        for name, value in zip(names, values, strict=True):
            try:
                parsed.append(Reading(value))
            except ValueError as error:
                raise ValueError(
                    f"checkpoint reads {key}.{name} by unknown reading {value!r}; {location}"
                ) from error
        readings[key] = tuple(parsed)
    return readings


# -- applying readings -------------------------------------------------------------


def signed_log(values: Array) -> Array:
    """sign(x) * log1p(|x|) / 10 in float64, returned as float32."""
    values = np.asarray(values, dtype=np.float64)
    return (np.sign(values) * np.log1p(np.abs(values)) / 10).astype(np.float32)


def numeric_columns(readings: Sequence[Reading]) -> tuple[int, ...]:
    """Indexes of the columns a numeric projection receives."""
    return tuple(i for i, reading in enumerate(readings) if reading in NUMERIC)


def duration_flags(duration: Array, reading: Reading) -> Array:
    """One float32 flag per duration code from 1 [..., F]; 0 (absent) sets none."""
    codes = np.arange(1, DURATION_FLAGS[reading] + 1)
    return (np.asarray(duration)[..., None] == codes).astype(np.float32)


def read_numeric(
    values: Array, readings: Sequence[Reading], fact_kinds: Array | None = None
) -> Array:
    """The numeric columns of `values` [..., C] read by their readings, float32.

    `fact_kinds` [...] holds each row's fact kind, which `fact_value` needs.
    Columns read `as_is` or `fact_value_raw` are copied exactly, so a model
    whose readings are all `as_is` computes what it did before readings existed.
    """
    columns = numeric_columns(readings)
    out = values[..., list(columns)].astype(np.float32, copy=False)
    for position, index in enumerate(columns):
        reading = readings[index]
        if reading is Reading.LOG:
            out[..., position] = signed_log(out[..., position])
        elif reading is Reading.RANK:
            out[..., position] /= np.float32(RANK_SCALE)
        elif reading is Reading.FACT_VALUE:
            if fact_kinds is None:
                raise ValueError("fact_value needs each row's fact kind")
            column = out[..., position]
            counts = np.isin(fact_kinds, FACT_COUNT_KINDS)
            column[counts] = signed_log(column[counts])
            column[np.isin(fact_kinds, PROGRAM_KINDS)] = 0
    return out
