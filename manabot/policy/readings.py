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

This module imports NumPy and the native layout description only.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

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
    MASK = "mask"  # the row's validity, which the legal mask carries; never a number
    # Stored checkpoints only; the rule never chooses these.
    FACT_VALUE_RAW = "fact_value_raw"  # a fact payload as stored, whatever its kind
    RANK_BY_EFFECT_ROWS = "rank_by_effect_rows"  # rank / EFFECT_ROWS


type Readings = Mapping[str, tuple[Reading, ...]]

# Every rank divides by this, whatever the capacity of its rows.
RANK_SCALE = 128
# Readings a numeric projection may receive; the others are joins.
NUMERIC = frozenset(Reading) - {Reading.GATHER, Reading.LOOKUP, Reading.MASK}
# Validity flags no projection has received: the buffer's own mask carries them.
MASK_COLUMNS = frozenset({("actions", "valid")})


@dataclass(frozen=True)
class ColumnKind:
    """One published column: its name, native kind and encoder divisor."""

    name: str
    kind: str
    scale: int | None


def published_columns(space: "ObservationSpace") -> dict[str, list[ColumnKind]]:
    """Every feature buffer's columns in index order, as the encoder declares them."""
    encoder = space.encoder
    layout = managym.observation_layout(
        policy_history=bool(encoder.config.policy_history_version),
        turn_state=encoder.turn_state,
        legacy=not encoder.one_token,
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
    raise ValueError(f"no reading for column kind {kind!r} (scale {scale})")


# Buffers whose fresh readings come from `kind_reading`; the others keep
# `legacy_readings` until their preparation reads by reading.
RULED_BUFFERS = (
    "agent_player",
    "opponent_player",
    "player_turn_state",
    "agent_cards",
    "opponent_cards",
    "agent_permanents",
    "opponent_permanents",
    "stack",
    "actions",
)


def fresh_readings(space: "ObservationSpace") -> dict[str, tuple[Reading, ...]]:
    """The readings of a model built now, over the runtime's full columns."""
    readings = legacy_readings(space)
    columns = published_columns(space)
    for key in RULED_BUFFERS:
        if key in columns:
            readings[key] = tuple(
                Reading.MASK
                if (key, c.name) in MASK_COLUMNS
                else kind_reading(c.kind, c.scale)
                for c in columns[key]
            )
    return readings


def legacy_readings(space: "ObservationSpace") -> dict[str, tuple[Reading, ...]]:
    """The readings of a checkpoint that records none: what every model did
    before readings were recorded. Never extended."""
    return {
        key: tuple(_legacy_reading(key, column) for column in columns)
        for key, columns in published_columns(space).items()
    }


def _legacy_reading(key: str, column: ColumnKind) -> Reading:
    if (key, column.name) in MASK_COLUMNS:
        return Reading.MASK
    if column.kind == "reference":
        return Reading.GATHER
    if column.kind == "id":
        return Reading.LOOKUP
    if (key, column.name) == ("facts", "timestamp"):
        return Reading.RANK
    if key == "facts" and column.name in ("value", "value_2"):
        return Reading.FACT_VALUE_RAW
    if (key, column.name) == ("effects", "timestamp"):
        return Reading.RANK_BY_EFFECT_ROWS
    return Reading.AS_IS


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
        try:
            readings[key] = tuple(Reading(value) for value in values)
        except ValueError:
            unknown = next(
                (name, value)
                for name, value in zip(names, values, strict=True)
                if value not in Reading.__members__.values()
            )
            raise ValueError(
                f"checkpoint reads {key}.{unknown[0]} by unknown reading "
                f"{unknown[1]!r}; {location}"
            ) from None
    return readings


def cut(
    readings: Readings, widths: Mapping[str, int]
) -> dict[str, tuple[Reading, ...]]:
    """Readings restricted to a checkpoint's saved column prefixes."""
    return {
        key: values[: widths.get(key, len(values))] for key, values in readings.items()
    }


# -- applying readings -------------------------------------------------------------


def signed_log(values: Array) -> Array:
    """sign(x) * log1p(|x|) / 10 in float64, returned as float32."""
    values = np.asarray(values, dtype=np.float64)
    return (np.sign(values) * np.log1p(np.abs(values)) / 10).astype(np.float32)


def numeric_columns(readings: Sequence[Reading]) -> tuple[int, ...]:
    """Indexes of the columns a numeric projection receives."""
    return tuple(i for i, reading in enumerate(readings) if reading in NUMERIC)


def read_numeric(values: Array, readings: Sequence[Reading]) -> Array:
    """The numeric columns of `values` [..., C] read by their readings, float32.

    Columns read `as_is` are copied exactly, so a model whose readings are all
    `as_is` computes what it did before readings existed.
    """
    columns = numeric_columns(readings)
    out = values[..., list(columns)].astype(np.float32)
    for position, index in enumerate(columns):
        reading = readings[index]
        if reading is Reading.LOG:
            out[..., position] = signed_log(out[..., position])
        elif reading is Reading.RANK:
            out[..., position] /= np.float32(RANK_SCALE)
        elif reading is not Reading.AS_IS:
            raise ValueError(f"reading {reading} applies to its own buffer only")
    return out
