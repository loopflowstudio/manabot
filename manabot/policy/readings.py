"""How the policy reads each numeric input column.

A column's *kind* is what the encoder wrote: the native `ValueLayout`
(`managym/src/agent/observation_layout.rs`), published with its `scale` by
`managym.observation_layout()`, and with its `size` when it is a category. A
column's *reading* is what host preparation does with it before a linear
projection: a `Reading`, or a `Sized` one (`one_hot:14`, `steps:8`) that turns
the column into several inputs. `Readings` holds one reading per column of
each buffer's saved prefix, so a buffer's width is the length of its readings
and its projection width the sum of their input widths (`input_width`).

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
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from manabot.env.observation import (
    FACT_ABILITY_RESOLUTION,
    FACT_CONTINUATION_PROGRAM,
    FACT_EFFECT_PROGRAM,
    FACT_TRANSFORMS,
    FACT_TRIGGERING_DAMAGE,
)

if TYPE_CHECKING:
    from manabot.env.observation import ObservationSpace

Array = np.ndarray


class Reading(StrEnum):
    """What preparation does with one column. Names are stored in checkpoints."""

    AS_IS = "as_is"  # the value as stored
    # Sized (`Sized`): n + 1 flags, one per code 1..n then one for "above n";
    # 0 (absent) sets none.
    ONE_HOT = "one_hot"
    # Sized: a column stored in tenths as stored, then n flags for the true
    # value being at least 1..n; a negative value sets none.
    STEPS = "steps"
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


SIZED = frozenset({Reading.ONE_HOT, Reading.STEPS})


@dataclass(frozen=True)
class Sized:
    """A reading that needs a size, stored as `name:size`. The size is
    recorded, so a stored model keeps its width when a category grows."""

    reading: Reading
    size: int

    def __post_init__(self) -> None:
        if self.reading not in SIZED or self.size < 1:
            raise ValueError(f"no sized reading {self.reading}:{self.size}")

    def __str__(self) -> str:
        return f"{self.reading}:{self.size}"


type ColumnReading = Reading | Sized
type Readings = Mapping[str, tuple[ColumnReading, ...]]


def parse_reading(text: object) -> ColumnReading:
    """A stored reading name (`as_is`, `one_hot:14`). Raises ValueError for an
    unknown name, a sized reading without its size or an unsized one with one."""
    if not isinstance(text, str):
        raise TypeError("reading name must be a string")
    name, separator, size = text.partition(":")
    reading = Reading(name)
    if reading in SIZED:
        if size.isascii() and size.isdigit():
            return Sized(reading, int(size))
    elif not separator:
        return reading
    raise ValueError(f"malformed reading {text!r}")


# Every rank divides by this, whatever the capacity of its rows.
RANK_SCALE = 128
# A selector's one-hot size: indexes 1..15, then "above 15".
SELECTOR_SIZE = 15
# Threshold flags of a column stored in tenths: at least 1..8.
TENTHS_STEPS = 8
# Unsized readings `read_numeric` applies, one input each; the others are
# joins or belong to their buffer's own code.
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
FACT_COUNT_KINDS = (
    FACT_TRANSFORMS,
    FACT_ABILITY_RESOLUTION,
    FACT_TRIGGERING_DAMAGE,
)

# Fresh readings determined by a column's role in its row.
PLACED_READINGS: Mapping[tuple[str, str], Reading] = {
    ("actions", "valid"): Reading.MASK,
    # These selectors identify relation rows used by frame pooling.
    ("relations", "frame"): Reading.GATHER,
    ("relations", "parent"): Reading.GATHER,
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
    "embedding by the count (normalized under stream_norm, / 60 otherwise)",
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
    """One published column: its name, native kind, encoder divisor and, for a
    category, its largest code."""

    name: str
    kind: str
    scale: int | None
    size: int | None = None


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
    layout = space.layout.arrays
    return {
        key: [
            ColumnKind(name, column["kind"], column.get("scale"), column.get("size"))
            for name, column in sorted(
                row["columns"].items(), key=lambda item: item[1]["index"]
            )
        ]
        for key, row in layout.items()
        if row["columns"]
    }


def kind_reading(column: ColumnKind) -> ColumnReading:
    """The reading a fresh model gives a column of this kind.

    Raises for a kind the rule does not know, or a category without its size,
    so a new native kind stops a fresh model from building until it is given a
    reading here.
    """
    scale = column.scale
    match column.kind:
        case "flag":
            return Reading.AS_IS
        case "count" | "number" if scale == 10:
            return Sized(Reading.STEPS, TENTHS_STEPS)
        case "count" | "number" | "mana" if scale is not None:
            return Reading.AS_IS if scale > 1 else Reading.LOG
        case "rank":
            return Reading.RANK
        case "category" if column.size is not None:
            return Sized(Reading.ONE_HOT, column.size)
        case "selector":
            return Sized(Reading.ONE_HOT, SELECTOR_SIZE)
        case "position":
            return Reading.LOG
        case "reference":
            return Reading.GATHER
        case "id":
            return Reading.LOOKUP
        case "tagged":
            return Reading.FACT_VALUE
    raise ValueError(
        f"no reading for column {column.name} of kind {column.kind!r} "
        f"(scale {scale}, size {column.size})"
    )


def fresh_readings(space: "ObservationSpace") -> dict[str, tuple[ColumnReading, ...]]:
    """The readings of a model built now, over the runtime's full columns."""
    return assign_readings(space, kind_reading, PLACED_READINGS)


def legacy_readings(space: "ObservationSpace") -> dict[str, tuple[ColumnReading, ...]]:
    """The readings of a checkpoint that records none: what every model did
    before readings were recorded. Never extended."""
    return assign_readings(space, lambda column: Reading.AS_IS, LEGACY_PLACED_READINGS)


def assign_readings(
    space: "ObservationSpace",
    rule: Callable[[ColumnKind], ColumnReading],
    placed: Mapping[tuple[str, str], Reading],
) -> dict[str, tuple[ColumnReading, ...]]:
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
) -> dict[str, tuple[ColumnReading, ...]]:
    """Validate a checkpoint's `input_readings` against its `input_columns`.

    Every buffer the columns name needs one known reading per column; an
    unknown name or a malformed size raises with the column and the writer's
    revision.
    """
    if not isinstance(recorded, dict) or recorded.keys() != columns.keys():
        raise ValueError(
            f"checkpoint input readings differ from its columns; {location}"
        )
    readings: dict[str, tuple[ColumnReading, ...]] = {}
    for key, names in columns.items():
        values = recorded[key]
        if not isinstance(values, list) or len(values) != len(names):
            raise ValueError(
                f"checkpoint {key} readings do not match its {len(names)} columns; {location}"
            )
        parsed: list[ColumnReading] = []
        for name, value in zip(names, values, strict=True):
            try:
                parsed.append(parse_reading(value))
            except (TypeError, ValueError):
                raise ValueError(
                    f"checkpoint reads {key}.{name} by unknown reading {value!r}; "
                    f"{location}"
                ) from None
        readings[key] = tuple(parsed)
    return readings


# -- applying readings -------------------------------------------------------------


def signed_log(values: Array) -> Array:
    """sign(x) * log1p(|x|) / 10 in float64, returned as float32."""
    values = np.asarray(values, dtype=np.float64)
    return (np.sign(values) * np.log1p(np.abs(values)) / 10).astype(np.float32)


def input_width(reading: ColumnReading) -> int:
    """How many projection inputs a column becomes; 0 when it is not numeric."""
    if isinstance(reading, Sized):
        return reading.size + 1
    return int(reading in NUMERIC)


def duration_flags(duration: Array, reading: Reading) -> Array:
    """One float32 flag per duration code from 1 [..., F]; 0 (absent) sets none."""
    codes = np.arange(1, DURATION_FLAGS[reading] + 1)
    return (np.asarray(duration)[..., None] == codes).astype(np.float32)


@dataclass(frozen=True)
class ReadingPlan:
    """Saved numeric projection order, grouped by the operation on each column."""

    width: int
    copies: tuple[tuple[slice, slice], ...]
    groups: tuple[tuple[ColumnReading, Array, Array], ...]

    def read(self, values: Array, fact_kinds: Array | None = None) -> Array:
        out = np.empty((*values.shape[:-1], self.width), dtype=np.float32)
        for source, target in self.copies:
            out[..., target] = values[..., source]
        for reading, sources, targets in self.groups:
            stored = values[..., sources]
            if isinstance(reading, Sized):
                if reading.reading is Reading.STEPS:
                    flags = np.arange(1, reading.size + 1)
                    out[..., targets] = stored
                    counts = np.rint(stored.astype(np.float64) * 10)
                    out[..., targets[:, None] + flags] = counts[..., None] >= flags
                else:
                    # Codes below one (including NaN) set no flag.
                    codes = np.trunc(np.clip(stored, 0, reading.size + 1))
                    flags = np.arange(1, reading.size + 2)
                    out[..., targets[:, None] + flags - 1] = codes[..., None] == flags
                continue
            columns = stored.astype(np.float32, copy=False)
            if reading is Reading.LOG:
                columns = signed_log(columns)
            elif reading is Reading.RANK:
                columns = columns / np.float32(RANK_SCALE)
            elif reading is Reading.FACT_VALUE:
                if fact_kinds is None:
                    raise ValueError("fact_value needs each row's fact kind")
                kinds = np.asarray(fact_kinds).reshape(values.shape[:-1])
                counts = np.isin(kinds, FACT_COUNT_KINDS)
                columns[counts] = signed_log(columns[counts])
                columns[np.isin(kinds, PROGRAM_KINDS)] = 0
            out[..., targets] = columns
        return out


@lru_cache(maxsize=256)
def reading_plan(readings: tuple[ColumnReading, ...]) -> ReadingPlan:
    copies: list[tuple[slice, slice]] = []
    groups: dict[ColumnReading, list[tuple[int, int]]] = {}
    at = 0
    for index, reading in enumerate(readings):
        width = input_width(reading)
        if not width:
            continue
        if reading in (Reading.AS_IS, Reading.FACT_VALUE_RAW):
            if copies and copies[-1][0].stop == index:
                source, target = copies.pop()
                copies.append(
                    (slice(source.start, index + 1), slice(target.start, at + 1))
                )
            else:
                copies.append((slice(index, index + 1), slice(at, at + 1)))
        else:
            groups.setdefault(reading, []).append((index, at))
        at += width
    return ReadingPlan(
        at,
        tuple(copies),
        tuple(
            (reading, *np.asarray(pairs, dtype=np.intp).T)
            for reading, pairs in groups.items()
        ),
    )


def read_numeric(
    values: Array, readings: Sequence[ColumnReading], fact_kinds: Array | None = None
) -> Array:
    """Read [..., columns] in saved projection order. Fact payloads use each row's kind."""
    return reading_plan(tuple(readings)).read(values, fact_kinds)
