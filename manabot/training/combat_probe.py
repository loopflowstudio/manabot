"""Measure combat declaration log-odds and their response to board edits.

combat_probe selects two-offer decisions in packed rollout order before
materializing observations. gap_shifts scores each edit against the same
unmodified positions with the same policy.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

import numpy as np

from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.policy.prepare import Prepared
from manabot.policy.protocol import BackendPolicy
from manabot.policy.readings import write_numeric_column
from manabot.policy.spans import ObservationSelection
from manabot.training.rollout.actions import PackedActions
from manabot.training.rollout.collection import RolloutBatch


@dataclass(frozen=True)
class PermanentColumns:
    is_creature: int
    power: int
    toughness: int
    tapped: int
    power_scale: int
    toughness_scale: int


@dataclass(frozen=True)
class CombatColumns:
    declare_attacker: int
    declare_blocker: int
    declared: int
    mine: PermanentColumns
    theirs: PermanentColumns


def combat_columns(space: ObservationSpace) -> CombatColumns | None:
    arrays = space.layout.arrays

    def permanents(key: str) -> PermanentColumns:
        cols = arrays[key]["columns"]
        power, toughness = cols["power"], cols["toughness"]
        return PermanentColumns(
            cols["is_creature"]["index"],
            power["index"],
            toughness["index"],
            cols["tapped"]["index"],
            power.get("scale", 1),
            toughness.get("scale", 1),
        )

    actions = arrays["actions"]["columns"]
    try:
        return CombatColumns(
            actions["declare_attacker"]["index"],
            actions["declare_blocker"]["index"],
            actions["declared"]["index"],
            permanents("agent_permanents"),
            permanents("opponent_permanents"),
        )
    except KeyError:
        # Earlier layouts do not identify creatures on the battlefield.
        return None


def combat_rows(
    packed: PackedActions, kind: int, declared: int, limit: int
) -> np.ndarray:
    """Evenly spaced decisions with one declare and one decline of this kind."""
    if limit < 1:
        raise ValueError("combat probe row limit must be positive")
    two = np.flatnonzero(np.diff(packed.offsets) == 2)
    first = packed.offsets[two]
    second = first + 1
    left = packed.action_columns(first, (kind, declared))
    right = packed.action_columns(second, (kind, declared))
    rows = two[
        (left[:, 0] > 0) & (right[:, 0] > 0) & ((left[:, 1] > 0) != (right[:, 1] > 0))
    ]
    if len(rows) <= limit:
        return rows
    return rows[np.linspace(0, len(rows) - 1, limit, dtype=np.intp)]


def gap_shifts(
    policy: BackendPolicy,
    chunks: Iterable[ObservationBuffers | ObservationSelection],
    columns: CombatColumns,
) -> dict[str, float | int]:
    """Pool per-decision readings across chunks, including partial final chunks."""
    samples: dict[tuple[str, str], list[np.ndarray]] = {}
    for chunk in chunks:
        for key, count_key, values in combat_samples(policy, chunk, columns):
            samples.setdefault((key, count_key), []).append(values)

    metrics: dict[str, float | int] = {}
    for (key, count_key), parts in samples.items():
        values = np.concatenate(parts)
        metrics[key] = float(values.mean())
        metrics[count_key] = len(values)
    return metrics


def combat_samples(
    policy: BackendPolicy,
    rows: ObservationBuffers | ObservationSelection,
    columns: CombatColumns,
) -> Iterator[tuple[str, str, np.ndarray]]:
    """Per-decision gaps and paired shifts for one execution chunk."""
    prepared = policy.prepare(rows)
    batch, slots = prepared.legal_batches, prepared.legal_slots
    if isinstance(rows, ObservationSelection):
        declared = rows.spans.tables["actions"].records["declared"] > 0
    else:
        declared = rows["actions"][batch, slots, columns.declared] > 0
    if not all(
        np.all(np.bincount(batch[mask], minlength=prepared.batch) == 1)
        for mask in (declared, ~declared)
    ):
        raise ValueError("gap_shifts requires two-offer declare/decline rows")
    yes, no = slots[declared], slots[~declared]
    indexes = np.arange(prepared.batch)

    def gaps(observation: Prepared[np.ndarray]) -> np.ndarray:
        logits, _ = policy.evaluate(observation)
        return logits[indexes, yes].astype(np.float64) - logits[indexes, no]

    base = gaps(prepared)
    yield "gap", "decisions", base
    # The rarer offer is selected about once in 10,000 at |gap| = 9.2.
    yield "saturated", "decisions", np.abs(base) > 9.2
    for key, cols, side, weak, strong in (
        (
            "agent_permanents",
            columns.mine,
            "mine",
            "my_creatures_0_1",
            "my_creatures_9_9",
        ),
        (
            "opponent_permanents",
            columns.theirs,
            "theirs",
            "their_creatures_0_1_tapped",
            "their_creatures_9_9_untapped",
        ),
    ):
        seat = int(side == "theirs")
        capacity = prepared.permanents.shape[2]
        if isinstance(rows, ObservationSelection):
            spans = rows.spans.tables[key]
            creatures = spans.scatter(spans.records["is_creature"] > 0, capacity)
        else:
            creatures = (rows[key][:, :capacity, cols.is_creature] > 0) & (
                rows[f"{key}_valid"][:, :capacity] > 0
            )
        present = creatures.any(axis=1)
        if not present.any():
            continue
        readings = policy.readings[key]
        for label, power, toughness in ((weak, 0, 1), (strong, 9, 9)):
            edited = replace(prepared, permanents=prepared.permanents.copy())
            board = edited.permanents[:, seat]
            write_numeric_column(
                board, creatures, readings, cols.power, power / cols.power_scale
            )
            write_numeric_column(
                board,
                creatures,
                readings,
                cols.toughness,
                toughness / cols.toughness_scale,
            )
            if seat == 1:
                write_numeric_column(
                    board, creatures, readings, cols.tapped, float(power == 0)
                )
            yield (
                f"shift/{label}",
                f"shift_decisions/{side}",
                (gaps(edited) - base)[present],
            )


def combat_probe(
    policy: BackendPolicy,
    batch: RolloutBatch,
    *,
    limit: int = 256,
    execution_rows: int | None = None,
) -> dict[str, float | int]:
    """Measure each available kind on at most limit collected decisions per kind."""
    if execution_rows is not None and execution_rows < 1:
        raise ValueError("execution_rows must be positive")
    columns = combat_columns(policy.observation_space)
    metrics: dict[str, float | int] = {}
    if columns is None:
        return metrics
    for kind, index in (
        ("attack", columns.declare_attacker),
        ("block", columns.declare_blocker),
    ):
        selected = combat_rows(batch.action_data, index, columns.declared, limit)
        width = execution_rows or limit
        chunks = (
            batch.observations(selected[start : start + width]).observation
            for start in range(0, len(selected), width)
        )
        for name, value in gap_shifts(policy, chunks, columns).items():
            metrics[f"combat/{kind}/{name}"] = value
    return metrics
