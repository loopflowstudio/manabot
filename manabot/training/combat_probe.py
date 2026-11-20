"""Measure combat declaration log-odds and their response to board edits.

combat_probe selects two-offer decisions in packed rollout order before
materializing observations. gap_shifts scores each edit against the same
unmodified positions with the same policy.
"""

from copy import copy
from dataclasses import dataclass

import numpy as np

from manabot.env.memory_store import select_rows
from manabot.env.observation import ObservationSpace
from manabot.policy.protocol import BackendPolicy
from manabot.policy.readings import published_columns
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
    layout = published_columns(space)
    indexes = {
        key: {col.name: i for i, col in enumerate(layout[key])}
        for key in ("actions", "agent_permanents", "opponent_permanents")
    }

    def permanents(key: str) -> PermanentColumns:
        cols = indexes[key]
        power, toughness = cols["power"], cols["toughness"]
        return PermanentColumns(
            cols["is_creature"],
            power,
            toughness,
            cols["tapped"],
            layout[key][power].scale or 1,
            layout[key][toughness].scale or 1,
        )

    actions = indexes["actions"]
    try:
        return CombatColumns(
            actions["declare_attacker"],
            actions["declare_blocker"],
            actions["declared"],
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
    actions = packed.fields["actions"]
    rows = two[
        (actions[first, kind] > 0)
        & (actions[second, kind] > 0)
        & ((actions[first, declared] > 0) != (actions[second, declared] > 0))
    ]
    if len(rows) <= limit:
        return rows
    return rows[np.linspace(0, len(rows) - 1, limit, dtype=np.intp)]


def gap_shifts(
    policy: BackendPolicy,
    rows: dict[str, np.ndarray],
    columns: CombatColumns,
    *,
    execution_rows: int | None = None,
) -> dict[str, float | int]:
    """Score a nonempty batch of two-offer combat rows in original action slots."""
    if execution_rows is not None and execution_rows < 1:
        raise ValueError("execution_rows must be positive")
    legal = rows["actions_valid"] > 0
    declared = rows["actions"][..., columns.declared] > 0
    yes = np.nonzero(legal & declared)
    no = np.nonzero(legal & ~declared)
    expected = np.arange(len(legal))
    if not len(expected) or not (
        np.array_equal(yes[0], expected) and np.array_equal(no[0], expected)
    ):
        raise ValueError(
            "gap_shifts requires a nonempty batch of two-offer declare/decline rows"
        )
    width = execution_rows or len(expected)

    def gaps(observation: dict[str, np.ndarray]) -> np.ndarray:
        if width >= len(expected):
            logits, _ = policy.evaluate(policy.prepare(observation))
        else:
            logits = np.concatenate(
                [
                    policy.evaluate(
                        policy.prepare(
                            select_rows(
                                observation,
                                np.arange(start, min(start + width, len(expected))),
                            )
                        )
                    )[0]
                    for start in range(0, len(expected), width)
                ]
            )
        return logits[yes].astype(np.float64) - logits[no]

    base = gaps(rows)
    metrics: dict[str, float | int] = {
        "gap": float(base.mean()),
        "decisions": len(base),
    }
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
        board = rows[key]
        creatures = (board[..., cols.is_creature] > 0) & (rows[f"{key}_valid"] > 0)
        present = creatures.any(axis=1)
        if not present.any():
            continue
        metrics[f"shift_decisions/{side}"] = int(present.sum())
        # Shallow-copy the observation to retain any decision-memory prefixes.
        edited = copy(rows)
        edited[key] = board.copy()
        for label, power, toughness in ((weak, 0, 1), (strong, 9, 9)):
            edited[key][..., cols.power][creatures] = power / cols.power_scale
            edited[key][..., cols.toughness][creatures] = (
                toughness / cols.toughness_scale
            )
            if side == "theirs":
                edited[key][..., cols.tapped][creatures] = power == 0
            metrics[f"shift/{label}"] = float((gaps(edited) - base)[present].mean())
    return metrics


def combat_probe(
    policy: BackendPolicy,
    batch: RolloutBatch,
    space: ObservationSpace,
    limit: int = 256,
    *,
    execution_rows: int | None = None,
) -> dict[str, float | int]:
    """Measure each available kind on at most limit collected decisions per kind."""
    if execution_rows is not None and execution_rows < 1:
        raise ValueError("execution_rows must be positive")
    columns = combat_columns(space)
    metrics: dict[str, float | int] = {}
    if columns is None:
        return metrics
    for kind, index in (
        ("attack", columns.declare_attacker),
        ("block", columns.declare_blocker),
    ):
        selected = combat_rows(batch.action_data, index, columns.declared, limit)
        if selected.size:
            rows = batch.observations(selected).observation
            for name, value in gap_shifts(
                policy, rows, columns, execution_rows=execution_rows
            ).items():
                metrics[f"combat/{kind}/{name}"] = value
    return metrics
