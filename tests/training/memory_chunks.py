"""Long native decision-memory logs that span several 32-record chunks."""

import numpy as np

from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.memory_store import MemoryStore, Prefix
from manabot.env.observation import ObservationSpace, encode_policy_batch
from manabot.policy.spans import ObservationSelection
import managym
from managym.decision_memory import (
    DecisionContext,
    DecisionRecord,
    DecisionReference,
    PublicChoice,
)
from tests.model.test_decision_memory import encode_with_records, memory_root


def reference(power: int, program: int = 1) -> DecisionReference:
    return DecisionReference(
        role=2,
        definition=None,
        appearance=None,
        name=None,
        description="",
        power=power,
        toughness=2,
        tapped=False,
        ready=True,
        attacking=False,
        blocking=False,
        program=[(4, program), (4, 7)],
        mana=[(0, 0, 0, 1, 0, 0)],
    )


def record(
    ordinal: int,
    *,
    count: int = 1,
    second: bool = True,
    program: int = 1,
    life: int = 20,
) -> DecisionRecord:
    """A priority pass whose mana source and responses are shared snapshots.

    Without `second`, both responses name the mana source's snapshot.
    """
    first = reference(1, program)
    second_reference = reference(3) if second else first
    return DecisionRecord(
        actor=2,
        ordinal=ordinal,
        first_position=ordinal,
        last_position=ordinal + count - 1,
        count=count,
        kind="Priority",
        context=DecisionContext(
            turn=3,
            phase=1,
            step=3,
            life=(life, 18),
            hand=(6, 3),
            board_counts=((0,) * 8, (0,) * 8),
            mana_sources=[first],
        ),
        response=[first, second_reference],
        choice=PublicChoice(
            kind="PassPriority",
            mode=None,
            ability=None,
            named=None,
            name_row=None,
            declared=None,
            references=[],
        ),
    )


def chunk_space() -> ObservationSpace:
    space, _ = memory_root()
    return ObservationSpace(
        space.config.model_copy(update={"max_decision_records": 96})
    )


def chunk_batch(
    space: ObservationSpace, records: list[DecisionRecord]
) -> managym.ObservationBatch:
    _, root = memory_root()
    if not records:
        # The opening viewer has no history; an empty decoded log is not canonical.
        return encode_policy_batch(space, [root])
    return encode_with_records(space, root, records)


def chunk_input(count: int = 65) -> tuple[ObservationSpace, ObservationSelection]:
    """`count` records: two full chunks and a partial tail by default."""
    space = chunk_space()
    batch = chunk_batch(space, [record(i) for i in range(count)])
    return space, ObservationSelection(space, ((batch, 0),))


def capacities(space: ObservationSpace) -> tuple[int, ...]:
    return tuple(space.shapes[key][0] for key in MEMORY_FIELDS)


def capture_streams(
    store: MemoryStore,
    selection: ObservationSelection,
    versions: list[int],
    *,
    seeds: list[int] | None = None,
    seats: list[int] | None = None,
    epochs: list[int] | None = None,
) -> tuple[Prefix, ...]:
    """Extend live streams the way collection does, from their native cursors.

    Row `i` of `selection` is stream slot `i`; its cursor is its own counts.
    """
    count = len(selection.coordinates)
    tables = selection.spans.tables
    cursors = []
    for row, version in enumerate(versions):
        cursor = [int(tables[key].counts.ravel()[row]) for key in MEMORY_FIELDS]
        cursors.append([[*cursor, version], [*cursor, version]])
    return store.capture_live_rows(
        selection,
        np.arange(count),
        seeds or list(range(1, count + 1)),
        seats or [0] * count,
        epochs or [1] * count,
        cursors,
    )


def capture_stream(
    store: MemoryStore,
    selection: ObservationSelection,
    version: int,
    *,
    epoch: int = 1,
) -> Prefix:
    """Extend one live stream (slot zero, seat zero) from its native cursor."""
    [prefix] = capture_streams(store, selection, [version], epochs=[epoch])
    return prefix
