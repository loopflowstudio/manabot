"""Collected rollout rows survive transport from an independent worker."""

from __future__ import annotations

import copy
from dataclasses import fields, replace
import multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
import pickle
import sys

import numpy as np
import pytest
import torch

from manabot.backends.torch.arrays import array_fields
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.policy import ActorCritic
from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.match import Match, Reward
from manabot.env.memory_store import (
    MemoryStore,
    Prefix,
    PrefixIdentity,
    RetainedMemory,
    record_tail,
    restore_prefix,
)
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.policy.spans import ObservationSelection
from manabot.sim.child import ChildProcess
from manabot.training.rollout.actions import PackedActions
from manabot.training.rollout.collection import (
    RecordedObservations,
    RolloutBatch,
    TrajectorySpan,
    close_collector,
    collect,
    collect_pair,
    open_collector,
)
from manabot.training.updates.targets import grouped_roots
import managym
from tests.fixtures.memory_rows import memory_rows
from tests.model.test_decision_memory import memory_input


def collected_game(memory: bool, seed: int = 17) -> tuple[ActorCritic, RolloutBatch]:
    torch.set_num_threads(1)
    torch.manual_seed(71)
    config = ObservationConfig(
        max_actions=64,
        max_decision_records=128 if memory else 0,
        max_decision_references=256 if memory else 0,
        max_decision_reference_details=16384 if memory else 0,
        max_decision_members=2048 if memory else 0,
    )
    space = ObservationSpace(config)
    model = ModelConfig(
        world=managym.WORLD_SETUP if memory else None,
        decision_memory="gru" if memory else "none",
        attention_on=memory,
        policy_film=memory,
        hidden_dim=8,
        num_attention_heads=2,
    )
    agent = ActorCritic(space, model)
    deck = {"Mountain": 8, "Raging Goblin": 8}
    collector = open_collector(
        space,
        Match(MatchConfig(hero_deck=deck, villain_deck=deck)),
        Reward(RewardConfig()),
        num_envs=1,
        seed=seed,
        games_per_stream=1,
        opponent_mode="learning" if memory else "self",
        opponent_agent=ActorCritic(space, model) if memory else None,
        numerics=CollectionSession(device="cpu", seed=seed, precision="float32"),
    )
    try:
        batch = (
            collect_pair(collector, agent, 1)[0]
            if memory
            else collect(collector, agent, 1)
        )
        return agent, batch
    finally:
        close_collector(collector)


def send_game(queue: Queue[RolloutBatch], memory: bool) -> None:
    _, batch = collected_game(memory)
    queue.put(batch)


def receive_worker_game(connection: Connection, memory: bool) -> None:
    # Start the consumer's spawn pool from a clean entry point, independent of pytest's main.
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    worker = ctx.Process(target=send_game, args=(queue, memory))
    worker.start()
    try:
        connection.send(queue.get(timeout=90))
        worker.join(10)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.kill()
        worker.join()
        queue.close()
        queue.join_thread()


def assert_same_inputs(
    agent: ActorCritic, first: ObservationSelection, second: ObservationSelection
) -> None:
    a = array_fields(agent.prepare(first), np.ndarray)
    b = array_fields(agent.prepare(second), np.ndarray)
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def chunk_count(batch: RolloutBatch) -> int:
    memory = batch.recorded.rows.memory
    return len(
        {
            id(node.chunk)
            for p in (memory.prefixes if memory is not None else ())
            if p.tail
            for node in p.tail.chunks()
        }
    )


@pytest.mark.parametrize("memory", [False, True])
def test_worker_queue_preserves_learner_rows(memory: bool) -> None:
    child = ChildProcess(
        "tests.training.test_rollout_transport:receive_worker_game", (memory,)
    )
    try:
        agent, original = collected_game(memory)
        assert child.connection.poll(120), "worker did not return a rollout"
        received = child.connection.recv()
        child.join(15)
        assert child.exitcode == 0
    finally:
        child.close()
    for other in (received, copy.deepcopy(original)):
        assert isinstance(other, RolloutBatch)
        for f in fields(original):
            value = getattr(original, f.name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(
                    getattr(other, f.name), value, err_msg=f.name
                )
        np.testing.assert_array_equal(other.recorded.index, original.recorded.index)
        assert other.dones[-1, 0] and other.next_done.all()
        assert chunk_count(other) == chunk_count(original)
        if memory:
            assert chunk_count(other) > 0
        for start in range(0, original.actions.size, 8):
            indexes = np.arange(start, min(start + 8, original.actions.size))
            a, b = original.observations(indexes), other.observations(indexes)
            np.testing.assert_array_equal(a.log_probabilities, b.log_probabilities)
            assert_same_inputs(agent, a.observation, b.observation)
        assert_same_inputs(agent, original.next_obs, other.next_obs)

    joined = RolloutBatch.concatenate([original, received])
    boundary = original.actions.size
    crossing = joined.observations(np.array([boundary - 1, boundary]))
    expected = original.observations(np.array([boundary - 1, 0]))
    np.testing.assert_array_equal(
        crossing.log_probabilities, expected.log_probabilities
    )
    assert_same_inputs(agent, crossing.observation, expected.observation)


@pytest.fixture
def batch() -> managym.ObservationBatch:
    env = managym.Env(seed=7)
    obs, _ = env.reset(
        [managym.PlayerConfig(name, {"Mountain": 8}) for name in ("a", "b")]
    )
    return managym.encode_observation_batch(
        [obs, env], managym.ObservationLayout(max_focus_objects=3)
    )


def test_packed_transport_shares_bytes_and_drops_selection_caches(
    batch: managym.ObservationBatch,
) -> None:
    other = batch.blank_decision_memory()
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        a, b = pickle.loads(pickle.dumps((batch, other), protocol))
        assert a.tables() == batch.tables()
        assert a.tables()["actions"][2] is b.tables()["actions"][2]
        assert (
            a.gather([(b, 1), (a, 0), (b, 1)]).tables()
            == batch.gather([(batch, 1), (batch, 0), (batch, 1)]).tables()
        )
    for cloned in (copy.copy(batch), copy.deepcopy(batch)):
        assert cloned.tables()["actions"][2] is batch.tables()["actions"][2]
    space = ObservationSpace(ObservationConfig(max_focus_objects=3))
    assert len(pickle.dumps(space)) < 4096
    for restored in (copy.deepcopy(space), pickle.loads(pickle.dumps(space))):
        assert restored.config == space.config and restored == space
    selected = ObservationSelection(space, ((batch, 1), (batch, 0), (batch, 1)))
    cold = pickle.dumps(selected)
    assert (
        selected.spans
        and selected.action_counts.tolist() == selected.batch.action_counts
    )
    warm = pickle.dumps(selected)
    assert cold == warm
    assert len(warm) < sum(selected.project(key).nbytes for key in selected.fields)
    assert pickle.loads(warm).batch.tables() == selected.batch.tables()


def test_restore_checks_packed_table_boundaries(
    batch: managym.ObservationBatch,
) -> None:
    restore, (layout, size, tables) = batch.__reduce__()
    seat, offsets, values = tables["actions"]
    # Keep the companion byte length consistent with the last offset in each case.
    stride = len(values) // sum(batch.action_counts)
    for ends in ([0], [1, 1, 1], [0, 1, 0], [0, 513, 513]):
        with pytest.raises(ValueError, match="actions"):
            restore(
                layout,
                size,
                tables
                | {
                    "actions": (
                        seat,
                        np.array(ends, np.int32).tobytes(),
                        bytes(ends[-1] * stride),
                    )
                },
            )
    for invalid in ((not seat, offsets, values), (seat, offsets, values[:-1])):
        with pytest.raises(ValueError, match="actions"):
            restore(layout, size, tables | {"actions": invalid})
    missing = tables.copy()
    del missing["actions"]
    for bad in (
        missing,
        tables | {"extra": tables["actions"]},
        missing | {"extra": tables["actions"]},
    ):
        with pytest.raises(ValueError):
            restore(layout, size, bad)


def test_deep_prefixes_restore_certificates() -> None:
    capacity = (96, 8, 8, 8)
    store = MemoryStore(capacity)
    identity = PrefixIdentity("game", 0, "root")
    # One new admitted record per node, so recursive tail pickling would overflow.
    rows = memory_rows(*memory_input())
    record = rows["decision_records"][:1].copy()
    empty_companions = tuple(rows[key][:0] for key in MEMORY_FIELDS[1:])
    tail = None
    for i in range(capacity[0]):
        record[0, [1, 7, 9]] = i
        tail = store._append(tail, store._intern((record, *empty_companions)))
    assert tail is not None
    prefix = Prefix(identity, tail, native_epoch=9)
    retained = RetainedMemory((Prefix(identity, None), prefix, prefix), capacity)
    cold = pickle.dumps(retained)
    assert record_tail(prefix) is not None
    assert pickle.dumps(retained) == cold
    limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(120)
        decoded = pickle.loads(pickle.dumps(retained))
    finally:
        sys.setrecursionlimit(limit)
    a, b, c = decoded.prefixes
    assert a.tail is None and b is c
    assert b.tail.key == tail.key and b.native_epoch is None
    assert b.serving.tail is None and b.tail.features.record_tail is None
    # A tampered archive still meets the constructor's capacity check on load.
    oversized = object.__new__(RetainedMemory)
    object.__setattr__(oversized, "prefixes", retained.prefixes)
    object.__setattr__(oversized, "capacities", (1, *capacity[1:]))
    with pytest.raises(ValueError, match="capacity"):
        pickle.loads(pickle.dumps(oversized))
    chain = tuple(tail.chunks())
    for invalid in (chain[1:], chain[:2] + chain[3:]):
        with pytest.raises(ValueError, match="disconnected"):
            restore_prefix(identity, invalid)
    with pytest.raises(ValueError, match="read-only|WRITEABLE"):
        b.tail.chunk.rows[0].setflags(write=True)


@pytest.fixture(scope="module")
def workers() -> tuple[ActorCritic, RolloutBatch, RolloutBatch]:
    agent, first = collected_game(True)
    _, second = collected_game(True, seed=19)
    return agent, first, pickle.loads(pickle.dumps(second))


def assert_worker_rows(
    agent: ActorCritic, merged: RolloutBatch, workers: tuple[RolloutBatch, ...]
) -> None:
    offset = 0
    for worker in workers:
        count = worker.actions.size
        for name in (
            "actions",
            "dones",
            "rewards",
            "logprobs",
            "values",
            "outcome_probabilities",
        ):
            value = getattr(worker, name)
            if value is not None:
                np.testing.assert_array_equal(
                    getattr(merged, name)[offset : offset + count], value
                )
        for start in range(0, count, 16):
            indexes = np.arange(start, min(start + 16, count))
            original = worker.observations(indexes)
            joined = merged.observations(indexes + offset)
            np.testing.assert_array_equal(
                joined.log_probabilities, original.log_probabilities
            )
            assert_same_inputs(agent, joined.observation, original.observation)
        offset += count
    assert merged.actions.shape == (offset, 1)


def test_concatenate_complete_games(
    workers: tuple[ActorCritic, RolloutBatch, RolloutBatch],
) -> None:
    agent, first, second = workers
    # Paired collection leaves both learners' rows in the same store.
    assert len(first.recorded.rows.coordinates) > first.actions.size
    first = replace(first, recorded=replace(first.recorded, needs=None))
    assert RolloutBatch.concatenate([first]) is first
    merged = RolloutBatch.concatenate([first, second])
    assert isinstance(merged.recorded.rows, ObservationSelection)
    assert (
        merged.recorded.rows.coordinates
        == first.recorded.rows.coordinates + second.recorded.rows.coordinates
    )
    assert merged.next_obs is second.next_obs and merged.next_done is second.next_done
    assert merged.trajectories is None
    row_count = len(first.recorded.rows.coordinates)
    np.testing.assert_array_equal(merged.recorded.needs[:row_count], -1)
    np.testing.assert_array_equal(
        merged.recorded.needs[row_count:], second.recorded.needs
    )
    assert_worker_rows(agent, merged, (first, second))


def snapshot_part(batch: RolloutBatch, snapshot: int, cut: bool) -> RolloutBatch:
    # Two sibling roots and one fresh trajectory. Retain real collected input,
    # action support and memory for each decision and for the frozen viewer.
    indexes = np.arange(3)
    shape = (3, 1)
    return RolloutBatch(
        recorded=replace(batch.recorded, index=batch.recorded.index[indexes]),
        actions=batch.actions[indexes],
        logprobs=batch.logprobs[indexes],
        rewards=batch.rewards[indexes],
        values=batch.values[indexes],
        dones=np.array([[True], [not cut], [True]]),
        action_data=PackedActions.concatenate(
            [batch.action_data.row(i) for i in indexes]
        ),
        next_obs=batch.recorded.gather(indexes[1:2])
        if cut
        else ObservationSelection(batch.recorded.rows.space, ()),
        next_done=np.zeros(int(cut), bool),
        trajectories=(
            TrajectorySpan(0, 1, 0),
            TrajectorySpan(1, 2, 0, 0 if cut else None),
            TrajectorySpan(2, 3, 1),
        ),
        branch_group_id=np.array([[4], [4], [-1]], np.int64),
        snapshot_id=np.array([[snapshot], [snapshot], [-1]], np.int64),
        episode_source=np.array([[1], [1], [0]], np.int8),
        is_root_decision=np.array([[True], [True], [False]]),
        sample_weight=np.ones(shape, np.float32),
    )


def test_concatenate_snapshot_trajectories(
    workers: tuple[ActorCritic, RolloutBatch, RolloutBatch],
) -> None:
    agent, first, second = workers
    a, ended, b = (
        snapshot_part(first, 21, True),
        snapshot_part(second, 22, False),
        snapshot_part(second, 23, True),
    )
    parts = (a, ended, b)
    merged = RolloutBatch.concatenate(parts)
    assert_worker_rows(agent, merged, parts)
    np.testing.assert_array_equal(merged.next_done, [False, False])
    assert isinstance(merged.next_obs, ObservationSelection)
    for i, worker in enumerate((a, b)):
        assert_same_inputs(
            agent, merged.next_obs.select(np.array([i])), worker.next_obs
        )
    assert merged.bootstrap_indexes().ravel().tolist() == [
        -1,
        0,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
    ]
    assert [[s.start for s in group] for group in grouped_roots(merged)] == [
        [0, 1],
        [3, 4],
        [6, 7],
    ]
    assert merged.branch_group_id.ravel().tolist() == [4, 4, -1, 9, 9, -1, 14, 14, -1]
    assert b.branch_group_id.ravel().tolist() == [4, 4, -1]
    terminal = RolloutBatch.concatenate([ended, ended])
    assert not len(terminal.next_done) and not terminal.next_obs.coordinates


def test_incompatible_worker_batches(
    workers: tuple[ActorCritic, RolloutBatch, RolloutBatch],
) -> None:
    _, a, b = workers
    for concatenate in (
        PackedActions.concatenate,
        RecordedObservations.concatenate,
        RolloutBatch.concatenate,
    ):
        with pytest.raises(ValueError, match="empty"):
            concatenate([])
    wide = replace(
        a, actions=np.tile(a.actions, (1, 2)), dones=np.tile(a.dones, (1, 2))
    )
    for parts in ([wide], [a, wide]):
        with pytest.raises(ValueError, match="packed"):
            RolloutBatch.concatenate(parts)
    for invalid in (
        replace(a, dones=np.zeros_like(a.dones)),
        replace(a, next_done=np.array([False])),
    ):
        with pytest.raises(ValueError, match="terminal"):
            RolloutBatch.concatenate([invalid])
    with pytest.raises(ValueError, match="trajectories"):
        RolloutBatch.concatenate([a, snapshot_part(b, 1, True)])
    with pytest.raises(ValueError, match="sample_weight"):
        RolloutBatch.concatenate([a, replace(b, sample_weight=np.ones_like(b.values))])
    record = a.recorded
    with pytest.raises(ValueError, match="memory capacities"):
        RecordedObservations.concatenate(
            [record, replace(record, rows=replace(record.rows, memory=None))]
        )
    config = record.rows.space.config.model_copy(
        update={"decision_memory_include_own": True}
    )
    space = ObservationSpace(config)
    with pytest.raises(ValueError, match="configs"):
        RecordedObservations.concatenate(
            [record, replace(record, rows=replace(record.rows, space=space))]
        )
    with pytest.raises(ValueError, match="configs"):
        PackedActions.concatenate(
            [
                a.action_data,
                replace(
                    a.action_data,
                    selection=replace(a.action_data.selection, space=space),
                ),
            ]
        )


def test_recorded_rows_validate_each_store_before_rebasing(
    batch: managym.ObservationBatch,
) -> None:
    space = ObservationSpace(ObservationConfig(max_focus_objects=3))
    rows = ObservationSelection(space, ((batch, 0), (batch, 1), (batch, 0)))
    record = RecordedObservations(rows, np.array([2, 0, 2]))
    for index in (np.array([3]), np.array([-1]), np.array([[2]]), np.array([1.5])):
        with pytest.raises(ValueError, match="index"):
            RecordedObservations.concatenate([replace(record, index=index)])
    with pytest.raises(ValueError, match="needs"):
        RecordedObservations.concatenate([replace(record, needs=np.zeros((2, 1)))])
