"""Collected rollout rows survive transport from an independent worker."""

from __future__ import annotations

import copy
from dataclasses import fields
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
    PrefixObservation,
    record_tail,
    restore_prefix,
)
from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.policy.spans import ObservationSelection
from manabot.sim.child import ChildProcess
from manabot.training.rollout.collection import (
    RolloutBatch,
    close_collector,
    collect,
    collect_pair,
    open_collector,
)
import managym
from tests.model.test_decision_memory import memory_input


def collected_game(memory: bool) -> tuple[ActorCritic, RolloutBatch]:
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
        seed=17,
        games_per_stream=1,
        opponent_mode="learning" if memory else "self",
        opponent_agent=ActorCritic(space, model) if memory else None,
        numerics=CollectionSession(device="cpu", seed=17, precision="float32"),
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
    agent: ActorCritic, first: ObservationBuffers, second: ObservationBuffers
) -> None:
    a = array_fields(agent.prepare(first), np.ndarray)
    b = array_fields(agent.prepare(second), np.ndarray)
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def chunk_count(batch: RolloutBatch) -> int:
    return len(
        {
            id(node.chunk)
            for p in batch.recorded.memory
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
            assert all(
                p is q
                for p, q in zip(
                    other.recorded.memory,
                    other.recorded.rows.memory.prefixes,
                    strict=True,
                )
            )
        for start in range(0, original.actions.size, 8):
            indexes = np.arange(start, min(start + 8, original.actions.size))
            a, b = original.policy_rows(indexes), other.policy_rows(indexes)
            np.testing.assert_array_equal(a.log_probabilities, b.log_probabilities)
            assert_same_inputs(agent, a.observation, b.observation)
        assert_same_inputs(agent, original.next_obs, other.next_obs)


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
    assert len(warm) < sum(a.nbytes for a in selected.export().values())
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
    with pytest.raises(ValueError, match="v11"):
        restore(managym.ObservationLayout(policy_input_version=10), size, tables)


def test_deep_prefixes_restore_certificates_and_preserve_edits() -> None:
    _, dense = memory_input()
    capacity = (96, 8, 8, 8)
    store = MemoryStore(capacity)
    identity = PrefixIdentity("game", 0, "root")
    # One new admitted record per node, so recursive tail pickling would overflow.
    record = dense["decision_records"][0, :1].copy()
    empty_companions = tuple(dense[key][0, :0] for key in MEMORY_FIELDS[1:])
    tail = None
    for i in range(capacity[0]):
        record[0, [1, 7, 9]] = i
        tail = store._append(tail, store._intern((record, *empty_companions)))
    assert tail is not None
    prefix = Prefix(identity, tail, native_epoch=9)
    selection = PrefixObservation(
        {}, [Prefix(identity, None), prefix, prefix], capacity
    )
    export = selection.materialize()
    export["decision_records"][1, 0, 3] = 21
    cold = pickle.dumps(selection)
    assert record_tail(prefix) is not None
    assert pickle.dumps(selection) == cold
    limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(120)
        decoded, edited = pickle.loads(pickle.dumps((selection, export)))
    finally:
        sys.setrecursionlimit(limit)
    a, b, c = decoded.prefixes
    assert a.tail is None and b is c is edited.prefixes[1]
    assert b.tail.key == tail.key and b.native_epoch is None
    assert edited["decision_records"][1, 0, 3] == 21
    assert decoded.materialize()["decision_records"][1, 0, 3] == 1
    assert b.serving.tail is None and b.tail.features.record_tail is None
    selection.memory_capacities = (1, *capacity[1:])
    with pytest.raises(ValueError, match="capacity"):
        pickle.loads(pickle.dumps(selection))
    chain = tuple(tail.chunks())
    for invalid in (chain[1:], chain[:2] + chain[3:]):
        with pytest.raises(ValueError, match="disconnected"):
            restore_prefix(identity, invalid)
    with pytest.raises(ValueError, match="read-only|WRITEABLE"):
        b.tail.chunk.rows[0].setflags(write=True)
