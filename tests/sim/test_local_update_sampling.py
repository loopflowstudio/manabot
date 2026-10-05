"""Fixture-only direct sampling, replay and lifecycle checks; no optimizer runs."""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch

from manabot.belief.likelihood import file_sha256
from manabot.belief.sampling import AutoregressiveBeliefSampler, SamplerSchema
from manabot.belief.sampling_data import _digest
from manabot.belief.state import ViewerHistory
from manabot.sim.distill import LOCAL_TARGET_KEY, generate_selfplay_shard, load_shards
from manabot.sim.local_sampling import (
    HandBatch,
    HandSample,
    PhysicalHandSampler,
    PreparedHands,
)
from manabot.sim.local_update import (
    LocalUpdatePlayer,
    LocalUpdateReceipt,
    LocalUpdateTeacher,
    SamplerArtifact,
)
from manabot.sim.search_supervised import _validate_dataset
from managym.decision import (
    PUBLIC_COMMITMENT_KINDS,
    Command,
    DecisionFrame,
    Observation,
)
from managym.possible_worlds import PossibleWorldSpace, WorldQuery
from tests.sim import test_local_update as fixtures
from tests.sim.test_local_update import make_teacher, small_match


def saved_sampler(teacher: LocalUpdateTeacher, path: Path) -> SamplerArtifact:
    """Serialize an initialized model in the ordinary format, without fitting."""
    names = ("Forest", "Gray Ogre", "Lightning Bolt", "Llanowar Elves", "Mountain")
    schema = SamplerSchema(
        "fixture-vocabulary", names, 2 * len(PUBLIC_COMMITMENT_KINDS) * (len(names) + 1)
    )
    model = AutoregressiveBeliefSampler(schema, hidden_size=8)
    identity = teacher.likelihood.checkpoint_sha256
    world = _digest(teacher.agent.world_binding)
    torch.save(
        {
            "format": "manabot.autoregressive-belief-sampler/v1",
            "schema": asdict(schema),
            "schema_identity": schema.identity,
            "dataset_identity": "untrained-fixture",
            "policy_identity": identity,
            "world_identity": world,
            "hidden_size": 8,
            "history_dropout": 0.0,
            "state_dict": model.state_dict(),
        },
        path,
    )
    return SamplerArtifact(
        path=path,
        sha256=file_sha256(path),
        dataset_identity="untrained-fixture",
        schema_identity=schema.identity,
        policy_identity=identity,
        world_identity=world,
    )


@pytest.mark.parametrize("learned", [False, True])
def test_direct_search_lifecycle_and_receipt(
    tmp_path: Path, learned: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    original, env = make_teacher(tmp_path)
    artifact = saved_sampler(original, tmp_path / "sampler.pt") if learned else None
    teacher = LocalUpdateTeacher(
        original.likelihood.checkpoint,
        original.likelihood.checkpoint_sha256,
        original.config.model_copy(
            update={"sampling": "learned" if learned else "compatible_prior"}
        ),
        sampler=artifact,
        hand_sampler=None if learned else PhysicalHandSampler(),
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("production search enumerated hidden support")

    monkeypatch.setattr(PossibleWorldSpace, "from_engine", forbidden)
    count = env._engine.possible_world_space_construction_count()
    player = LocalUpdatePlayer(teacher, seed=8)
    viewer = env._engine.current_agent_index()
    player.start_game(env, viewer)
    assert player.history is not None and player.tracker is None
    receipt = teacher.search(env._engine, player.history, seed=13)
    teacher.verify_replay(env._engine, player.history, receipt)
    loaded = LocalUpdateReceipt.from_json(receipt.to_json())
    assert loaded.replay_identity() == receipt.replay_identity()
    assert (
        (loaded.learned_belief is not None)
        if learned
        else (loaded.direct_belief is not None)
    )
    assert sum(loaded.target) == pytest.approx(1)
    assert not loaded.sampling_probabilities
    sampled = loaded.learned_belief if learned else loaded.direct_belief
    assert sampled is not None
    constraints = (
        sampled.constraints_json if learned else sampled.prepared.constraints_json
    )
    swapped = env._engine.materialize_sampled_hand(
        viewer, constraints, dict(sampled.hands[0].counts), 882
    )
    assert (
        teacher.search(swapped, player.history, seed=13).replay_identity()
        == receipt.replay_identity()
    )
    with pytest.raises(ValueError, match="query mass"):
        teacher.search(
            env._engine, player.history, seed=13, query=WorldQuery.has("Mountain")
        )
    with pytest.raises(TimeoutError):
        teacher.search(env._engine, player.history, seed=13, deadline=1)
    # An actual canonical transition exercises history refresh without a tracker.
    choice = player.act(env, {})
    player.prepare_step(env, viewer, choice)
    frame = DecisionFrame.from_json(env._engine.semantic_decision_frame_json())
    command = Command("sampling-test", frame.revision, int(frame.offers[choice]["id"]))
    _, _, _, _, _, transition = env.step_semantic(command)
    player.observe_step(env, viewer, transition)
    assert env._engine.possible_world_space_construction_count() == count


class InvalidSampler(PhysicalHandSampler):
    def sample(
        self,
        prepared: PreparedHands,
        *,
        count: int,
        seed: int,
        check: Callable[[], None],
    ) -> HandBatch:
        return HandBatch(
            prepared, seed, (HandSample((("Mountain", 999),), 0.0),) * count
        )


def test_injected_sampler_is_checked_before_execution(tmp_path: Path) -> None:
    original, env = make_teacher(tmp_path)
    teacher = LocalUpdateTeacher(
        original.likelihood.checkpoint,
        original.likelihood.checkpoint_sha256,
        original.config.model_copy(update={"sampling": "compatible_prior"}),
        hand_sampler=InvalidSampler(),
    )
    history = ViewerHistory.from_observation(
        Observation.from_json(
            env._engine.semantic_observation_json(env._engine.current_agent_index())
        )
    )
    before = env._engine.search_witness_json()
    with pytest.raises(ValueError, match="violates public constraints"):
        teacher.search(env._engine, history, seed=1)
    assert env._engine.search_witness_json() == before
    with pytest.raises(ValueError, match="history"):
        teacher.search(
            env._engine,
            replace(history, current_revision=history.current_revision + 1),
            seed=1,
        )


def test_physical_draw_prefix_and_tiny_exact_reference(tmp_path: Path) -> None:
    _, env = make_teacher(tmp_path)
    viewer = env._engine.current_agent_index()
    history = ViewerHistory.from_observation(
        Observation.from_json(env._engine.semantic_observation_json(viewer))
    )
    sampler = PhysicalHandSampler()
    prepared = sampler.prepare(env._engine, history)

    def check() -> None:
        pass

    short = sampler.sample(prepared, count=2, seed=7, check=check)
    long = sampler.sample(prepared, count=512, seed=7, check=check)
    assert short.hands == long.hands[:2]
    space = PossibleWorldSpace.from_engine(env._engine, viewer)
    expected = {
        tuple(world.hand): world.weight / space.total_weight for world in space.worlds
    }
    for hand in long.hands:
        assert np.exp(hand.log_probability) == pytest.approx(expected[hand.counts])
    observed = {
        key: sum(hand.counts == key for hand in long.hands) / len(long.hands)
        for key in expected
    }
    assert max(abs(observed[key] - expected[key]) for key in expected) < 0.08


def test_learned_arena_replay_without_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def attach(teacher: LocalUpdateTeacher, directory: Path) -> LocalUpdateTeacher:
        return LocalUpdateTeacher(
            teacher.likelihood.checkpoint,
            teacher.likelihood.checkpoint_sha256,
            teacher.config.model_copy(update={"sampling": "learned"}),
            sampler=saved_sampler(teacher, directory / "saved-sampler.pt"),
        )

    monkeypatch.setattr(fixtures, "attach_learned_sampler", attach)
    fixtures.test_arena_uses_exact_range_lifecycle_and_replays(tmp_path, learned=True)


def test_prior_targets_reload_through_distillation_reader(tmp_path: Path) -> None:
    teacher, _ = make_teacher(tmp_path)
    shard = tmp_path / "targets.npz"
    generate_selfplay_shard(
        num_games=1,
        seed=44,
        out_path=shard,
        max_steps_per_game=1000,
        match_hypers=small_match(),
        teacher_spec={
            "kind": "local_update",
            "checkpoint": str(teacher.likelihood.checkpoint),
            "checkpoint_sha256": teacher.likelihood.checkpoint_sha256,
            "config": teacher.config.model_copy(
                update={"sampling": "compatible_prior", "worlds": 1, "depth": 1}
            ).model_dump(),
        },
    )
    dataset = load_shards([shard])
    targets = _validate_dataset(
        dataset, policy_target_kind="local_soft", value_target_kind="terminal_outcome"
    )
    np.testing.assert_array_equal(targets, dataset[LOCAL_TARGET_KEY])


class ExpiringSampler(PhysicalHandSampler):
    def __init__(self, clock: list[float]) -> None:
        self.clock = clock

    def sample(
        self,
        prepared: PreparedHands,
        *,
        count: int,
        seed: int,
        check: Callable[[], None],
    ) -> HandBatch:
        result = super().sample(prepared, count=count, seed=seed, check=check)
        self.clock[0] += 20
        return result


def test_deadline_after_injected_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original, env = make_teacher(tmp_path)
    clock = [100.0]
    teacher = LocalUpdateTeacher(
        original.likelihood.checkpoint,
        original.likelihood.checkpoint_sha256,
        original.config.model_copy(update={"sampling": "compatible_prior"}),
        hand_sampler=ExpiringSampler(clock),
    )
    history = ViewerHistory.from_observation(
        Observation.from_json(
            env._engine.semantic_observation_json(env._engine.current_agent_index())
        )
    )
    monkeypatch.setattr("manabot.sim.local_update.time.perf_counter", lambda: clock[0])
    with pytest.raises(TimeoutError, match="target unavailable"):
        teacher.search(env._engine, history, seed=4)
