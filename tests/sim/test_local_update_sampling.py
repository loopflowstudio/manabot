"""Fixture-only direct sampling, replay and lifecycle checks; no optimizer runs."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence, cast

from managym.decision import (
    PUBLIC_COMMITMENT_KINDS,
    Command,
    DecisionFrame,
    Observation,
)
from managym.possible_worlds import PossibleWorldSpace, WorldQuery
import numpy as np
import pytest
import torch

from manabot.belief.sampling import AutoregressiveBeliefSampler, SamplerSchema
from manabot.belief.sampling_data import _digest
from manabot.belief.state import ViewerHistory
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.infra.identity import file_sha256
from manabot.sim.search.branch import BranchSession
from manabot.sim.search.local_update import (
    FloatArray,
    LocalSearchConfig,
    LocalUpdatePlayer,
    LocalUpdateReceipt,
    LocalUpdateTeacher,
    RolloutReceipt,
    SamplerArtifact,
    _Rollout,
)
from manabot.sim.search.sampling import (
    HandBatch,
    HandSample,
    PhysicalHandSampler,
    PreparedHands,
)
import managym
from tests.sim import test_local_update as fixtures
from tests.sim.test_local_update import make_teacher


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
    monkeypatch.setattr(
        "manabot.sim.search.local_update.time.perf_counter", lambda: clock[0]
    )
    with pytest.raises(TimeoutError, match="target unavailable"):
        teacher.search(env._engine, history, seed=4)


def _assert_rollouts_match(
    actual: Sequence[RolloutReceipt], expected: Sequence[RolloutReceipt]
) -> None:
    """Only critic estimates may differ with network batch size."""
    for left, right in zip(actual, expected, strict=True):
        assert replace(left, signed_value=right.signed_value) == right
    np.testing.assert_allclose(
        [row.signed_value for row in actual],
        [row.signed_value for row in expected],
        atol=1e-6,
        rtol=1e-5,
    )


@pytest.mark.parametrize("depth", [1, 5, 20])
def test_batched_rollouts_match_sequential_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, depth: int
) -> None:
    """Different scheduling and network batch sizes preserve replay witnesses."""
    teacher, env = make_teacher(
        tmp_path, config=LocalSearchConfig(worlds=4, depth=depth, decision_seconds=10)
    )
    engine = env._engine
    history = ViewerHistory.from_observation(
        Observation.from_json(
            engine.semantic_observation_json(engine.current_agent_index())
        )
    )
    before = engine.search_witness_json()
    batch_sizes: list[int] = []

    def observe_forward(module: torch.nn.Module, args: tuple[object, ...]) -> None:
        tensors = args[0]
        assert isinstance(tensors, dict)
        batch_sizes.append(len(tensors["actions_valid"]))

    handle = teacher.agent.register_forward_pre_hook(observe_forward)
    try:
        batched = teacher.search(engine, history, seed=17)
    finally:
        handle.remove()
    assert batch_sizes == [1] + [len(batched.rollouts)] * depth
    run_rollouts = teacher._run_rollouts

    def serial_rollouts(rows: list[_Rollout], check: Callable[[], None]) -> None:
        # A single live branch also makes every network forward a singleton.
        for row in rows:
            run_rollouts([row], check)

    with monkeypatch.context() as patch:
        patch.setattr(teacher, "_run_rollouts", serial_rollouts)
        sequential = teacher.search(engine, history, seed=17)
    assert engine.search_witness_json() == before
    assert batched.base == sequential.base
    assert batched.allocation_counts == sequential.allocation_counts
    np.testing.assert_allclose(batched.target, sequential.target, atol=1e-6, rtol=1e-5)
    _assert_rollouts_match(batched.rollouts, sequential.rollouts)
    teacher.verify_replay(engine, history, batched)
    assert LocalUpdateReceipt.from_json(batched.to_json()) == batched

    teacher.config = teacher.config.model_copy(
        update={"full_probability": 0, "cheap_worlds": 1}
    )
    cheap = teacher.search(engine, history, seed=17)
    assert len(cheap.rollouts) * teacher.config.worlds == len(batched.rollouts)
    _assert_rollouts_match(cheap.rollouts, batched.rollouts[: len(cheap.rollouts)])


@dataclass
class _ScriptedBranch:
    """A bounded decision trace for terminal dropout and actor-relative leaves."""

    terminal_after: int | None
    winner: int | None = None
    steps: int = 0

    def current_agent_index(self) -> int:
        return self.steps % 2

    def is_game_over(self) -> bool:
        return self.terminal_after is not None and self.steps >= self.terminal_after

    def winner_index(self) -> int | None:
        return self.winner

    def semantic_viewer_state_hash(self, actor: int) -> str:
        return f"actor-{actor}-step-{self.steps}"


@dataclass
class _ScriptedSession:
    truncated: bool = False

    def apply_policy_choice(
        self, branch: _ScriptedBranch, *, site: str, policy_index: int
    ) -> tuple[None, float, bool, bool, dict[str, object]]:
        assert site == "child" and policy_index in (0, 1)
        branch.steps += 1
        return None, 0.0, branch.is_game_over(), self.truncated, {}

    def snapshot(self) -> dict[str, object]:
        return {"fixture": True}


def test_frontier_drops_terminal_branches_and_signs_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher, _ = make_teacher(tmp_path)
    branches = [
        _ScriptedBranch(1, 0),
        _ScriptedBranch(2, 1),
        _ScriptedBranch(2),
        _ScriptedBranch(None),
    ]
    # The fixture supplies only the native methods the rollout runner consumes.
    rows = [
        _Rollout(
            branch=cast(managym.Env, branch),
            session=cast(BranchSession, _ScriptedSession()),
            world_index=0,
            world_seed=1,
            action_index=0,
            rollout_seed=i,
            sampled_hand=None,
            probabilities=np.array([0.5, 0.5]),
        )
        for i, branch in enumerate(branches)
    ]
    batch_sizes: list[int] = []

    def predict(engines: Sequence[managym.Env]) -> list[tuple[FloatArray, float]]:
        assert all(not branch.is_game_over() for branch in engines)
        batch_sizes.append(len(engines))
        return [(np.array([0.5, 0.5]), 0.25) for _ in engines]

    monkeypatch.setattr(teacher, "predict_batch", predict)
    teacher._run_rollouts(rows, lambda: None)
    receipts = [row.receipt(viewer=0) for row in rows]
    assert batch_sizes == [3, 1, 1]
    assert [row.canonical_commands for row in receipts] == [1, 2, 2, 3]
    assert [row.terminal for row in receipts] == [True, True, True, False]
    assert [row.signed_value for row in receipts] == [1, -1, 0, -0.25]
    assert rows[-1].receipt(viewer=1).signed_value == 0.25
    rows[-1].session = cast(BranchSession, _ScriptedSession(truncated=True))
    with pytest.raises(RuntimeError, match="authoritative outcome"):
        teacher._run_rollouts([rows[-1]], lambda: None)


def test_batched_search_checks_deadline_after_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher, env = make_teacher(
        tmp_path, config=LocalSearchConfig(worlds=2, depth=3, decision_seconds=10)
    )
    history = ViewerHistory.from_observation(
        Observation.from_json(
            env._engine.semantic_observation_json(env._engine.current_agent_index())
        )
    )
    clock = [100.0]
    predict_batch = teacher.predict_batch

    def predict(engines: Sequence[managym.Env]) -> list[tuple[FloatArray, float]]:
        result = predict_batch(engines)
        if len(engines) > 1:
            clock[0] += 20
        return result

    monkeypatch.setattr(
        "manabot.sim.search.local_update.time.perf_counter", lambda: clock[0]
    )
    monkeypatch.setattr(teacher, "predict_batch", predict)
    with pytest.raises(TimeoutError, match="target unavailable"):
        teacher.search(env._engine, history, seed=17)


def test_batch_rejects_incomplete_and_nonfinite_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher, env = make_teacher(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(
            teacher,
            "space",
            ObservationSpace(
                teacher.space.encoder.config.model_copy(
                    update={"max_actions": 1, "action_overflow": "first-legal"}
                )
            ),
        )
        with pytest.raises(ValueError, match="complete legal offers"):
            teacher.predict_batch([env._engine, env._engine])
    forward = teacher.agent.forward

    def nonfinite(
        tensors: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = forward(tensors)
        values[-1] = float("nan")
        return logits, values

    monkeypatch.setattr(teacher.agent, "forward", nonfinite)
    with pytest.raises(ValueError, match="nonfinite leaf value"):
        teacher.predict_batch([env._engine, env._engine])


@pytest.mark.parametrize("version", range(1, 10))
@pytest.mark.parametrize("history", [0, 1])
def test_batch_inputs_match_python_at_selected_capacities(
    tmp_path: Path, version: int, history: int
) -> None:
    """Every selected key and row reaches inference, with no shared buffer state."""
    teacher, env = make_teacher(
        tmp_path,
        observation=ObservationConfig(
            policy_input_version=version,
            policy_history_version=history,
            max_facts=17 if version >= 6 else None,
            max_cards_per_player=33,
            max_permanents_per_player=7,
            max_actions=37,
            max_focus_objects=5,
            max_events=3,
        ),
    )
    branch = env._engine.clone_env()
    branch.step(0)
    engines = [env._engine, branch, env._engine]
    witnesses = [engine.search_witness_json() for engine in engines]
    expected = [
        teacher.space.encode(
            engine.observation_for_player(engine.current_agent_index())
        )
        for engine in engines
    ]
    tensors = {
        key: torch.as_tensor(np.stack([row[key] for row in expected]))
        for key in teacher.space.keys()
    }

    def check_inputs(module: torch.nn.Module, args: tuple[object, ...]) -> None:
        actual = args[0]
        assert isinstance(actual, dict)
        assert actual.keys() == tensors.keys()
        for key, tensor in tensors.items():
            np.testing.assert_array_equal(
                actual[key].numpy(), tensor.numpy(), err_msg=key
            )

    handle = teacher.agent.register_forward_pre_hook(check_inputs)
    try:
        predictions = teacher.predict_batch(engines)
    finally:
        handle.remove()
    with torch.inference_mode():
        logits, values = teacher.agent(tensors)
    for i, (probabilities, value) in enumerate(predictions):
        count = int(expected[i]["actions_valid"].sum())
        np.testing.assert_array_equal(
            probabilities, logits[i, :count].double().softmax(-1).numpy()
        )
        assert value == float(values[i])
    assert [engine.search_witness_json() for engine in engines] == witnesses
    assert teacher.predict_batch([]) == []


def test_batch_retains_root_zero_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher, env = make_teacher(
        tmp_path, config=LocalSearchConfig(worlds=2, depth=3, decision_seconds=10)
    )
    history = ViewerHistory.from_observation(
        Observation.from_json(
            env._engine.semantic_observation_json(env._engine.current_agent_index())
        )
    )
    predict_batch = teacher.predict_batch

    def predict(engines: Sequence[managym.Env]) -> list[tuple[FloatArray, float]]:
        result = predict_batch(engines)
        if len(engines) == 1 and engines[0] is env._engine:
            probabilities, value = result[0]
            probabilities[0] = 0
            probabilities /= probabilities.sum()
            result[0] = probabilities, value
        return result

    monkeypatch.setattr(teacher, "predict_batch", predict)
    receipt = teacher.search(env._engine, history, seed=17)
    assert receipt.target[0] == 0
    assert receipt.values[0] is None
    assert receipt.allocation_counts[0] == 0
    assert all(row.action_index != 0 for row in receipt.rollouts)
    assert LocalUpdateReceipt.from_json(receipt.to_json()) == receipt
