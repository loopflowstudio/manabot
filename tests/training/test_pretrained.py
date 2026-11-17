"""A self-play stage can start from published weights under its own recipe."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.backends import resolve
from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.env.match import Match
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.checkpoint import pretrained_memory_branch
from manabot.training import execution
from manabot.training.records import TrainingRun
from manabot.training.regime import PretrainedPolicy, TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore
import managym
from tests.training.test_learning_state import continuation, producer, recipe, reference
from tests.training.test_recovery import tracked_updates

__all__ = ["producer"]


def fork(parent: TrainingRun, export: Path, updates: int = 2) -> TrainingRegime:
    """The producer's model and world with a deliberately different recipe."""
    recipe = parent.regime.model_copy(deep=True)
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = updates
    stage.streams = 4
    stage.learning_state = None
    stage.learning = stage.learning.model_copy(
        update={"learning_rate_min": 3e-4, "learning_rate_max": 3e-4}
    )
    stage.pretrained = PretrainedPolicy(
        source_run=reference(export),
        source_stage=parent.stages[0].id,
        checkpoint=parent.stages[0].artifacts["ema"],
        weights="ema",
    )
    return recipe


def run(recipe: TrainingRegime, tmp_path: Path, name: str, seed: int) -> TrainingRun:
    with VerifyStore(tmp_path / "training.sqlite") as store:
        return execution.execute_regime(
            recipe,
            seed,
            tmp_path / name,
            store,
            checkpoint_updates=1,
            backend=resolve("torch"),
        )


def test_changed_recipe_starts_from_published_weights(
    tmp_path: Path, producer: TrainingRun
) -> None:
    export = tmp_path / "producer/run.json"
    frozen = export.read_bytes()
    forked = run(fork(producer, export), tmp_path, "fork", 198)
    assert forked.status == "completed", forked.error
    record = forked.stages[0]
    expected, _ = load_checkpoint(producer.stages[0].artifacts["ema"]["path"])
    initial, _ = load_checkpoint(record.artifacts["initial_raw"]["path"])
    assert initial.value_semantic == "signed_outcome"
    for key, value in initial.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)
    # Schedules, counters and Adam belong to this run, not to the producer.
    assert record.learning_state_origin is None
    assert [d["iteration"] for d in tracked_updates(forked, tmp_path)] == [1, 2]
    assert forked.updates_through() == 2
    adam = torch.load(record.artifacts["optimizer"]["path"], weights_only=True)
    steps = {int(state["step"].item()) for state in adam["state"].values()}
    assert max(steps) == record.optimizer_updates
    assert record.producer_cost is not None
    assert record.producer_cost.run_id == producer.id
    assert record.producer_cost.weights == "ema"
    assert record.producer_cost.cumulative_seconds == pytest.approx(
        producer.stages[0].cumulative_seconds
    )
    assert (
        record.inputs["pretrained/checkpoint"]["sha256"]
        == (producer.stages[0].artifacts["ema"]["sha256"])
    )
    assert export.read_bytes() == frozen


def test_ancestor_costs_reach_the_newest_run(
    tmp_path: Path, producer: TrainingRun
) -> None:
    export = tmp_path / "producer/run.json"
    with VerifyStore(tmp_path / "training.sqlite") as store:
        extended = execution.execute_regime(
            continuation(producer, export, 3),
            197,
            tmp_path / "extended",
            store,
            backend=resolve("torch"),
        )
    assert extended.status == "completed", extended.error
    first = run(fork(extended, tmp_path / "extended/run.json"), tmp_path, "first", 5)
    assert first.status == "completed", first.error
    through_extension = (
        producer.stages[0].cumulative_seconds + extended.stages[0].cumulative_seconds
    )
    assert first.stages[0].producer_cost is not None
    assert first.stages[0].producer_cost.cumulative_seconds == pytest.approx(
        through_extension
    )
    second = run(fork(first, tmp_path / "first/run.json"), tmp_path, "second", 6)
    assert second.status == "completed", second.error
    assert second.stages[0].producer_cost is not None
    assert second.stages[0].producer_cost.cumulative_seconds == pytest.approx(
        through_extension + first.stages[0].cumulative_seconds
    )


@pytest.mark.parametrize("fault", ["architecture", "hash", "weights"])
def test_rejects_weights_this_model_cannot_hold(
    tmp_path: Path, producer: TrainingRun, fault: str
) -> None:
    recipe = fork(producer, tmp_path / "producer/run.json")
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay) and stage.pretrained is not None
    if fault == "architecture":
        recipe.agent.hidden_dim = 16
    elif fault == "hash":
        stage.pretrained.checkpoint["sha256"] = "0" * 64
    else:
        stage.pretrained.weights = "raw"
    with VerifyStore(tmp_path / "training.sqlite") as store:
        with pytest.raises(ValueError):
            execution.execute_regime(
                recipe, 198, tmp_path / "rejected", store, backend=resolve("torch")
            )
    failed = TrainingRun.model_validate_json(
        (tmp_path / "rejected/run.json").read_text()
    )
    assert failed.status == "failed" and not failed.stages[0].updates


def test_one_starting_state_per_stage(tmp_path: Path, producer: TrainingRun) -> None:
    export = tmp_path / "producer/run.json"
    payload = fork(producer, export, 4).model_dump(mode="json")
    extended = continuation(producer, export, 4).stages[0]
    assert isinstance(extended, TrainSelfPlay) and extended.learning_state is not None
    payload["stages"][0]["learning_state"] = extended.learning_state.model_dump()
    with pytest.raises(ValueError, match="another starting state"):
        TrainingRegime.model_validate(payload)


@pytest.fixture
def memory_trunk(tmp_path: Path) -> TrainingRun:
    configuration = recipe(2)
    configuration.observation = ObservationConfig(
        max_decision_records=64,
        max_decision_reference_details=65536,
        max_decision_references=1024,
        max_decision_members=16384,
    )
    configuration.agent = configuration.agent.model_copy(
        update={
            "attention_on": True,
            "policy_film": True,
            "world": managym.WORLD_SETUP,
        }
    )
    stage = configuration.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.streams = 2
    stage.transitions = 8
    stage.learning_seats = "one"
    stage.batch_closure = "exact-quota"
    stage.execution.collection_rows = 2
    stage.execution.bootstrap_rows = 2
    stage.execution.learner_rows = 2
    return run(configuration, tmp_path, "memory-trunk", 197)


@pytest.mark.parametrize("encoder", ["none", "transformer", "gru"])
def test_executor_branches_from_control_with_fresh_learning_state(
    tmp_path: Path,
    memory_trunk: TrainingRun,
    encoder: Literal["none", "transformer", "gru"],
) -> None:
    assert memory_trunk.status == "completed", memory_trunk.error
    configuration = fork(memory_trunk, tmp_path / "memory-trunk/run.json", updates=1)
    configuration.agent.decision_memory = encoder
    if encoder != "none":
        configuration.agent.decision_memory_gate_scale = 2.0
    # fork normally varies stream count; keep this bounded proof to two streams.
    stage = configuration.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.streams = 2
    child = run(configuration, tmp_path, "memory-branch", 198)
    assert child.status == "completed", child.error
    record = child.stages[0]
    initial, _ = load_checkpoint(record.artifacts["initial_raw"]["path"])
    trunk, _ = load_checkpoint(memory_trunk.stages[0].artifacts["ema"]["path"])
    for key, value in trunk.state_dict().items():
        torch.testing.assert_close(initial.state_dict()[key], value, rtol=0, atol=0)
    assert all(value == 0 for value in initial.memory_gates().values())
    # Compare the executor's exported initialization on a real nonempty prefix.
    config = initial.observation_space.encoder.config
    env = managym.Env(seed=198, decision_memory=config.max_decision_records > 0)
    obs, _ = env.reset(Match(configuration.root_match).to_rust())
    for _ in range(8):
        if obs.game_over:
            break
        obs, _, _, _, _ = env.step(0)
    buffers = initial.observation_space.encoder.allocate()
    env.encode_observation_into(obs, buffers)
    rows = {key: value[None] for key, value in buffers.items()}
    assert np.any(rows["decision_records"])
    with torch.inference_mode():
        expected = trunk.forward_distribution(on_device(trunk.prepare(rows), "cpu"))
        actual = initial.forward_distribution(on_device(initial.prepare(rows), "cpu"))
    for value, reference_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference_value, rtol=0, atol=0)
    assert record.learning_state_origin is None
    assert record.producer_cost is not None
    assert record.producer_cost.run_id == memory_trunk.id
    assert [d["iteration"] for d in tracked_updates(child, tmp_path)] == [1]
    # An actual branch checkpoint cannot seed a different memory architecture.
    if encoder != "none":
        rejected = fork(child, tmp_path / "memory-branch/run.json", updates=1)
        rejected.agent.decision_memory = (
            "gru" if encoder == "transformer" else "transformer"
        )
        with pytest.raises(ValueError, match="model settings"):
            run(rejected, tmp_path, "memory-rejected", 199)


def test_memory_pretrained_admission_refuses_other_model_settings(
    tmp_path: Path,
    memory_trunk: TrainingRun,
) -> None:
    configuration = fork(memory_trunk, tmp_path / "memory-trunk/run.json", updates=1)
    configuration.agent.decision_memory = "gru"
    configuration.agent.hidden_dim *= 2
    with pytest.raises(ValueError, match="model settings"):
        run(configuration, tmp_path, "rejected", 198)


def test_memory_pretrained_requires_log_enabled_control() -> None:
    control = ModelConfig(policy_film=True)
    arm = control.model_copy(update={"decision_memory": "gru"})
    with pytest.raises(ValueError, match="capacities"):
        pretrained_memory_branch(control, arm, ObservationConfig())
    with pytest.raises(ValueError, match="model settings"):
        pretrained_memory_branch(
            arm,
            arm.model_copy(update={"decision_memory": "transformer"}),
            ObservationConfig(),
        )


def test_producer_layout_refusal_names_observation_fields(
    tmp_path: Path,
    memory_trunk: TrainingRun,
) -> None:
    """Offering a trunk over another layout names the observation fields."""
    configuration = fork(memory_trunk, tmp_path / "memory-trunk/run.json", updates=1)
    configuration.observation.max_decision_records *= 2
    with pytest.raises(ValueError, match=r"observation\.max_decision_records"):
        run(configuration, tmp_path, "rejected-layout", 198)
