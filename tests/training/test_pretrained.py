"""A self-play stage can start from published weights under its own recipe."""

from pathlib import Path

import pytest
import torch

from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training import execution
from manabot.training.models import (
    PretrainedPolicy,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore
from tests.training.test_learning_state import continuation, producer, reference

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
            recipe, seed, tmp_path / name, store, checkpoint_updates=1
        )


def test_changed_recipe_starts_from_published_weights(
    tmp_path: Path, producer: TrainingRun
) -> None:
    export = tmp_path / "producer/run.json"
    frozen = export.read_bytes()
    forked = run(fork(producer, export), tmp_path, "fork", 198)
    assert forked.status == "completed", forked.error
    record = forked.stages[0]
    expected, _ = load_checkpoint_agent(producer.stages[0].artifacts["ema"]["path"])
    initial, _ = load_checkpoint_agent(record.artifacts["initial_raw"]["path"])
    for key, value in initial.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)
    # Schedules, counters and Adam belong to this run, not to the producer.
    assert record.learning_state_origin is None
    assert [d["iteration"] for d in record.diagnostics] == [1, 2]
    assert forked.updates_through() == 2
    adam = torch.load(record.artifacts["optimizer"]["path"], weights_only=True)
    steps = {int(state["step"].item()) for state in adam["state"].values()}
    assert max(steps) == record.observed_optimizer_updates()
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
            continuation(producer, export, 3), 197, tmp_path / "extended", store
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
            execution.execute_regime(recipe, 198, tmp_path / "rejected", store)
    failed = TrainingRun.model_validate_json(
        (tmp_path / "rejected/run.json").read_text()
    )
    assert failed.status == "failed" and not failed.stages[0].diagnostics


def test_one_starting_state_per_stage(tmp_path: Path, producer: TrainingRun) -> None:
    export = tmp_path / "producer/run.json"
    payload = fork(producer, export, 4).model_dump(mode="json")
    extended = continuation(producer, export, 4).stages[0]
    assert isinstance(extended, TrainSelfPlay) and extended.learning_state is not None
    payload["stages"][0]["learning_state"] = extended.learning_state.model_dump()
    with pytest.raises(ValueError, match="another starting state"):
        TrainingRegime.model_validate(payload)
