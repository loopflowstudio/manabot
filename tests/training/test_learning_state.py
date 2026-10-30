"""Portable segments retain learned state but deliberately reset game streams."""

import json
import os
from pathlib import Path

import pytest
import torch

from manabot.experiment.checkpoint_evaluation import stage_checkpoint
from manabot.experiment.evaluation_queue import checkpoints
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint
from manabot.training import execution
from manabot.training.records import TrainingRun
from manabot.training.regime import (
    ArtifactReference,
    DampedPolicyLearning,
    Execution,
    LearningStateImport,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.state.learning_state import admit_learning_state
from manabot.verify.store import VerifyStore
import managym
from tests.training.test_recovery import tracked_updates


def reference(path: Path) -> ArtifactReference:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def recipe(updates: int) -> TrainingRegime:
    return TrainingRegime(
        id="portable-test",
        world=managym.WORLD_VERSION,
        match=MatchConfig(
            hero_deck={"Mountain": 8, "Gray Ogre": 8},
            villain_deck={"Forest": 8, "Llanowar Elves": 8},
        ),
        agent=ModelConfig(
            hidden_dim=8, num_attention_heads=2, value_kind="categorical_wdl"
        ),
        schedule_clock="iteration_fraction",
        stages=[
            TrainSelfPlay(
                id="fit",
                operation="train_self_play",
                updates=updates,
                execution=Execution(
                    device=os.environ.get("MANABOT_NUMERICS_DEVICE", "cpu")
                ),
                streams=2,
                transitions=16,
                learning=DampedPolicyLearning(
                    gradient="damped_policy",
                    ema=0.9,
                    min_advantage=0,
                    advantage_quantile=0,
                ),
            )
        ],
    )


@pytest.fixture
def producer(tmp_path: Path) -> TrainingRun:
    with VerifyStore(tmp_path / "training.sqlite") as store:
        return execution.execute_regime(recipe(2), 197, tmp_path / "producer", store)


def continuation(parent: TrainingRun, export: Path, endpoint: int) -> TrainingRegime:
    recipe = parent.regime.model_copy(deep=True)
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = endpoint
    stage.learning_state = LearningStateImport(
        source_run=reference(export),
        source_stage=parent.stages[0].id,
        **{
            role: parent.stages[0].artifacts[role]
            for role in ("raw", "ema", "optimizer")
        },
    )
    return recipe


def test_segments_keep_adam_ema_and_absolute_coordinates(
    tmp_path: Path,
    producer: TrainingRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_export = tmp_path / "producer/run.json"
    frozen = parent_export.read_bytes()
    recipe = continuation(producer, parent_export, 4)
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    admitted_dir = tmp_path / "admission"
    admitted_dir.mkdir()
    admitted = admit_learning_state(stage, recipe, admitted_dir)
    for role, agent in (("raw", admitted.raw), ("ema", admitted.ema)):
        payload = torch.load(
            producer.stages[0].artifacts[role]["path"], weights_only=False
        )
        # Use ordinary reader state to avoid coupling to checkpoint wire keys.
        expected, _ = load_checkpoint(producer.stages[0].artifacts[role]["path"])
        for key, value in agent.state_dict().items():
            torch.testing.assert_close(
                value, expected.state_dict()[key], rtol=0, atol=0
            )
        assert payload["bc"]["weights"] == role
    initial_steps = [s["step"].item() for s in admitted.optimizer["state"].values()]
    assert initial_steps and min(initial_steps) > 0
    original_update_ema = execution.update_ema
    observed_ema_updates = 0

    def checked_ema(averaged: ActorCritic, learner: ActorCritic, rate: float) -> None:
        nonlocal observed_ema_updates
        if observed_ema_updates == 0:
            for key, value in averaged.state_dict().items():
                torch.testing.assert_close(
                    value,
                    admitted.ema.state_dict()[key].to(value.device),
                    rtol=0,
                    atol=0,
                )
        original_update_ema(averaged, learner, rate)
        observed_ema_updates += 1

    monkeypatch.setattr(execution, "update_ema", checked_ema)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        second = execution.execute_regime(
            recipe, 197, tmp_path / "second", store, checkpoint_seconds=1e9
        )
        third = execution.execute_regime(
            continuation(second, tmp_path / "second/run.json", 5),
            197,
            tmp_path / "third",
            store,
        )
    assert second.status == third.status == "completed"
    assert observed_ema_updates == 3
    initial, endpoint = checkpoints(second, include_initial=True)
    assert initial.coordinates.updates == 2
    assert endpoint.coordinates.updates == 4
    final_checkpoint = stage_checkpoint(third, "fit")
    assert final_checkpoint is not None
    assert final_checkpoint.coordinates.updates == 5
    # Complete-game two-seat batches overshoot the decision floor. Continuation
    # must accumulate recorded rows rather than reconstruct them from quotas.
    assert final_checkpoint.coordinates.learner_transitions == sum(
        run.stages[0].learner_transitions for run in (producer, second, third)
    )
    assert (
        initial.coordinates.learner_transitions
        == producer.stages[0].learner_transitions
    )
    assert second.updates_through() == 4 and third.updates_through() == 5
    rows = {run.id: tracked_updates(run, tmp_path) for run in (second, third)}
    assert [d["iteration"] for d in rows[second.id]] == [3, 4]
    assert [d["iteration"] for d in rows[third.id]] == [5]
    for child, parent in ((second, producer), (third, second)):
        record = child.stages[0]
        origin = record.learning_state_origin
        assert origin is not None and origin.streams == "fresh"
        assert origin.run_id == parent.id
        assert child.seed_streams["collection"] != parent.seed_streams["collection"]
        assert child.recovery_artifact is None and child.parent_run_id is None
        for name in (
            "games",
            "environment_decisions",
            "learner_transitions",
            "optimizer_exposures",
        ):
            assert rows[child.id][-1][f"coordinates/{name}"] == getattr(
                origin, name
            ) + getattr(record, name)
        assert rows[child.id][0]["tau"] == stage.learning.rates(origin.iteration + 1)[1]
        assert (
            child.prior_seconds == 0
        )  # inherited cost is separate, not allowance consumption
    final = torch.load(
        third.stages[0].artifacts["optimizer"]["path"], weights_only=True
    )
    for key, value in admitted.optimizer["state"].items():
        assert final["state"][key]["step"].item() > value["step"].item()
    assert parent_export.read_bytes() == frozen


@pytest.mark.parametrize(
    "fault", ["hash", "missing", "empty_adam", "nan_adam", "ema", "recipe", "endpoint"]
)
def test_rejects_incomplete_or_mismatched_learning_state(
    tmp_path: Path,
    producer: TrainingRun,
    fault: str,
) -> None:
    recipe = continuation(producer, tmp_path / "producer/run.json", 4)
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay) and stage.learning_state is not None
    spec = stage.learning_state
    if fault == "hash":
        spec.optimizer["sha256"] = "0" * 64
    elif fault == "missing":
        spec.optimizer["path"] = str(tmp_path / "missing.pt")
    elif fault == "recipe":
        stage.learning.tau_scale *= 2
    elif fault == "endpoint":
        stage.updates = 2
    else:
        role = "ema" if fault == "ema" else "optimizer"
        path = Path(getattr(spec, role)["path"])
        payload = torch.load(path, weights_only=False)
        if fault == "empty_adam":
            payload["state"] = {}
        elif fault == "nan_adam":
            next(iter(payload["state"].values()))["exp_avg"].fill_(float("nan"))
        else:
            payload["bc"]["averaging"]["iteration"] = 1
        torch.save(payload, path)
        setattr(spec, role, reference(path))
        producer.stages[0].artifacts[role] = reference(path)
        execution.atomic_json(
            tmp_path / "producer/run.json", producer.model_dump(mode="json")
        )
        spec.source_run = reference(tmp_path / "producer/run.json")
    with VerifyStore(tmp_path / "training.sqlite") as store:
        with pytest.raises((ValueError, FileNotFoundError)):
            execution.execute_regime(recipe, 197, tmp_path / "rejected", store)
    failed = TrainingRun.model_validate_json(
        (tmp_path / "rejected/run.json").read_text()
    )
    assert failed.status == "failed" and not failed.stages[0].updates
    assert failed.stages[0].inputs["learning_state/source_run"] == spec.source_run


def test_historical_wire_digest_and_next_iteration(
    tmp_path: Path,
    producer: TrainingRun,
) -> None:
    """Historical source paths/defaults stay immutable while staged paths differ."""
    export = tmp_path / "producer/run.json"
    payload = json.loads(export.read_text())
    payload["regime"]["observation"]["policy_history_version"] = 0
    payload["regime_digest"] = canonical_sha256(payload["regime"])
    # A synthetic coordinate fixture, not a claim that this test trained 26k updates.
    payload["stages"][0]["updates"] = 26000
    for role in ("raw", "ema"):
        path = Path(producer.stages[0].artifacts[role]["path"])
        weights = torch.load(path, weights_only=False)
        weights["bc"]["regime_digest"] = payload["regime_digest"]
        if role == "ema":
            weights["bc"]["averaging"]["iteration"] = 26000
        torch.save(weights, path)
        payload["stages"][0]["artifacts"][role] = reference(path)
    execution.atomic_json(export, payload)
    parent = TrainingRun.model_validate(payload)
    recipe = continuation(parent, export, 100000)
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    out = tmp_path / "admit"
    out.mkdir()
    state = admit_learning_state(stage, recipe, out)
    assert state.origin.iteration == 26000
    assert stage.learning.rates(state.origin.iteration + 1) == pytest.approx(
        (6.957944302072213e-6, 0.0023684978179155515)
    )
    assert (
        Path(state.artifacts["parent_run"]["path"]).read_bytes() == export.read_bytes()
    )


@pytest.fixture
def interrupted(tmp_path: Path) -> TrainingRun:
    """A parent with per-update monitoring exports, retained as interrupted.

    The status is the synthetic part, standing in for a watchdog kill after
    update 3; a raw-only import reads only the record and one monitoring export.
    """
    with VerifyStore(tmp_path / "interrupted.sqlite") as store:
        execution.execute_regime(
            recipe(3), 197, tmp_path / "interrupted", store, checkpoint_updates=1
        )
    export = tmp_path / "interrupted/run.json"
    payload = json.loads(export.read_text())
    payload["status"] = payload["stages"][0]["status"] = "interrupted"
    payload["error"] = "TimeoutError: training resource wall deadline exceeded"
    execution.atomic_json(export, payload)
    return TrainingRun.model_validate(payload)


def from_monitoring(
    parent: Path, run: TrainingRun, update: int, endpoint: int
) -> TrainingRegime:
    continued = recipe(endpoint)
    stage = continued.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    (receipt,) = [c for c in run.monitoring_checkpoints if c.updates == update]
    assert receipt.artifact is not None
    stage.learning_state = LearningStateImport(
        source_run=reference(parent),
        source_stage="fit",
        raw=receipt.artifact,
        monitoring_update=update,
    )
    return continued


def test_raw_only_import_resets_ema_and_adam_at_the_monitoring_export(
    tmp_path: Path,
    interrupted: TrainingRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "interrupted/run.json"
    continued = from_monitoring(parent, interrupted, 2, 4)
    receipt = interrupted.monitoring_checkpoints[1]
    assert receipt.artifact is not None
    expected, _ = load_checkpoint(receipt.artifact["path"])
    seeded: list[dict[str, torch.Tensor]] = []
    original_update_ema = execution.update_ema

    def checked_ema(averaged: ActorCritic, learner: ActorCritic, rate: float) -> None:
        if not seeded:
            seeded.append(
                {k: v.detach().cpu().clone() for k, v in averaged.state_dict().items()}
            )
        original_update_ema(averaged, learner, rate)

    monkeypatch.setattr(execution, "update_ema", checked_ema)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        run = execution.execute_regime(continued, 197, tmp_path / "continued", store)
    assert run.status == "completed" and run.updates_through() == 4
    # The EMA starts as the monitoring export's raw weights.
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(seeded[0][key], value, rtol=0, atol=0)
    record = run.stages[0]
    origin = record.learning_state_origin
    assert origin is not None and origin.reset == ("ema", "optimizer")
    assert origin.iteration == 2 and origin.run_id == interrupted.id
    for name in ("games", "learner_transitions", "optimizer_exposures"):
        assert getattr(origin, name) == getattr(receipt, name)
    assert origin.optimizer_updates == receipt.optimizer_updates
    assert set(record.inputs) == {"learning_state/source_run", "learning_state/raw"}
    # Fresh Adam: its step counters count only this segment's optimizer steps.
    final = torch.load(record.artifacts["optimizer"]["path"], weights_only=True)
    assert record.optimizer_updates is not None
    assert {s["step"].item() for s in final["state"].values()} <= {
        float(record.optimizer_updates)
    }


@pytest.mark.parametrize("fault", ["receipt", "bytes", "completed_roles", "finished"])
def test_raw_only_import_requires_the_declared_monitoring_export(
    tmp_path: Path, interrupted: TrainingRun, fault: str
) -> None:
    parent = tmp_path / "interrupted/run.json"
    if fault == "completed_roles":
        with pytest.raises(ValueError, match="raw alone"):
            LearningStateImport(
                source_run=reference(parent),
                source_stage="fit",
                raw=interrupted.stages[0].artifacts["raw"],
                ema=interrupted.stages[0].artifacts["ema"],
                monitoring_update=2,
            )
        return
    continued = from_monitoring(parent, interrupted, 2, 4)
    stage = continued.stages[0]
    assert isinstance(stage, TrainSelfPlay) and stage.learning_state is not None
    if fault == "receipt":
        stage.learning_state.monitoring_update = 3
    elif fault == "bytes":
        # The final raw export is not the monitoring export at update 2.
        stage.learning_state.raw = interrupted.stages[0].artifacts["raw"]
    else:
        payload = json.loads(parent.read_text())
        payload["status"] = payload["stages"][0]["status"] = "failed"
        execution.atomic_json(parent, payload)
        stage.learning_state.source_run = reference(parent)
    out = tmp_path / "admit"
    out.mkdir()
    with pytest.raises(ValueError):
        admit_learning_state(stage, continued, out)


def test_wall_bounded_stage_pauses_with_learner_state_a_successor_continues(
    tmp_path: Path,
) -> None:
    bounded = recipe(5)
    watchdog = min(bounded.wall_seconds, bounded.stages[0].execution.wall_seconds)
    # A reserve just inside the watchdog: no second update ever fits.
    reserve = watchdog - 1e-3
    with VerifyStore(tmp_path / "training.sqlite") as store:
        run = execution.execute_regime(
            bounded, 197, tmp_path / "bounded", store, stop_reserve_seconds=reserve
        )
        record = run.stages[0]
        assert run.status == record.status == "paused"
        assert run.updates_through() == 1
        assert {"raw", "ema", "optimizer"} <= set(record.artifacts)
        assert run.selected_artifact == record.artifacts["raw"]
        successor = execution.execute_regime(
            continuation(run, tmp_path / "bounded/run.json", 3),
            197,
            tmp_path / "successor",
            store,
        )
    assert successor.status == "completed" and successor.updates_through() == 3
    with pytest.raises(ValueError, match="wall-bounded"):
        with VerifyStore(tmp_path / "other.sqlite") as store:
            execution.execute_regime(
                bounded,
                197,
                tmp_path / "refused",
                store,
                stop_reserve_seconds=watchdog,
            )
