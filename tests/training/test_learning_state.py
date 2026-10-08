"""Portable segments retain learned state but deliberately reset game streams."""

import json
import os
from pathlib import Path

import pytest
import torch

from manabot.arena.models import canonical_sha256, file_sha256
from manabot.infra.hypers import AgentSpec, MatchHypers
from manabot.model.agent import Agent
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training import execution
from manabot.training.checkpoint_queue import checkpoints
from manabot.training.learning_state import admit_learning_state
from manabot.training.models import (
    ArtifactReference,
    AtaraxosMoveLearning,
    Execution,
    LearningStateImport,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.training.monitor_evaluation import stage_checkpoint
from manabot.verify.store import VerifyStore
import managym
from tests.remote.test_job_spec import job_spec
from tests.training.test_allocation import AllocationClock


def reference(path: Path) -> ArtifactReference:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


@pytest.fixture
def producer(tmp_path: Path) -> TrainingRun:
    recipe = TrainingRegime(
        id="portable-test",
        world=managym.WORLD_VERSION,
        match=MatchHypers(
            hero_deck={"Mountain": 8, "Gray Ogre": 8},
            villain_deck={"Forest": 8, "Llanowar Elves": 8},
        ),
        agent=AgentSpec(
            hidden_dim=8, num_attention_heads=2, value_kind="categorical_wdl"
        ),
        schedule_clock="iteration_fraction",
        stages=[
            TrainSelfPlay(
                id="fit",
                operation="train_self_play",
                updates=2,
                execution=Execution(
                    device=os.environ.get("MANABOT_NUMERICS_DEVICE", "cpu")
                ),
                streams=2,
                transitions=16,
                learning=AtaraxosMoveLearning(
                    gradient="ataraxos_move",
                    ema=0.9,
                    min_advantage=0,
                    advantage_quantile=0,
                ),
            )
        ],
    )
    with VerifyStore(tmp_path / "training.sqlite") as store:
        return execution.execute_regime(recipe, 197, tmp_path / "producer", store)


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
        expected, _ = load_checkpoint_agent(producer.stages[0].artifacts[role]["path"])
        for key, value in agent.state_dict().items():
            torch.testing.assert_close(
                value, expected.state_dict()[key], rtol=0, atol=0
            )
        assert payload["bc"]["weights"] == role
    initial_steps = [s["step"].item() for s in admitted.optimizer["state"].values()]
    assert initial_steps and min(initial_steps) > 0
    original_update_ema = execution.update_ema
    observed_ema_updates = 0

    def checked_ema(averaged: Agent, learner: Agent, rate: float) -> None:
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
    assert final_checkpoint.coordinates.learner_transitions == 5 * 16 * 2
    assert initial.coordinates.learner_transitions == 2 * 16 * 2
    assert second.updates_through() == 4 and third.updates_through() == 5
    assert [d["iteration"] for d in second.stages[0].diagnostics] == [3, 4]
    assert [d["iteration"] for d in third.stages[0].diagnostics] == [5]
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
            assert record.diagnostics[-1]["coordinates"][name] == getattr(
                origin, name
            ) + getattr(record, name)
        assert (
            record.diagnostics[0]["tau"]
            == stage.learning.rates(origin.iteration + 1)[1]
        )
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
    assert failed.status == "failed" and not failed.stages[0].diagnostics
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
    payload["stages"][0]["diagnostics"] = [{"iteration": i} for i in range(1, 26001)]
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


def test_allocation_pause_exports_next_portable_segment(
    tmp_path: Path,
    producer: TrainingRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = AllocationClock()
    allocation = job_spec().admit(clock.time())
    export = execution.export_training_run

    def pause_after_update(
        run_id: str,
        owner: VerifyStore,
        out: str | Path,
    ) -> TrainingRun:
        run = export(run_id, owner, out)
        if run.status == "running" and run.stages and run.stages[0].diagnostics:
            clock.now = allocation.pause_at
        return run

    with VerifyStore(tmp_path / "training.sqlite") as store:
        with monkeypatch.context() as patch:
            patch.setattr(execution, "time", clock)
            patch.setattr(execution, "PROGRESS_EXPORT_SECONDS", 0.0)
            patch.setattr(execution, "PROGRESS_EXPORT_SHARE", float("inf"))
            patch.setattr(execution, "export_training_run", pause_after_update)
            paused = execution.execute_regime(
                continuation(producer, tmp_path / "producer/run.json", 4),
                197,
                tmp_path / "paused",
                store,
                allocation=allocation,
            )
        assert paused.status == "paused" and paused.updates_through() == 3
        assert paused.recovery_artifact is None
        continued = execution.execute_regime(
            continuation(paused, tmp_path / "paused/run.json", 4),
            197,
            tmp_path / "continued",
            store,
        )
    assert continued.status == "completed" and continued.updates_through() == 4
    assert [d["iteration"] for d in continued.stages[0].diagnostics] == [4]
