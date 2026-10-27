"""The four-arm declaration binds one recipe, disjoint evaluations and hard caps."""

from pathlib import Path
from typing import Literal

from etude.experiment import Checkpoint, Checkpoints, ExperimentState, TrainingRun
from etude.jobs import evaluation_spec
from etude.run.records import ArtifactRef

from experiments import curriculum_adapters as study
from experiments.elemental_long import recipe as baseline_recipe
from experiments.elemental_long_eval import Conditions
from manabot.experiment.job import TrainingJob
from manabot.training.regime import TrainSelfPlay


def test_four_arms_share_learning_and_initialization_with_inclusive_reserves() -> None:
    runs = study.runs("HEAD")
    assert len(runs) == 3
    assert len(study.runs("HEAD", reuse_uniform=False)) == 4
    assert [run.params["arm"] for run in study.runs("HEAD", arm="ME")] == ["ME"]
    assert {run.seed for run in runs} == {18301}
    assert [run.params["arm"] for run in runs] == list(study.ARMS[1:])
    specs = [TrainingJob.model_validate_json(run.job.config or "") for run in runs]
    controls = []
    for run, spec in zip(runs, specs, strict=True):
        assert run.job.max_attempts == 1
        assert run.job.max_usd + study.RESERVE_USD <= 10
        assert spec.milestones == study.CADENCE
        assert spec.initial_checkpoint is not None
        assert spec.initial_checkpoint.key == (
            "jobs/mtg183-elemental-long-w1/runtime/artifacts/sha256/"
            "1004ccdf46f373e833dadcb5278351a758e5328ed7c28049259c8c06be614688"
        )
        assert run.job.inputs[0].key == spec.initial_checkpoint.key
        stage = spec.regime.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        assert stage.adaptive is not None and len(stage.adaptive.cells) == 100
        assert stage.learning_seats == "both"
        assert (
            stage.game_errors
            == baseline_recipe(study.UPDATES, hours=study.HOURS).stages[0].game_errors
        )
        control = spec.regime.model_dump(mode="json")
        del control["id"]
        control["stages"][0]["adaptive"].pop("signal")
        controls.append(control)
    assert all(control == controls[0] for control in controls)
    assert sum(run.job.max_usd + study.RESERVE_USD for run in runs) <= 30


def _run(
    tmp_path: Path, name: str, state: Literal["running", "finished", "failed"]
) -> TrainingRun:
    checkpoints = []
    for ordinal, update in enumerate(study.MONITOR_UPDATES, start=1):
        checkpoint = tmp_path / f"{name}-{update}.pt"
        checkpoint.write_bytes(b"identity fixture, never played")
        checkpoints.append(
            Checkpoint(
                run=name,
                experiment=study.EXPERIMENT,
                ordinal=ordinal,
                artifact=ArtifactRef(
                    uri=checkpoint.as_uri(), sha256="0" * 64, size_bytes=1
                ),
                clocks={"updates": float(update)},
            )
        )
    return TrainingRun(
        name=name,
        experiment=study.EXPERIMENT,
        state=state,
        checkpoints=Checkpoints(checkpoints),
    )


def test_development_and_final_deals_are_disjoint_and_failure_is_not_an_endpoint(
    tmp_path: Path,
) -> None:
    state = ExperimentState(
        experiment=study.EXPERIMENT,
        runs=(
            _run(tmp_path, "completed", "finished"),
            _run(tmp_path, "failed", "failed"),
        ),
    )
    specs = [evaluation_spec(job) for job in study.eval(state)]
    development = [s for s in specs if s.tournament.name == study.MONITOR]
    held_out = [s for s in specs if s.tournament.name == study.FINAL]
    assert sum(len(s.checkpoints) for s in development) == 14
    assert sum(len(s.checkpoints) for s in held_out) == 3
    assert all(
        cp.run == "completed"
        for s in held_out
        for cp in s.checkpoints
        if int(cp.clocks["updates"]) != 0
    )
    deals = []
    for spec, count in ((development[0], 400), (held_out[0], 2000)):
        matchup = spec.tournament.matchups[0]
        rows = [Conditions.model_validate(value) for value in matchup.conditions]
        assert len(rows) == 10
        selected = [
            deal for row in rows for group in row.groups for deal in group.deals
        ]
        assert len(selected) == count
        assert len(set(selected)) == count // 2
        assert {row.deck for row in rows} == set(study.DECKS)
        deals.append(set(selected))
    assert deals[0].isdisjoint(deals[1])
    assert study.eval(ExperimentState(experiment=study.EXPERIMENT)) == []


def test_reused_uniform_is_a_labelled_projection_of_real_export_references(
    tmp_path: Path,
) -> None:
    from etude.train.read import read

    from experiments.curriculum_uniform import (
        EXPERIMENT,
        UNIFORM_RUN,
        load_control,
        publish_uniform,
    )

    control = load_control()
    assert control.estimated_inclusive_dollars < 10
    assert control.endpoint_updates == 1325
    receipt = publish_uniform(tmp_path, control=control)
    first = read(tmp_path, EXPERIMENT, UNIFORM_RUN)
    assert len(first.checkpoints) == 28
    assert [int(cp.at["updates"]) for cp in first.checkpoints] == [
        cp.updates for cp in control.checkpoints
    ]
    assert [cp.artifact.sha256 for cp in first.checkpoints] == [
        cp.artifact.sha256 for cp in control.checkpoints
    ]
    before = receipt.read_bytes()
    publish_uniform(tmp_path, control=control)
    second = read(tmp_path, EXPERIMENT, UNIFORM_RUN)
    assert len(second.checkpoints) == 28
    assert before == receipt.read_bytes()


def test_frozen_183_input_version_survives_job_and_bound_input_roundtrips() -> None:
    from manabot.infra.config import ObservationConfig
    from manabot.training.regime import TrainingRegime

    assert ObservationConfig().policy_input_version == 2
    regime = study.recipe("M")
    assert regime.observation.policy_input_version == 1
    raw = regime.model_dump(mode="json")
    assert "policy_input_version" not in raw["observation"]
    assert TrainingRegime.model_validate(raw).observation.policy_input_version == 1
    spec = TrainingJob.model_validate_json(study.runs("HEAD")[0].job.config)
    assert spec.regime.observation.policy_input_version == 1
    assert spec.bound_regime(Path("/inputs")).observation.policy_input_version == 1


def test_cost_selection_uses_first_seen_cost_and_never_backdates_exports(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime
    from decimal import Decimal

    from etude.launch.records import Status
    import pytest

    from experiments.curriculum_costs import CostLedger

    run = _run(tmp_path, "adaptive-m", "running")
    now = datetime.now(UTC)
    status = Status(
        run=run.name,
        phase="running",
        reason=None,
        outcome=None,
        cause=None,
        until="updates=1400",
        clocks={"updates": 200},
        usd=Decimal("6"),
        attempts=1,
        rental_ended=False,
        estimated=True,
    )
    ledger = CostLedger(arms={run.name: "M"})
    early = TrainingRun(
        name=run.name,
        experiment=run.experiment,
        state="running",
        checkpoints=Checkpoints(run.checkpoints[:3]),
    )
    ledger.add(
        ExperimentState(experiment=study.EXPERIMENT, runs=(early,)),
        {run.name: status},
        now,
    )
    assert ledger.selections()[run.name].clocks["updates"] == 200
    prior = CostLedger(arms={run.name: "M"}, prior_dollars={run.name: Decimal("0.80")})
    prior.add(
        ExperimentState(experiment=study.EXPERIMENT, runs=(early,)),
        {run.name: status},
        now,
    )
    assert prior.selections() == {}
    ledger.add(
        ExperimentState(experiment=study.EXPERIMENT, runs=(run,)),
        {
            run.name: status.model_copy(
                update={"usd": Decimal("8"), "clocks": {"updates": 1400}}
            )
        },
        now,
    )
    assert len(ledger.observations) == len(run.checkpoints)
    assert ledger.selections()[run.name].clocks["updates"] == 200
    altered = run.checkpoints[0].model_copy(
        update={"artifact": ArtifactRef(uri="file:///changed", sha256="1" * 64)}
    )
    conflict = TrainingRun(
        name=run.name,
        experiment=run.experiment,
        state="running",
        checkpoints=Checkpoints((altered,)),
    )
    with pytest.raises(ValueError, match="bytes changed"):
        ledger.add(
            ExperimentState(experiment=study.EXPERIMENT, runs=(conflict,)),
            {run.name: status},
            now,
        )
