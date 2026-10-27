"""The four-arm declaration binds one recipe, disjoint evaluations and hard caps."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Literal

from etude.experiment import Checkpoint, Checkpoints, ExperimentState, TrainingRun
from etude.jobs import evaluation_spec
from etude.run.records import ArtifactRef
from etude.train.read import read
import pytest

from experiments import curriculum_adapters as study, curriculum_uniform as uniform
from experiments.curriculum_gpu import COMPUTE, GPUConditions, benchmark
from experiments.elemental_long import recipe as baseline_recipe
from experiments.elemental_long_eval import Conditions
from manabot.experiment.job import TrainingJob
from manabot.training.regime import TrainSelfPlay


@pytest.mark.parametrize("held_out", (False, True))
def test_gpu_benchmark_preserves_deals_seats_and_anchor_with_distinct_identity(
    held_out: bool,
) -> None:
    original = study._benchmark(held_out)
    before = original.model_dump_json()
    gpu = benchmark(original)
    assert original.model_dump_json() == before
    assert gpu.environment == original.environment
    assert gpu.name == original.name + "-gpu"
    for scalar, batched in zip(original.matchups, gpu.matchups, strict=True):
        assert scalar.players[1] == batched.players[1]
        assert not {"development", "held-out"}.intersection(batched.tags)
        assert ("held-out-gpu" if held_out else "development-gpu") in batched.tags
        assert scalar.games == batched.games
        assert scalar.definition != batched.definition
        for first, second in zip(scalar.conditions, batched.conditions, strict=True):
            condition = GPUConditions.model_validate(second)
            assert condition.compute == COMPUTE
            assert (
                condition.model_dump(exclude={"compute"})
                == Conditions.model_validate(first).model_dump()
            )


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


@pytest.mark.parametrize("state", ["running", "finished", "failed"])
def test_off_cadence_endpoint_requires_clean_training_finish(
    tmp_path: Path, state: Literal["running", "finished", "failed"]
) -> None:
    run = _run(tmp_path, "time-limited", state)
    endpoint = run.checkpoints[-1].model_copy(update={"clocks": {"updates": 675.0}})
    run = replace(run, checkpoints=Checkpoints([run.checkpoints[0], endpoint]))
    specs = [
        evaluation_spec(job)
        for job in study.eval(ExperimentState(experiment=study.EXPERIMENT, runs=(run,)))
    ]
    for name in (study.MONITOR, study.FINAL):
        actual = [
            cp.clocks["updates"]
            for spec in specs
            if spec.tournament.name == name
            for cp in spec.checkpoints
        ]
        assert actual == ([0.0, 675.0] if state == "finished" else [0.0])


def test_initialization_only_finish_does_not_duplicate_initial_evaluations(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, "initial-only", "finished")
    run = replace(run, checkpoints=Checkpoints([run.checkpoints[0]]))
    specs = [
        evaluation_spec(job)
        for job in study.eval(ExperimentState(experiment=study.EXPERIMENT, runs=(run,)))
    ]
    assert sorted(spec.tournament.name for spec in specs) == sorted(
        [study.MONITOR, study.FINAL]
    )
    assert all(len(spec.checkpoints) == 1 for spec in specs)


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


def test_imported_export_bytes_are_discoverable_as_etude_job_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Tiny identity fixtures, never admitted or played as actual checkpoints.
    cache = tmp_path / "cache"
    cache.mkdir()
    original = uniform.load_control()
    checkpoints = []
    for cp in original.checkpoints:
        payload = f"checkpoint identity fixture {cp.updates}".encode()
        digest = sha256(payload).hexdigest()
        (cache / digest).write_bytes(payload)
        checkpoints.append(
            cp.model_copy(
                update={
                    "artifact": ArtifactRef(
                        uri=f"r2://manabot/jobs/fixture/{digest}",
                        sha256=digest,
                        size_bytes=len(payload),
                    )
                }
            )
        )
    control = original.model_copy(update={"checkpoints": tuple(checkpoints)})

    def fixture_control() -> uniform.UniformControl:
        return control

    monkeypatch.setattr(uniform, "load_control", fixture_control)
    directory = tmp_path / "projection"
    uniform.publish_uniform(directory)
    uniform.materialize_uniform(directory, cache)
    found = read(directory, uniform.EXPERIMENT, uniform.UNIFORM_RUN)
    for cp in found.checkpoints:
        assert cp.artifact.sha256 is not None
        path = directory / uniform.UNIFORM_RUN / "checkpoints" / cp.artifact.sha256
        assert cp.artifact.uri == path.as_uri()
        assert sha256(path.read_bytes()).hexdigest() == cp.artifact.sha256
    first = (
        directory / uniform.UNIFORM_RUN / "checkpoints" / str(control.initial.sha256)
    )
    before = first.read_bytes()
    (cache / control.initial.sha256).write_bytes(b"corrupt cache fixture")
    with pytest.raises(ValueError, match="cached uniform bytes differ"):
        uniform.materialize_uniform(directory, cache)
    assert first.read_bytes() == before


def test_frozen_183_input_version_survives_job_and_bound_input_roundtrips() -> None:
    from manabot.infra.config import ObservationConfig
    from manabot.training.regime import TrainingRegime

    # Fresh defaults evolve; the historical campaign must still round-trip v1.
    assert ObservationConfig().policy_input_version > 1
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
