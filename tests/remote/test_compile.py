"""Rental placement and cost admission without GPU or provider access."""

import json
from pathlib import Path

import pytest

from manabot.remote.plan import DeploymentPlan, JobSpec, Source, compile_plan

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Source(commit="a" * 40, tree="b" * 40, lock_sha256="c" * 64)


def test_placement_is_declared_and_input_is_preserved() -> None:
    raw = (ROOT / "ops/examples/step-target.json").read_text()
    spec = JobSpec.model_validate_json(
        (ROOT / "ops/jobs/runpod-small.json").read_text()
    )
    spec = spec.model_copy(
        update={"machine": spec.machine.model_copy(update={"vcpus": 2})}
    )
    data = json.loads(raw)
    for stage in data["stages"]:
        stage["execution"]["threads"] = 4
    raw = json.dumps(data)
    plan = compile_plan(raw, spec, SOURCE, 197)
    assert plan.input_json == raw
    assert all(
        s.execution.device == "cuda" and s.execution.threads == 2
        for s in plan.regime.stages
    )
    assert plan.projected_dollars == 0.31
    assert DeploymentPlan.model_validate_json(plan.model_dump_json()) == plan
    forged = plan.model_dump()
    forged["regime"]["stages"][0]["updates"] = 99
    with pytest.raises(ValueError, match="differs"):
        DeploymentPlan.model_validate(forged)


def test_cost_and_watchdog_admission() -> None:
    data = json.loads((ROOT / "ops/jobs/runpod-small.json").read_text())
    with pytest.raises(ValueError, match="spending limit"):
        JobSpec.model_validate(data | {"spending_limit": 0.01})
    with pytest.raises(ValueError, match="reserves"):
        JobSpec.model_validate(data | {"lifetime_hours": 700 / 3600})


def test_cuda_rejects_recovery_and_mixed_device_continuation() -> None:
    from manabot.training.models import TrainingRegime

    data = json.loads((ROOT / "experiments/regimes/direct-self-play.json").read_text())
    data["stages"][0]["execution"]["device"] = "cuda"
    with pytest.raises(ValueError, match="continuation"):
        TrainingRegime.model_validate(data)
    data["stages"][1]["execution"]["device"] = "cuda"
    data["recovery_max_microsteps"] = 1000
    data["schedule_clock"] = "iteration_fraction"
    with pytest.raises(ValueError, match="CUDA process recovery"):
        TrainingRegime.model_validate(data)
