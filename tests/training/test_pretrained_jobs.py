"""Published trunks reach fresh control/memory learners through staged job inputs."""

from decimal import Decimal
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from experiments.opponent_memory import branch_jobs
from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.env.match import Match
from manabot.experiment.job import (
    LearnerContinuation,
    PretrainedInitialization,
    StagedExport,
    TrainingJob,
    run_worker,
)
from manabot.infra.identity import file_sha256
from manabot.training.records import TrainingRun
from manabot.training.regime import TrainingRegime, TrainSelfPlay
import managym
from tests.training.test_etude_job import ema_regime, worker_environment
from tests.training.test_pretrained import memory_trunk
from tests.training.test_recovery import tracked_updates

__all__ = ["memory_trunk"]


def staged_trunk(parent: TrainingRun, directory: Path) -> PretrainedInitialization:
    """Use the run record and EMA bytes published by the actual executor."""
    checkpoint = Path(parent.stages[0].artifacts["ema"]["path"])
    sources = {
        "source_run": directory / "memory-trunk/run.json",
        "checkpoint": checkpoint,
    }
    staged = {
        role: StagedExport(
            key=f"trunks/39101/{path.name}",
            sha256=file_sha256(path),
            bytes=path.stat().st_size,
        )
        for role, path in sources.items()
    }
    return PretrainedInitialization(
        source_stage=parent.stages[0].id, weights="ema", **staged
    )


def branch_recipe(parent: TrainingRun) -> TrainingRegime:
    recipe = parent.regime.model_copy(deep=True)
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 1
    return recipe


def deliver(parent: TrainingRun, directory: Path, spec: TrainingJob) -> Path:
    """Lay out exactly the SDK inputs, matching their declared digests/sizes."""
    assert spec.pretrained is not None
    sources = {
        spec.pretrained.source_run.key: directory / "memory-trunk/run.json",
        spec.pretrained.checkpoint.key: Path(parent.stages[0].artifacts["ema"]["path"]),
    }
    inputs = directory / "inputs"
    inputs.mkdir()
    for given in spec.pretrained.inputs():
        source = sources[given.key]
        assert (file_sha256(source), source.stat().st_size) == (
            given.sha256,
            given.size_bytes,
        )
        (inputs / given.name).write_bytes(source.read_bytes())
    return inputs


@pytest.mark.parametrize("encoder", ["none", "gru"])
def test_worker_branches_from_one_published_trunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory_trunk: TrainingRun,
    encoder: Literal["none", "gru"],
) -> None:
    trunk = staged_trunk(memory_trunk, tmp_path)
    requests = branch_jobs(
        "HEAD",
        regime=branch_recipe(memory_trunk),
        trunks={39101: trunk},
        max_hours=1,
        max_usd=Decimal("0.25"),
    )
    specs = [
        TrainingJob.model_validate_json(each.job.config or "") for each in requests
    ]
    assert {spec.regime.agent.decision_memory for spec in specs} == {
        "none",
        "transformer",
        "gru",
    }
    assert all(each.seed == 39101 for each in requests)
    assert all(spec.pretrained == trunk for spec in specs)
    assert all(each.job.inputs == requests[0].job.inputs for each in requests)
    spec = next(spec for spec in specs if spec.regime.agent.decision_memory == encoder)
    # Local workers read this checkout's source instead of an unpacked archive.
    spec.source_bundle = None
    stage = spec.regime.stages[0]
    assert isinstance(stage, TrainSelfPlay) and stage.pretrained is None
    bound = spec.bound_regime(Path("/inputs")).stages[0]
    assert isinstance(bound, TrainSelfPlay) and bound.pretrained is not None
    assert bound.pretrained.source_run["path"] == "/inputs/producer-run.json"
    inputs = deliver(memory_trunk, tmp_path, spec)
    worker_environment(tmp_path, monkeypatch, spec)
    monkeypatch.setenv("ETUDE_INPUTS_DIR", str(inputs))
    monkeypatch.setenv("ETUDE_SEED", "39101")
    child = run_worker()
    record = child.stages[0]
    initial, _ = load_checkpoint(record.artifacts["initial_raw"]["path"])
    expected, _ = load_checkpoint(memory_trunk.stages[0].artifacts["ema"]["path"])
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(initial.state_dict()[key], value, rtol=0, atol=0)
    assert all(gate == 0 for gate in initial.memory_gates().values())
    env = managym.Env(seed=198)
    observation, _ = env.reset(Match(spec.regime.root_match).to_rust())
    for _ in range(8):
        if observation.game_over:
            break
        observation, _, _, _, _ = env.step(0)
    buffers = initial.observation_space.encoder.allocate()
    env.encode_observation_into(observation, buffers)
    rows = {key: value[None] for key, value in buffers.items()}
    assert np.any(rows["decision_records"])
    with torch.inference_mode():
        actual = initial.forward_distribution(on_device(initial.prepare(rows), "cpu"))
        reference = expected.forward_distribution(
            on_device(expected.prepare(rows), "cpu")
        )
    for value, target in zip(actual, reference, strict=True):
        torch.testing.assert_close(value, target, rtol=0, atol=0)
    assert record.learning_state_origin is None
    assert record.producer_cost is not None
    assert record.producer_cost.run_id == memory_trunk.id
    assert record.producer_cost.weights == "ema"
    assert record.producer_cost.cumulative_seconds == pytest.approx(
        memory_trunk.stages[0].cumulative_seconds
    )
    assert [row["iteration"] for row in tracked_updates(child, tmp_path)] == [1]
    optimizer = torch.load(record.artifacts["optimizer"]["path"], weights_only=True)
    assert max(int(value["step"].item()) for value in optimizer["state"].values()) == (
        record.optimizer_updates
    )
    assert record.inputs["pretrained/checkpoint"]["sha256"] == trunk.checkpoint.sha256


@pytest.mark.parametrize("fault", ["source_run", "checkpoint", "weights", "model"])
def test_pretrained_worker_refuses_invalid_producer_before_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory_trunk: TrainingRun,
    fault: str,
) -> None:
    spec = TrainingJob(
        regime=branch_recipe(memory_trunk),
        experiment="fixture",
        pretrained=staged_trunk(memory_trunk, tmp_path),
    )
    inputs = deliver(memory_trunk, tmp_path, spec)
    assert spec.pretrained is not None
    if fault in {"source_run", "checkpoint"}:
        name, _ = spec.pretrained.files()[fault]
        with (inputs / name).open("ab") as handle:
            handle.write(b"corrupt")
        message = "hash/size mismatch"
    elif fault == "weights":
        spec.pretrained.weights = "raw"
        message = "weight artifact"
    else:
        spec.regime.agent.hidden_dim *= 2
        message = "model settings"
    worker_environment(tmp_path, monkeypatch, spec)
    monkeypatch.setenv("ETUDE_INPUTS_DIR", str(inputs))
    with pytest.raises(ValueError, match=message):
        run_worker()
    failed = TrainingRun.model_validate_json(
        (tmp_path / "output/training/run.json").read_text()
    )
    assert failed.status == "failed" and not failed.stages[0].updates


def test_pretrained_job_requires_one_unbound_starting_state() -> None:
    export = StagedExport(key="trunk", sha256="0" * 64, bytes=1)
    pretrained = PretrainedInitialization(
        source_stage="fit", source_run=export, checkpoint=export
    )
    for other in (
        {"initial_checkpoint": export},
        {
            "continuation": LearnerContinuation(
                from_update=1,
                source_stage="fit",
                source_run=export,
                raw=export,
                ema=export,
                optimizer=export,
            )
        },
    ):
        with pytest.raises(ValueError, match="one starting state"):
            TrainingJob(
                regime=ema_regime(2),
                experiment="fixture",
                pretrained=pretrained,
                **other,
            )
    bound = TrainingJob(
        regime=ema_regime(2), experiment="fixture", pretrained=pretrained
    ).bound_regime(Path("/inputs"))
    with pytest.raises(ValueError, match="bound from its inputs"):
        TrainingJob(regime=bound, experiment="fixture", pretrained=pretrained)
