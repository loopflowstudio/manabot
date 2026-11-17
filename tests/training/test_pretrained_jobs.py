"""Published trunks reach fresh control/memory learners through staged job inputs."""

from decimal import Decimal
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from etude.jobs import Checkpoints, ExperimentState, TrainingRunSummary
from etude.storage import DirectoryStorage, S3Storage, run_prefix
import numpy as np
import pytest
import torch

from experiments import opponent_memory
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
from manabot.infra.config import ObservationConfig
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
    storage = published_trunk(parent, directory)
    return PretrainedInitialization.from_storage(
        storage,
        "trunks/run.json",
        source_stage=parent.stages[0].id,
        weights="ema",
    )


def published_trunk(parent: TrainingRun, directory: Path) -> DirectoryStorage:
    storage = DirectoryStorage(directory / "published")
    storage.put_file("trunks/run.json", directory / "memory-trunk/run.json")
    for role in ("raw", "ema"):
        path = Path(parent.stages[0].artifacts[role]["path"])
        storage.put_file(f"trunks/{path.name}", path)
    return storage


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
    recipe = branch_recipe(memory_trunk)
    recipe.agent.decision_memory = encoder
    spec = TrainingJob(regime=recipe, experiment="fixture", pretrained=trunk)
    stage = spec.regime.stages[0]
    assert isinstance(stage, TrainSelfPlay) and stage.pretrained is None
    bound = spec.bound_regime(Path("/inputs")).stages[0]
    assert isinstance(bound, TrainSelfPlay) and bound.pretrained is not None
    assert bound.pretrained.source_run["path"] == "/inputs/producer-run.json"
    inputs = deliver(memory_trunk, tmp_path, spec)
    worker_environment(tmp_path, monkeypatch, spec)
    monkeypatch.setenv("ETUDE_INPUTS_DIR", str(inputs))
    monkeypatch.setenv("ETUDE_SEED", str(memory_trunk.seed))
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


@pytest.mark.parametrize(
    "fault", ["source_run", "checkpoint", "weights", "model", "seed"]
)
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
    elif fault == "seed":
        spec.pretrained.source_seed = memory_trunk.seed + 1
        message = "different seed than declared"
    else:
        spec.regime.agent.hidden_dim *= 2
        message = "model settings"
    worker_environment(tmp_path, monkeypatch, spec)
    monkeypatch.setenv("ETUDE_INPUTS_DIR", str(inputs))
    with pytest.raises(ValueError, match=message):
        run_worker()
    if fault == "seed":
        # Refused before execution: no training record or update exists.
        assert not (tmp_path / "output/training").exists()
        return
    failed = TrainingRun.model_validate_json(
        (tmp_path / "output/training/run.json").read_text()
    )
    assert failed.status == "failed" and not failed.stages[0].updates


def test_pretrained_job_requires_one_unbound_starting_state() -> None:
    export = StagedExport(key="trunk", sha256="0" * 64, bytes=1)
    pretrained = PretrainedInitialization(
        source_stage="fit", source_seed=197, source_run=export, checkpoint=export
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


@pytest.mark.parametrize("weights", ["raw", "ema"])
@pytest.mark.parametrize("prefix", ["", "workspace/"])
def test_reads_published_pretrained_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory_trunk: TrainingRun,
    weights: Literal["raw", "ema"],
    prefix: str,
) -> None:
    storage = published_trunk(memory_trunk, tmp_path)
    # Exercise bucket-relative input keys without credentials or a remote read.
    bucket = S3Storage(
        "fixture",
        prefix=prefix,
        client=SimpleNamespace(
            meta=SimpleNamespace(endpoint_url="https://fixture.invalid")
        ),
    )
    monkeypatch.setattr(bucket, "get", storage.get)
    trunk = PretrainedInitialization.from_storage(
        bucket,
        "trunks/run.json",
        source_stage=memory_trunk.stages[0].id,
        weights=weights,
    )
    assert trunk.source_seed == memory_trunk.seed
    assert trunk.weights == weights
    assert trunk.source_stage == memory_trunk.stages[0].id
    for export in (trunk.source_run, trunk.checkpoint):
        assert export.key.startswith(prefix + "trunks/")
        data = storage.get(export.key.removeprefix(prefix))
        assert (hashlib.sha256(data).hexdigest(), len(data)) == (
            export.sha256,
            export.bytes,
        )


@pytest.mark.parametrize(
    "fault", ["run", "stage", "missing-stage", "role", "digest", "size"]
)
def test_published_pretrained_reader_refuses_invalid_exports(
    tmp_path: Path,
    memory_trunk: TrainingRun,
    fault: str,
) -> None:
    storage = published_trunk(memory_trunk, tmp_path)
    producer = memory_trunk.model_copy(deep=True)
    stage = producer.stages[0]
    source_stage = stage.id
    message = "completed"
    if fault == "run":
        producer.status = "failed"
    elif fault == "stage":
        stage.status = "failed"
    elif fault == "missing-stage":
        source_stage = "absent"
    elif fault == "role":
        del stage.artifacts["ema"]
        message = "no ema weight export"
    elif fault == "digest":
        path = Path(stage.artifacts["ema"]["path"])
        data = storage.get(f"trunks/{path.name}")
        storage.put(f"trunks/{path.name}", bytes([data[0] ^ 1]) + data[1:])
        message = "hash/size mismatch"
    else:
        stage.artifacts["ema"]["bytes"] += 1
        message = "hash/size mismatch"
    storage.put("trunks/run.json", producer.model_dump_json().encode())
    with pytest.raises(ValueError, match=message):
        PretrainedInitialization.from_storage(
            storage,
            "trunks/run.json",
            source_stage=source_stage,
            weights="ema",
        )


def test_branch_builder_declares_shared_inputs_and_matching_seed_labels() -> None:
    export = StagedExport(key="trunk", sha256="0" * 64, bytes=1)
    trunk = PretrainedInitialization(
        source_stage="fit",
        source_seed=39101,
        source_run=export,
        checkpoint=export,
        weights="ema",
    )
    recipe = ema_regime(2)
    recipe.observation = ObservationConfig(
        max_decision_records=64,
        max_decision_reference_details=65536,
        max_decision_references=1024,
        max_decision_members=16384,
    )
    recipe.agent.attention_on = True
    recipe.agent.policy_film = True
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.execution.collection_rows = 2
    stage.execution.bootstrap_rows = 2
    stage.execution.learner_rows = 2
    requests = branch_jobs(
        "HEAD",
        regime=recipe,
        trunks={39101: trunk},
        max_hours=1,
        max_usd=Decimal("0.25"),
        milestones=(1,),
    )
    specs = [
        TrainingJob.model_validate_json(each.job.config or "") for each in requests
    ]
    assert {spec.regime.agent.decision_memory for spec in specs} == {
        "none",
        "gru",
    }
    assert all(each.seed == trunk.source_seed for each in requests)
    assert all(spec.pretrained == trunk for spec in specs)
    assert all(spec.experiment == "opponent_memory" for spec in specs)
    assert all(spec.milestones == (1,) for spec in specs)
    assert all(each.job.inputs == requests[0].job.inputs for each in requests)
    with pytest.raises(ValueError, match="match its producer seed"):
        branch_jobs(
            "HEAD",
            regime=recipe,
            trunks={39102: trunk},
            max_hours=1,
            max_usd=Decimal("0.25"),
            milestones=(),
        )


@pytest.mark.parametrize("state", ["missing", "running", "finished"])
def test_pilot_requires_completed_trunk_for_every_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: Literal["missing", "running", "finished"],
) -> None:

    runs = (
        ()
        if state == "missing"
        else (
            TrainingRunSummary(
                name="trunk",
                experiment="opponent_memory",
                job="decision-memory-trunk",
                launch="decision-memory-trunk",
                line="trunk",
                state=state,
                seed=39101,
                checkpoints=Checkpoints(),
            ),
        )
    )

    def observe(source: object, experiment: str) -> ExperimentState:
        return ExperimentState(experiment=experiment, runs=runs)

    monkeypatch.setattr(opponent_memory, "observe", observe)
    message = "seed 39101:.*" + (
        "unfinished" if state == "running" else "expected one|completed attempt records"
    )
    with pytest.raises(ValueError, match=message):
        opponent_memory.training_jobs("HEAD", trunks=str(tmp_path))


@pytest.mark.parametrize("retried", [False, True])
def test_pilot_branches_from_three_published_ema_trunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory_trunk: TrainingRun,
    retried: bool,
) -> None:

    storage = DirectoryStorage(tmp_path / "pilot")
    runs: list[TrainingRunSummary] = []
    for seed in opponent_memory.SEEDS:
        name = f"trunk-{seed}"
        parent = memory_trunk.model_copy(deep=True)
        parent.seed = seed
        parent.stages[0].id = "self-play"
        folder = f"{run_prefix(name)}attempts/one/training/"
        storage.put(folder + "run.json", parent.model_dump_json().encode())
        artifact = Path(parent.stages[0].artifacts["ema"]["path"])
        storage.put_file(folder + artifact.name, artifact)
        if retried:
            failed = parent.model_copy(deep=True)
            failed.status = "failed"
            # No weight export exists for this failed attempt.
            storage.put(
                f"{run_prefix(name)}attempts/failed/training/run.json",
                failed.model_dump_json().encode(),
            )
        runs.append(
            TrainingRunSummary(
                name=name,
                experiment="opponent_memory",
                job="decision-memory-trunk",
                launch="decision-memory-trunk",
                line=name,
                state="finished",
                checkpoints=Checkpoints(),
                seed=seed,
            )
        )

    def observe(source: object, experiment: str) -> ExperimentState:
        return ExperimentState(experiment=experiment, runs=tuple(runs))

    monkeypatch.setattr(opponent_memory, "observe", observe)
    requests = opponent_memory.training_jobs("HEAD", trunks=str(storage.root))
    assert len(requests) == 6
    for seed in opponent_memory.SEEDS:
        specs = [
            TrainingJob.model_validate_json(request.job.config or "")
            for request in requests
            if request.seed == seed
        ]
        assert len(specs) == 2
        assert specs[0].pretrained == specs[1].pretrained
        assert specs[0].pretrained.source_seed == seed
        assert specs[0].pretrained.source_run.key.endswith(
            "attempts/one/training/run.json"
        )
        assert specs[0].pretrained.checkpoint.key.endswith(
            f"attempts/one/training/{artifact.name}"
        )
        for spec in specs:
            stage = spec.regime.stages[0]
            assert isinstance(stage, TrainSelfPlay)
            assert stage.target_learner_transitions == 4_400_000
            assert spec.milestones == (20, 40, 60)
        assert specs[1].regime.agent.decision_memory_gate_scale == 2.0
    runs.pop()
    with pytest.raises(ValueError, match="seed 39103: expected one trunk"):
        opponent_memory.training_jobs("HEAD", trunks=str(storage.root))


@pytest.mark.parametrize("completed", [0, 2])
def test_pilot_refuses_no_or_multiple_completed_trunk_attempts(
    tmp_path: Path,
    memory_trunk: TrainingRun,
    completed: int,
) -> None:
    storage = DirectoryStorage(tmp_path / "pilot")
    parent = memory_trunk.model_copy(deep=True)
    parent.seed = 39101
    parent.stages[0].id = "self-play"
    for attempt in range(2):
        parent.status = "completed" if attempt < completed else "failed"
        storage.put(
            f"{run_prefix('trunk')}attempts/{attempt}/training/run.json",
            parent.model_dump_json().encode(),
        )
    with pytest.raises(
        ValueError, match=f"seed 39101: trunk has {completed} completed attempt records"
    ):
        opponent_memory._published_trunk(storage, "trunk", 39101)
