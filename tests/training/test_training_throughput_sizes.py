"""Step 3's size variants and launch legs (Task b015d5)."""

import hashlib
from pathlib import Path

from etude.launch.job_file import read_job_file
import pytest
from torch import nn

from experiments import training_throughput_sizes as sizes
from experiments.elemental_variants import elemental_recipe
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.experiment.job import TrainingJob
from manabot.infra.identity import canonical_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint
from manabot.training.records import StageRecord, TrainingRun
from manabot.training.regime import TrainingRegime
from scripts.check_throughput_admission import check
from tests.training.test_training_throughput import _worker

ROOT = Path(__file__).parents[2]
SIZE_FIELDS = {
    "hidden_dim",
    "attention_layers",
    "num_attention_heads",
    "attention_feedforward_dim",
}


def _compute_parameters(regime: TrainingRegime) -> int:
    """Parameters outside the embedding lookup tables, as the protocol counts them."""
    agent = ActorCritic(ObservationSpace(regime.observation), regime.agent)
    lookups = sum(
        m.weight.numel() for m in agent.modules() if isinstance(m, nn.Embedding)
    )
    return sum(p.numel() for p in agent.parameters()) - lookups


def _without_size(regime: TrainingRegime) -> dict[str, object]:
    return regime.model_dump(mode="json", exclude={"id": True, "agent": SIZE_FIELDS})


@pytest.mark.parametrize(
    ("size", "parameters"),
    [
        ("w64", 243_842),
        ("w128l4", 1_334_786),
        ("w160l8", 4_953_602),
        ("a384l8", 18_868_226),
    ],
)
def test_sizes_change_only_the_network_and_count_as_the_protocol(
    size: sizes.Size, parameters: int
) -> None:
    variant = sizes.recipe(size, "float32", sizes.SPEED_UPDATES)
    control = elemental_recipe("control", sizes.SPEED_UPDATES)

    assert _without_size(variant) == _without_size(control)
    assert _compute_parameters(variant) == parameters
    shape = sizes.SIZES[size]
    agent = variant.agent
    assert (agent.hidden_dim, agent.attention_layers, agent.num_attention_heads) == (
        shape.width,
        shape.layers,
        shape.heads,
    )


def test_w64_float32_keeps_the_control_bytes_and_precisions_differ_only_there() -> None:
    control = elemental_recipe("control", sizes.SPEED_UPDATES)
    assert sizes.recipe("w64", "float32", 4).model_dump(
        exclude={"id"}
    ) == control.model_dump(exclude={"id"})

    bf16 = sizes.recipe("w160l8", "bf16", 4).model_dump(mode="json")
    fp32 = sizes.recipe("w160l8", "float32", 4).model_dump(mode="json")
    assert bf16["stages"][0]["execution"]["precision"] == "bf16"
    for declared in (bf16, fp32):
        declared["id"] = None
        declared["stages"][0]["execution"]["precision"] = None
    assert bf16 == fp32


def _launched(*arguments: str) -> list[sizes.Run]:
    return read_job_file(
        ROOT / "experiments/training_throughput_sizes.py", list(arguments)
    )


def test_launch_file_declares_every_cell() -> None:
    found = {leg: _launched("--revision", "HEAD", "--leg", leg) for leg in sizes.LEGS}

    assert len(found) == 42
    for name, runs in found.items():
        leg = sizes.LEGS[name]
        assert [run.seed for run in runs] == list(sizes.SEEDS[: leg.packed])
        job = TrainingJob.model_validate_json(runs[0].job.config or "")
        stage = job.regime.stages[0]
        assert stage.execution.device == "cuda"
        assert stage.execution.workers == 1
        assert stage.streams == 256
        assert stage.learning_seats == "both"
        assert sizes.GPU_SAMPLER in runs[0].job.command
    checks = {name for name, leg in sizes.LEGS.items() if leg.updates == 12}
    assert checks == {
        "fast-w160l8-k1-float32",
        "fast-w160l8-k1-float32-repeat",
        "fast-w160l8-k1-tf32",
        "fast-w160l8-k1-bf16",
    }
    assert not any(name.startswith(("a4000-w64", "l4-w64")) for name in sizes.LEGS)


def test_the_repeat_shares_regime_bytes_and_is_its_own_job() -> None:
    first, repeat = (
        sizes.runs("HEAD", leg)[0]
        for leg in ("fast-w160l8-k1-float32", "fast-w160l8-k1-float32-repeat")
    )

    def regime(run: sizes.Run) -> str:
        job = TrainingJob.model_validate_json(run.job.config or "")
        return canonical_sha256(job.regime.model_dump(mode="json"))

    assert regime(first) == regime(repeat)
    assert first.job.environment != repeat.job.environment


def test_limits_follow_the_pace_and_cover_setup_at_the_ceiling() -> None:
    small = sizes.limits(sizes.LEGS["a4000-w128l4-k1-float32"], 37)
    assert (small.watchdog_seconds, small.max_hours, small.stall_seconds) == (
        1500,
        0.75,
        600,
    )

    large = sizes.limits(sizes.LEGS["l4-a384l8-k2-float32"], 350)
    assert large.watchdog_seconds >= 6 * 350 * 4
    assert large.stall_seconds == 3 * 350
    assert large.max_usd >= (large.max_hours + 0.25) * 0.60

    run = sizes.runs("HEAD", "l4-a384l8-k2-float32", pace="350")[0]
    frozen = TrainingJob.model_validate_json(run.job.config or "")
    assert frozen.regime.wall_seconds == large.watchdog_seconds
    assert run.job.max_hours == large.max_hours


def test_unknown_legs_and_seeds_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown leg"):
        sizes.runs("HEAD", "a4000-w64-k1-float32")
    with pytest.raises(ValueError, match="declares no seeds"):
        sizes.runs("HEAD", "a4000-w128l4-k2-float32", seeds="26303")
    with pytest.raises(ValueError, match="pace"):
        sizes.limits(sizes.LEGS["l4-w128l4-k1-float32"], 0)


@pytest.mark.training
@pytest.mark.parametrize("leg", ["a4000-w128l4-k1-float32", "a4000-a384l8-k1-float32"])
def test_a_size_cell_trains_through_the_real_worker_on_cpu(
    tmp_path: Path, leg: str
) -> None:
    """The leg's regime at 8 streams and 1 update, moved to CPU."""
    chosen = sizes.LEGS[leg]
    regime = sizes.recipe(chosen.size, chosen.precision, 1)
    regime.wall_seconds = 600
    stage = regime.stages[0]
    stage.streams = 8
    stage.execution.device = "cpu"
    stage.execution.wall_seconds = 600
    spec = TrainingJob(regime=regime, experiment=sizes.EXPERIMENT)

    process, training = _worker(tmp_path, spec, "cell", 26301)

    assert process.wait() == 0, (tmp_path / "cell" / "log.txt").read_text()[-4000:]
    receipt = TrainingRun.model_validate_json((training / "run.json").read_bytes())
    assert receipt.status == "completed"
    assert receipt.stages[0].updates == 1


def test_c1_discovery_preserves_controls_and_declares_endpoints() -> None:
    for name, leg in sizes.C1_LEGS.items():
        runs = _launched("--revision", "HEAD", "--leg", name)
        seeds = (26400,) if leg.admission else sizes.C1_SEEDS[: leg.packed]
        assert tuple(r.seed for r in runs) == seeds
        spec = TrainingJob.model_validate_json(runs[0].job.config or "")
        assert spec.experiment == sizes.C1_EXPERIMENT
        actual = spec.regime.model_dump(mode="json")
        expected = sizes.recipe(leg.size, leg.precision, leg.updates).model_dump(
            mode="json"
        )
        assert (
            actual.pop("id")
            == f"elemental-throughput-c1-{leg.size}-{leg.precision}-u{leg.updates}-v1"
        )
        expected.pop("id")
        for regime in (actual, expected):
            regime.pop("wall_seconds", None)
            regime["stages"][0]["execution"].pop("wall_seconds", None)
        assert actual == expected
        assert sizes.limits(leg, 37).watchdog_seconds >= 6 * 37 * leg.updates
        if leg.admission:
            assert sizes.CUDA_ADMISSION.rstrip() in runs[0].job.command
            assert "|| exit $?" in runs[0].job.command
            assert runs[0].job.command.endswith("scripts/check_throughput_admission.py")
            assert leg.updates == 2
        elif leg.overlap_retry or (leg.gpu, leg.size, leg.packed) == (
            "fast",
            "w160l8",
            1,
        ):
            assert leg.updates == 12
        else:
            assert leg.updates == 8


@pytest.mark.parametrize("pace", [float("nan"), float("inf"), -1])
def test_c1_refuses_nonfinite_or_negative_limits(pace: float) -> None:
    with pytest.raises(ValueError, match="pace"):
        sizes.limits(sizes.C1_LEGS["c1-fast-a384l8-k1-float32-u8"], pace)


def test_admission_reloads_receipt_artifact_and_rejects_unhealthy_or_changed_bytes(
    tmp_path: Path,
) -> None:
    """A synthetic CUDA receipt exercises admission plumbing, not CUDA execution."""
    regime = sizes.leg_recipe(sizes.C1_LEGS["c1-fast-w160l8-k1-bf16-u2-admission"])
    regime.agent.hidden_dim = 8
    regime.agent.num_attention_heads = 2
    regime.agent.attention_layers = 1
    regime.agent.attention_feedforward_dim = 32
    space = ObservationSpace(regime.observation)
    agent = ActorCritic(space, regime.agent)
    checkpoint = tmp_path / "training/raw.pt"
    save_checkpoint(
        agent, space, checkpoint, player_configs=Match(regime.match).to_rust()
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    stage = StageRecord(
        id=regime.stages[0].id,
        actual_device="cuda",
        updates=2,
        optimizer_updates=2,
        artifacts={
            "raw": {
                "path": "/rented/worker/raw.pt",
                "sha256": digest,
                "bytes": checkpoint.stat().st_size,
            }
        },
    )
    receipt = TrainingRun(
        id="admission-fixture",
        seed=26400,
        seed_streams={},
        identities={},
        regime=regime,
        regime_digest=canonical_sha256(regime.model_dump(mode="json")),
        status="completed",
        stages=[stage],
    )
    receipt_path = tmp_path / "training/run.json"
    receipt_path.write_text(receipt.model_dump_json())
    assert check(tmp_path)["checkpoint_sha256"] == digest

    for updates in (None, 0):
        stage.optimizer_updates = updates
        receipt_path.write_text(receipt.model_dump_json())
        with pytest.raises(ValueError, match="healthy CUDA BF16"):
            check(tmp_path)
    stage.optimizer_updates = 2
    receipt_path.write_text(receipt.model_dump_json())
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="artifact bytes differ"):
        check(tmp_path)
