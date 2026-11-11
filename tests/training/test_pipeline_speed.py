"""Speed claims require complete measured populations and matched behavior."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Literal, cast

from pydantic import JsonValue
import pytest
import torch

from manabot._process import ProcessOutcome
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.files import atomic_json
from manabot.infra.host import HostFacts
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.training.records import PhaseTiming, StageRecord, TrainingRun
from manabot.training.regime import TrainSelfPlay
from manabot.training.update_metrics import MetricRow
import scripts.bench_pipeline as pipeline
from scripts.bench_pipeline import (
    EVALUATION_MATCHUP,
    INFERENCE,
    Attempt,
    Host,
    Report,
    Sample,
    Source,
    SuiteConfig,
    admit_reference,
    compare,
    describe,
    evaluation_behavior,
    regime_for,
    run_suite,
    training_reading,
    validate_config,
)


def _source() -> Source:
    return Source(commit="a" * 40, dirty=False, diff_sha256="b" * 64)


def _sample(
    seconds: float = 10,
    *,
    workload: str = "training",
    size: str = "w64",
    device: str = "cpu",
) -> Sample:
    return Sample(
        workload=workload,
        size=size,
        source=_source(),
        host=Host(
            machine_sha256="host",
            platform="test",
            python_version="3.13.0",
            torch_version="test",
            device_name="cpu",
            facts=HostFacts(
                cpu_model="test",
                logical_cpus=1,
                schedulable_cpus=1,
                cpu_quota=None,
            ),
        ),
        device=device,
        world="test-world",
        input={"policy_input_version": 6},
        workload_sha256="work",
        behavior_sha256="behavior",
        seconds=seconds,
        units=100,
        unit="learner decisions",
        metrics={},
        phases={"inference": seconds / 2},
        evidence={"run": "training.json"},
    )


def _attempt(
    side: Literal["candidate", "reference"],
    repetition: int,
    sample: Sample | None = None,
) -> Attempt:
    sample = sample or _sample()
    return Attempt(
        side=side,
        repetition=repetition,
        workload=sample.workload,
        size=sample.size,
        directory=f"{side}/{repetition}",
        sample=sample,
    )


def _baseline(config: SuiteConfig | None = None) -> Report:
    config = config or SuiteConfig(sizes=("w64",), workloads=("training",))
    return Report(
        config=config,
        source=_source(),
        status="completed",
        attempts=[
            _attempt(
                "candidate",
                0,
                _sample(workload=workload, size=size, device=config.device),
            )
            for size in config.sizes
            for workload in config.workloads
        ],
    )


def _rows() -> list[MetricRow]:
    return [
        {
            "time/collection": 100,
            "time/learning": 100,
            "time/collection/policy_inference_and_action_selection": 90,
            "coordinates/learner_transitions": 50,
        },
        {
            "time/collection": 2,
            "time/learning": 3,
            "time/collection/policy_inference_and_action_selection": 1,
            "coordinates/learner_transitions": 100,
        },
        {
            "time/collection": 4,
            "time/learning": 1,
            "time/collection/policy_inference_and_action_selection": 2,
            "coordinates/learner_transitions": 180,
        },
    ]


def _run() -> TrainingRun:
    # This projection reads measured fields, not recipe/admission identity.
    return TrainingRun.model_construct(
        status="completed",
        stages=[
            StageRecord(
                id="self-play",
                status="completed",
                updates=3,
                environment_decisions=200,
                idle_stream_steps=120,
                collection_timings={INFERENCE: PhaseTiming(seconds=93, calls=10)},
            )
        ],
        setup_seconds=1,
        seconds=212,
    )


def test_training_reading_excludes_warmup_without_erasing_idle_tail() -> None:
    reading = training_reading(_run(), _rows(), 32)
    metrics, phases = reading.metrics, reading.phases
    assert (reading.seconds, reading.units) == (10, 130)
    assert metrics["seconds_per_update"] == 5
    assert metrics["inference_calls_per_update_all_updates"] == 10 / 3
    assert metrics["mean_rows_per_call_all_updates"] == 20
    assert metrics["idle_stream_fraction_all_updates"] == 0.375
    assert phases["time/collection"] == 6
    assert phases["time/collection/policy_inference_and_action_selection"] == 3


def test_training_reading_rejects_missing_updates() -> None:
    with pytest.raises(ValueError, match="every update row"):
        training_reading(_run(), _rows()[1:], 32)


def test_comparison_selects_whole_minimum_samples() -> None:
    attempts = [
        _attempt("reference", n, _sample(s)) for n, s in enumerate((12, 10, 11))
    ]
    attempts += [_attempt("candidate", n, _sample(s)) for n, s in enumerate((6, 7, 8))]
    (result,) = compare(attempts, _baseline())
    assert result.status == "comparable"
    assert result.reference_sample == "reference/1"
    assert result.candidate_sample == "candidate/0"
    assert result.speedup == pytest.approx(10 / 6)
    assert attempts[3].sample is not None
    assert attempts[3].sample.phases == {"inference": 3}


@pytest.mark.parametrize(
    "field,value",
    [
        ("behavior_sha256", "different"),
        ("workload_sha256", "different"),
        ("world", "different"),
        ("device", "cuda"),
        ("input", {"policy_input_version": 7}),
    ],
)
def test_comparison_rejects_mismatched_evidence(field: str, value: JsonValue) -> None:
    attempts = [
        _attempt(side, n) for n in range(3) for side in ("reference", "candidate")
    ]
    assert attempts[-1].sample is not None
    changed = attempts[-1].sample.model_copy(update={field: value})
    attempts[-1] = attempts[-1].model_copy(update={"sample": changed})
    (result,) = compare(attempts, _baseline())
    assert result.status == "incomparable"
    assert result.reason == f"{field} mismatch"
    assert result.speedup is None


@pytest.mark.parametrize(
    "field,value",
    [("machine_sha256", "another"), ("python_version", "3.12.0")],
)
def test_comparison_rejects_different_host_even_with_same_cpu_label(
    field: str, value: str
) -> None:
    attempts = [
        _attempt(side, n) for n in range(3) for side in ("reference", "candidate")
    ]
    sample = attempts[-1].sample
    assert sample is not None
    changed = sample.model_copy(
        update={"host": sample.host.model_copy(update={field: value})}
    )
    attempts[-1] = attempts[-1].model_copy(update={"sample": changed})
    assert compare(attempts, _baseline())[0].reason == "host mismatch"


def test_missing_or_failed_samples_cannot_establish_speedup() -> None:
    attempts = [_attempt("candidate", n) for n in range(3)]
    assert compare(attempts, _baseline())[0].status == "incomparable"
    attempts += [_attempt("reference", n) for n in range(3)]
    attempts[-1] = attempts[-1].model_copy(update={"error": "timeout"})
    assert compare(attempts, _baseline())[0].speedup is None


def test_behavior_ignores_wall_clock_but_checks_commands() -> None:
    game: dict[str, JsonValue] = {
        "deal_seed": 101,
        "leg": 0,
        "winner": 1,
        "terminated": True,
        "truncated": False,
        "termination_reason": "normal",
        "execution_seconds": 20,
        "decisions": [
            {
                "actor": 0,
                "command": {"offer_id": 2},
                "pre_state_digest": "pre",
                "post_state_digest": "post",
                "latency_seconds": 1,
            }
        ],
    }
    later = deepcopy(game)
    later["execution_seconds"] = 40
    decisions = cast(list[dict[str, JsonValue]], later["decisions"])
    decisions[0]["latency_seconds"] = 2
    assert evaluation_behavior([game]) == evaluation_behavior([later])
    decisions[0]["command"] = {"offer_id": 3}
    assert evaluation_behavior([game]) != evaluation_behavior([later])


def test_weight_identity_survives_checkpoint_reload(tmp_path: Path) -> None:
    regime = regime_for(SuiteConfig(), "w64")
    observation = ObservationSpace(regime.observation)
    agent = ActorCritic(observation, regime.agent)
    digest = pipeline._weights_digest(agent)
    path = tmp_path / "policy.pt"
    save_checkpoint(
        agent,
        observation,
        path,
        player_configs=Match(EVALUATION_MATCHUP.match()).to_rust(),
    )
    restored, _ = load_checkpoint(str(path))
    assert pipeline._weights_digest(restored) == digest
    with torch.no_grad():
        next(restored.parameters()).add_(1)
    assert pipeline._weights_digest(restored) != digest


def test_suite_regime_keeps_ladder_and_learning_contract() -> None:
    regime = regime_for(SuiteConfig(), "a384l8")
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    assert regime.id == "pipeline-speed-a384l8-v1"
    assert regime.agent.hidden_dim == 384
    assert regime.agent.attention_feedforward_dim == 1536
    assert regime.observation.policy_input_version == 6
    assert stage.learning_seats == "both"
    assert stage.batch_closure == "complete-games"
    assert stage.matchup_curriculum == "elemental-uniform"
    assert stage.execution.precision == "float32"
    assert stage.streams == 256


def test_mps_is_explicitly_training_only() -> None:
    with pytest.raises(ValueError, match="evaluation does not support mps"):
        validate_config(SuiteConfig(device="mps"))
    validate_config(SuiteConfig(device="mps", workloads=("training",)))


def test_failed_reference_admission_keeps_receipt_and_baseline(tmp_path: Path) -> None:
    target = tmp_path / "baseline.json"
    target.write_text("previous reference")
    report = run_suite(
        SuiteConfig(),
        tmp_path / "reading",
        seconds=10,
        hourly_price=None,
        baseline_path=tmp_path / "missing.json",
        reference_root=None,
        record_reference=target,
    )
    assert report.status == "incomplete"
    assert report.attempts == []
    assert (tmp_path / "reading/report.json").is_file()
    assert target.read_text() == "previous reference"


@pytest.mark.parametrize("mode", ["record", "compare"])
@pytest.mark.parametrize(
    "config",
    [
        SuiteConfig(sizes=("w64",)),
        SuiteConfig(workloads=("training",)),
        SuiteConfig(device="cuda", sizes=("w64",)),
        SuiteConfig(device="cuda", workloads=("training",)),
        SuiteConfig(device="mps", sizes=("w64",), workloads=("training",)),
    ],
)
def test_subset_reference_rejected_before_workers_with_retained_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    config: SuiteConfig,
) -> None:
    baseline = _baseline(config)
    target = tmp_path / "baseline.json"
    retained = baseline.model_dump_json()
    target.write_text(retained)
    monkeypatch.setattr(pipeline, "source_identity", lambda root: _source())
    monkeypatch.setattr(pipeline, "host_identity", lambda device: _sample().host)

    def unexpected_worker(*args: object, **kwargs: object) -> ProcessOutcome:
        pytest.fail("reference admission must finish before starting a worker")

    monkeypatch.setattr(pipeline, "bounded_call", unexpected_worker)
    report = run_suite(
        config,
        tmp_path / "reading",
        seconds=30,
        hourly_price=None,
        baseline_path=target if mode == "compare" else None,
        reference_root=None,
        record_reference=target if mode == "record" else None,
    )
    assert report.status == "incomplete"
    assert report.error == (
        "ValueError: a reference requires the full ladder and every supported workload"
    )
    assert report.attempts == []
    assert (
        Report.model_validate_json((tmp_path / "reading/report.json").read_bytes())
        == report
    )
    assert target.read_text() == retained


@pytest.mark.parametrize(
    "config",
    [
        SuiteConfig(),
        SuiteConfig(device="cuda"),
        SuiteConfig(device="mps", workloads=("training",)),
    ],
)
def test_reference_admits_full_device_ladder(config: SuiteConfig) -> None:
    admit_reference(_baseline(config), _sample().host)


def test_suite_interleaves_sources_under_one_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    baseline = _baseline(SuiteConfig())
    reference = tmp_path / "reference"
    (reference / "scripts").mkdir(parents=True)
    (reference / "scripts/bench_pipeline.py").touch()
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(baseline.model_dump_json())
    monkeypatch.setattr(pipeline, "source_identity", lambda root: _source())
    monkeypatch.setattr(pipeline, "host_identity", lambda device: _sample().host)
    deadlines: list[float] = []
    order: list[str] = []

    def measured(
        target: object,
        args: tuple[Path, Path, float],
        *,
        deadline_unix: float,
        process_tree_bytes: int,
    ) -> ProcessOutcome:
        root, directory, worker_deadline = args
        order.append("reference" if root == reference else "candidate")
        deadlines.append(deadline_unix)
        assert worker_deadline < deadline_unix
        request = SuiteConfig.model_validate_json(
            (directory / "request.json").read_bytes()
        )
        atomic_json(
            directory / "sample.json",
            _sample(
                10 if root == reference else 5,
                workload=request.workloads[0],
                size=request.sizes[0],
                device=request.device,
            ).model_dump(mode="json"),
        )
        return ProcessOutcome({"returncode": 0}, None, 0.1, 0.01, 100, 0, True)

    monkeypatch.setattr(pipeline, "bounded_call", measured)
    report = run_suite(
        baseline.config,
        tmp_path / "reading",
        seconds=30,
        hourly_price=0.5,
        baseline_path=baseline_path,
        reference_root=reference,
        record_reference=None,
    )
    assert report.status == "completed", report.error
    assert order == (
        ["reference", "candidate"] * 8
        + ["candidate", "reference"] * 8
        + ["reference", "candidate"] * 8
    )
    assert len(set(deadlines)) == 1
    assert len(report.comparisons) == 8
    assert all(result.speedup == 2 for result in report.comparisons)
    assert baseline_path.read_text() == baseline.model_dump_json()


def test_timeout_retains_failure_and_stops_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    monkeypatch.setattr(pipeline, "source_identity", lambda root: _source())
    monkeypatch.setattr(pipeline, "host_identity", lambda device: _sample().host)

    def timeout(
        target: object,
        args: tuple[object, ...],
        *,
        deadline_unix: float,
        process_tree_bytes: int,
    ) -> ProcessOutcome:
        return ProcessOutcome(None, "TimeoutError: deadline", 1, 0.01, 100, -9, True)

    monkeypatch.setattr(pipeline, "bounded_call", timeout)
    target = tmp_path / "baseline.json"
    target.write_text("previous")
    report = run_suite(
        SuiteConfig(),
        tmp_path / "reading",
        seconds=30,
        hourly_price=None,
        baseline_path=None,
        reference_root=None,
        record_reference=target,
    )
    assert report.status == "incomplete"
    assert len(report.attempts) == 1
    assert report.attempts[0].sample is None
    assert report.attempts[0].error == "TimeoutError: deadline"
    assert target.read_text() == "previous"


def test_evaluation_panel_admits_actual_elemental_deck_keys() -> None:
    match = EVALUATION_MATCHUP.match()
    assert sum(match.hero_deck.values()) == 40
    assert sum(match.villain_deck.values()) == 40
    assert match.hero_deck != match.villain_deck


def test_reference_rejects_missing_or_mixed_source_samples() -> None:
    baseline = _baseline(SuiteConfig())
    host = _sample().host
    with pytest.raises(ValueError, match="complete"):
        admit_reference(baseline.model_copy(update={"attempts": []}), host)
    sample = _sample().model_copy(
        update={"source": _source().model_copy(update={"commit": "changed"})}
    )
    attempt = _attempt("candidate", 0, sample)
    with pytest.raises(ValueError, match="mixed-source"):
        admit_reference(
            baseline.model_copy(update={"attempts": [attempt, *baseline.attempts[1:]]}),
            host,
        )


def test_supplied_price_converts_rates_without_printing_invalid_samples(
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = _baseline().model_copy(update={"hourly_price": 0.5})
    describe(baseline)
    row = json.loads(capsys.readouterr().out.splitlines()[0])
    assert row["per_dollar_at_supplied_price"] == 100 / 10 * 3600 / 0.5
    bad = baseline.attempts[0].model_copy(update={"error": "source changed"})
    describe(baseline.model_copy(update={"attempts": [bad], "status": "incomplete"}))
    output = capsys.readouterr().out
    assert "source changed" in output
    assert "per_second" not in output
