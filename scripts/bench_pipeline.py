"""Standing pipeline measurements and same-host comparisons.

Workload runners project ordinary training and evaluation receipts into Sample.
The coordinator owns one bounded child tree at a time, retains every attempt,
and compares whole samples only after workload, host and behavior admission.
Training execution is injected from bench_training_update's existing executor.
"""

import ast
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import inspect
from itertools import product
import json
import math
from multiprocessing.connection import Connection
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Literal

from pydantic import Field, JsonValue
import torch

from experiments.training_throughput_sizes import SIZES, Size
from manabot._process import bounded_call
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.eval.evaluation import evaluate
from manabot.eval.local import checkpoint_request
from manabot.infra.files import atomic_json
from manabot.infra.host import HostFacts, host_facts
from manabot.infra.identity import StrictModel, canonical_sha256, file_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.sim.execution import EvalExecution
from manabot.sim.matchup import AuthoredMatchup
from manabot.sim.replay import read_trace
from manabot.training.execution import validate_regime
from manabot.training.records import TrainingRun
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.training.update_metrics import MetricRow, read_updates, stage_rows
import managym

ROOT = Path(__file__).resolve().parents[1]
INFERENCE = "Policy inference and action selection"
REPETITIONS = 3
# Source admission reads this literal without importing the other checkout.
type SuiteInterface = Literal[1]
SUITE_INTERFACE: SuiteInterface = 1
EVALUATION_MATCHUP = AuthoredMatchup(
    pack_key="elemental-world-v2", role_a="elemental_bg", role_b="elemental_gr"
)
# A callback keeps the script's single-recipe and suite paths on one executor.
type Execute = Callable[[TrainingRegime, int, Path | None, Path | None], TrainingRun]


class SuiteConfig(StrictModel):
    version: Literal[1] = 1
    sizes: tuple[Size, ...] = tuple(SIZES)
    workloads: tuple[str, ...] = ("training", "evaluation")
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    streams: int = Field(default=256, gt=0)
    updates: int = Field(default=3, ge=2)
    seed: int = 25201
    deal_seeds: tuple[int, ...] = (101,)
    process_tree_bytes: int = Field(default=16 * 2**30, gt=0)


class Source(StrictModel):
    commit: str
    dirty: bool
    diff_sha256: str


class Reference(StrictModel):
    """Reviewed source and worker protocol, independent of any measurement host."""

    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    suite_interface: SuiteInterface


class Host(StrictModel):
    machine_sha256: str
    platform: str
    python_version: str
    torch_version: str
    device_name: str
    facts: HostFacts


class Sample(StrictModel):
    workload: str
    size: str
    source: Source
    host: Host
    device: str
    precision: Literal["float32"] = "float32"
    world: str
    input: dict[str, JsonValue]
    workload_sha256: str
    behavior_sha256: str
    # The measured interval and population define rate; never sum child clocks.
    seconds: float = Field(gt=0, allow_inf_nan=False)
    units: int = Field(gt=0)
    unit: str
    metrics: dict[str, float]
    phases: dict[str, float]
    evidence: dict[str, str]

    @property
    def rate(self) -> float:
        return self.units / self.seconds


class Attempt(StrictModel, frozen=False):
    """Progress of one worker; its admitted Sample remains immutable."""

    side: Literal["reference", "candidate"]
    repetition: int
    workload: str
    size: str
    directory: str
    sample: Sample | None = None
    error: str | None = None
    elapsed_seconds: float = 0
    cleanup_seconds: float = 0


class Comparison(StrictModel):
    workload: str
    size: str
    status: Literal["comparable", "incomparable"]
    reason: str | None = None
    reference_sample: str | None = None
    candidate_sample: str | None = None
    speedup: float | None = None


class Report(StrictModel, frozen=False):
    """Coordinator-owned progress, persisted after each attempt and at exit."""

    version: Literal[1] = 1
    config: SuiteConfig
    source: Source
    reference: Reference | None = None
    status: Literal["running", "completed", "incomplete"] = "running"
    elapsed_seconds: float = 0
    hourly_price: float | None = None
    attempts: list[Attempt] = []
    comparisons: list[Comparison] = []
    error: str | None = None


def source_identity(root: Path) -> Source:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *args])

    changes = git("status", "--porcelain", "--untracked-files=normal")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    files = {
        name: file_sha256(root / name)
        for name in untracked.decode().split("\0")
        if name and (root / name).is_file()
    }
    return Source(
        commit=git("rev-parse", "HEAD").decode().strip(),
        dirty=bool(changes),
        diff_sha256=hashlib.sha256(
            git("diff", "HEAD", "--binary") + changes + canonical_sha256(files).encode()
        ).hexdigest(),
    )


def host_identity(device: str, facts: HostFacts | None = None) -> Host:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; no device substitution")
        properties = torch.cuda.get_device_properties(0)
        device_name = str(properties)
    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable; no device substitution")
        device_name = "Apple MPS"
    else:
        device_name = platform.processor() or platform.machine()
    machine = Path("/etc/machine-id")
    identity = machine.read_bytes() if machine.exists() else platform.node().encode()
    if not identity.strip():
        raise ValueError("host identity is unavailable")
    return Host(
        machine_sha256=hashlib.sha256(identity).hexdigest(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        device_name=device_name,
        facts=facts or host_facts(),
    )


def regime_for(config: SuiteConfig, size: Size) -> TrainingRegime:
    """Own benchmark identity, fixed v6 input, unchanged Elemental learning rule."""
    regime = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes/elemental-long-183-v1.json").read_bytes()
    )
    regime.id = f"pipeline-speed-{size}-v1"
    regime.world = managym.WORLD_VERSION
    regime.observation.policy_input_version = 6
    regime.observation.max_facts = 512
    shape = SIZES[size]
    regime.agent.hidden_dim = shape.width
    regime.agent.attention_layers = shape.layers
    regime.agent.num_attention_heads = shape.heads
    regime.agent.attention_feedforward_dim = shape.feedforward
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = config.updates
    stage.streams = config.streams
    stage.execution.device = config.device
    stage.execution.precision = "float32"
    stage.execution.wall_seconds = regime.wall_seconds = 3600
    stage.execution.memory_bytes = config.process_tree_bytes
    return validate_regime(regime)


def _number(row: MetricRow, key: str) -> float:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {key} is not numeric")
    return float(value)


@dataclass(frozen=True)
class TrainingReading:
    seconds: float
    units: int
    metrics: dict[str, float]
    phases: dict[str, float]


def training_reading(
    run: TrainingRun, rows: list[MetricRow], streams: int
) -> TrainingReading:
    """Exclude update one from timing; batching counters cover the whole stage."""
    record = run.stages[0]
    if run.status != "completed" or record.updates < 2 or len(rows) != record.updates:
        raise ValueError("training needs every update row and a post-warmup update")
    if record.numerical_failure is not None:
        raise ValueError("training rejected a numerical update")
    if run.game_errors is not None and (
        run.game_errors.crashed or run.game_errors.discarded_samples
    ):
        raise ValueError(
            "training discarded failed games; not a complete speed reading"
        )
    phases = {
        key: sum(_number(row, key) for row in rows[1:])
        for key in rows[-1]
        if key.startswith("time/")
    }
    # Subphase clocks overlap their parents; only these two totals form the rate.
    seconds = phases["time/collection"] + phases["time/learning"]
    units = int(
        _number(rows[-1], "coordinates/learner_transitions")
        - _number(rows[0], "coordinates/learner_transitions")
    )
    timing = (record.collection_timings or {})[INFERENCE]
    if timing.calls <= 0 or units <= 0 or seconds <= 0:
        raise ValueError("empty training measurement")
    return TrainingReading(
        seconds,
        units,
        {
            "seconds_per_update": seconds / (record.updates - 1),
            "measured_updates": float(record.updates - 1),
            "warmup_updates": 1.0,
            "inference_calls_all_updates": float(timing.calls),
            "inference_calls_per_update_all_updates": timing.calls / record.updates,
            "mean_rows_per_call_all_updates": record.environment_decisions
            / timing.calls,
            "idle_stream_fraction_all_updates": record.idle_stream_steps
            / (timing.calls * streams),
            "setup_seconds": run.setup_seconds,
            "export_seconds": record.export_seconds,
            "run_seconds": run.seconds,
        },
        phases,
    )


def _weights_digest(agent: ActorCritic) -> str:
    """Tensor content, not torch.save container bytes or source metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(agent.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(
            canonical_sha256([name, str(value.dtype), list(value.shape)]).encode()
        )
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _train(
    config: SuiteConfig, size: Size, out: Path, deadline: float, execute: Execute
) -> Sample:
    regime = regime_for(config, size)
    workload = regime.model_dump(mode="json")
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.execution.wall_seconds = regime.wall_seconds = max(
        0.01, deadline - time.time()
    )
    run = execute(regime, config.seed, out / "training", None)
    atomic_json(out / "training.json", run.model_dump(mode="json"))
    rows = stage_rows(run, read_updates(out, [run]))[stage.id]
    atomic_json(out / "updates.json", rows)
    reading = training_reading(run, rows, config.streams)
    if run.selected_artifact is None or run.host is None:
        raise ValueError("training did not retain a policy and host facts")
    policy = Path(run.selected_artifact["path"])
    record = run.stages[0]
    behavior = {
        "weights": _weights_digest(load_checkpoint(str(policy))[0]),
        "environment_decisions": record.environment_decisions,
        "learner_transitions": record.learner_transitions,
        "games": record.games,
        "optimizer_updates": record.optimizer_updates,
    }
    return Sample(
        workload="training",
        size=size,
        source=source_identity(ROOT),
        host=host_identity(config.device, run.host),
        device=config.device,
        world=regime.world,
        input=regime.observation.model_dump(mode="json"),
        workload_sha256=canonical_sha256({"regime": workload, "seed": config.seed}),
        behavior_sha256=canonical_sha256(behavior),
        seconds=reading.seconds,
        units=reading.units,
        unit="learner decisions",
        metrics=reading.metrics,
        phases=reading.phases,
        evidence={
            "run": "training.json",
            "updates": "updates.json",
            "policy": str(policy),
        },
    )


def evaluation_behavior(games: list[dict[str, JsonValue]]) -> str:
    """Retained Commands and state transitions; omit scheduling and clocks."""
    projected: list[dict[str, JsonValue]] = []
    for game in games:
        decisions = game["decisions"]
        if not isinstance(decisions, list):
            raise ValueError("trace decisions must be a list")
        steps: list[JsonValue] = []
        for decision in decisions:
            if not isinstance(decision, dict):
                raise ValueError("trace decision must be an object")
            steps.append(
                {
                    key: decision[key]
                    for key in (
                        "actor",
                        "command",
                        "pre_state_digest",
                        "post_state_digest",
                    )
                }
            )
        projected.append(
            {
                **{
                    key: game[key]
                    for key in (
                        "deal_seed",
                        "leg",
                        "winner",
                        "terminated",
                        "truncated",
                        "termination_reason",
                    )
                },
                "decisions": steps,
            }
        )
    projected.sort(key=lambda row: (int(str(row["deal_seed"])), int(str(row["leg"]))))
    return canonical_sha256(projected)


def _evaluate(
    config: SuiteConfig, size: Size, out: Path, deadline: float, execute: Execute
) -> Sample:
    regime = regime_for(config, size)
    obs = ObservationSpace(regime.observation)
    matchup = EVALUATION_MATCHUP
    checkpoints: list[Path] = []
    identities: list[str] = []
    started = time.monotonic()
    for offset in range(2):
        torch.manual_seed(config.seed + offset)
        agent = ActorCritic(obs, regime.agent)
        path = out / f"initialized-{offset}.pt"
        save_checkpoint(
            agent, obs, path, player_configs=Match(matchup.match()).to_rust()
        )
        checkpoints.append(path)
        identities.append(_weights_digest(agent))
    # The two named Elemental roles are a fixed four-leg panel, not the whole
    # training curriculum. Admission resolves their actual deck lists.
    request = checkpoint_request(
        checkpoints[0],
        opponent=checkpoints[1],
        deal_seeds=config.deal_seeds,
        authored_matchup=matchup,
        game_seconds=300,
        max_commands=40_000,
    )
    # Sampling must depend on the actual policies, not serialization details of
    # their checkpoint containers. Full registrations still admit exact bytes.
    request = request.model_copy(
        update={
            "key": request.key.model_copy(
                update={
                    "anchor_cohort_sha256": canonical_sha256(
                        {"initialized_weights": identities}
                    ),
                }
            )
        }
    )
    limits = EvalExecution(device=config.device, deadline_unix=deadline)
    atomic_json(out / "evaluation-request.json", request.model_dump(mode="json"))
    atomic_json(out / "execution.json", limits.model_dump(mode="json"))
    setup = time.monotonic() - started
    result = evaluate(request, out_dir=out / "panel", execution=limits)
    if result.status != "completed" or not result.evaluation_seconds:
        raise ValueError(f"evaluation {result.status}: {result.error}")
    if any(row.truncated for row in result.rows):
        raise ValueError("evaluation reached an execution limit before game completion")
    trace = json.loads((out / "panel/trace.json").read_bytes())
    games = read_trace(out / "panel" / trace["artifact_path"])
    replay = json.loads((out / "panel/replay.json").read_bytes())
    if not replay["passed"]:
        raise ValueError("evaluation replay failed")
    workload = {
        "agent": regime.agent.model_dump(mode="json"),
        "observation": regime.observation.model_dump(mode="json"),
        "match": request.authored_matchup.model_dump(mode="json"),
        "deals": config.deal_seeds,
        "seed": config.seed,
        "execution": limits.model_dump(mode="json", exclude={"deadline_unix"}),
        "game_seconds": request.game_seconds,
        "max_commands": request.max_commands,
    }
    return Sample(
        workload="evaluation",
        size=size,
        source=source_identity(ROOT),
        host=host_identity(config.device),
        device=config.device,
        world=regime.world,
        input=regime.observation.model_dump(mode="json"),
        workload_sha256=canonical_sha256(workload),
        behavior_sha256=canonical_sha256(
            {
                "initialized_weights": identities,
                "games": evaluation_behavior(games),
            }
        ),
        seconds=result.evaluation_seconds,
        units=len(result.rows),
        unit="games",
        metrics={
            "preparation_seconds": setup,
            "replay_seconds": float(replay["seconds"]),
        },
        phases={},
        evidence={
            "evaluation": "panel/evaluation.json",
            "performance": "panel/performance.json",
            "serving": "panel/allocation/serving.json",
            "replay": "panel/replay.json",
            "trace": str(out / "panel" / trace["artifact_path"]),
            **{f"checkpoint_{i}": file_sha256(p) for i, p in enumerate(checkpoints)},
        },
    )


@dataclass(frozen=True)
class Workload:
    run: Callable[[SuiteConfig, Size, Path, float, Execute], Sample]
    devices: tuple[str, ...]


# A new real workload adds one runner and registration, not another coordinator.
WORKLOADS = {
    "training": Workload(_train, ("cpu", "cuda", "mps")),
    "evaluation": Workload(_evaluate, ("cpu", "cuda")),
}


def validate_config(config: SuiteConfig) -> None:
    for values in (config.sizes, config.workloads, config.deal_seeds):
        if not values or len(set(values)) != len(values):
            raise ValueError("suite dimensions must be nonempty and unique")
    for name in config.workloads:
        if name not in WORKLOADS:
            raise ValueError(f"unknown workload {name}")
        if config.device not in WORKLOADS[name].devices:
            raise ValueError(
                f"{name} does not support {config.device}; no device substitution"
            )


def run_worker(request: Path, out: Path, deadline: float, execute: Execute) -> None:
    config = SuiteConfig.model_validate_json(request.read_bytes())
    validate_config(config)
    if len(config.sizes) != 1 or len(config.workloads) != 1:
        raise ValueError("worker requires one workload and size")
    torch.set_num_threads(1)
    if not Path(inspect.getfile(validate_regime)).resolve().is_relative_to(ROOT):
        raise ValueError("imported training library belongs to another checkout")
    if not Path(managym._managym.__file__).resolve().is_relative_to(ROOT):
        raise ValueError("native extension belongs to another checkout")
    host_identity(config.device)
    sample = WORKLOADS[config.workloads[0]].run(
        config,
        config.sizes[0],
        out,
        deadline,
        execute,
    )
    atomic_json(out / "sample.json", sample.model_dump(mode="json"))


def _command_worker(sender: Connection, root: Path, out: Path, deadline: float) -> None:
    """Run source in its own locked environment inside the supervisor's group."""
    with (out / "worker.log").open("w") as log:
        result = subprocess.run(
            [
                "uv",
                "run",
                "--locked",
                "--extra",
                "evaluation",
                "python",
                str(root / "scripts/bench_training_update.py"),
                "--suite-request",
                str(out / "request.json"),
                "--out",
                str(out),
                "--deadline",
                str(deadline),
            ],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    sender.send({"returncode": result.returncode})


def _measure_attempt(
    attempt: Attempt,
    config: SuiteConfig,
    root: Path,
    source: Source,
    deadline: float,
) -> None:
    """Fill one attempt from a bounded worker and admit its retained sample."""
    directory = Path(attempt.directory)
    directory.mkdir()
    request = config.model_copy(
        update={"sizes": (attempt.size,), "workloads": (attempt.workload,)}
    )
    atomic_json(directory / "request.json", request.model_dump(mode="json"))
    outcome = bounded_call(
        _command_worker,
        (root, directory, deadline - 2),
        deadline_unix=deadline,
        process_tree_bytes=config.process_tree_bytes,
    )
    attempt.elapsed_seconds = outcome.seconds
    attempt.cleanup_seconds = outcome.cleanup_seconds
    attempt.error = outcome.error
    if not outcome.cleanup_complete:
        attempt.error = "process cleanup incomplete"
    if not attempt.error and (outcome.payload or {}).get("returncode") != 0:
        attempt.error = "worker failed; see worker.log and native receipts"
    if attempt.error:
        return
    try:
        sample = Sample.model_validate_json((directory / "sample.json").read_bytes())
        if sample.source != source or source_identity(root) != source:
            raise ValueError("source changed during measurement")
        if (sample.workload, sample.size) != (attempt.workload, attempt.size):
            raise ValueError("worker returned a different workload")
        attempt.sample = sample
    except (ValueError, OSError) as exc:
        attempt.error = str(exc)


def _compatible(samples: list[Sample]) -> str | None:
    for field in (
        "host",
        "device",
        "precision",
        "world",
        "input",
        "workload_sha256",
        "behavior_sha256",
        "units",
        "unit",
    ):
        if any(
            getattr(sample, field) != getattr(samples[0], field)
            for sample in samples[1:]
        ):
            return f"{field} mismatch"
    return None


def compare(attempts: list[Attempt], config: SuiteConfig) -> list[Comparison]:
    """Admit only this job's fresh samples; historical rates never participate."""
    results: list[Comparison] = []
    for workload, size in product(config.workloads, config.sizes):
        group = [a for a in attempts if (a.workload, a.size) == (workload, size)]
        refs = [a for a in group if a.side == "reference"]
        candidates = [a for a in group if a.side == "candidate"]
        if (
            len(refs) != REPETITIONS
            or len(candidates) != REPETITIONS
            or any(a.error or a.sample is None for a in group)
        ):
            reason = "incomplete interleaved samples"
        else:
            reason = _compatible([a.sample for a in group if a.sample])
        if reason:
            results.append(
                Comparison(
                    workload=workload,
                    size=size,
                    status="incomparable",
                    reason=reason,
                )
            )
            continue
        reference = min(
            refs, key=lambda a: a.sample.seconds if a.sample else float("inf")
        )
        candidate = min(
            candidates, key=lambda a: a.sample.seconds if a.sample else float("inf")
        )
        assert reference.sample and candidate.sample
        results.append(
            Comparison(
                workload=workload,
                size=size,
                status="comparable",
                reference_sample=reference.directory,
                candidate_sample=candidate.directory,
                speedup=candidate.sample.rate / reference.sample.rate,
            )
        )
    return results


def _require_full_ladder(config: SuiteConfig) -> None:
    """References use the canonical ladder; subsets remain diagnostic readings."""
    if config.sizes != tuple(SIZES) or config.workloads != tuple(
        name
        for name, workload in WORKLOADS.items()
        if config.device in workload.devices
    ):
        raise ValueError(
            "a reference requires the full ladder and every supported workload"
        )


def reference_from_report(report: Report) -> Reference:
    """Propose source metadata only after a complete clean standalone reading."""
    validate_config(report.config)
    _require_full_ladder(report.config)
    expected = {(w, s) for w in report.config.workloads for s in report.config.sizes}
    actual = [(a.workload, a.size) for a in report.attempts]
    if (
        report.status != "completed"
        or report.error is not None
        or report.reference is not None
        or report.source.dirty
        or len(actual) != len(expected)
        or set(actual) != expected
    ):
        raise ValueError("reference must contain one complete clean-source ladder")
    first = report.attempts[0].sample
    for attempt in report.attempts:
        sample = attempt.sample
        if (
            attempt.error
            or sample is None
            or first is None
            or sample.host != first.host
            or sample.source != report.source
            or attempt.side != "candidate"
            or attempt.repetition != 0
            or (sample.workload, sample.size) != (attempt.workload, attempt.size)
            or sample.device != report.config.device
        ):
            raise ValueError(
                "incomparable: reference samples are incomplete, mixed-source or from another host"
            )
    return Reference(commit=report.source.commit, suite_interface=SUITE_INTERFACE)


def admit_reference(reference: Reference, root: Path) -> Source:
    """Check the declared worker interface without executing reference code."""
    source = source_identity(root)
    if source.dirty or source.commit != reference.commit:
        raise ValueError(
            "reference checkout must be clean at the recorded source commit"
        )
    module = root / "scripts/bench_pipeline.py"
    command = root / "scripts/bench_training_update.py"
    if module.is_file() and command.is_file():
        tree = ast.parse(module.read_text())
        for node in tree.body:
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "SUITE_INTERFACE"
                and isinstance(node.value, ast.Constant)
                and type(node.value.value) is int
                and node.value.value == reference.suite_interface
            ):
                return source
    raise ValueError("reference source does not support the recorded suite interface")


def run_suite(
    config: SuiteConfig,
    out: Path,
    *,
    seconds: float,
    hourly_price: float | None,
    baseline_path: Path | None,
    reference_root: Path | None,
    record_reference: Path | None,
    started_at: float | None = None,
) -> Report:
    start = time.monotonic()
    startup_seconds = (
        max(0.0, time.time() - started_at) if started_at is not None else 0.0
    )
    deadline = time.time() + seconds - startup_seconds
    source = source_identity(ROOT)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = Report(config=config, source=source, hourly_price=hourly_price)
    atomic_json(out / "report.json", report.model_dump(mode="json"))
    try:
        validate_config(config)
        if not math.isfinite(seconds) or seconds <= 4:
            raise ValueError("suite needs a finite lifetime greater than four seconds")
        if hourly_price is not None and (
            not math.isfinite(hourly_price) or hourly_price <= 0
        ):
            raise ValueError("hourly price must be finite and positive")
        if baseline_path or record_reference:
            _require_full_ladder(config)
        if record_reference and (baseline_path or source.dirty):
            raise ValueError("record a reference only from a clean standalone reading")
        host_identity(config.device)
        if baseline_path:
            report.reference = Reference.model_validate_json(baseline_path.read_bytes())
            if source.dirty:
                raise ValueError("candidate checkout must be clean for comparison")
            if reference_root is None:
                raise ValueError(
                    "--compare requires --reference-root at the recorded source"
                )
            reference_root = reference_root.resolve()
            ref = admit_reference(report.reference, reference_root)
        elif reference_root:
            raise ValueError("--reference-root requires --compare")
        repetitions = REPETITIONS if report.reference else 1
        sides_per_cell = 2 if report.reference else 1
        total = repetitions * sides_per_cell * len(config.sizes) * len(config.workloads)
        for repetition in range(repetitions):
            sides = ("reference", "candidate") if report.reference else ("candidate",)
            if repetition % 2:
                sides = tuple(reversed(sides))
            for size, workload, side in product(config.sizes, config.workloads, sides):
                root = reference_root if side == "reference" else ROOT
                assert root is not None
                attempt = Attempt(
                    side=side,
                    repetition=repetition,
                    workload=workload,
                    size=size,
                    directory=str(out / f"{repetition}-{side}-{workload}-{size}"),
                )
                _measure_attempt(
                    attempt,
                    config,
                    root,
                    ref if side == "reference" else source,
                    deadline,
                )
                report.attempts.append(attempt)
                atomic_json(out / "report.json", report.model_dump(mode="json"))
                print(
                    f"[{len(report.attempts)}/{total}] {side} {workload}/{size}: "
                    f"{attempt.error or 'measured'} in {attempt.elapsed_seconds:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt.error:
                    raise RuntimeError(attempt.error)
        report.comparisons = (
            compare(report.attempts, config) if report.reference else []
        )
        if any(c.status != "comparable" for c in report.comparisons):
            raise ValueError(
                "comparison is incomparable; see comparisons in report.json"
            )
        if source_identity(ROOT) != source:
            raise ValueError("candidate source changed during measurement")
        report.status = "completed"
        if record_reference:
            reference = reference_from_report(report)
            record_reference.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(record_reference, reference.model_dump(mode="json"))
    except Exception as exc:
        report.status = "incomplete"
        report.error = f"{type(exc).__name__}: {exc}"
    report.elapsed_seconds = time.monotonic() - start + startup_seconds
    atomic_json(out / "report.json", report.model_dump(mode="json"))
    return report


_SUMMARY_PHASES = (
    ("step+enc s", "time/collection/environment_step_and_encoding"),
    ("infer s", "time/collection/policy_inference_and_action_selection"),
    ("learn s", "time/learning"),
)


def _print_summary(report: Report) -> None:
    """Print one aligned line per sample; phase seconds are per measured update.

    The JSON rows above and report.json remain the complete reading.
    """
    samples = [a for a in report.attempts if a.sample is not None and not a.error]
    if not samples:
        return
    header = ["side", "rep", "workload", "size", "rate/s", "unit", "s/update"]
    header += [label for label, _ in _SUMMARY_PHASES]
    header += ["calls/upd*", "rows/call*", "idle*"]
    lines = [header]
    for attempt in samples:
        sample = attempt.sample
        assert sample is not None
        metrics = sample.metrics
        updates = metrics.get("measured_updates")
        line = [attempt.side, str(attempt.repetition), sample.workload, sample.size]
        line += [f"{sample.rate:.3f}", sample.unit]
        if isinstance(updates, int | float) and updates > 0:
            line.append(f"{metrics['seconds_per_update']:.1f}")
            line += [
                f"{sample.phases[key] / updates:.1f}" if key in sample.phases else "-"
                for _, key in _SUMMARY_PHASES
            ]
            line.append(f"{metrics['inference_calls_per_update_all_updates']:.0f}")
            line.append(f"{metrics['mean_rows_per_call_all_updates']:.1f}")
            line.append(f"{metrics['idle_stream_fraction_all_updates']:.3f}")
        else:
            line += ["-"] * (len(header) - len(line))
        lines.append(line)
    widths = [max(len(row[i]) for row in lines) for i in range(len(header))]
    for row in lines:
        print("  ".join(cell.rjust(width) for cell, width in zip(row, widths)))
    print("* whole stage, including the warmup update")


def describe(report: Report) -> None:
    for attempt in report.attempts:
        sample = attempt.sample
        if attempt.error or sample is None:
            print(f"{attempt.workload}/{attempt.size}: {attempt.error}")
            continue
        row = {
            "side": attempt.side,
            "repetition": attempt.repetition,
            "workload": sample.workload,
            "size": sample.size,
            "unit": sample.unit,
            "per_second": sample.rate,
            **sample.metrics,
            "phases_seconds": sample.phases,
        }
        if report.hourly_price is not None:
            row["per_dollar_at_supplied_price"] = (
                sample.rate * 3600 / report.hourly_price
            )
        print(json.dumps(row))
    for comparison in report.comparisons:
        print(comparison.model_dump_json())
    _print_summary(report)
    print(
        f"{report.status}; elapsed {report.elapsed_seconds:.2f}s; {report.error or 'receipts retained'}"
    )
    print(
        "Rates omit rental setup/idle/failures/billing. "
        "CPU readings are harness proofs, not speed claims or GPU evidence."
    )
