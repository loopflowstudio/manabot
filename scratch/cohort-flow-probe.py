"""Couple the existing inference queue to one measured synchronous learner cohort.

Only leased actors collect a version-current global batch. All inference drains
before one optimizer owner updates; publication precedes the next lease. This
models phase sharing without inventing stale-policy learning or multi-GPU Adam.
Frozen tape repetition is a stationary workload assumption, not training quality.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from experiments.runners.pipeline_benchmark import Replay, Tape, load_services
from experiments.runners.serving_cost import ServiceMeter, load_serving
from manabot.training.models import TrainingRegime


class Service(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    case: str
    shape: tuple[int, int]
    raw_rows_per_iteration: int = Field(gt=0)
    optimizer_exposures_per_iteration: int = Field(gt=0)
    optimizer_steps_per_iteration: int = Field(gt=0)
    whole_learning_seconds_per_iteration: float = Field(gt=0)


class Collection(BaseModel):
    seconds: float = Field(gt=0)
    cpu_busy_seconds: float = Field(ge=0)
    inference_server_and_transfer_busy_fraction: float = Field(ge=0, le=1.000000001)
    decisions: int = Field(gt=0)
    requests: int = Field(gt=0)
    peak_queued_rows: int = Field(ge=0)
    padded_batch_counts: dict[int, int]
    useful_row_fraction: float = Field(gt=0, le=1)
    mean_request_seconds: float = Field(ge=0)
    max_request_seconds: float = Field(ge=0)


@dataclass(frozen=True)
class Settings:
    available_actors: int
    cpu_workers: int
    gpus: int
    max_batch: int
    host_speed: float = 1.0
    aggregation_seconds: float = 0.0
    publication_seconds: float = 0.0
    horizon_seconds: float = 10.0


@dataclass(frozen=True)
class Outcome:
    settings: Settings
    service_case: str
    leased_actors: int
    idle_actors: int
    collection: Collection
    learning_seconds: float
    epoch_seconds: float
    stationary_raw_rows_per_second: float
    stationary_optimizer_exposures_per_second: float
    stationary_server_and_learning_reservation_fraction: float
    learning_start_policy_age_seconds: float
    peak_array_bytes_under_copy_assumption: int
    completed_updates: int
    published_versions: int
    learned_rows_at_horizon: int
    optimizer_exposures_at_horizon: int
    pending_phase: Literal[
        "collection", "aggregation", "learning", "publication", "none"
    ]
    pending_phase_seconds: float
    completed_epoch_cpu_work_seconds: float


def leases(point: Service, tape: Tape) -> int:
    steps, streams = point.shape
    if steps != tape.transitions or streams % tape.streams:
        raise ValueError("learner shape cannot be composed from complete actor tapes")
    if point.raw_rows_per_iteration != steps * streams:
        raise ValueError("learner shape and raw-row count disagree")
    return streams // tape.streams


def couple(
    collected: Collection,
    point: Service,
    settings: Settings,
    leased: int,
    rollout_bytes: int,
    buffer_bytes: int,
) -> Outcome:
    s = settings
    if min(leased, s.cpu_workers, s.gpus) < 1 or s.available_actors < leased:
        raise ValueError("insufficient actors or invalid resources")
    times = (s.aggregation_seconds, s.publication_seconds, s.horizon_seconds)
    if any(not math.isfinite(t) or t < 0 for t in times) or s.horizon_seconds == 0:
        raise ValueError("finite nonnegative delays and positive horizon required")
    learning = point.whole_learning_seconds_per_iteration
    epoch = collected.seconds + s.aggregation_seconds + learning + s.publication_seconds
    cycles = int(s.horizon_seconds // epoch)
    remaining = s.horizon_seconds - cycles * epoch
    if math.isclose(remaining, epoch, rel_tol=0, abs_tol=1e-12):
        cycles += 1
        remaining = 0.0
    if remaining < 1e-12:
        remaining = 0.0
    phases: tuple[
        tuple[Literal["collection", "aggregation", "learning", "publication"], float],
        ...,
    ] = (
        ("collection", collected.seconds),
        ("aggregation", s.aggregation_seconds),
        ("learning", learning),
        ("publication", s.publication_seconds),
    )
    pending: Literal["collection", "aggregation", "learning", "publication", "none"] = (
        "none"
    )
    elapsed = remaining
    learned = cycles
    for phase, duration in phases:
        if elapsed < duration:
            pending = phase if remaining else "none"
            break
        elapsed -= duration
        if phase == "learning" and remaining:
            learned += 1
    busy = (
        collected.inference_server_and_transfer_busy_fraction
        * collected.seconds
        * s.gpus
        + learning
    ) / (epoch * s.gpus)
    assert busy <= 1 + 1e-9
    return Outcome(
        s,
        point.case,
        leased,
        s.available_actors - leased,
        collected,
        learning,
        epoch,
        point.raw_rows_per_iteration / epoch,
        point.optimizer_exposures_per_iteration / epoch,
        busy,
        collected.seconds + s.aggregation_seconds,
        leased * (2 * rollout_bytes + buffer_bytes),
        learned,
        cycles,
        learned * point.raw_rows_per_iteration,
        learned * point.optimizer_exposures_per_iteration,
        pending,
        elapsed if pending != "none" else 0.0,
        cycles * collected.cpu_busy_seconds,
    )


def queue_module() -> ModuleType:
    """Load the existing scratch instrument; validate its result at this boundary."""
    spec = importlib.util.spec_from_file_location(
        "cohort_queue", "scratch/accelerator-queue-probe.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select(points: list[Service], streams: int, repeat: str = "a") -> Service:
    matches = [
        p
        for p in points
        if p.shape == (8, streams) and p.case.endswith(f"reference-{repeat}")
    ]
    if len(matches) != 1:
        raise ValueError("exact completed learner shape/repeat is unavailable")
    return matches[0]


def check() -> None:
    point = Service(
        case="fixture",
        shape=(8, 64),
        raw_rows_per_iteration=512,
        optimizer_exposures_per_iteration=128,
        optimizer_steps_per_iteration=8,
        whole_learning_seconds_per_iteration=1,
    )
    collection = Collection(
        seconds=2,
        cpu_busy_seconds=1,
        inference_server_and_transfer_busy_fraction=0.5,
        decisions=1024,
        requests=4,
        peak_queued_rows=64,
        padded_batch_counts={64: 4},
        useful_row_fraction=1.0,
        mean_request_seconds=0.5,
        max_request_seconds=0.5,
    )
    settings = Settings(8, 1, 1, 64, horizon_seconds=9)
    result = couple(collection, point, settings, 1, 100, 20)
    assert result.completed_updates == result.published_versions == 3
    assert result.learned_rows_at_horizon == 1536 and result.pending_phase == "none"
    assert result.peak_array_bytes_under_copy_assumption == 220
    assert result.idle_actors == 7 and result.completed_epoch_cpu_work_seconds == 3
    partial = couple(
        collection, point, Settings(8, 1, 1, 64, horizon_seconds=2.5), 1, 100, 20
    )
    assert partial.completed_updates == 0 and partial.pending_phase == "learning"
    assert partial.pending_phase_seconds == 0.5
    publication = couple(
        collection,
        point,
        Settings(8, 1, 1, 64, publication_seconds=2, horizon_seconds=4),
        1,
        100,
        20,
    )
    assert publication.completed_updates == 1 and publication.published_versions == 0
    assert (
        publication.pending_phase == "publication"
        and publication.pending_phase_seconds == 1
    )
    try:
        select([point], 4096)
    except ValueError:
        pass
    else:
        raise AssertionError("unmeasured learner shape was admitted")
    print(
        "Cohort timing, resource, lease, memory, horizon and unsupported-shape checks passed"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card", default="l4")
    parser.add_argument(
        "--serving-costs",
        type=Path,
        help="Validated full-service model; replaces idealized links",
    )
    parser.add_argument("--global-streams", type=int, choices=(512, 4096), default=512)
    parser.add_argument(
        "--hardware",
        type=Path,
        default=Path("scratch/gpu-probes/l4-002/capacity-report.json"),
    )
    parser.add_argument(
        "--learner-service",
        type=Path,
        default=Path("scratch/gpu-probes/l4-004/learner-service.json"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("fresh output required")
    check()
    root = Path("scratch/accelerator-probes/cohort-t8-s64")
    tape_path, replay_path = root / "tape.json", root / "replay.json"
    service_path = args.learner_service
    hardware_path = args.hardware
    data = json.loads(service_path.read_text())
    summary_path = service_path.parent / data["source_summary"]
    assert (
        hashlib.sha256(summary_path.read_bytes()).hexdigest() == data["source_sha256"]
    )
    points = [Service.model_validate(p) for p in data["points"]]
    tape = Tape.model_validate_json(tape_path.read_text())
    summary = json.loads(summary_path.read_text())
    for point in points:
        cell = next(c for c in summary["cells"] if c["name"] == point.case)
        if cell["status"] != "completed" or cell["completed_update_iterations"] != 3:
            raise ValueError("learner service has no completed three-update evidence")
        if not math.isclose(
            point.whole_learning_seconds_per_iteration,
            cell["learning_seconds"] / 3,
            rel_tol=1e-12,
        ):
            raise ValueError("service time disagrees with the retained run")
        regime_path = (
            service_path.parent
            / "evidence"
            / point.case
            / f"w64-d1-loop-b{point.raw_rows_per_iteration}-s{point.shape[1]}"
            / "regime.json"
        )
        regime = TrainingRegime.model_validate_json(regime_path.read_text())
        if any(
            getattr(regime, key) != getattr(tape.recipe, key)
            for key in ("agent", "observation", "match", "world")
        ):
            raise ValueError(
                "learner recipe and inference tape have different model/world contracts"
            )
    replay_data = json.loads(replay_path.read_text())
    assert (
        replay_data["tape_sha256"] == hashlib.sha256(tape_path.read_bytes()).hexdigest()
    )
    sample = Replay.model_validate(replay_data["samples"][1])
    assert (
        sample.batch_sha256,
        sample.buffers_sha256,
        sample.journal_sha256,
        sample.game_seeds,
    ) == (tape.batch_sha256, tape.buffers_sha256, tape.journal_sha256, tape.game_seeds)
    assert (
        sample.request_host_seconds is not None and sample.tail_host_seconds is not None
    )
    assert math.isclose(
        sum(sample.request_host_seconds) + sample.tail_host_seconds,
        sample.seconds,
        abs_tol=1e-12,
    )
    assert sample.demand.learner_transitions == tape.streams * tape.transitions
    hardware = {
        s.batch: s.inference_seconds
        for s in load_services(hardware_path)
        if s.card == args.card and s.capacity == "w64-d1"
    }
    serving = (
        load_serving(
            args.serving_costs,
            hardware_path,
            args.card,
            tape.recipe,
            (sample.demand.input_bytes + sample.demand.output_bytes)
            / sample.demand.decisions,
        )
        if args.serving_costs is not None
        else None
    )
    serving_usage = []
    queue = queue_module()
    queue.check()
    outcomes: list[Outcome] = []
    global_streams = args.global_streams
    global_actors = leases(select(points, global_streams), tape)
    for streams, actors, cpus, gpus, speed, delay, repeat in [
        (64, 1, 1, 1, 1, 0, "a"),
        (64, 1, 1, 1, 1, 0, "b"),
        (global_streams, global_actors, 1, 1, 1, 0, "a"),
        (global_streams, global_actors, global_actors, 1, 1, 0, "a"),
        (global_streams, global_actors, global_actors, global_actors, 1, 0, "a"),
        (global_streams, 4096, global_actors, global_actors, 1, 0, "a"),
        (global_streams, 4096, 512, 64, 1, 0, "a"),
        (global_streams, global_actors, global_actors, 1, 2, 0, "a"),
        (global_streams, global_actors, global_actors, 1, 1, 0.01, "a"),
    ]:
        point = select(points, streams, repeat)
        leased = leases(point, tape)
        for batch in (64, 256, 1024):
            settings = Settings(
                actors,
                cpus,
                gpus,
                batch,
                host_speed=speed,
                aggregation_seconds=delay,
                publication_seconds=delay,
            )
            # Dynamic scratch-module API is narrowed immediately to Collection.
            meter = ServiceMeter(serving.model) if serving is not None else None
            result = queue.simulate(
                [f.rows for f in tape.frames],
                0,
                hardware,
                (sample.demand.input_bytes + sample.demand.output_bytes)
                / sample.demand.decisions,
                queue.Settings(
                    actors=leased,
                    cpu_workers=cpus,
                    gpus=gpus,
                    max_batch=batch,
                    wait_seconds=0.00025,
                    host_speed=speed,
                ),
                request_host_seconds=sample.request_host_seconds,
                tail_host_seconds=sample.tail_host_seconds,
                batch_service=meter,
            )
            if meter is not None:
                serving_usage.append(
                    {"settings": asdict(settings), **asdict(meter.usage)}
                )
            collection = Collection.model_validate(asdict(result))
            assert collection.decisions == leased * sample.demand.decisions
            outcomes.append(
                couple(
                    collection,
                    point,
                    settings,
                    leased,
                    sample.demand.rollout_bytes,
                    sample.demand.buffer_bytes,
                )
            )
    reference = next(
        o
        for o in outcomes
        if o.settings == Settings(global_actors, global_actors, global_actors, 256)
    )
    idle = next(
        o
        for o in outcomes
        if o.settings == Settings(4096, global_actors, global_actors, 256)
    )
    assert (
        reference.stationary_raw_rows_per_second == idle.stationary_raw_rows_per_second
    )
    paths = (
        Path(__file__),
        Path("scratch/accelerator-queue-probe.py"),
        tape_path,
        replay_path,
        service_path,
        summary_path,
        hardware_path,
    )
    report = {
        "card": args.card,
        "global_streams": global_streams,
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "host_repeat": 1,
        "serving_boundary": "complete_measured_worker"
        if serving
        else "model_plus_assumed_links",
        "serving_cost_sources": serving.sources if serving else {},
        "serving_cost_usage": serving_usage,
        "outcomes": [
            asdict(o) | {"collection": o.collection.model_dump()} for o in outcomes
        ],
        "limits": [
            "Stationary cloned untrained CPU tape; not real multi-version training.",
            "One centrally filtered update with a completed measured shape; no interpolation or multi-GPU learning.",
            "Inference and learning share GPUs in exclusive phases; zero-lag cohort barrier prevents overlap.",
            "Aggregation/publication zero is an optimistic network bound;10ms sensitivity is assumed.",
            "GPU service comes from the selected named hardware report and learner receipt; coupled performance is modeled, not measured.",
            "Array-memory scenario retains leased actor arrays while copying a global batch; idle capacity has no instantiated collectors. Native games, models, Python, staging excluded.",
            "Horizon counts completed updates, published versions and pending phase; CPU work excludes the partial epoch.",
            "Extra inference-serving CPU thread perGPU; target CPU contention and all-reduce unmodeled.",
            "Full serving and learner measurements may come from different rentals; this is an explicit composition, not simultaneous measured throughput. Each server assumes independent calibrated CPU/memory/link capacity.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Recorded {len(outcomes)} coupled cases; idle actor invariance passed")


if __name__ == "__main__":
    main()
