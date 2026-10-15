"""Fit complete serving costs and check excluded physical replay cases.

The immutable partition is actor8/repeat0 from run006. Four non-forward phases
use nonnegative affine costs in padded rows and request count; forward service
keeps the same-rental calibration. Existing queue scheduling owns all simulated
completion times. Excluded cases are retrospective checks, not blind validation.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
from statistics import median
import sys
from types import ModuleType
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from experiments.runners.pipeline_benchmark import Replay, Tape
from experiments.runners.serving_cost import PHASES, Coefficients, Demand, ServingModel

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
SIZES = (1, 4, 16, 64, 256, 1024)


class Delivery(BaseModel):
    rows: Annotated[int, Field(gt=0)]


class PhysicalBatch(BaseModel):
    padded_rows: Annotated[int, Field(gt=0)]
    phases: dict[str, Seconds]
    deliveries: list[Delivery]


class Sample(BaseModel):
    seconds: Seconds
    decisions: int
    requests: int
    batches: list[PhysicalBatch]


class PhysicalCase(BaseModel):
    samples: list[Sample]
    prediction: dict[str, object]
    settings: dict[str, int | float | str]
    payload_bytes_per_row: float


class Window(BaseModel):
    observations: int
    seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Inference(BaseModel):
    windows: list[Window]


class CalibrationPoint(BaseModel):
    device: str
    input_sha256: str
    inference: Inference


def queue_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cost_queue", "scratch/accelerator-queue-probe.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def nnls(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Enumerate the eight active sets of this three-coefficient convex fit."""
    if x.ndim != 2 or x.shape[1] != 3 or y.shape != (len(x),):
        raise ValueError("three features and one duration per observation required")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or (x < 0).any()
        or (y < 0).any()
    ):
        raise ValueError("finite nonnegative observations required")
    best = np.zeros(3, dtype=np.float64)
    loss = float(y @ y)
    for count in (1, 2, 3):
        for active in combinations(range(3), count):
            indices = list(active)
            values = np.linalg.lstsq(x[:, indices], y, rcond=None)[0]
            if (values < 0).any():
                continue
            candidate = np.zeros(3, dtype=np.float64)
            candidate[indices] = values
            residual = x @ candidate - y
            error = float(residual @ residual)
            if error < loss:
                best, loss = candidate, error
    return best


def fit(batches: list[PhysicalBatch], forward: dict[int, float]) -> ServingModel:
    # Scale features for numerical conditioning; coefficients are converted back
    # to seconds per row/request in the persisted model, never unitless factors.
    x = np.array(
        [[1, b.padded_rows / 1024, len(b.deliveries) / 64] for b in batches],
        dtype=np.float64,
    )
    phases = {}
    for phase in PHASES:
        coefficients = nnls(
            x, np.array([b.phases[phase] for b in batches], dtype=np.float64)
        )
        phases[phase] = Coefficients(
            batch_seconds=float(coefficients[0]),
            padded_row_seconds=float(coefficients[1]) / 1024,
            request_seconds=float(coefficients[2]) / 64,
        )
    return ServingModel(
        phases=phases,
        forward_seconds=forward,
        calibration_rows_range=(
            min(b.padded_rows for b in batches),
            max(b.padded_rows for b in batches),
        ),
        calibration_requests_range=(
            min(len(b.deliveries) for b in batches),
            max(len(b.deliveries) for b in batches),
        ),
        calibration_batches=len(batches),
    )


def check(module: ModuleType) -> None:
    module.check()
    x = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 2, 3]], dtype=np.float64)
    truth = np.array([0.2, 0.03, 0.004], dtype=np.float64)
    assert np.allclose(nnls(x, x @ truth), truth, rtol=0, atol=1e-12)
    constrained = nnls(x, np.array([1, 0, 1, 0], dtype=np.float64))
    assert (constrained >= 0).all() and constrained[1] == 0
    setting = module.Settings(actors=2, cpu_workers=2, max_batch=8, wait_seconds=1)

    def complete(demand: Demand) -> float:
        return 0.1 + demand.padded_rows * 0.001 + demand.requests * 0.003

    outcome = module.simulate(
        [4], 0, {4: 99, 8: 99}, 1e12, setting, batch_service=complete
    )
    assert abs(outcome.seconds - 0.114) < 1e-12
    assert outcome.requests == 2 and outcome.decisions == 8
    for invalid in (-1.0, float("nan"), float("inf")):

        def bad(_: Demand) -> float:
            return invalid

        try:
            module.simulate([4], 0, {4: 99, 8: 99}, 1e12, setting, batch_service=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid service accepted")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("fresh output required")
    module = (
        queue_module()
    )  # Scratch module boundary; its dataclasses remain its owner.
    check(module)
    root = Path("scratch/gpu-probes/serving-006/evidence")
    manifest = {
        entry["path"]: entry
        for entry in json.loads((root / "manifest.json").read_text())
    }
    sources: dict[str, str] = {}

    def verified(path: Path) -> str:
        body = path.read_bytes()
        sha = hashlib.sha256(body).hexdigest()
        expected = manifest[str(path.relative_to(root))]
        assert sha == expected["sha256"] and len(body) == expected["bytes"]
        sources[str(path)] = sha
        return body.decode()

    forward = {}
    for size in SIZES:
        point = CalibrationPoint.model_validate_json(
            verified(root / f"w64-d1-model-b{size}-s0/result.json")
        )
        assert (
            point.device == "NVIDIA H100 80GB HBM3"
            and point.input_sha256
            == "424ab00b9f07e56f47a1072d9fa9a080b99a93d785197a8ddc70be7f4f03dfa2"
        )
        forward[size] = size / median(
            w.observations / w.seconds for w in point.inference.windows
        )
    calibration = [
        PhysicalCase.model_validate_json(
            verified(root / f"serve-a8-b{size}/result.json")
        )
        for size in (64, 256, 1024)
    ]
    model = fit(
        [batch for case in calibration for batch in case.samples[0].batches], forward
    )
    # Model is frozen before reading excluded records; their aggregate times were
    # previously inspected, so this partition still is not a blind test.
    tape_path = Path("scratch/accelerator-probes/cohort-t8-s64/tape.json")
    replay_path = tape_path.parent / "replay.json"
    tape = Tape.model_validate_json(tape_path.read_text())
    replay_data = json.loads(replay_path.read_text())
    assert replay_data["tape_sha256"] == digest(tape_path)
    replay = Replay.model_validate(replay_data["samples"][1])
    assert (
        replay.batch_sha256 == tape.batch_sha256
        and replay.request_host_seconds is not None
        and replay.tail_host_seconds is not None
    )
    cases = []
    for actors in (1, 8, 64):
        for maximum in (64, 256, 1024):
            name = f"serve-a{actors}-b{maximum}"
            case = PhysicalCase.model_validate_json(
                verified(root / name / "result.json")
            )
            settings = module.Settings(**case.settings)
            arguments = (
                [f.rows for f in tape.frames],
                0,
                forward,
                case.payload_bytes_per_row,
                settings,
            )
            options = dict(
                request_host_seconds=replay.request_host_seconds,
                tail_host_seconds=replay.tail_host_seconds,
            )
            baseline = module.simulate(*arguments, **options)
            assert json.loads(json.dumps(asdict(baseline))) == case.prediction
            predicted = module.simulate(
                *arguments, **options, batch_service=model.duration
            )
            assert (
                predicted.requests == actors * 59
                and predicted.decisions == actors * 1099
            )
            ratios = [s.seconds / predicted.seconds for s in case.samples]
            outside = [
                sum(
                    len(b.deliveries) > model.calibration_requests_range[1]
                    for b in s.batches
                )
                for s in case.samples
            ]
            cases.append(
                dict(
                    name=name,
                    actors=actors,
                    maximum=maximum,
                    partition="excluded_actor"
                    if actors != 8
                    else "repeat0_fit_repeats1_2_check",
                    predicted=asdict(predicted),
                    original_predicted_seconds=baseline.seconds,
                    physical_seconds=[s.seconds for s in case.samples],
                    actual_over_predicted=ratios,
                    median_actual_over_predicted=median(ratios),
                    median_within_20_percent=0.8 <= median(ratios) <= 1.2,
                    batches_outside_calibration_request_range=outside,
                )
            )
            print(
                name,
                "ratios",
                [round(r, 3) for r in ratios],
                "extrapolated batches",
                outside,
            )
    rankings = []
    for actors in (1, 8, 64):
        group = [c for c in cases if c["actors"] == actors]
        best = min(c["predicted"]["seconds"] for c in group)
        predicted_best = [
            c["maximum"] for c in group if abs(c["predicted"]["seconds"] - best) < 1e-9
        ]
        actual_best = min(group, key=lambda c: median(c["physical_seconds"]))["maximum"]
        rankings.append(
            dict(
                actors=actors,
                predicted_best=predicted_best,
                actual_best=actual_best,
                preserved=actual_best in predicted_best,
            )
        )
    for path in (
        Path(__file__),
        Path("scratch/accelerator-queue-probe.py"),
        Path("scratch/serving-calibration-protocol.txt"),
        tape_path,
        replay_path,
    ):
        sources[str(path)] = digest(path)
    report = dict(
        schema="manabot.research.serving-cost.v1",
        model=model.model_dump(mode="json"),
        fit="NNLS phases from actor8 repeat0, all batches, no clipping/refitting",
        sources=sources,
        cases=cases,
        rankings=rankings,
        limits=[
            "Retrospective excluded-data checks; aggregate evaluation times were seen before partition selection.",
            "Same machine, driver, fixed inputs and finite stationary traffic; no network, actual host compute, learning or distributed validation.",
            "Request counts beyond calibration range are extrapolation, reported per case; no forward batch-size interpolation.",
            "Timer release and manager delivery delays remain omitted.",
        ],
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("Calibration batches", model.calibration_batches, "rankings", rankings)


if __name__ == "__main__":
    main()
