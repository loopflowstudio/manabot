"""Complete serving costs for the research queue and cohort instruments.

ServingModel owns fitted phase coefficients and exact forward points. load_serving
checks a prospective validation bundle against the selected hardware and workload.
ServiceMeter counts extrapolated batch shapes without changing scheduling. These
costs describe one isolated serving worker, not distributed CPU/memory contention.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.runners.pipeline_benchmark import load_services
from manabot.training.models import TrainingRegime

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Phase = Literal["prepare", "h2d", "d2h", "validate"]
PHASES: tuple[Phase, ...] = ("prepare", "h2d", "d2h", "validate")


class Coefficients(BaseModel):
    """Nonnegative seconds per batch, padded row and logical request."""

    model_config = ConfigDict(extra="forbid")
    batch_seconds: Seconds
    padded_row_seconds: Seconds
    request_seconds: Seconds

    def duration(self, rows: int, requests: int) -> float:
        return (
            self.batch_seconds
            + rows * self.padded_row_seconds
            + requests * self.request_seconds
        )


class Demand(Protocol):
    @property
    def padded_rows(self) -> int: ...

    @property
    def requests(self) -> int: ...


class ServingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phases: dict[Phase, Coefficients]
    forward_seconds: dict[int, Seconds]
    calibration_rows_range: tuple[int, int]
    calibration_requests_range: tuple[int, int]
    calibration_batches: int = Field(gt=0)

    @model_validator(mode="after")
    def complete(self) -> "ServingModel":
        if (
            set(self.phases) != set(PHASES)
            or not self.forward_seconds
            or min(self.forward_seconds) < 1
        ):
            raise ValueError("complete phases and positive batch sizes required")
        if any(
            low < 1 or low > high
            for low, high in (
                self.calibration_rows_range,
                self.calibration_requests_range,
            )
        ):
            raise ValueError("invalid calibration support")
        return self

    def duration(self, demand: Demand) -> float:
        return self.forward_seconds[demand.padded_rows] + sum(
            phase.duration(demand.padded_rows, demand.requests)
            for phase in self.phases.values()
        )


class ModelRecord(BaseModel):
    schema_version: Literal["manabot.research.serving-cost.v1"] = Field(alias="schema")
    model: ServingModel


class ValidatedRecord(ModelRecord):
    overhead_source: Path
    overhead_source_sha256: str
    forward_run: str
    summary_sha256: str


class FileWitness(BaseModel):
    path: str
    bytes: int
    sha256: str


@dataclass
class ServiceUsage:
    calls: int = 0
    outside_padded_row_range: int = 0
    outside_request_range: int = 0
    maximum_requests: int = 0


@dataclass
class ServiceMeter:
    model: ServingModel
    extra_seconds: dict[int, float] = field(default_factory=dict)
    usage: ServiceUsage = field(default_factory=ServiceUsage)

    def __call__(self, demand: Demand) -> float:
        self.usage.calls += 1
        low, high = self.model.calibration_rows_range
        self.usage.outside_padded_row_range += not low <= demand.padded_rows <= high
        low, high = self.model.calibration_requests_range
        self.usage.outside_request_range += not low <= demand.requests <= high
        self.usage.maximum_requests = max(self.usage.maximum_requests, demand.requests)
        return self.model.duration(demand) + self.extra_seconds.get(
            demand.padded_rows, 0.0
        )


@dataclass(frozen=True)
class ServingEvidence:
    model: ServingModel
    sources: dict[str, str]
    forward_run: str


def load_serving(
    path: Path,
    hardware_path: Path,
    card: str,
    recipe: TrainingRegime,
    payload_bytes_per_row: float,
) -> ServingEvidence:
    """Admit one measured boundary; reject incompatible workload or source inputs.

    The bundle's current forward curve is used exactly. Recalibration belongs in
    an explicit physical experiment, never implicitly in this CLI loader.
    """
    sources: dict[str, str] = {
        str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }

    def read(p: Path, expected: str | None = None) -> str:
        body = p.read_bytes()
        sha = hashlib.sha256(body).hexdigest()
        if expected is not None and sha != expected:
            raise ValueError(f"serving evidence digest mismatch: {p}")
        sources[str(p)] = sha
        return body.decode()

    record = ValidatedRecord.model_validate_json(read(path))
    overhead = ModelRecord.model_validate_json(
        read(record.overhead_source, record.overhead_source_sha256)
    )
    if record.model.phases != overhead.model.phases:
        raise ValueError("overhead coefficients differ from frozen calibration")
    if any(
        getattr(record.model, field) != getattr(overhead.model, field)
        for field in (
            "calibration_rows_range",
            "calibration_requests_range",
            "calibration_batches",
        )
    ):
        raise ValueError("calibration support differs from frozen evidence")
    root = path.parent
    summary = json.loads(read(root / "summary.json", record.summary_sha256))
    if summary["run"] != record.forward_run:
        raise ValueError("forward and validation runs differ")
    expected_hardware = root / "capacity-report.json"
    read(expected_hardware)
    read(hardware_path, sources[str(expected_hardware)])
    hardware = json.loads(hardware_path.read_text())
    if hardware["run"] != record.forward_run:
        raise ValueError("hardware run differs")
    forward = {
        s.batch: s.inference_seconds
        for s in load_services(hardware_path)
        if s.card == card and s.capacity == "w64-d1"
    }
    if forward != record.model.forward_seconds:
        raise ValueError("selected forward curve differs from serving calibration")
    manifest = [
        FileWitness.model_validate(item)
        for item in json.loads(read(root / "evidence/manifest.json"))
    ]
    witnesses = {item.path: item for item in manifest}
    baseline = TrainingRegime.model_validate_json(
        read(root / "evidence/baseline.json", witnesses["baseline.json"].sha256)
    )
    if any(
        getattr(recipe, key) != getattr(baseline, key)
        for key in ("agent", "observation", "match", "world")
    ):
        raise ValueError("serving and tape model/world/observation contracts differ")
    # One completed physical result supplies exact input+output bytes per row.
    first = next(c for c in summary["cells"] if c["status"] == "completed")
    name = first["name"] + "/result.json"
    physical = json.loads(read(root / "evidence" / name, witnesses[name].sha256))
    if not math.isclose(
        payload_bytes_per_row,
        physical["payload_bytes_per_row"],
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError("observation payload differs from serving calibration")
    return ServingEvidence(record.model, sources, record.forward_run)
