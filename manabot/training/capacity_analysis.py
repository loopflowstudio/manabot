"""Capacity summaries derived from ordinary study measurements, never new scores.

The shared cost integrator owns step-function area. Crossings are observations
at scheduled checkpoints, not exact latent learning times or stopping rules.
"""

from pathlib import Path
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from experiments.runners.training_protocol import EvaluationProtocol
from manabot.training import analysis
from manabot.training.execution import atomic_json
from manabot.training.models import ArtifactReference


class Measurement(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    regime: str
    seed: int
    variant: str = "raw"
    phase: str = "development"
    cutoff: int
    opponent: str
    checkpoint: ArtifactReference
    training_seconds: float = Field(ge=0)
    decisions: int = Field(ge=0)
    optimizer_exposures: int | None = Field(default=None, ge=0)
    score: float | None = Field(ge=0, le=1)
    complete: bool


class Crossing(BaseModel):
    regime: str
    seed: int
    status: Literal["observed", "right-censored", "unavailable"]
    lower_seconds: float | None = None
    upper_seconds: float | None = None
    checkpoint: ArtifactReference | None = None


def threshold_crossing(
    points: list[Measurement], horizon: float, threshold: float
) -> Crossing:
    """First observed hit; right censor at the last observed non-hit in the window."""
    if not points:
        raise ValueError("crossing requires an identified series")
    identity = {(p.regime, p.seed, p.variant) for p in points}
    if len(identity) != 1:
        raise ValueError("crossing requires one regime/seed/variant")
    result = Crossing(
        regime=points[0].regime, seed=points[0].seed, status="unavailable"
    )
    previous = None
    for point in sorted(points, key=lambda p: p.training_seconds):
        if (
            point.phase != "development"
            or not point.complete
            or point.score is None
            or point.training_seconds > horizon
        ):
            continue
        if point.score >= threshold:
            return result.model_copy(
                update={
                    "status": "observed",
                    "lower_seconds": previous,
                    "upper_seconds": point.training_seconds,
                    "checkpoint": point.checkpoint,
                }
            )
        previous = point.training_seconds
    return result.model_copy(
        update={
            "status": "right-censored" if previous is not None else "unavailable",
            "lower_seconds": previous,
        }
    )


class CapacitySummary(TypedDict):
    status: str
    crossings: list[dict[str, Any]]
    threshold: float
    early_window_seconds: float
    # Existing cost_comparison owns these heterogeneous JSON metric records.
    early_area: dict[str, Any]
    curves: dict[str, object]
    terminal: list[dict[str, Any]]
    decision: str
    limits: str


def capacity_summary(
    rows: list[Measurement],
    protocol: EvaluationProtocol,
    regimes: tuple[str, ...],
    completed: bool,
) -> CapacitySummary:
    """Require the frozen cohort; keep endpoint observations separate from monitoring."""
    assert protocol.early_progress_seconds is not None
    assert protocol.progress_score is not None
    anchor = "random-smoke-anchor"
    development = [p for p in rows if p.opponent == anchor and p.phase == "development"]
    expected = {(r, s) for r in regimes for s in protocol.training_seeds}
    observed = {
        (p.regime, p.seed) for p in development if p.complete and p.score is not None
    }
    available = completed and observed == expected
    crossings = []
    for regime, seed in sorted(expected):
        points = [p for p in development if (p.regime, p.seed) == (regime, seed)]
        crossing = (
            threshold_crossing(
                points, protocol.early_progress_seconds, protocol.progress_score
            )
            if points
            else Crossing(regime=regime, seed=seed, status="unavailable")
        )
        crossings.append(crossing.model_dump())
    curves: dict[str, object] = {}
    for axis in ("training_seconds", "decisions", "optimizer_exposures"):
        if not available or any(getattr(p, axis) is None for p in development):
            curves[axis] = {
                "status": "unavailable",
                "reason": "incomplete cohort or missing counter",
            }
            continue
        # Reuse the authoritative integration, expressing its coordinate in this
        # axis's units. No time extrapolation or invented exposure counters.
        mapped = [
            {**p.model_dump(), "training_seconds": getattr(p, axis)}
            for p in development
        ]
        comparison = analysis.cost_comparison(mapped, anchor)
        if comparison["status"] == "available":
            start, end = comparison.pop("start_seconds"), comparison.pop("end_seconds")
            comparison.update(start=start, end=end, axis=axis)
            for item in comparison["rows"]:
                item["checkpoint_coordinate"] = item.pop("checkpoint_seconds")
            comparison["points"] = mapped
        curves[axis] = comparison
    early = (
        analysis.cost_comparison(
            [p.model_dump() for p in development],
            anchor,
            end_seconds=protocol.early_progress_seconds,
        )
        if available
        else {"status": "unavailable", "reason": "incomplete cohort"}
    )
    if early["status"] == "available":
        duration = early["end_seconds"] - early["start_seconds"]
        for item in early["rows"]:
            item["score_seconds"] = (
                item["mean_score"] * duration if duration > 0 else None
            )
    endpoints = [
        p
        for p in rows
        if p.phase == "endpoint" and p.cutoff == protocol.checkpoint_count - 1
    ]
    return {
        "status": "available" if available else "unavailable",
        "crossings": crossings,
        "threshold": protocol.progress_score,
        "early_window_seconds": protocol.early_progress_seconds,
        "early_area": early,
        "curves": curves,
        "terminal": [p.model_dump() for p in endpoints],
        "decision": "Unresolved: no automatic fast-model or promotion selection; inspect seed/deal uncertainty and complete endpoint cohort.",
        "limits": "Crossings are descriptive scheduled observations; censoring is at last observed checkpoint. Areas use common observed support. Endpoint index is a fixed calibrated workload, not convergence or equal actual time. Incomplete rows remain retained, never strength evidence.",
    }


def write_capacity_report(
    out: Path,
    rows: list[Measurement],
    protocol: EvaluationProtocol,
    regimes: tuple[str, ...],
    completed: bool,
) -> None:
    summary = capacity_summary(rows, protocol, regimes, completed)
    atomic_json(out / "capacity-analysis.json", summary)
    with (out / "report.md").open("a") as stream:
        stream.write(
            "\n## Capacity comparison\n\n"
            + str(summary["decision"])
            + "\n\n"
            + str(summary["limits"])
            + "\n\n[Derived curves, censoring and terminal observations](capacity-analysis.json).\n"
        )
