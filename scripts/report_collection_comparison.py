"""Compare newly retained MTG-281 arms against their pre-execution contract.

Inputs are the prepared directory plus before/after metric captures and each
arm's delivered training/run.json. Read receipts without importing an executor
or the native engine. No launcher success is treated as scientific completion.
Slopes and tails are descriptive; one paired seed supplies no causal confidence
interval. A 500-update result says nothing about the later frontier regime.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

from etude.launch.records import Run
import numpy as np
from pydantic import BaseModel, ConfigDict, JsonValue

from manabot.infra.host import HostFacts
from scripts.collection_comparison import (
    SEED,
    SOURCES,
    UPDATES,
    Analysis,
    prepared_runs,
)
from scripts.measure_collection_growth import Capture
from scripts.report_collection_growth import PHASES, FloatArray, Reading, reading


class Projection(BaseModel):
    # Retain original files; this reader only narrows the fields it interprets.
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class Stage(Projection):
    updates: int
    status: str
    actual_device: str | None = None
    actual_threads: int | None = None
    games: int
    learner_transitions: int


class Errors(Projection):
    attempted: int
    completed: int
    crashed: int
    abandoned: int
    discarded_samples: int
    replacement_samples: int


class Receipt(Projection):
    id: str
    seed: int
    metrics_run: str | None = None
    status: str
    regime: dict[str, JsonValue]
    identities: dict[str, JsonValue]
    host: HostFacts | None = None
    stages: list[Stage]
    game_errors: Errors | None = None
    parent_run_id: str | None = None
    recovery_seconds: float


class LoggedTraining(Projection):
    run_id: str
    regime: dict[str, JsonValue]
    identities: dict[str, JsonValue]


class LoggedConfig(Projection):
    training: LoggedTraining


class JobConfig(Projection):
    regime: dict[str, JsonValue]


@dataclass(frozen=True)
class Statistics:
    slope_per_100_updates: float
    slope_per_million_learner_rows: float
    fitted_window_rise: float
    mean: float
    p90: float
    early_mean: float
    early_median: float
    early_p90: float
    late_mean: float
    late_median: float
    late_p90: float


def _statistics(
    run: Reading, values: FloatArray, plan: Analysis, end: int
) -> Statistics:
    # reading() admits consecutive updates starting at one, so inclusive update
    # windows map directly to slices without rebuilding coordinate masks.
    window = slice(plan.first - 1, end)
    selected = values[window]
    early = values[plan.first - 1 : plan.early_last]
    late = values[plan.late_first - 1 : end]
    if len(selected) < 2 or not len(early) or not len(late):
        raise ValueError("both declared endpoint windows must be reached")
    if not np.isfinite(selected).all():
        raise ValueError("missing or nonfinite metric within analysis window")
    rows = run.metrics["rows"]
    if not np.isfinite(rows).all() or np.any(rows <= 0):
        raise ValueError("learner rows must be finite and positive")
    slope = float(np.polyfit(run.update[window] / 100, selected, 1)[0])
    work = np.cumsum(rows) / 1e6
    return Statistics(
        slope_per_100_updates=slope,
        slope_per_million_learner_rows=float(np.polyfit(work[window], selected, 1)[0]),
        fitted_window_rise=slope * (end - plan.first) / 100,
        mean=float(np.mean(selected)),
        p90=float(np.quantile(selected, 0.9)),
        early_mean=float(np.mean(early)),
        early_median=float(np.median(early)),
        early_p90=float(np.quantile(early, 0.9)),
        late_mean=float(np.mean(late)),
        late_median=float(np.median(late)),
        late_p90=float(np.quantile(late, 0.9)),
    )


def _growth_score(
    stats: Statistics, kind: Literal["time", "rss"], plan: Analysis
) -> float:
    """Largest positive growth divided by its declared practical-flatness limit."""
    if kind == "rss":
        return (
            max(0, stats.fitted_window_rise, stats.late_median - stats.early_median)
            / plan.rss_growth_gb
        )
    if stats.early_mean <= 0:
        raise ValueError("primary timing baseline must be positive")
    return max(0, stats.fitted_window_rise, stats.late_p90 - stats.early_p90) / (
        stats.early_mean * plan.time_growth_fraction
    )


def _verdict(
    scores: dict[str, dict[str, float]], issues: list[str], plan: Analysis
) -> str:
    if issues:
        return "inconclusive"
    before, after = scores["before"], scores["after"]
    if any(value <= 1 for value in before.values()):
        return "inconclusive: baseline growth did not recur in every primary"
    if all(value <= 1 for value in after.values()):
        return "flattens through 500 updates"
    if all(
        after[key] <= value * (1 - plan.partial_reduction_fraction)
        for key, value in before.items()
    ):
        return "partly flattens through 500 updates"
    return "does not meet the declared flattening criteria through 500 updates"


def _validate_arm(arm: str, run: Reading, receipt: Receipt, job: Run) -> list[str]:
    """Reject unbound evidence; retain comparability limits for a descriptive report."""
    issues: list[str] = []
    declaration = run.source.declaration.declaration
    if (
        hashlib.sha256((declaration.config_json or "").encode()).hexdigest()
        != declaration.config_sha256
    ):
        raise ValueError(f"{arm}: declaration config digest differs")
    logged = LoggedConfig.model_validate_json(declaration.config_json or "").training
    expected = JobConfig.model_validate_json(job.job.config or "").regime
    if (
        receipt.regime != expected
        or logged.regime != expected
        or logged.run_id != receipt.id
        or logged.identities != receipt.identities
        or receipt.metrics_run != run.source.run
        or declaration.seed != SEED
        or receipt.seed != SEED
        or receipt.identities.get("source_commit") != SOURCES[arm]
    ):
        raise ValueError(f"{arm}: capture/training/prepared identities disagree")
    if (
        len(run.source.attempts) != 1
        or run.source.attempts[0].resumed_at
        or receipt.parent_run_id is not None
    ):
        raise ValueError(f"{arm}: comparison requires one fresh attempt")
    if len(receipt.stages) != 1:
        raise ValueError(f"{arm}: comparison requires one training stage")
    stage = receipt.stages[0]
    if (
        receipt.status != "completed"
        or stage.status != "completed"
        or stage.updates != UPDATES
        or run.update[-1] != UPDATES
    ):
        issues.append(f"{arm}: incomplete scientific target or metric capture")
    if stage.actual_device != "cuda" or stage.actual_threads != 1:
        issues.append(f"{arm}: device/thread mismatch")
    if stage.updates == run.update[-1] and (
        stage.games != run.metrics["coordinates/games"][-1]
        or stage.learner_transitions != np.sum(run.metrics["rows"])
    ):
        issues.append(f"{arm}: receipt and metric completed work disagree")
    if (
        receipt.host is None
        or receipt.host.cpu_model is None
        or receipt.host.schedulable_cpus is None
    ):
        issues.append(f"{arm}: missing host CPU evidence")
    if receipt.game_errors is None or receipt.game_errors.attempted <= 0:
        issues.append(f"{arm}: missing game-error evidence")
    if receipt.recovery_seconds != 0:
        issues.append(f"{arm}: recovery occurred")
    return issues


def _pair_limitations(before: Receipt, after: Receipt, plan: Analysis) -> list[str]:
    issues: list[str] = []
    if before.host != after.host:
        issues.append("host CPU/quota/memory facts differ; descriptive comparison only")
    for key in (
        "world",
        "engine_source_sha256",
        "content_digest",
        "observation_abi_sha256",
        "action_abi_sha256",
        "deck_source_sha256",
        "architecture",
        "torch",
        "cuda",
    ):
        if key not in before.identities or before.identities.get(
            key
        ) != after.identities.get(key):
            issues.append(f"runtime identity differs or is unavailable: {key}")
    cuda = before.identities.get("cuda")
    if not isinstance(cuda, dict) or cuda.get("device") != "NVIDIA L4":
        issues.append("runtime GPU is not the declared NVIDIA L4")
    if before.game_errors is not None and after.game_errors is not None:
        before_rate = before.game_errors.crashed / max(1, before.game_errors.attempted)
        after_rate = after.game_errors.crashed / max(1, after.game_errors.attempted)
        if abs(before_rate - after_rate) > plan.error_rate_difference:
            issues.append("game error rates differ by more than one percentage point")
    return issues


@dataclass(frozen=True)
class ArmAnalysis:
    statistics: dict[str, Statistics]
    growth_scores: dict[str, float]
    limitations: list[str]


def _analyze(run: Reading, plan: Analysis, end: int) -> ArmAnalysis:
    result = ArmAnalysis(statistics={}, growth_scores={}, limitations=[])
    rows = run.metrics["rows"]
    for metric in PHASES.values():
        if metric not in run.metrics:
            result.limitations.append(f"{run.name}: missing phase {metric}")
    series = {
        key: value
        for key, value in run.metrics.items()
        if key.startswith("time/collection") or key == "time/learning"
    }
    series.update(
        {f"{key}/per_1000_rows": value / rows * 1000 for key, value in series.items()}
    )
    series.update(
        {
            "rss_gb": run.metrics["coordinates/rss_bytes"] / 1e9,
            "rows": rows,
            "games": run.games,
            "decisions_per_game": run.decisions_per_game,
        }
    )
    for key, values in series.items():
        try:
            result.statistics[key] = _statistics(run, values, plan, end)
        except ValueError as error:
            result.limitations.append(f"{run.name}/{key}: {error}")
    for key, kind in (
        ("time/collection", "time"),
        ("time/collection/per_1000_rows", "time"),
        ("rss_gb", "rss"),
    ):
        if key in result.statistics:
            result.growth_scores[key] = _growth_score(
                result.statistics[key], kind, plan
            )
        else:
            result.limitations.append(f"{run.name}: missing primary {key}")
    return result


def compare(prepared: Path, data: Path, out: Path) -> None:
    """Write a new report; never overwrite prepared inputs or measured evidence."""
    contract, jobs = prepared_runs(prepared)
    plan = contract.analysis
    runs = {
        arm: reading(
            arm, Capture.model_validate_json((data / f"{arm}.json").read_bytes())
        )
        for arm in SOURCES
    }
    receipts = {
        arm: Receipt.model_validate_json((data / f"{arm}-training.json").read_bytes())
        for arm in SOURCES
    }
    issues = [
        issue
        for arm in SOURCES
        for issue in _validate_arm(arm, runs[arm], receipts[arm], jobs[arm])
    ]
    issues.extend(_pair_limitations(receipts["before"], receipts["after"], plan))
    end = int(min(plan.last, *(r.update[-1] for r in runs.values())))
    analyses = {arm: _analyze(run, plan, end) for arm, run in runs.items()}
    for analysis in analyses.values():
        issues.extend(analysis.limitations)
    for key in ("rows", "games", "decisions_per_game"):
        if all(key in analyses[arm].statistics for arm in SOURCES):
            a, b = (analyses[arm].statistics[key].mean for arm in SOURCES)
            if a <= 0 or abs(b / a - 1) > plan.work_difference_fraction:
                issues.append(f"completed work differs by more than 10%: {key}")
    scores = {arm: analysis.growth_scores for arm, analysis in analyses.items()}
    result = {
        "analysis": plan.model_dump(),
        "common_last_update": end,
        "verdict": _verdict(scores, issues, plan),
        "limitations": issues,
        "scope": "Descriptive single-seed early-growth comparison; no later-regime or causal claim.",
        "growth_scores": scores,
        "arms": {
            arm: {
                "run": runs[arm].source.run,
                "receipt": receipt.model_dump(mode="json"),
                "statistics": {
                    key: asdict(value)
                    for key, value in analyses[arm].statistics.items()
                },
            }
            for arm, receipt in receipts.items()
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(result["verdict"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare(args.prepared, args.data, args.out)


if __name__ == "__main__":
    main()
