"""Read-only TrainingRun projection and resumable W&B dashboard delivery.

VerifyStore (or a retained run.json export) owns training facts. This module
never trains, evaluates, edits a recipe or rewrites historical coordinates.
Offline mode writes portable JSON; later online backfill uses the same run ID
and W&B's acknowledged next step rather than a speculative local upload cursor.
"""

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import time
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue
import wandb

from manabot.arena.models import canonical_sha256
from manabot.infra.experiment import flatten_config
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRun
from manabot.training.monitor_evaluation import MonitorResult
from manabot.training.recovery import attempt_lock
from manabot.verify.store import VerifyStore

Scalar = int | float | str | bool


class Dashboard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    run_id: str
    config: dict[str, JsonValue]
    summary: dict[str, JsonValue]
    rows: list[dict[str, Scalar]] = Field(default_factory=list)


class HistorySink(Protocol):
    @property
    def step(self) -> int: ...

    def log(self, data: dict[str, object], *, step: int) -> None: ...


def append_history(
    dashboard: Dashboard, sink: HistorySink, panels: dict[str, object] | None = None
) -> None:
    """Publish only the suffix not already acknowledged by the remote run."""
    start = sink.step
    if start > len(dashboard.rows):
        raise ValueError("remote history is longer than this retained snapshot")
    for index in range(start, len(dashboard.rows)):
        payload: dict[str, object] = dict(dashboard.rows[index])
        if index == len(dashboard.rows) - 1:
            payload.update(panels or {})
        sink.log(payload, step=index)


def default_panels(dashboard: Dashboard) -> dict[str, object]:
    """Ready-to-use W&B panels containing measured samples, never invented zeros."""
    groups = {
        "Teacher cross-entropy (nats)": [
            f"distillation/{name}_cross_entropy"
            for name in ("train", "growing", "fixed")
        ],
        "Teacher KL (nats)": [
            f"distillation/{name}_teacher_kl" for name in ("train", "growing", "fixed")
        ],
        "RL clipped policy objective (not log loss)": ["rl/policy_loss"],
        "RL value loss": ["rl/value_loss"],
        "RL regularization KL": ["rl/collection_kl", "rl/reference_kl"],
        "RL entropy": ["rl/entropy"],
        "Advantage retention": ["rl/retained_fraction"],
        "Lambda value residual": ["rl/value_residual_abs_mean"],
        "Learning rate": ["rl/learning_rate"],
        "Regularization schedules": ["rl/tau", "rl/collection_kl_coefficient"],
    }
    panels: dict[str, object] = {}
    for title, metrics in groups.items():
        series = [
            (
                metric,
                [
                    (r["progress/observation"], r[metric])
                    for r in dashboard.rows
                    if isinstance(r.get(metric), (int, float))
                ],
            )
            for metric in metrics
        ]
        series = [(metric, points) for metric, points in series if points]
        if series:
            panels["dashboard/" + title] = wandb.plot.line_series(
                xs=[[p[0] for p in points] for _, points in series],
                ys=[[p[1] for p in points] for _, points in series],
                keys=[metric for metric, _ in series],
                title=title,
                xname="Recorded epoch/update ordinal",
            )
    for axis in ("updates", "training_seconds"):
        for name in ("win", "draw", "score"):
            complete = [r for r in dashboard.rows if f"monitor/{name}/mean" in r]
            if complete:
                title = f"Monitoring {name} and deal-cluster 95% interval vs {axis}"
                panels["dashboard/" + title] = wandb.plot.line_series(
                    xs=[r[f"progress/{axis}"] for r in complete],
                    ys=[
                        [r[f"monitor/{name}/{bound}"] for r in complete]
                        for bound in ("mean", "lower", "upper")
                    ],
                    keys=["mean", "lower", "upper"],
                    title=title,
                    xname=axis,
                )
    return panels


def _scalars(value: Mapping[str, object], prefix: str = "") -> dict[str, Scalar]:
    result: dict[str, Scalar] = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, Mapping):
            result.update(_scalars(item, name))
        elif isinstance(item, (int, float, str, bool)):
            if not isinstance(item, float) or math.isfinite(item):
                result[name] = item
    return result


def training_dashboard(run: TrainingRun) -> Dashboard:
    rows: list[dict[str, Scalar]] = []
    missing_coordinates = 0
    for stage in run.stages:
        specification = next(s for s in run.regime.stages if s.id == stage.id)
        for ordinal, diagnostic in enumerate(stage.diagnostics):
            row: dict[str, Scalar] = {
                "progress/observation": len(rows),
                "stage/id": stage.id,
                "stage/operation": specification.operation,
                "stage/ordinal": ordinal,
            }
            coordinate = diagnostic.get("coordinates")
            if isinstance(coordinate, dict):
                row.update(_scalars(coordinate, "progress"))
                elapsed = coordinate.get("training_seconds")
                if isinstance(elapsed, (int, float)) and elapsed > 0:
                    for counter in (
                        "environment_decisions",
                        "learner_transitions",
                        "optimizer_exposures",
                        "games",
                    ):
                        count = coordinate.get(counter)
                        if isinstance(count, (int, float)):
                            row[f"throughput/{counter}_per_second"] = count / elapsed
                    admitted = [
                        c.training_seconds
                        for c in run.monitoring_checkpoints
                        if c.artifact is not None and c.training_seconds <= elapsed
                    ]
                    if admitted:
                        row["progress/checkpoint_age_seconds"] = elapsed - max(admitted)
            else:
                missing_coordinates += 1
            row["availability/original_coordinates"] = isinstance(coordinate, dict)
            if specification.operation == "train_supervised":
                row["objective/meaning"] = "teacher cross-entropy (nats)"
                row["objective/training_target"] = specification.target
                if run.fixed_validation is not None:
                    row["objective/fixed_validation_target"] = (
                        run.fixed_validation.policy_target_kind
                    )
                mapping = {
                    "train_policy_loss": "distillation/train_cross_entropy",
                    "train_policy_kl": "distillation/train_teacher_kl",
                    "validation/policy_loss": "distillation/growing_cross_entropy",
                    "validation/policy_kl": "distillation/growing_teacher_kl",
                    "fixed_validation/policy_loss": "distillation/fixed_cross_entropy",
                    "fixed_validation/policy_kl": "distillation/fixed_teacher_kl",
                }
                flattened = _scalars(diagnostic)
                for source, target in mapping.items():
                    row[f"availability/{target}"] = source in flattened
                    if source in flattened:
                        row[target] = flattened[source]
            elif specification.operation in {"train_self_play", "train_compound"}:
                row["objective/meaning"] = "clipped policy objective; not log loss"
                for metric in (
                    "loss",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "collection_kl",
                    "reference_kl",
                    "value_residual_abs_mean",
                    "advantage_abs_mean",
                    "learning_rate",
                    "tau",
                    "collection_kl_coefficient",
                    "retained",
                    "rows",
                    "schedule_progress",
                    "behavior_iteration",
                ):
                    value = diagnostic.get(metric)
                    row[f"availability/rl/{metric}"] = isinstance(value, (float, int))
                    if isinstance(value, (float, int)) and math.isfinite(value):
                        row[f"rl/{metric}"] = value
                if diagnostic.get("rows"):
                    row["rl/retained_fraction"] = (
                        diagnostic.get("retained", 0) / diagnostic["rows"]
                    )
            rows.append(row)
    return Dashboard(
        run_id="training-" + canonical_sha256({"run": run.id})[:24],
        config={
            "training_run_id": run.id,
            "regime": run.regime.model_dump(mode="json"),
            "regime_digest": run.regime_digest,
            "seed": run.seed,
            "seed_streams": run.seed_streams,
            "identities": run.identities,
            "purpose": "training-monitoring-not-scientific-evaluation",
        },
        summary={
            "parent_run_id": run.parent_run_id,
            "status": run.status,
            "error": run.error,
            "history_incomplete": bool(missing_coordinates),
            "rows_without_original_coordinates": missing_coordinates,
            "fixed_validation": run.fixed_validation.model_dump(mode="json")
            if run.fixed_validation
            else None,
            "checkpoint_interval_seconds": run.monitoring_checkpoint_seconds,
            "stages": [
                s.model_dump(mode="json", exclude={"diagnostics"}) for s in run.stages
            ],
            "monitoring_checkpoints": [
                c.model_dump(mode="json") for c in run.monitoring_checkpoints
            ],
            "elapsed_seconds": run.prior_seconds + run.seconds,
            "monitoring_export_seconds": run.monitoring_export_seconds,
            "metric_limits": "RL loss/KL/entropy are last optimized minibatch values; residuals use lambda targets. Missing values are unavailable, never zero. Historical stage totals do not reconstruct update coordinates.",
        },
        rows=rows,
    )


def publish_dashboard(
    dashboard: Dashboard,
    out: Path,
    *,
    project: str = "manabot",
    entity: str | None = None,
) -> str | None:
    """Resume one remote projection. Caller holds a single publisher lease.

    Network failure leaves local JSON intact. There is no offline W&B resume
    assumption: offline JSON is backfilled online using the remote step cursor.
    """
    tracked = wandb.init(
        project=project,
        entity=entity,
        id=dashboard.run_id,
        resume="allow",
        group=str(dashboard.config.get("regime_digest", dashboard.run_id)),
        name=f"{dashboard.config.get('training_run_id', dashboard.run_id)}",
        job_type="training-monitor",
        dir=str(out),
        config=flatten_config(dashboard.config),
        mode="online",
        settings=wandb.Settings(init_timeout=30),
    )
    try:
        acknowledged = tracked.summary.get("projection_rows", 0)
        digest = tracked.summary.get("projection_sha256")
        if digest is not None and (
            acknowledged > len(dashboard.rows)
            or canonical_sha256(dashboard.rows[:acknowledged]) != digest
        ):
            raise ValueError("retained projection changed an already published prefix")
        # Default workspace sections are generated by metric prefixes. Each
        # metric also has an elapsed-time copy for comparison across recipes.
        tracked.define_metric("progress/observation", hidden=True)
        tracked.define_metric("progress/training_seconds")
        for family in ("distillation/*", "rl/*", "progress/*", "throughput/*"):
            tracked.define_metric(family, step_metric="progress/observation")
        tracked.define_metric("monitor/*", step_metric="progress/updates")
        tracked.define_metric("elapsed/*", step_metric="progress/training_seconds")
        tracked.define_metric("availability/*", hidden=True)
        rows = []
        for source in dashboard.rows:
            row = dict(source)
            if "progress/training_seconds" in row:
                row.update(
                    {
                        f"elapsed/{k}": v
                        for k, v in source.items()
                        if k.startswith(("distillation/", "rl/", "monitor/"))
                    }
                )
            rows.append(row)
        append_history(
            dashboard.model_copy(update={"rows": rows}),
            tracked,
            default_panels(dashboard) if tracked.step < len(rows) else None,
        )
        tracked.summary.update(dashboard.summary)
        tracked.summary.update(
            {
                "projection_rows": len(dashboard.rows),
                "projection_sha256": canonical_sha256(dashboard.rows),
            }
        )
        return tracked.url
    finally:
        tracked.finish()


def evaluation_dashboard(results: list[MonitorResult]) -> Dashboard:
    """Separate monitor stream permits late evaluation without rewriting training."""
    if not results:
        raise ValueError("at least one saved monitor result is required")
    first = results[0]
    binding = (
        first.run_id,
        first.protocol,
        first.opponent,
        first.key,
        first.evaluation_identities,
    )
    if any(
        (r.run_id, r.protocol, r.opponent, r.key, r.evaluation_identities) != binding
        for r in results
    ):
        raise ValueError(
            "monitor curves require the same run, frozen protocol and opponent"
        )
    rows: list[dict[str, Scalar]] = []
    for result in results:
        row: dict[str, Scalar] = {
            "progress/observation": len(rows),
            **_scalars(result.coordinates.model_dump(), "progress"),
            "monitor/status": result.status,
            "monitor/checkpoint_sha256": result.artifact["sha256"],
            "monitor/attempted_games": len(result.rows),
            "monitor/expected_games": result.expected_games,
            "availability/monitor_rates": result.status == "completed",
        }
        for name in ("win", "draw", "score"):
            rate = getattr(result, name)
            if rate is not None:
                row.update(_scalars(rate.model_dump(), f"monitor/{name}"))
        if result.evaluation_seconds is not None:
            row["monitor/evaluation_seconds"] = result.evaluation_seconds
        rows.append(row)
    return Dashboard(
        run_id="monitor-"
        + canonical_sha256(
            {
                "run": first.run_id,
                "protocol": first.protocol.model_dump(),
                "opponent": first.opponent.identity_sha256,
            }
        )[:24],
        config={
            "training_run_id": first.run_id,
            "regime_digest": first.regime_digest,
            "seed": first.training_seed,
            "protocol": first.protocol.model_dump(mode="json"),
            "opponent": first.opponent.model_dump(mode="json"),
            "purpose": first.purpose,
        },
        summary={
            "attempts": [
                {
                    **r.model_dump(mode="json", exclude={"rows"}),
                    "attempted_games": len(r.rows),
                    "failed_games": sum(not row.valid for row in r.rows),
                    "failure_reasons": sorted(
                        {row.failure for row in r.rows if row.failure}
                    ),
                }
                for r in results
            ],
            "uncertainty": "95% percentile bootstrap of complete four-leg deals; not independent training seeds",
        },
        rows=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="Retained run.json export")
    source.add_argument("--run-id", help="TrainingRun ID in VerifyStore")
    source.add_argument(
        "--evaluations",
        type=Path,
        nargs="+",
        help="Saved monitor.json attempts in immutable append order",
    )
    parser.add_argument("--store", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--online", action="store_true", help="Upload; default is local/offline"
    )
    parser.add_argument("--project", default="manabot")
    parser.add_argument("--entity")
    parser.add_argument("--watch-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.run_id and not args.store:
        parser.error("--run-id requires --store")
    if args.watch_seconds < 0 or not math.isfinite(args.watch_seconds):
        parser.error("--watch-seconds must be finite and nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    with attempt_lock(args.out / "publisher.lock"):
        while True:
            if args.evaluations:
                dashboard = evaluation_dashboard(
                    [
                        MonitorResult.model_validate_json(p.read_text())
                        for p in args.evaluations
                    ]
                )
                status = "completed"
            elif args.run:
                run = TrainingRun.model_validate_json(args.run.read_text())
                dashboard = training_dashboard(run)
                status = run.status
            else:
                with VerifyStore(args.store) as store:
                    run = store.training_run(args.run_id)
                dashboard = training_dashboard(run)
                status = run.status
            atomic_json(args.out / "dashboard.json", dashboard.model_dump(mode="json"))
            if args.online:
                try:
                    url = publish_dashboard(
                        dashboard, args.out, project=args.project, entity=args.entity
                    )
                    atomic_json(
                        args.out / "delivery.json", {"status": "uploaded", "url": url}
                    )
                except Exception as error:
                    # Do not dump SDK errors, which can contain credential URLs.
                    atomic_json(
                        args.out / "delivery.json",
                        {"status": "failed", "error_type": type(error).__name__},
                    )
                    if not args.watch_seconds:
                        raise SystemExit(
                            "W&B upload failed; local dashboard retained (see delivery.json)"
                        ) from None
            print(
                json.dumps(
                    {
                        "run": dashboard.run_id,
                        "status": status,
                        "rows": len(dashboard.rows),
                    }
                ),
                flush=True,
            )
            if not args.watch_seconds or status not in {"pending", "running"}:
                break
            time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
