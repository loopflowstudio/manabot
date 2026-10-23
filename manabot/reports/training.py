"""Read retained training facts into Etude metric runs and portable dashboards.

VerifyStore or a retained run.json owns training facts. Settled rows are appended
once, checked against Etude's durable row digests on retry. In-progress evaluation
stays in the summary until settled. Etude owns metric storage and delivery.
"""

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import time
from typing import Literal

from etude.run.storage import storage_from_uri
from etude.train.read import read
from etude.train.records import Step
from etude.train.run import seal_runs, track
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from manabot.arena.models import canonical_sha256
from manabot.arena.monitor import MonitorResult
from manabot.experiment.artifacts import ArtifactManifest
from manabot.infra.artifacts import verify_file
from manabot.infra.files import atomic_json
from manabot.reports.records import ReportRun, RetainedRun
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


def training_dashboard(run: ReportRun) -> Dashboard:
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
                        "optimizer_updates",
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
                if diagnostic.get("rows") and isinstance(
                    diagnostic.get("retained"), (int, float)
                ):
                    row["rl/retained_fraction"] = (
                        diagnostic["retained"] / diagnostic["rows"]
                    )
            numerical = diagnostic.get("numerical")
            if isinstance(numerical, dict):
                for name, value in numerical.items():
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        row[f"numerical/{name}"] = value
            rows.append(row)
        if stage.numerical_failure is not None:
            failure = stage.numerical_failure
            rows.append(
                {
                    "progress/observation": len(rows),
                    "stage/id": stage.id,
                    "stage/operation": specification.operation,
                    "progress/updates": failure.iteration - 1,
                    "failure/invariant": failure.invariant,
                    **{
                        f"numerical/{key}": value
                        for key, value in failure.health.items()
                    },
                }
            )
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
            "checkpoint_interval_updates": run.monitoring_checkpoint_updates,
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


def _recorded_rows(directory: Path, experiment: str, run_id: str) -> dict[int, str]:
    """Read acknowledged row identities from Etude, including sealed crash tails."""
    recorded: dict[int, str] = {}
    for table in read(directory, experiment, run_id).tables:
        for index in range(table.rows):
            row = table.row(index)
            values, clocks = row["values"], row["clocks"]
            if not isinstance(values, dict) or not isinstance(clocks, dict):
                raise ValueError("invalid Etude projection row")
            digest = values.get("projection/row_sha256")
            if digest is None:
                continue  # Summary observations are not settled history rows.
            ordinal = clocks.get("observation")
            if not isinstance(digest, str) or not isinstance(ordinal, (int, float)):
                raise ValueError("invalid Etude projection identity")
            if not math.isfinite(ordinal) or ordinal < 0 or int(ordinal) != ordinal:
                raise ValueError("invalid Etude projection ordinal")
            position = int(ordinal)
            if position in recorded and recorded[position] != digest:
                raise ValueError("Etude projection has conflicting row identities")
            recorded[position] = digest
    if set(recorded) != set(range(len(recorded))):
        raise ValueError("Etude projection history is not a contiguous prefix")
    return recorded


def publish_dashboard(
    dashboard: Dashboard,
    out: Path,
    *,
    experiment: str = "manabot-monitoring",
    destination: str | None = None,
) -> str:
    """Append a retained projection to Etude and return its local run URI.

    A settled row cannot be rewritten. Retry reads Etude's actual durable rows;
    no speculative local cursor advances before persistence. A destination is
    an explicit Etude directory or S3 prefix, never an implicit cloud upload.
    """
    directory = out / "etude"
    # Separate per-projection locks also cover callers without a publisher lease.
    with attempt_lock(out / "etude-projection.lock"):
        seal_runs(directory)
        recorded = _recorded_rows(directory, experiment, dashboard.run_id)
        rows = dashboard.rows
        for index, row in enumerate(rows):
            if row.get("monitor/status") == "running":
                rows = rows[:index]
                break
        if len(recorded) > len(rows) or any(
            canonical_sha256(rows[index]) != digest
            for index, digest in recorded.items()
        ):
            raise ValueError("retained projection changed an already published prefix")
        tracked = track(
            experiment=experiment,
            run=dashboard.run_id,
            kind="retained-monitoring",
            clocks=[Step("observation", "retained epoch/update ordinal")],
            config=dashboard.config,
            directory=directory,
            storages=[storage_from_uri(destination)] if destination else (),
            strict=True,
        )
        try:
            for index in range(len(recorded), len(rows)):
                tracked.log(
                    {
                        **rows[index],
                        "projection/row_sha256": canonical_sha256(rows[index]),
                    },
                    observation=index,
                )
            tracked.log(
                {
                    "projection/summary_json": json.dumps(
                        dashboard.summary, sort_keys=True
                    )
                },
                observation=max(0, len(rows) - 1),
            )
        except Exception:
            tracked.finish("failed")
            raise
        tracked.finish()
        if tracked.dropped or tracked.incomplete or tracked.problems:
            raise RuntimeError(
                "Etude delivery incomplete; retained files can be retried"
            )
        if len(_recorded_rows(directory, experiment, dashboard.run_id)) != len(rows):
            raise RuntimeError("Etude did not retain every projection row")
        return tracked.files.root.resolve().as_uri()


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
    parser.add_argument("--experiment", default="manabot-monitoring")
    parser.add_argument("--destination", help="Etude directory or s3://bucket/prefix")
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        help="S3 locations for this exact --run export; metadata only",
    )
    parser.add_argument("--watch-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.run_id and not args.store:
        parser.error("--run-id requires --store")
    if args.artifact_manifest and not args.run:
        parser.error("--artifact-manifest requires --run")
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
                run = RetainedRun.model_validate_json(args.run.read_text())
                dashboard = training_dashboard(run)
                status = run.status
            else:
                with VerifyStore(args.store) as store:
                    run = store.training_run(args.run_id)
                dashboard = training_dashboard(run)
                status = run.status
            if args.artifact_manifest:
                manifest = ArtifactManifest.model_validate_json(
                    args.artifact_manifest.read_text()
                )
                verify_file(
                    args.run, manifest.source_run.sha256, manifest.source_run.bytes
                )
                if manifest.training_run_id != run.id:
                    raise ValueError(
                        "artifact manifest belongs to another training run"
                    )
                dashboard.summary["artifact_storage"] = manifest.model_dump(mode="json")
            atomic_json(args.out / "dashboard.json", dashboard.model_dump(mode="json"))
            try:
                url = publish_dashboard(
                    dashboard,
                    args.out,
                    experiment=args.experiment,
                    destination=args.destination,
                )
                atomic_json(
                    args.out / "delivery.json",
                    {
                        "status": "delivered" if args.destination else "recorded",
                        "url": url,
                    },
                )
            except Exception as error:
                # Do not dump SDK errors, which can contain credential URLs.
                atomic_json(
                    args.out / "delivery.json",
                    {"status": "failed", "error_type": type(error).__name__},
                )
                if not args.watch_seconds:
                    raise SystemExit(
                        "Etude delivery failed; local dashboard retained (see delivery.json)"
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
