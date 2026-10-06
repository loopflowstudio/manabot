"""Evaluate saved checkpoints on the monitoring cohort without invoking training."""

import argparse
import json
import math
from pathlib import Path
import time

from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRun
from manabot.training.monitor_evaluation import (
    MonitorProtocol,
    MonitorResult,
    TrainingCoordinates,
    evaluate_checkpoint,
    import_saved_rows,
    stage_checkpoint,
)
from manabot.training.monitoring import evaluation_dashboard, publish_dashboard
from manabot.training.recovery import attempt_lock


def follow_checkpoints(
    source: Path,
    out: Path,
    protocol: MonitorProtocol,
    *,
    poll_seconds: float,
    concurrent_activity: str,
    online: bool = False,
    project: str = "manabot",
    entity: str | None = None,
) -> None:
    """Evaluate every admitted hourly export once; stopped attempts are never replaced."""
    out.mkdir(parents=True, exist_ok=True)
    with attempt_lock(out / "evaluator.lock"):
        while True:
            run = TrainingRun.model_validate_json(source.read_text())
            binding = {"run_id": run.id, "protocol": protocol.model_dump(mode="json")}
            manifest = out / "cohort.json"
            if manifest.exists():
                if json.loads(manifest.read_text()) != binding:
                    raise ValueError(
                        "monitor directory is bound to another run or protocol"
                    )
            else:
                atomic_json(manifest, binding)
            for checkpoint in run.monitoring_checkpoints:
                directory = out / f"checkpoint-{checkpoint.ordinal:08d}"
                if (
                    directory.exists()
                    or checkpoint.artifact is None
                    or checkpoint.error
                ):
                    continue
                coordinates = TrainingCoordinates.model_validate(
                    checkpoint.model_dump(exclude={"ordinal", "artifact", "error"})
                )
                try:
                    evaluate_checkpoint(
                        run,
                        checkpoint.artifact,
                        coordinates,
                        directory,
                        protocol=protocol,
                        concurrent_activity=concurrent_activity,
                    )
                except Exception:
                    # evaluate_checkpoint owns the admission-failure receipt.
                    # The next checkpoint is independent; this one is not retried.
                    if not (directory / "admission-failure.json").exists():
                        raise
            results = [
                MonitorResult.model_validate_json(p.read_text())
                for p in sorted(out.glob("checkpoint-*/monitor.json"))
            ]
            if results:
                dashboard = evaluation_dashboard(results)
                dashboard.summary["admission_failures"] = [
                    json.loads(p.read_text())
                    for p in sorted(out.glob("checkpoint-*/admission-failure.json"))
                ]
                atomic_json(out / "dashboard.json", dashboard.model_dump(mode="json"))
                if online:
                    try:
                        url = publish_dashboard(
                            dashboard, out, project=project, entity=entity
                        )
                        atomic_json(
                            out / "delivery.json", {"status": "uploaded", "url": url}
                        )
                    except Exception as error:
                        atomic_json(
                            out / "delivery.json",
                            {"status": "failed", "error_type": type(error).__name__},
                        )
            if run.status not in {"running", "pending"}:
                return
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="TrainingRun JSON export")
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0,
        help="Follow new monitoring exports in a separate process",
    )
    parser.add_argument(
        "--online", action="store_true", help="Publish the watched evaluation dashboard"
    )
    parser.add_argument("--project", default="manabot")
    parser.add_argument("--entity")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--checkpoint", type=int, help="Monitoring checkpoint ordinal"
    )
    selection.add_argument("--stage", help="Completed stage raw checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="New attempt directory")
    parser.add_argument(
        "--protocol",
        type=Path,
        help="Frozen MonitorProtocol JSON; defaults to 25 deals / 100 games",
    )
    parser.add_argument(
        "--concurrent-activity", default="unknown; no contention correction"
    )
    parser.add_argument(
        "--import-manifest",
        type=Path,
        help="Saved MonitorResult with original coordinates and registrations",
    )
    parser.add_argument(
        "--import-rows", type=Path, help="Saved arena rows.json; never plays games"
    )
    args = parser.parse_args()
    if args.watch_seconds < 0 or not math.isfinite(args.watch_seconds):
        parser.error("watch interval must be finite and nonnegative")
    if args.watch_seconds:
        if (
            not args.run
            or args.stage
            or args.checkpoint is not None
            or args.import_manifest
            or args.import_rows
        ):
            parser.error(
                "watch requires only --run, without a checkpoint selection or import"
            )
        follow_checkpoints(
            args.run,
            args.out,
            MonitorProtocol.model_validate_json(args.protocol.read_text())
            if args.protocol
            else MonitorProtocol(),
            poll_seconds=args.watch_seconds,
            concurrent_activity=args.concurrent_activity,
            online=args.online,
            project=args.project,
            entity=args.entity,
        )
        return
    if args.online:
        parser.error(
            "--online requires --watch-seconds; saved attempts use manabot.training.monitoring"
        )
    if args.import_manifest or args.import_rows:
        if not (args.import_manifest and args.import_rows) or args.run or args.protocol:
            parser.error(
                "import requires manifest and rows, without --run or --protocol"
            )
        result = import_saved_rows(
            MonitorResult.model_validate_json(args.import_manifest.read_text()),
            args.import_rows,
            args.out,
        )
    else:
        if not args.run or (args.checkpoint is None and args.stage is None):
            parser.error("evaluation requires --run and --checkpoint or --stage")
        run = TrainingRun.model_validate_json(args.run.read_text())
        if args.checkpoint is not None:
            receipt = next(
                (c for c in run.monitoring_checkpoints if c.ordinal == args.checkpoint),
                None,
            )
            if receipt is None or receipt.artifact is None or receipt.error:
                parser.error("monitoring checkpoint is absent or failed admission")
            artifact = receipt.artifact
            coordinates = TrainingCoordinates.model_validate(
                receipt.model_dump(exclude={"ordinal", "artifact", "error"})
            )
        else:
            checkpoint = stage_checkpoint(run, args.stage)
            if checkpoint is None:
                parser.error(
                    "stage lacks a completed raw checkpoint with original elapsed coordinate"
                )
            artifact, coordinates = checkpoint.artifact, checkpoint.coordinates
        protocol = (
            MonitorProtocol.model_validate_json(args.protocol.read_text())
            if args.protocol
            else MonitorProtocol()
        )
        result = evaluate_checkpoint(
            run,
            artifact,
            coordinates,
            args.out,
            protocol=protocol,
            concurrent_activity=args.concurrent_activity,
        )
    print(
        f"{result.status}: {len(result.rows)}/{result.expected_games} attempted games; {args.out / 'monitor.json'}"
    )
    if result.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
