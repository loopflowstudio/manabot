"""Publish saved monitor results to Etude as evaluation curves.

`evaluation_dashboard` projects saved monitor results, and `publish_dashboard`
appends settled rows once, checked against Etude's durable row digests on
retry. In-progress evaluation stays in the summary until settled.
"""

import argparse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
from typing import Literal

from etude.records import Clock
from etude.run import declare, destinations
from etude.storage import storage_from_uri
from etude.train.read import RunData, read
from etude.train.tracking import TrainingRun, seal_runs
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from manabot.eval.monitor import MonitorResult
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256

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


def _recorded_rows(data: RunData) -> dict[int, str]:
    """Read acknowledged row identities from Etude, including sealed crash tails."""
    recorded: dict[int, str] = {}
    for table in data.columns:
        for index in range(table.rows):
            row = table.row(index)
            values, clocks = row.values, row.clocks
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
    with dashboard_lock(out / "etude-projection.lock"):
        seal_runs(directory)
        prior = read(directory, experiment, dashboard.run_id)
        recorded = _recorded_rows(prior)
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
        declared = declare(
            experiment=experiment,
            run=dashboard.run_id,
            kind="retained-monitoring",
            clocks=(Clock(name="observation", kind="step", meaning="retained row"),),
            config=dashboard.config,
            directory=directory,
            resume=prior.file is not None,
        )
        tracked = TrainingRun(
            experiment=experiment,
            name=declared.name,
            root=declared.root,
            clocks=declared.record.clocks,
            destinations=destinations(
                declared.base, [storage_from_uri(destination)] if destination else ()
            ),
            strict=True,
            closed_clocks=True,
            inherited=None,
            seed=declared.record.seed or 0,
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
                    ),
                },
                # Summary-only retries must not supersede the last settled row.
                observation=len(rows),
            )
        except Exception:
            tracked.finish("failed")
            raise
        tracked.finish()
        if tracked.dropped or tracked.incomplete or tracked.problems:
            raise RuntimeError(
                "Etude delivery incomplete; retained files can be retried"
            )
        if len(_recorded_rows(read(directory, experiment, dashboard.run_id))) != len(
            rows
        ):
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
    parser = argparse.ArgumentParser(
        description="Publish saved monitor results as an Etude evaluation curve."
    )
    parser.add_argument(
        "--evaluations",
        type=Path,
        nargs="+",
        required=True,
        help="Saved monitor.json attempts in immutable append order",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--experiment", default="manabot-monitoring")
    parser.add_argument("--destination", help="Etude directory or s3://bucket/prefix")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with dashboard_lock(args.out / "publisher.lock"):
        dashboard = evaluation_dashboard(
            [MonitorResult.model_validate_json(p.read_text()) for p in args.evaluations]
        )
        atomic_json(args.out / "dashboard.json", dashboard.model_dump(mode="json"))
        try:
            url = publish_dashboard(
                dashboard,
                args.out,
                experiment=args.experiment,
                destination=args.destination,
            )
        except Exception as error:
            # Do not dump SDK errors, which can contain credential URLs.
            atomic_json(
                args.out / "delivery.json",
                {"status": "failed", "error_type": type(error).__name__},
            )
            raise SystemExit(
                "Etude delivery failed; local dashboard retained (see delivery.json)"
            ) from None
        atomic_json(
            args.out / "delivery.json",
            {"status": "delivered" if args.destination else "recorded", "url": url},
        )
        print(json.dumps({"run": dashboard.run_id, "rows": len(dashboard.rows)}))


@contextmanager
def dashboard_lock(path: Path) -> Iterator[None]:
    """Hold a local POSIX publisher lease; process death releases it.

    Never unlink a lock file: replacing its inode could admit two writers.
    """
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("dashboard still has a live publisher") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
