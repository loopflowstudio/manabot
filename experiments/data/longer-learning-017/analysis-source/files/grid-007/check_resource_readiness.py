"""Exercise long-study resource extraction on real retained short-study records."""

from collections.abc import Callable
from datetime import UTC, datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from learning_resources import RetainedCell, cell_resources, digest, read_intervals

ROOT = Path(".runs/grid-007/grouped-feasibility")
OUT = Path(".runs/grid-007/learning-017/resource-readiness.json")


def requires_rejection(action: Callable[[], object]) -> None:
    try:
        action()
    except ValueError:
        return
    raise AssertionError("Invalid evidence was accepted")


def main() -> None:
    retained = ROOT / "verified-results.json"
    evidence = json.loads(retained.read_text())
    results = []
    sources = {str(retained): digest(retained)}
    groups = sorted({c["group"] for c in evidence["cells"]})
    rejected: list[str] = []
    for group in groups:
        log = ROOT / group / "job.log"
        status = ROOT / group / "results/worker-status.json"
        completed = json.loads(status.read_text())["completed"]
        timing = read_intervals(log, completed)
        sources.update({str(p): digest(p) for p in (log, status)})
        for row in evidence["cells"]:
            if row["group"] != group:
                continue
            size, stream_count = row["name"].rsplit("-s", 1)
            cell = RetainedCell.model_validate(
                {
                    **row,
                    "cell": row["name"],
                    "seed": row["summary"]["seed"],
                    "size": size,
                    "streams": int(stream_count),
                }
            )
            interval = next(i for i in timing.cells if i.cell == cell.cell)
            resource = cell_resources(cell, interval)
            results.append(resource.model_dump(mode="json"))
            run = cell.directory / "study/training/run/run.json"
            sources[str(run)] = digest(run)
            damaged = cell.model_copy(deep=True)
            damaged.numerical_audit.diagnostic_counts.optimizer_steps += 1
            requires_rejection(lambda: cell_resources(damaged, interval))
        with TemporaryDirectory() as directory:
            damaged_log = Path(directory) / "events.log"
            lines = log.read_text().splitlines()
            damaged_log.write_text("\n".join([lines[0], *lines]) + "\n")
            requires_rejection(lambda: read_intervals(damaged_log, completed))
            damaged_log.write_text("\n".join(reversed(lines)) + "\n")
            requires_rejection(lambda: read_intervals(damaged_log, completed))
        rejected.extend([f"{group}:duplicate_event", f"{group}:backwards_clock"])
    if len(results) != 11:
        raise AssertionError("Expected all eleven complete added-screen trajectories")
    output = {
        "status": "passed",
        "observed_at": datetime.now(UTC).isoformat(),
        "scope": "Resource extraction from prior screen013; no protocol017 outcome or rental-cost verification",
        "actual_prior_cells": len(results),
        "actual_prior_groups": len(groups),
        "adam_counter_mismatches_rejected": len(results),
        "invalid_event_logs_rejected": rejected,
        "input_sha256": sources,
        "resources": results,
        "collector_sha256": digest(Path(__file__).with_name("learning_resources.py")),
        "readiness_script_sha256": digest(Path(__file__)),
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: output[k]
                for k in (
                    "status",
                    "scope",
                    "actual_prior_cells",
                    "actual_prior_groups",
                    "adam_counter_mismatches_rejected",
                    "invalid_event_logs_rejected",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
