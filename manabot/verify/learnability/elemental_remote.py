"""Read MTG-174's verified remote snapshots across the Etude SDK boundary.

The isolated evaluation owner invokes this module with Manabot's environment.
It can observe or cancel its one named job; it never allocates a machine.
"""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path, PurePosixPath
import time

from manabot.experiment.evaluation_queue import checkpoints
from manabot.remote.job_client import (
    cancel_job,
    fetch_job_file,
    job_manifest,
    job_status,
    load_job,
)
from manabot.remote.job_store import S3JobStore
from manabot.remote.jobs import Deletion, Resource
from manabot.training.records import TrainingRun


def observe(job_id: str, cache: Path) -> dict[str, object]:
    spec = load_job(job_id)
    status = job_status(spec)
    store = S3JobStore(spec.prefix)
    cumulative = 0.0
    for purpose in ("guardian", "training"):
        resource = store.read(f"{purpose}.json")
        if resource is None:
            continue
        deletion = store.read(f"{purpose}-deletion.json")
        if deletion is not None:
            cumulative += Deletion.model_validate_json(deletion.data).estimated_dollars
        else:
            rented = Resource.model_validate_json(resource.data)
            cumulative += (
                (time.time() - rented.claim.intent_time)
                * (rented.pod.rate + spec.plan.spec.machine.storage_hourly_allowance)
                / 3600
            )
    result: dict[str, object] = {
        "observed_at": datetime.now(UTC).isoformat(),
        "source": spec.plan.source.model_dump(),
        "status": status.model_dump(mode="json"),
        "checkpoints": [],
        "estimated_cumulative_dollars": cumulative,
    }
    record = status.record
    if record is None or record.manifest is None:
        return result
    manifest = job_manifest(spec, record, cache)
    run_path = fetch_job_file(spec, record, "run/run.json", cache)
    if run_path is None:
        return result
    run = TrainingRun.model_validate_json(run_path.read_bytes())
    result["run"] = run.model_dump(mode="json")
    found: list[dict[str, object]] = []
    for checkpoint in checkpoints(run, include_initial=True):
        reference = checkpoint.artifact
        entries = [
            entry
            for entry in manifest.bundle.files
            if entry.sha256 == reference["sha256"]
            and entry.size == reference["bytes"]
            and PurePosixPath(entry.producer_path) == PurePosixPath(reference["path"])
        ]
        if len(entries) != 1:
            raise ValueError("checkpoint missing or ambiguous in verified generation")
        path = fetch_job_file(spec, record, entries[0].relative_path, cache)
        if path is None:
            raise ValueError("published checkpoint unavailable")
        found.append(
            {
                "path": str(path.resolve()),
                "sha256": reference["sha256"],
                "coordinates": checkpoint.coordinates.model_dump(mode="json"),
            }
        )
    result["checkpoints"] = found
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cancel", action="store_true")
    args = parser.parse_args()
    if args.cancel:
        cancel_job(load_job(args.job_id))
    print(json.dumps(observe(args.job_id, args.cache)))


if __name__ == "__main__":
    main()
