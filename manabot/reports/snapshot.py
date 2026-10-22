"""Make an offline report snapshot without opening a live store for writing.

SQLite's online backup owns consistency of run, stage and diagnostic rows. Atomic
monitor JSON exports are copied separately; the receipt states that capture window
rather than claiming a transaction across the filesystem and database. Checkpoint
bytes, credentials and training control files are deliberately outside this export.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from manabot.arena.models import file_sha256
from manabot.verify.store import training_stage_payloads


def export_database(database: Path, output: Path) -> None:
    """Export exact stored payloads from an immutable backup; preserve extension fields."""
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as con:
        con.execute("BEGIN")
        for run_id, text in con.execute("SELECT id, payload FROM training_runs"):
            payload = json.loads(text)
            payload["stages"] = training_stage_payloads(con, run_id)
            # IDs are not trusted filesystem paths.
            path = output / "training" / _directory_key(run_id) / "run.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2))
        for execution_id, text in con.execute(
            "SELECT id, payload FROM experiment_runs"
        ):
            path = (
                output / "executions" / _directory_key(execution_id) / "experiment.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)


def _directory_key(value: str) -> str:
    """Use a stable directory key without changing the identity in the payload."""
    return hashlib.sha256(value.encode()).hexdigest()


def snapshot_report(source: Path, output: Path) -> Path:
    """Create once; never mutate source or overwrite an earlier snapshot."""
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    database = output / "source.sqlite"
    with sqlite3.connect(
        f"{(source / 'experiment.sqlite').resolve().as_uri()}?mode=ro", uri=True
    ) as live:
        with sqlite3.connect(database) as backup:
            live.backup(backup, pages=128, sleep=0.05)
    export_database(database, output)
    for name in ("monitor.json", "attempt.json"):
        for path in sorted((source / "monitoring").rglob(name)):
            destination = output / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    files = [
        {"path": str(p.relative_to(output)), "sha256": file_sha256(p)}
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.suffix != ".sqlite"
    ]
    receipt = {
        "source_root": str(source.resolve()),
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "Read-only SQLite online backup; atomic monitoring files captured separately in this interval",
        "database_sha256": file_sha256(database),
        "files": files,
    }
    (output / "snapshot.json").write_text(json.dumps(receipt, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(snapshot_report(args.source, args.output))


if __name__ == "__main__":
    main()
