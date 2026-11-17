"""Make an offline report snapshot without opening a live store for writing.

SQLite's online backup owns consistency of run and stage rows. Atomic monitor
JSON exports and the sealed files of training's Etude runs, which hold self-play
update metrics, are copied separately; the receipt states that capture window
rather than claiming a transaction across the filesystem and database. Checkpoint
bytes, credentials and training control files are deliberately outside this export.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from manabot.infra.identity import file_sha256


def training_stage_payloads(con: sqlite3.Connection, run_id: str) -> list[dict]:
    """Read one saved run's retained stage records in execution order."""
    return [
        json.loads(payload)
        for (payload,) in con.execute(
            "SELECT payload FROM training_stages WHERE run_id=? ORDER BY rowid",
            (run_id,),
        )
    ]


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
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    database = output / "source.sqlite"
    # Open the source before creating the output, so a wrong source leaves
    # no empty snapshot directory behind to refuse the corrected retry.
    with sqlite3.connect(
        f"{(source / 'experiment.sqlite').resolve().as_uri()}?mode=ro", uri=True
    ) as live:
        output.mkdir(parents=True, exist_ok=False)
        with sqlite3.connect(database) as backup:
            live.backup(backup, pages=128, sleep=0.05)
    export_database(database, output)
    copied = [
        path
        for name in ("monitor.json", "attempt.json")
        for path in (source / "monitoring").rglob(name)
    ]
    # Etude writes blocks whole by rename; open logs and partial files are
    # not yet rows a reader would count.
    copied += [
        path
        for path in (source / "training").glob("**/etude/**/*")
        if path.is_file()
        and path.suffix not in {".open", ".partial"}
        and "checkpoints" not in path.relative_to(source).parts
    ]
    for path in sorted(copied):
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
