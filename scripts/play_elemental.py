"""Pin the latest published MTG-183 policy, then launch local CPU play.

Uses Etude's checkpoint records and manabot's existing download/admission path.
Only checkpoint discovery needs storage credentials; the play server receives
local bytes and all games in that server keep the same opponent identity.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from urllib.parse import urlencode

from etude.run.storage import storage_from_uri
from etude.train.read import read
from etude.train.records import CheckpointRecord

from manabot.infra.artifacts import verify_file
from manabot.model.catalog import inspect_checkpoint
from manabot.model.checkpoint import fetch_etude_checkpoint, load_checkpoint

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = "f3748bcabac8"


def latest_checkpoint(records: tuple[CheckpointRecord, ...]) -> CheckpointRecord:
    """Select a declared training update, never infer freshness from filenames."""
    candidates = [record for record in records if "updates" in record.at]
    if not candidates:
        raise ValueError("MTG-183 has no published checkpoint with a training update.")
    return max(candidates, key=lambda record: (record.at["updates"], record.wall_ms))


def prepare_checkpoint(directory: Path, source: str, run: str, offline: bool) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "checkpoint.json"
    if offline and not receipt.is_file():
        raise ValueError("No pinned checkpoint; run once with storage access.")
    record = (
        CheckpointRecord.model_validate_json(receipt.read_bytes())
        if offline
        else latest_checkpoint(
            read(storage_from_uri(source), "self-play", run).checkpoints
        )
    )
    digest = record.artifact.sha256
    if not digest or record.artifact.size_bytes is None:
        raise ValueError(
            "The published checkpoint needs a SHA-256 identity and byte count."
        )
    checkpoint = (
        directory / f"MTG-183-update-{int(record.at['updates'])}-{digest[:12]}.pt"
    )
    if checkpoint.exists():
        verify_file(checkpoint, digest, record.artifact.size_bytes)
    elif offline:
        raise ValueError(
            "The pinned checkpoint is missing; run once with storage access."
        )
    else:
        fetch_etude_checkpoint(record.artifact, checkpoint)
    summary = inspect_checkpoint(checkpoint)
    if not summary.compatible:
        raise ValueError(
            f"MTG-183 cannot play in this runtime: {summary.incompatibility}"
        )
    agent, _ = load_checkpoint(str(checkpoint))
    if any(parameter.device.type != "cpu" for parameter in agent.parameters()):
        raise ValueError("Local Elemental play requires a CPU model.")
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(record.model_dump_json(indent=2) + "\n")
    temporary.replace(receipt)
    print(
        f"MTG-183 · update {int(record.at['updates']):,} · {digest[:12]} · CPU",
        flush=True,
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="r2://manabot/jobs")
    parser.add_argument("--run", default=DEFAULT_RUN, help="MTG-183 training segment")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="reuse verified pinned bytes without storage access",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="download and admit the model without starting servers",
    )
    parser.add_argument("--port", type=int, default=18479)
    parser.add_argument("--frontend-port", type=int, default=18480)
    args = parser.parse_args()
    directory = ROOT / ".runs" / "elemental-play"
    checkpoint = prepare_checkpoint(
        directory / "checkpoints", args.source, args.run, args.offline
    )
    if args.prepare_only:
        return
    if not args.offline:
        subprocess.run(
            ["node", str(ROOT / "academy/ui/scripts/fetch-card-art.mjs")],
            cwd=ROOT,
            check=True,
        )
    digest = inspect_checkpoint(checkpoint).sha256
    query = urlencode(
        {"hero": "elemental_ur", "villain": "elemental_gw", "opponent": digest}
    )
    print(
        f"\nElemental deck browser: http://localhost:{args.frontend_port}/?{query}\n",
        flush=True,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "ACADEMY_PLAY_CHECKPOINTS": str(checkpoint),
            "MANABOT_RUNS_DIR": str(directory / "catalog"),
            "ACADEMY_TRACES_DIR": str(directory / "human-play"),
            "ACADEMY_PLAY_RECORD_ORIGIN": "human_exploratory",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    os.execve(
        ROOT / "scripts" / "play",
        [
            str(ROOT / "scripts" / "play"),
            "--port",
            str(args.port),
            "--frontend-port",
            str(args.frontend_port),
        ],
        environment,
    )


if __name__ == "__main__":
    main()
