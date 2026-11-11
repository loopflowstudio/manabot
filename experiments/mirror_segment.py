"""Seal the stopped v2 segment before a separately identified uniform continuation.

Etude owns stopping and allocating jobs. This module only reconciles retained
files after that stop. Shared predecessor locks prevent the original runner
from resuming while a continuation uses its completed keys. Interrupted uniform
attempts remain evidence and may restart; scored failures are never replayed.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from experiments.search_mirror import (
    CHECKPOINT_SHA256,
    extension_arms,
    planned_units,
    reconcile_extension,
)
from manabot.infra.identity import file_sha256


class Predecessor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: Literal["mirror-predecessor/v1"] = "mirror-predecessor/v1"
    root: Path
    etude_run: Literal["mtg269-mirror-v2"] = "mtg269-mirror-v2"
    etude_attempt: Literal["3a2b62144b9a"] = "3a2b62144b9a"
    etude_outcome: Literal["cancelled", "succeeded"]
    deadline: float
    workers: int
    files: dict[str, str]
    completed_keys: list[str]
    interrupted_attempts: dict[str, str]
    attempt_sha256: dict[str, str]


@contextmanager
def hold_predecessor(receipt: Path) -> Iterator[None]:
    """Exclude every original shard for the lifetime of each new worker."""
    prior = Predecessor.model_validate_json(receipt.read_text())
    with ExitStack() as stack:
        for worker in range(prior.workers):
            lock = stack.enter_context((prior.root / f"worker-{worker}.lock").open("r"))
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def read_predecessor(receipt: Path, deadline: float) -> set[str]:
    """Check the sealed ledger without rescanning all private search receipts."""
    prior = Predecessor.model_validate_json(receipt.read_text())
    if prior.deadline != deadline or prior.workers != 7:
        raise ValueError("predecessor deadline or worker assignment changed")
    files = {"identity.json"} | {p.name for p in prior.root.glob("games-*.jsonl")}
    if files != prior.files.keys():
        raise ValueError("predecessor ledger files changed")
    for name, digest in prior.files.items():
        if file_sha256(prior.root / name) != digest:
            raise ValueError(f"predecessor bytes changed: {name}")
    identity = json.loads((prior.root / "identity.json").read_text())
    if (
        identity["checkpoint_sha256"] != CHECKPOINT_SHA256
        or identity["deadline"] != deadline
    ):
        raise ValueError("predecessor checkpoint or deadline mismatch")
    done = reconcile_extension(prior.root, owned_keys=set())
    if done != set(prior.completed_keys):
        raise ValueError("predecessor outcome keys changed")
    required = {f"{a.key}/{g.deal}" for g, a in planned_units(extension_arms("oracle"))}
    if not required <= done:
        raise ValueError("uniform stage requires all oracle outcomes first")
    return done


def seal_predecessor(root: Path, etude_receipt: Path, destination: Path) -> Predecessor:
    """Retain every closed outcome and explicit interruption after Etude stops.

    A truncated/unreadable trace or an uncommitted outcome stops reconciliation.
    Those conditions require recovery, never an inferred failed-game score.
    """
    status = json.loads(etude_receipt.read_text())
    if (
        status["status"]["run"] != "mtg269-mirror-v2"
        or status["status"]["phase"] != "done"
    ):
        raise ValueError("original Etude job is not stopped")
    attempts = status["attempts"]
    if (
        len(attempts) != 1
        or attempts[0]["id"] != "3a2b62144b9a"
        or not attempts[0]["finished_at"]
    ):
        raise ValueError("original Etude attempt is not reconciled")
    identity = json.loads((root / "identity.json").read_text())
    done = reconcile_extension(root, owned_keys=set())
    rows = {
        row["key"]: row
        for path in root.glob("games-*.jsonl")
        for line in path.read_text().splitlines()
        if (row := json.loads(line))
    }
    uniform = {f"{a.key}/{g.deal}" for g, a in planned_units(extension_arms("uniform"))}
    interrupted: dict[str, str] = {}
    digests: dict[str, str] = {}
    observed: set[str] = set()
    for path in sorted((root / "attempts").glob("*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            start = json.loads(next(stream))
            last = start
            for line in stream:
                last = json.loads(line)
        key = start["key"]
        if (
            key in observed
            or path.name != hashlib.sha256(key.encode()).hexdigest() + ".jsonl.gz"
        ):
            raise ValueError(f"conflicting attempt identity: {path}")
        observed.add(key)
        if (
            start["checkpoint_sha256"] != CHECKPOINT_SHA256
            or start["deadline"] != identity["deadline"]
        ):
            raise ValueError(f"attempt input mismatch: {path}")
        if key in done:
            if last != {"event": "outcome", "row": rows[key]}:
                raise ValueError(f"unsealed outcome: {path}")
        elif key in uniform and last.get("event") == "interrupted":
            interrupted[key] = str(path.relative_to(root))
        else:
            raise ValueError(f"unreconciled attempt: {path}")
        digests[str(path.relative_to(root))] = file_sha256(path)
    if not done <= observed:
        raise ValueError("completed outcome lacks its sealed attempt")
    prior = Predecessor(
        root=root.resolve(),
        etude_outcome=status["status"]["outcome"],
        deadline=identity["deadline"],
        workers=identity["workers"],
        files={
            p.name: file_sha256(p)
            for p in [root / "identity.json", *sorted(root.glob("games-*.jsonl"))]
        },
        completed_keys=sorted(done),
        interrupted_attempts=interrupted,
        attempt_sha256=digests,
    )
    with destination.open("x") as stream:
        stream.write(prior.model_dump_json(indent=2) + "\n")
    read_predecessor(destination, prior.deadline)
    return prior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--etude-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Exclusive locks also prevent another worker starting during the seal.
    with ExitStack() as stack:
        for worker in range(7):
            lock = stack.enter_context((args.root / f"worker-{worker}.lock").open("r"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = seal_predecessor(args.root, args.etude_receipt, args.out)
    print(
        json.dumps(
            {
                "completed": len(prior.completed_keys),
                "interrupted": len(prior.interrupted_attempts),
            }
        )
    )


if __name__ == "__main__":
    main()
