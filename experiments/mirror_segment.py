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
from datetime import datetime
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


class InterruptionRecovery(BaseModel):
    """External stop evidence for readable traces missing their terminal callback.

    This does not add events or scores to the original streams. Etude's confirmed
    cancellation, exact trace bytes and all seven worker logs remain the authority.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: Literal["mirror-interruption-recovery/v1"] = (
        "mirror-interruption-recovery/v1"
    )
    etude_receipt_sha256: str
    pending_attempts: dict[str, str]
    worker_logs: dict[str, str]


class Predecessor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: Literal["mirror-predecessor/v1", "mirror-predecessor/v2"] = (
        "mirror-predecessor/v1"
    )
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
    interruption_recovery: InterruptionRecovery | None = None


def _verify_recovery_logs(root: Path, recovery: InterruptionRecovery) -> None:
    expected = {f"logs/uniform-{worker}.log" for worker in range(7)}
    if recovery.worker_logs.keys() != expected:
        raise ValueError("interruption recovery requires all seven worker logs")
    for name, digest in recovery.worker_logs.items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"recovery log bytes changed: {name}")


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
    if (prior.schema_name == "mirror-predecessor/v2") != (
        prior.interruption_recovery is not None
    ):
        raise ValueError("predecessor recovery schema mismatch")
    if recovery := prior.interruption_recovery:
        _verify_recovery_logs(prior.root, recovery)
        for key, digest in recovery.pending_attempts.items():
            name = f"attempts/{hashlib.sha256(key.encode()).hexdigest()}.jsonl.gz"
            if (
                prior.interrupted_attempts.get(key) != name
                or prior.attempt_sha256.get(name) != digest
                or file_sha256(prior.root / name) != digest
            ):
                raise ValueError(f"recovered interruption bytes changed: {key}")
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


def seal_predecessor(
    root: Path,
    etude_receipt: Path,
    destination: Path,
    *,
    interruption_recovery: Path | None = None,
) -> Predecessor:
    """Retain every closed outcome and explicit interruption after Etude stops.

    Truncated/unreadable traces and uncommitted outcomes always stop admission.
    A readable pending trace needs a separate, exact-hash recovery receipt bound
    to Etude's cancellation; a missing callback alone never authorizes a retry.
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
    recovery = (
        InterruptionRecovery.model_validate_json(interruption_recovery.read_text())
        if interruption_recovery is not None
        else None
    )
    recovery_window: tuple[float, float] | None = None
    if recovery is not None:
        if (
            status["status"]["outcome"] != "cancelled"
            or attempts[0].get("outcome") != "cancelled"
            or recovery.etude_receipt_sha256 != file_sha256(etude_receipt)
            or not recovery.pending_attempts
        ):
            raise ValueError("recovery requires the exact confirmed Etude cancellation")
        _verify_recovery_logs(root, recovery)
        recovery_window = (
            datetime.fromisoformat(attempts[0]["started_at"]).timestamp(),
            datetime.fromisoformat(attempts[0]["finished_at"]).timestamp(),
        )
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
    recovered: set[str] = set()
    for path in sorted((root / "attempts").glob("*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            start = json.loads(next(stream))
            last = start
            events = {start["event"]}
            for line in stream:
                last = json.loads(line)
                events.add(last["event"])
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
        elif recovery is not None and key in recovery.pending_attempts:
            assert recovery_window is not None
            if (
                key not in uniform
                or last.get("event") != "pending"
                or not isinstance(last.get("frame"), dict)
                or not isinstance(last.get("observation"), dict)
                or events & {"failure", "outcome", "interrupted"}
                or not recovery_window[0] <= start["started"] <= recovery_window[1]
                or file_sha256(path) != recovery.pending_attempts[key]
            ):
                raise ValueError(
                    f"attempt does not match cancellation recovery: {path}"
                )
            interrupted[key] = str(path.relative_to(root))
            recovered.add(key)
        else:
            raise ValueError(f"unreconciled attempt: {path}")
        digests[str(path.relative_to(root))] = file_sha256(path)
    if not done <= observed:
        raise ValueError("completed outcome lacks its sealed attempt")
    if recovery is not None and recovered != recovery.pending_attempts.keys():
        raise ValueError("recovery must name exactly the missing terminal callbacks")
    prior = Predecessor(
        schema_name="mirror-predecessor/v2" if recovery else "mirror-predecessor/v1",
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
        interruption_recovery=recovery,
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
    parser.add_argument("--interruption-recovery", type=Path)
    args = parser.parse_args()
    # Exclusive locks also prevent another worker starting during the seal.
    with ExitStack() as stack:
        for worker in range(7):
            lock = stack.enter_context((args.root / f"worker-{worker}.lock").open("r"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = seal_predecessor(
            args.root,
            args.etude_receipt,
            args.out,
            interruption_recovery=args.interruption_recovery,
        )
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
