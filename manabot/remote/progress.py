"""Retrieve committed remote exports while their ordinary TrainingRun continues.

LiveExports preserves producer JSON and hash-bound policy bytes. Local path
resolution is transport metadata; it never rewrites TrainingRun or its database.
The final closed bundle remains required for complete-run admission.
"""

import json
from pathlib import Path, PurePosixPath
import time

from manabot.remote.plan import digest
from manabot.remote.transport import Transport
from manabot.training.models import ArtifactReference, TrainingRun


class LiveExports:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.references: dict[str, ArtifactReference] = {}
        self.seconds = 0.0
        self.latest: Path | None = None

    def retrieve(self, transport: Transport) -> Path | None:
        began = time.monotonic()
        try:
            raw = transport.shell(
                "if test -f /workspace/evidence/run/run.json; then cat /workspace/evidence/run/run.json; fi"
            )
            if not raw:
                return None
            run = TrainingRun.model_validate_json(raw)
            references = [
                c.artifact
                for c in run.monitoring_checkpoints
                if c.artifact is not None and c.error is None
            ]
            references.extend(
                reference
                for stage in run.stages
                for name, reference in stage.artifacts.items()
                if name in ("initial_raw", "raw")
            )
            for reference in references:
                self._retrieve(transport, reference)
            snapshots = self.out / "snapshots"
            snapshots.mkdir(exist_ok=True)
            target = snapshots / f"{digest(raw)}.json"
            if not target.exists():
                target.write_bytes(raw)
            self.latest = target
            return target
        finally:
            self.seconds += time.monotonic() - began
            (self.out / "transport.json").write_text(
                json.dumps(
                    {
                        "seconds": self.seconds,
                        "latest": str(self.latest) if self.latest else None,
                        "references": self.references,
                    },
                    indent=2,
                )
            )

    def _retrieve(self, transport: Transport, reference: ArtifactReference) -> None:
        producer = PurePosixPath(reference["path"])
        if (
            not producer.is_relative_to("/workspace/evidence/run")
            or ".." in producer.parts
            or any(
                c
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
                for c in str(producer)
            )
        ):
            raise ValueError("unrecognized remote policy path")
        identity = reference["sha256"]
        target = self.out / f"{identity}.pt"
        if not target.exists():
            temporary = target.with_suffix(".partial")
            transport.get(str(producer), temporary)
            if (
                temporary.stat().st_size != reference["bytes"]
                or digest(temporary.read_bytes()) != identity
            ):
                raise ValueError("live artifact differs from committed receipt")
            temporary.replace(target)
        self.references[identity] = {**reference, "path": str(target)}

    def resolve(self, reference: ArtifactReference) -> ArtifactReference:
        local = self.references[reference["sha256"]]
        if (
            local["sha256"] != reference["sha256"]
            or local["bytes"] != reference["bytes"]
        ):
            raise ValueError("live artifact identity differs")
        return local
