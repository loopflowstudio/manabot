"""The follower keeps cohort-era checkpoints and settles on its one Etude segment."""

from decimal import Decimal
import hashlib
from pathlib import Path

from etude.launch.records import Status
from etude.run.records import ArtifactRef
from etude.train.records import CheckpointRecord

from experiments.elemental_long_follow import Checkpoint, Ledger, SegmentSource, _admit


def _record(path: Path, updates: int) -> CheckpointRecord:
    data = path.read_bytes()
    return CheckpointRecord(
        ordinal=1,
        artifact=ArtifactRef(
            uri=path.as_uri(),
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        ),
        at={"updates": updates, "learner_transitions": updates * 10},
        wall_ms=0,
    )


def _status(phase: str, outcome: str | None) -> Status:
    return Status.model_validate(
        {
            "run": "w3",
            "phase": phase,
            "reason": None,
            "outcome": outcome,
            "cause": None,
            "until": "updates=2700",
            "clocks": {},
            "usd": Decimal(1),
            "attempts": 1,
            "rental_ended": None,
        }
    )


def test_segment_joins_retained_windows_and_settles_when_terminal(
    tmp_path: Path,
) -> None:
    exports = []
    for updates in (2550, 2600, 2650):
        path = tmp_path / f"{updates}.pt"
        path.write_bytes(str(updates).encode())
        exports.append(_record(path, updates))
    ledger = Ledger(
        heuristic_schedule_sha256="x",
        h2h_schedule_sha256="y",
        checkpoints={
            2550: Checkpoint(
                updates=2550,
                job="mtg183-elemental-long-w2",
                training_run="w2-run",
                path="w2.pt",
                sha256="s",
                coordinates={},
            )
        },
    )
    phase = ["running", None]
    segment = SegmentSource(
        run="w3",
        checkpoints=lambda: exports[:2] if phase[0] == "running" else exports,
        status=lambda: _status(phase[0], phase[1]),
    )
    _admit(tmp_path, ledger, segment)
    assert ledger.checkpoints[2550].job == "mtg183-elemental-long-w2"
    assert ledger.checkpoints[2600].job == "w3"
    assert ledger.segment_outcome is None and ledger.endpoint is None
    # A segment that stops short of its target still ends the seed.
    phase[:] = ["done", "failed"]
    _admit(tmp_path, ledger, segment)
    assert ledger.segment_outcome == "failed" and ledger.endpoint == 2650
    assert ledger.error is None
