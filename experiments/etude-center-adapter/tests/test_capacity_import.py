from __future__ import annotations

from pathlib import Path

import pytest
from etude_client.models import JobSnapshot, QueueObservation, utcnow
from etude_fantasia.fantasia import FantasiaImport

from etude.admission import Queue


def test_real_capacity_import_replay_and_continuation(
    queue: Queue, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[2] / "fantasia"
    imported = FantasiaImport(root, "capacity-recovery-v3")
    receipt = queue.register(imported.registration())
    for batch in imported.batches(receipt.experiment_id):
        assert queue.ingest(batch).inserted > 0
        assert queue.ingest(batch).inserted == 0
    now = utcnow()
    monkeypatch.setattr("etude.store.utcnow", lambda: now)
    detail = queue.get(receipt.experiment_id)
    cohorts = [
        f.observation for f in detail.current_facts if isinstance(f.observation, QueueObservation)
    ]
    assert len(cohorts) == 1 and len(cohorts[0].jobs) == 8
    continuations = [j for j in cohorts[0].jobs if j.continuation]
    assert [j.target_step for j in continuations] == [63000, 100000]
    assert [j.continuation.origin_step for j in continuations if j.continuation] == [26000, 63000]
    observations = [f.observation for f in detail.facts if isinstance(f.observation, JobSnapshot)]
    assert {146, 242} <= {o.progress.absolute_step for o in observations}
    assert any(
        o.evaluation.completed_games == 100 and o.evaluation.completed_cohorts == 1
        for o in observations
    )
    assert any("Original remote endpoint omitted" in o.missing_evidence for o in observations)
    assert not queue.resources()  # Import grants no execution ownership.
    assert Queue(queue.dsn, queue.organization_id).get(receipt.experiment_id) == detail
