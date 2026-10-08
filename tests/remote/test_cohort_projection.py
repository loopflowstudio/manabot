"""Telemetry faults retain local reports and cannot become scheduling dependencies."""

from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from manabot.remote import cohort_projection as projection, cohort_service, job_client
from manabot.remote.bundle import Bundle, BundleFile
from manabot.remote.cohort import CohortAttempt, CohortState
from manabot.remote.jobs import JobRecord
from manabot.remote.snapshots import JobManifest
from manabot.training.monitoring import Dashboard
from tests.remote.job_fixtures import FileStore
from tests.remote.test_cohort_service import plan
from tests.remote.test_job_fetch import Artifacts


@pytest.mark.parametrize(
    "dashboard_path",
    [
        "training-dashboard.json",
        "monitoring/run-one/dashboard.json",
        "monitoring/run-one/dashboard-protocol.json",
    ],
)
def test_offline_report_survives_telemetry_outage_and_retries_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dashboard_path: str,
) -> None:
    cohort = plan()
    spec = cohort.entries[0].job(time.time())
    state = CohortState(
        cohort_sha256=cohort.identity,
        owner="scheduler",
        phase="running",
        heartbeat_at=time.time(),
        attempts=(CohortAttempt(spec=spec),),
    )
    store = FileStore(tmp_path / "cohort.sqlite")
    store.create("state.json", state.model_dump_json().encode())
    jobs = FileStore(tmp_path / "job.sqlite")
    artifacts = Artifacts()
    dashboard = Dashboard(
        run_id="run-one", config={}, summary={}, rows=[{"rl/loss": 1.25}]
    )
    reference = artifacts.add(dashboard.model_dump_json().encode())
    # Real snapshots contain this queue index before the per-run Dashboards.
    queue = artifacts.add(b'{"attempts": [], "pending_checkpoints": 0}')
    manifest = JobManifest(
        spec_sha256=spec.identity,
        generation=1,
        complete=False,
        bundle=Bundle(
            files=(
                BundleFile(
                    producer_path="/worker/monitoring/dashboard.json",
                    relative_path="monitoring/dashboard.json",
                    size=queue.bytes,
                    sha256=queue.sha256,
                ),
                BundleFile(
                    producer_path=f"/worker/{dashboard_path}",
                    relative_path=dashboard_path,
                    size=reference.bytes,
                    sha256=reference.sha256,
                ),
            )
        ),
        artifacts=(queue, reference),
    )
    record = JobRecord(
        spec_sha256=spec.identity,
        pod_id="pod",
        phase="running",
        accepted_at=time.time(),
        heartbeat_at=time.time(),
        generation=1,
        manifest=artifacts.add(manifest.model_dump_json().encode()),
    )
    jobs.create("runtime/record.json", record.model_dump_json().encode())
    monkeypatch.setattr(
        projection,
        "S3JobStore",
        lambda prefix: store if prefix == cohort.prefix else jobs,
    )
    monkeypatch.setattr(job_client, "S3ArtifactStore", lambda: artifacts)
    config = projection.ProjectionConfig(project="etude", entity="loopflow-studio")
    called: list[str] = []

    def unavailable(
        dashboard: Dashboard, out: Path, *, project: str, entity: str | None = None
    ) -> str:
        assert (out / "report.html").exists()  # Local report precedes telemetry.
        called.append(dashboard.run_id)
        raise ConnectionError("do not expose credential-bearing response")

    output = tmp_path / "projection"
    with pytest.raises(RuntimeError, match="local evidence retained"):
        projection.project_once(cohort, output, config, publish=unavailable)
    assert "ConnectionError" in (output / "telemetry.json").read_text()
    assert "credential-bearing" not in (output / "telemetry.json").read_text()
    assert (
        output / spec.job_id / f"{reference.sha256}.json"
    ).read_bytes() == dashboard.model_dump_json().encode()
    assert store.read("state.json").data == state.model_dump_json().encode()

    def available(
        dashboard: Dashboard, out: Path, *, project: str, entity: str | None = None
    ) -> str:
        called.append(dashboard.run_id)
        return "https://wandb.ai/loopflow-studio/etude/runs/run-one"

    projection.project_once(cohort, output, config, publish=available)
    projection.project_once(cohort, output, config, publish=available)
    assert called == ["run-one", "run-one"]
    assert len(artifacts.downloads) == 2
    assert len(list((output / "telemetry-history").glob("*.json"))) == 3
    with pytest.raises(ValueError, match="another owner"):
        projection.project_once(
            cohort, tmp_path / "competing", config, publish=available
        )


def test_interrupted_projection_reservation_survives_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = plan()
    config = projection.ProjectionConfig(total_seconds=120)
    ledger = projection.ProjectionLedger(
        cohort_sha256=cohort.identity,
        config=config,
        attempts=(
            projection.ProjectionAttempt(
                started_at=time.time() - 10, reserved_seconds=120
            ),
        ),
    )
    (tmp_path / "ledger.json").write_text(ledger.model_dump_json())
    monkeypatch.setattr(cohort_service, "_machine_identity", lambda: "fixture-host")
    launch = Mock(side_effect=AssertionError("spent allowance must not reset"))
    monkeypatch.setattr(projection.subprocess, "Popen", launch)
    projection.follow_projection(cohort, tmp_path, config, ["unused"])
    launch.assert_not_called()
    assert (
        projection.ProjectionLedger.model_validate_json(
            (tmp_path / "ledger.json").read_bytes()
        )
        == ledger
    )


def test_projection_child_owns_deadline_even_without_parent_guardian(
    tmp_path: Path,
) -> None:
    cohort = plan()
    file = tmp_path / "cohort.json"
    file.write_text(cohort.model_dump_json())
    code = """from pathlib import Path
import sys,time
from manabot.remote.cohort import Cohort
from manabot.remote import cohort_projection as p
p.project_once=lambda *args: time.sleep(30)
p.project_before(Cohort.model_validate_json(Path(sys.argv[1]).read_bytes()),Path(sys.argv[2]),p.ProjectionConfig(),time.time()+.1)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(file), str(tmp_path)],
        start_new_session=True,
        timeout=15,
        capture_output=True,
    )
    assert result.returncode == -9, result.stderr
