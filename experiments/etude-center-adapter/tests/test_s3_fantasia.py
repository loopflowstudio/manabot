"""GET-only current selection feeds the existing native atomic ingestion path."""

from __future__ import annotations

import json
import stat
import subprocess
import time
from pathlib import Path

import pytest
from etude_client.api import Client
from etude_client.errors import EtudeError
from etude_client.models import JobSnapshot
from etude_fantasia.live_fantasia import NativeJob, NativeTranslation, refresh_native
from etude_fantasia.reader import Reader
from etude_fantasia.transports import S3Source
from test_native_fantasia import ReadOnlySource, native_fixture


class FakeAWS:
    def __init__(self) -> None:
        self.source = ReadOnlySource()
        self.reads: list[tuple[str, str | None]] = []
        self.paths: list[Path] = []
        self.error = False
        self.version: str | None = None
        self.fail_key: str | None = None
        self.advance(1791450100, 63010)

    def advance(self, heartbeat: int, steps: int) -> None:
        native_fixture(self.source, heartbeat=heartbeat, updates=steps)

    def run(
        self,
        command: list[str],
        *,
        capture_output: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        assert command[:3] == ["aws", "s3api", "get-object"]
        assert timeout <= 1200 and capture_output and not check
        path = Path(command[-1])
        self.paths.append(path)
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        key = (
            "s3://"
            + command[command.index("--bucket") + 1]
            + "/"
            + command[command.index("--key") + 1]
        )
        requested_version = (
            command[command.index("--version-id") + 1] if "--version-id" in command else None
        )
        self.reads.append((key, requested_version))
        if self.error or key == self.fail_key:
            return subprocess.CompletedProcess(command, 1, b"", b"SECRET-CANARY")
        versions = [v for k, v in self.source.objects if k == key]
        version = requested_version or versions[-1]
        path.write_bytes(self.source.objects[key, version])
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"VersionId": self.version or version}).encode(),
            b"",
        )


@pytest.fixture
def aws(monkeypatch: pytest.MonkeyPatch) -> FakeAWS:
    fake = FakeAWS()
    monkeypatch.setattr("etude_fantasia.transports.subprocess.run", fake.run)
    return fake


def test_current_cut_pins_bytes_and_only_follows_run(aws: FakeAWS) -> None:
    reader = Reader(S3Source("s3://fixture/jobs", ("continued",)))
    job = NativeJob(
        spec=reader.select("s3://fixture/jobs/continued/spec.json"),
        record=reader.select("s3://fixture/jobs/continued/runtime/record.json"),
    )
    translation = NativeTranslation(reader)
    observed = translation.job(job)
    before = reader.bytes(job.record)
    aws.advance(1791450200, 63100)
    assert reader.select(job.record.key) == job.record
    assert reader.bytes(job.record) == before
    assert translation.job(job) == observed
    assert len(aws.reads) == 4
    assert [version for _, version in aws.reads] == [None, None, "2", "2"]
    newer = Reader(S3Source("s3://fixture/jobs", ("continued",))).select(job.record.key)
    assert newer.version != job.record.version


@pytest.mark.parametrize("version", ["null", "wrong-version"])
def test_unversioned_or_wrong_version_rejected(aws: FakeAWS, version: str) -> None:
    aws.version = version
    source = S3Source("s3://fixture/jobs", ("continued",))
    with pytest.raises(EtudeError):
        source.select(
            "s3://fixture/jobs/continued/spec.json",
            version="spec-version",
            deadline=time.monotonic() + 45,
        )
    assert all(not path.parent.exists() for path in aws.paths)


def test_foreign_dependency_and_error_body_rejected(aws: FakeAWS) -> None:
    source = S3Source("s3://fixture/jobs", ("continued",))
    with pytest.raises(EtudeError):
        Reader(source).select("s3://fixture/jobs/other/runtime/record.json")
    aws.error = True
    with pytest.raises(EtudeError) as error:
        Reader(source).select("s3://fixture/jobs/continued/spec.json")
    assert "SECRET-CANARY" not in str(error.value)
    assert all(not path.parent.exists() for path in aws.paths)


def test_current_refresh_and_dependency_failure_are_atomic(
    aws: FakeAWS,
    service: tuple[str, str],
) -> None:
    with Client(*service) as client:
        experiment = client.register("s3-current")
        refresh_native(
            client, S3Source("s3://fixture/jobs", ("continued",)), experiment.experiment_id
        )
        before = client.get(experiment.experiment_id)
        aws.advance(1791450200, 63100)
        aws.fail_key = "s3://fixture/jobs/continued/runtime/manifest.json"
        with pytest.raises(EtudeError):
            refresh_native(
                client, S3Source("s3://fixture/jobs", ("continued",)), experiment.experiment_id
            )
        failed = client.get(experiment.experiment_id)
        assert failed.facts == before.facts and failed.evidence == before.evidence
        assert failed.collectors[0].error == "unavailable"
        aws.fail_key = None
        refresh_native(
            client, S3Source("s3://fixture/jobs", ("continued",)), experiment.experiment_id
        )
        after = client.get(experiment.experiment_id)
        fact = after.jobs[0].fact
        assert fact is not None and isinstance(fact.observation, JobSnapshot)
        assert fact.observation.progress.absolute_step == 63100
        assert fact.observation.continuation is not None
        assert fact.observation.continuation.origin_step == 63000


def large_run(aws: FakeAWS) -> str:
    from etude_client.models import MAX_BYTES
    from etude_fantasia.values import obj, seq

    source = aws.source
    run_key = "s3://fixture/jobs/continued/runtime/run.json"
    run = obj(json.loads(source.objects[run_key, "2"]))
    run["parent_run_id"] = "recovery-parent"
    run["recovery_artifact"] = {
        "uri": "s3://fixture/recovery",
        "sha256": "c" * 64,
        "bytes": 200,
        "version_id": "recovery-version",
    }
    obj(obj(seq(run["stages"])[0])["learning_state_origin"]).update(
        {
            "stage_id": "policy",
            "games": 900,
            "environment_decisions": 1200,
            "learner_transitions": 1500,
            "optimizer_exposures": 2000,
            "active_training_seconds": 40.5,
            "cumulative_seconds": 50.5,
            "streams": "fresh",
        }
    )
    run["dense_series"] = ["x" * 1024] * (MAX_BYTES // 1024 + 1)
    run_ref = source.put_fixture(run_key, "2", run)
    manifest_key = "s3://fixture/jobs/continued/runtime/manifest.json"
    manifest = obj(json.loads(source.objects[manifest_key, "2"]))
    obj(seq(manifest["artifacts"])[0]).update(source.stored(run_ref))
    obj(seq(obj(manifest["bundle"])["files"])[0]).update(
        {"size": len(source.objects[run_key, "2"]), "sha256": run_ref.sha256}
    )
    manifest_ref = source.put_fixture(manifest_key, "2", manifest)
    record_key = "s3://fixture/jobs/continued/runtime/record.json"
    record = obj(json.loads(source.objects[record_key, "1791450100"]))
    record["manifest"] = source.stored(manifest_ref)
    source.put_fixture(record_key, "1791450100", record)
    return run_key


def test_large_run_reference_retains_complete_ancestry_and_replays(
    aws: FakeAWS,
    service: tuple[str, str],
) -> None:
    import httpx

    run_key = large_run(aws)
    with Client(*service) as client:
        experiment = client.register("large-run")
        refresh_native(
            client, S3Source("s3://fixture/jobs", ("continued",)), experiment.experiment_id
        )
        before = client.get(experiment.experiment_id)
        evidence = next(e for e in before.evidence if e.source_uri == run_key)
        assert evidence.retention == "referenced"
        assert evidence.size_bytes == len(aws.source.objects[run_key, "2"])
        assert all(not path.parent.exists() for path in aws.paths)
        assert evidence.omission_reason is not None and "Dense per-step" in evidence.omission_reason
        fact = before.jobs[0].fact
        assert fact is not None and isinstance(fact.observation, JobSnapshot)
        ancestry = fact.observation.continuation
        assert ancestry is not None and ancestry.origin_step == 63000
        assert ancestry.parent_run is not None and ancestry.parent_run.source_id == "parent-run"
        assert ancestry.origin_stage == "policy" and ancestry.origin_games == 900
        assert ancestry.origin_environment_decisions == 1200
        assert ancestry.origin_learner_transitions == 1500
        assert ancestry.origin_optimizer_updates is None
        assert ancestry.origin_optimizer_exposures == 2000
        assert ancestry.origin_active_training_seconds == 40.5
        assert ancestry.origin_cumulative_seconds == 50.5 and ancestry.origin_streams == "fresh"
        assert ancestry.recovery_parent_run is not None
        assert ancestry.recovery_parent_run.source_id == "recovery-parent"
        assert (
            ancestry.recovery_artifact is not None
            and ancestry.recovery_artifact.version == "recovery-version"
        )
        assert all(
            (
                ancestry.checkpoint_sha256,
                ancestry.ema_sha256,
                ancestry.optimizer_sha256,
                ancestry.source_run_sha256,
            )
        )
        response = httpx.get(
            service[0] + f"/evidence/{evidence.evidence_id}",
            headers={"Authorization": f"Bearer {service[1]}"},
        )
        assert response.status_code == 409
        assert response.json()["error"] == "evidence_referenced_not_retained"
        refresh_native(
            client, S3Source("s3://fixture/jobs", ("continued",)), experiment.experiment_id
        )
        after = client.get(experiment.experiment_id)
        assert after.evidence == before.evidence and after.facts == before.facts
        # Corruption in discarded series must veto the whole cut too.
        aws.source.objects[run_key, "2"] += b" "
        with pytest.raises(EtudeError, match="invalid"):
            refresh_native(
                client, S3Source("s3://fixture/jobs", ("continued",)), experiment.experiment_id
            )
        failed = client.get(experiment.experiment_id)
        assert failed.evidence == before.evidence and failed.facts == before.facts
        assert failed.collectors[0].error == "invalid"
        assert all(not path.parent.exists() for path in aws.paths)


def test_cached_dependencies_still_validate_each_assertion(aws: FakeAWS) -> None:
    reader = Reader(S3Source("s3://fixture/jobs", ("continued",)))
    ref = aws.source.put_fixture("s3://fixture/jobs/continued/runtime/example.json", "1", {"x": 1})
    value = aws.source.stored(ref)
    assert reader.dependency(value)[0] == ref
    with pytest.raises(EtudeError, match="invalid"):
        reader.dependency(value | {"bytes": len(aws.source.objects[ref.key, ref.version]) + 1})
    with pytest.raises(EtudeError, match="invalid"):
        reader.dependency(value | {"sha256": "f" * 64})


def test_cut_bound_covers_all_objects(aws: FakeAWS, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = "s3://fixture/jobs/continued/spec.json"
    limit = len(aws.source.objects[spec, "spec-version"])
    monkeypatch.setattr("etude_fantasia.reader.MAX_BYTES", limit)
    reader = Reader(S3Source("s3://fixture/jobs", ("continued",)))
    ref = reader.select(spec)
    with pytest.raises(EtudeError, match="invalid"):
        reader.select("s3://fixture/jobs/continued/runtime/record.json")
    assert reader.bytes(ref) == aws.source.objects[spec, "spec-version"]


def test_expired_cut_does_not_fetch(aws: FakeAWS) -> None:
    reader = Reader(S3Source("s3://fixture/jobs", ("continued",)), deadline=time.monotonic() - 1)
    with pytest.raises(EtudeError, match="timeout"):
        reader.select("s3://fixture/jobs/continued/spec.json")
    assert not aws.reads


def test_pinned_selection_does_not_gain_s3_dependency_discovery(aws: FakeAWS) -> None:
    reader = Reader(S3Source("s3://fixture/jobs", ("continued",)), dependencies=())
    ref = aws.source.put_fixture("s3://fixture/jobs/continued/runtime/extra.json", "1", {"x": 1})
    with pytest.raises(EtudeError, match="forbidden"):
        reader.dependency(aws.source.stored(ref))
    assert not aws.reads
