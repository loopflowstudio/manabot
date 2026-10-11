"""Synthetic native wire receipts, not evidence of a current Fantasia refresh."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from etude_client.api import Client
from etude_client.collector import HTTPReadSource, ObjectRef
from etude_client.errors import EtudeError
from etude_client.models import JobSnapshot, sha256
from etude_fantasia.live_fantasia import (
    NativeJob,
    NativeSelection,
    continuation,
    manifest_run,
    refresh_native,
)
from etude_fantasia.values import Object


class ReadOnlySource:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.error: str | None = None
        self.failed_key: str | None = None
        self.reads: list[tuple[str, str]] = []

    def put_fixture(self, uri: str, version: str, data: Object) -> ObjectRef:
        raw = json.dumps(data, separators=(",", ":")).encode()
        self.objects[uri, version] = raw
        return ObjectRef(key=uri, version=version, sha256=sha256(raw))

    def read(self, key: str, version: str, *, deadline: float) -> bytes:
        self.reads.append((key, version))
        if self.error and key == self.failed_key:
            raise EtudeError(
                self.error, {"missing": 404, "forbidden": 403, "timeout": 504}[self.error]
            )
        return self.objects[key, version]

    def stored(self, ref: ObjectRef) -> Object:
        return {
            "schema_version": 1,
            "uri": ref.key,
            "version_id": ref.version,
            "sha256": ref.sha256,
            "bytes": len(self.objects[ref.key, ref.version]),
        }


def native_fixture(
    source: ReadOnlySource,
    generation: int = 2,
    heartbeat: int = 1791450100,
    updates: int | None = 63010,
) -> NativeSelection:
    prefix = "s3://fixture/jobs/continued"
    learning_inputs: Object = {
        role: {
            "schema_version": 1,
            "uri": "s3://fixture/parent/" + role,
            "sha256": str(index) * 64,
            "bytes": 10,
            "version_id": "parent",
        }
        for index, role in enumerate(("raw", "ema", "optimizer", "source_run"), 1)
    }
    spec = source.put_fixture(
        prefix + "/spec.json",
        "spec-version",
        {
            "job_id": "continued",
            "destination": "s3://fixture/jobs",
            "publish_seconds": 30,
            "created_at": 1791450000,
            "deadline": 1791453600,
            "plan": {"schema_version": 2, "source": {"commit": "a" * 40}},
            "learning_inputs": learning_inputs,
        },
    )
    run = source.put_fixture(
        f"{prefix}/runtime/run.json",
        str(generation),
        {
            "schema_version": 1,
            "id": "continued-run",
            "stages": [
                {
                    "id": "policy",
                    "learning_state_origin": {"run_id": "parent-run", "iteration": 63000},
                }
            ],
        },
    )
    manifest = source.put_fixture(
        f"{prefix}/runtime/manifest.json",
        str(generation),
        {
            "spec_sha256": spec.sha256,
            "generation": generation,
            "complete": False,
            "bundle": {
                "files": [
                    {
                        "relative_path": "run/run.json",
                        "producer_path": "/workspace/evidence/run/run.json",
                        "sha256": run.sha256,
                        "size": len(source.objects[run.key, run.version]),
                    }
                ]
            },
            "artifacts": [source.stored(run)],
        },
    )
    data: Object = {
        "spec_sha256": spec.sha256,
        "pod_id": "original-pod",
        "phase": "running",
        "accepted_at": 1791450001,
        "heartbeat_at": heartbeat,
        "generation": generation,
        "manifest": source.stored(manifest),
        "run_id": "continued-run",
        "evaluations_completed": 1,
    }
    if updates is not None:
        data["updates"] = updates
    record = source.put_fixture(prefix + "/runtime/record.json", str(heartbeat), data)
    return NativeSelection(
        jobs=(NativeJob(spec=spec, record=record),), dependencies=(manifest, run)
    )


def test_native_replay_advances_without_resetting_origin(service: tuple[str, str]) -> None:
    source = ReadOnlySource()
    initial = native_fixture(source)
    with Client(*service) as client:
        experiment = client.register("native")
        refresh_native(client, source, experiment.experiment_id, initial)
        first = client.get(experiment.experiment_id)
        assert source.reads == [
            (ref.key, ref.version)
            for ref in (initial.jobs[0].spec, initial.jobs[0].record, *initial.dependencies)
        ]
        refresh_native(client, source, experiment.experiment_id, initial)
        assert client.get(experiment.experiment_id).facts == first.facts
        assert client.get(experiment.experiment_id).evidence == first.evidence
        # Same manifest generation, newer heartbeat/progress is not a conflict.
        advanced = native_fixture(source, heartbeat=1791450110, updates=63020)
        refresh_native(client, source, experiment.experiment_id, advanced)
        same_generation = client.get(experiment.experiment_id)
        assert not same_generation.conflicts
        same_fact = same_generation.jobs[0].fact
        assert same_fact is not None and isinstance(same_fact.observation, JobSnapshot)
        assert same_fact.observation.progress.absolute_step == 63020
        newer = native_fixture(source, generation=3, heartbeat=1791450200, updates=63100)
        refresh_native(client, source, experiment.experiment_id, newer)
        refresh_native(client, source, experiment.experiment_id, initial)
        # Reconnect through the real API/store, not a process-local adapter cache.
    with Client(*service) as client:
        detail = client.get(experiment.experiment_id)
        assert not detail.conflicts
        assert len(detail.jobs) == 1
        fact = detail.jobs[0].fact
        assert fact is not None and isinstance(fact.observation, JobSnapshot)
        job = fact.observation
        assert job.source_key.source_id == "continued" and job.run is not None
        assert job.run.source_id == "continued-run"
        assert job.progress.absolute_step == 63100
        assert job.progress.optimizer_updates is None and job.progress.learner_transitions is None
        assert job.evaluation.completed_cohorts == 1 and job.evaluation.completed_games is None
        assert job.continuation is not None and job.continuation.origin_step == 63000
        assert (
            job.continuation.parent_run is not None
            and job.continuation.parent_run.source_id == "parent-run"
        )
        assert job.continuation.optimizer_sha256 == "3" * 64
        assert job.source_revision == "a" * 40 and job.evidence_state == "partial"
        assert fact.source_time == datetime.fromtimestamp(1791450200, UTC)
        assert all(
            e.source_revision is None for e in detail.evidence
        )  # Runtime bytes are not source files.
        assert {e.source_version for e in detail.evidence} >= {"2", "3", "1791450100", "1791450200"}
        assert all(age.generated_age_seconds is None for age in detail.report_ages)


@pytest.mark.parametrize("error", ["missing", "forbidden", "timeout"])
def test_dependency_outage_is_atomic(service: tuple[str, str], error: str) -> None:
    source = ReadOnlySource()
    selection = native_fixture(source)
    with Client(*service) as client:
        experiment = client.register("outage")
        refresh_native(client, source, experiment.experiment_id, selection)
        before = client.get(experiment.experiment_id)
        advanced = native_fixture(source, 3, 1791450200, 63100)
        source.error, source.failed_key = error, advanced.dependencies[-1].key
        with pytest.raises(EtudeError, match=error):
            refresh_native(client, source, experiment.experiment_id, advanced)
        after = client.get(experiment.experiment_id)
        assert before.facts == after.facts and before.evidence == after.evidence
        assert after.collectors[0].error == error
        assert after.collectors[0].last_success == before.collectors[0].last_success
        source.error = None
        refresh_native(client, source, experiment.experiment_id, advanced)
        assert client.get(experiment.experiment_id).collectors[0].error is None


@pytest.mark.parametrize(
    "corruption", ["digest", "generation", "identity", "not_enumerated", "unversioned"]
)
def test_native_cut_rejects_mismatch(service: tuple[str, str], corruption: str) -> None:
    source = ReadOnlySource()
    selection = native_fixture(source)
    manifest = selection.dependencies[0]
    if corruption == "unversioned":
        selection = selection.model_copy(
            update={
                "jobs": (
                    NativeJob(
                        spec=selection.jobs[0].spec.model_copy(update={"version": "null"}),
                        record=selection.jobs[0].record,
                    ),
                )
            }
        )
    elif corruption == "not_enumerated":
        selection = selection.model_copy(update={"dependencies": ()})
    elif corruption == "digest":
        source.objects[manifest.key, manifest.version] = b"{}"
    else:
        record_ref = selection.jobs[0].record
        record: Object = json.loads(source.objects[record_ref.key, record_ref.version])
        record["generation" if corruption == "generation" else "spec_sha256"] = (
            99 if corruption == "generation" else "b" * 64
        )
        replacement = source.put_fixture(record_ref.key, record_ref.version, record)
        selection = selection.model_copy(
            update={"jobs": (NativeJob(spec=selection.jobs[0].spec, record=replacement),)}
        )
    with Client(*service) as client:
        experiment = client.register("invalid")
        with pytest.raises(EtudeError):
            refresh_native(client, source, experiment.experiment_id, selection)
        detail = client.get(experiment.experiment_id)
        assert not detail.facts and not detail.evidence
        assert detail.collectors[0].error in {"invalid", "forbidden"}


def test_absent_counter_stays_unknown(service: tuple[str, str]) -> None:
    source = ReadOnlySource()
    selection = native_fixture(source, updates=None)
    with Client(*service) as client:
        experiment = client.register("unknown")
        refresh_native(client, source, experiment.experiment_id, selection)
        detail = client.get(experiment.experiment_id)
        assert detail.jobs[0].fact is not None
        job = detail.jobs[0].fact.observation
        assert isinstance(job, JobSnapshot) and job.progress.absolute_step is None


def test_unenumerated_transport_records_forbidden(service: tuple[str, str]) -> None:
    selection = native_fixture(ReadOnlySource())
    with Client(*service) as client:
        experiment = client.register("unenumerated")
        with pytest.raises(EtudeError, match="source_not_enumerated"):
            refresh_native(client, HTTPReadSource({}), experiment.experiment_id, selection)
        detail = client.get(experiment.experiment_id)
        assert not detail.evidence and detail.collectors[0].error == "forbidden"


def test_native_input_digests_before_manifest(service: tuple[str, str]) -> None:
    source = ReadOnlySource()
    selection = native_fixture(source)
    job = selection.jobs[0]
    record: Object = json.loads(source.objects[job.record.key, job.record.version])
    record["manifest"] = None
    selected = NativeSelection(
        jobs=(NativeJob(spec=job.spec, record=source.put_fixture(job.record.key, "early", record)),)
    )
    with Client(*service) as client:
        experiment = client.register("before-manifest")
        refresh_native(client, source, experiment.experiment_id, selected)
        fact = client.get(experiment.experiment_id).jobs[0].fact
        assert fact is not None and isinstance(fact.observation, JobSnapshot)
        ancestry = fact.observation.continuation
        assert ancestry is not None and ancestry.parent_run is None and ancestry.origin_step is None
        assert ancestry.checkpoint_sha256 == "1" * 64
        assert ancestry.ema_sha256 == "2" * 64
        assert ancestry.optimizer_sha256 == "3" * 64
        assert ancestry.source_run_sha256 == "4" * 64
        assert len(source.reads) == 2  # Only spec and record; no inferred source access.


@pytest.mark.parametrize(
    "case", ["multiple_origins", "before_origin", "missing_inputs", "unknown_origin_field"]
)
def test_continuation_rejects_unrepresentable_ancestry(case: str) -> None:
    source = ReadOnlySource()
    selection = native_fixture(source)
    spec_ref, run_ref = selection.jobs[0].spec, selection.dependencies[1]
    spec: Object = json.loads(source.objects[spec_ref.key, spec_ref.version])
    run: Object = json.loads(source.objects[run_ref.key, run_ref.version])
    stage: Object = {"learning_state_origin": {"run_id": "parent-run", "iteration": 63000}}
    if case == "multiple_origins":
        run["stages"] = [stage, stage]
    elif case == "missing_inputs":
        spec["learning_inputs"] = None
    elif case == "unknown_origin_field":
        run["stages"] = [
            {"learning_state_origin": {"run_id": "parent-run", "iteration": 63000, "future": 1}}
        ]
    with pytest.raises(EtudeError):
        continuation(spec, run, 62999 if case == "before_origin" else 63010)
    assert continuation({}, None, None) is None


@pytest.mark.parametrize("case", ["size", "digest", "duplicate_path", "missing_artifact"])
def test_manifest_validates_non_run_entries(case: str) -> None:
    file: Object = {"relative_path": "private.log", "sha256": "a" * 64, "size": 1}
    reference: Object = {"uri": "s3://fixture/private.log", "sha256": "a" * 64, "bytes": 1}
    manifest: Object = {"bundle": {"files": [file]}, "artifacts": [reference]}
    assert manifest_run(manifest) is None
    if case == "size":
        reference["bytes"] = 2
    elif case == "digest":
        reference["sha256"] = "b" * 64
    elif case == "duplicate_path":
        manifest = {"bundle": {"files": [file, file]}, "artifacts": [reference, reference]}
    else:
        manifest["artifacts"] = []
    with pytest.raises(EtudeError):
        manifest_run(manifest)
