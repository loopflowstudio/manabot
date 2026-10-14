"""Read native, version-pinned Fantasia evidence without importing its runtime.

This private adapter runs beside the tenant's read credentials, not in dashboard
GETs. The selection is transport metadata only: observations come from native
job, cohort and report bytes. It cannot list, write, claim or reconcile jobs.
Schemas follow Fantasia 220c5730; plan v1/v2, TrainingRun/1 and Cohort/1 are supported.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from etude_client.api import Client
from etude_client.collector import ObjectRef, ReadSource, refresh_batch
from etude_client.errors import EtudeError
from etude_client.models import (
    Continuation,
    JobSnapshot,
    Model,
    Observation,
    ObservationBatch,
    ObservationEnvelope,
    QueueObservation,
    ReportObservation,
    SourceKey,
)
from pydantic import Field, model_validator

from .fantasia import artifact, job_record
from .reader import Reader
from .transports import S3Source
from .values import Object, integer, key, money, number, obj, seq, string, timestamp


class NativeJob(Model):
    spec: ObjectRef
    record: ObjectRef


class NativeCohort(Model):
    spec: ObjectRef
    state: ObjectRef
    # Explicit versioned export of the projection owner's local report.json.
    report: ObjectRef | None = None


class NativeSelection(Model):
    schema_version: Literal[1] = 1
    jobs: tuple[NativeJob, ...] = Field(default=(), max_length=100)
    cohort: NativeCohort | None = None
    # Exact S3 URIs/version IDs/digests; no URLs or observations enter evidence.
    dependencies: tuple[ObjectRef, ...] = Field(default=(), max_length=500)

    @model_validator(mode="after")
    def nonempty(self) -> Self:
        if not self.jobs and self.cohort is None:
            raise ValueError("native_selection_empty")
        return self


class NativeTranslation:
    """Schema translation only; Reader owns bytes, dependencies and envelopes."""

    def __init__(self, reader: Reader) -> None:
        self.reader = reader
        self.records: list[ObservationEnvelope] = []

    def retain(
        self,
        ref: ObjectRef,
        observation: Observation,
        schema: str,
        observed: datetime | None = None,
    ) -> None:
        self.records.append(
            self.reader.envelope(ref, observation, schema=schema, source_time=observed)
        )

    def receipt(self, ref: ObjectRef, schema: str) -> None:
        # A receipt is not a freshly generated scientific report. No source clock
        # is invented for manifests/run exports that do not publish one.
        self.retain(
            ref,
            ReportObservation(
                source_key=key("receipt", ref.key),
                title="Native " + schema,
            ),
            schema,
        )

    def job(self, selected: NativeJob) -> JobSnapshot:
        spec, record = self.reader.read(selected.spec), self.reader.read(selected.record)
        identifier = string(spec["job_id"])
        prefix = string(spec["destination"]).rstrip("/") + "/" + identifier
        if (
            selected.spec.key != prefix + "/spec.json"
            or selected.record.key != prefix + "/runtime/record.json"
        ):
            raise EtudeError("invalid", 422)
        # Native spec.json is the exact compact serialization hashed by Job.identity.
        # Never reserialize it (historical optional-field presence affects identity).
        if string(record["spec_sha256"]) != selected.spec.sha256:
            raise EtudeError("invalid", 422)
        plan = obj(spec["plan"])
        if integer(plan["schema_version"]) not in (1, 2):
            raise EtudeError("invalid", 422)
        revision = string(obj(plan["source"])["commit"])
        observation = job_record(identifier, record, revision=revision)
        observed = timestamp(record["heartbeat_at"])
        generation = integer(record["generation"])
        self.receipt(selected.spec, "Job")
        run: Object | None = None
        if record.get("manifest") is not None:
            manifest_ref, manifest = self.reader.dependency(obj(record["manifest"]))
            if (
                manifest["spec_sha256"] != selected.spec.sha256
                or integer(manifest["generation"]) != generation
            ):
                raise EtudeError("invalid", 422)
            self.receipt(manifest_ref, "JobManifest")
            run_reference = manifest_run(manifest)
            if run_reference is not None:
                run_ref, run = self.reader.dependency(run_reference, run=True)
                if integer(run["schema_version"]) != 1 or run["id"] != record.get("run_id"):
                    raise EtudeError("invalid", 422)
                self.receipt(run_ref, "TrainingRun/1")
        # Validate derived fields directly; model_copy would skip validation.
        observation = JobSnapshot.model_validate(
            observation.model_dump()
            | {
                "continuation": continuation(spec, run, observation.progress.absolute_step),
                "publish_seconds": number(spec["publish_seconds"]),
                "missing_evidence": (
                    "Native receipts and run identity/ancestry imported; full artifacts and dense series not retained",
                ),
            }
        )
        # JobRecord.generation counts manifest publications, not heartbeats.
        # Order this stream by heartbeat; retain the artifact generation in bytes.
        self.retain(selected.record, observation, "JobRecord", observed)
        return observation

    def cohort(
        self, selected: NativeCohort, definitions: tuple[JobSnapshot, ...] = ()
    ) -> QueueObservation:
        spec, state = self.reader.read(selected.spec), self.reader.read(selected.state)
        identifier = string(spec["cohort_id"])
        prefix = string(spec["destination"]).rstrip("/") + "/" + identifier
        if (
            integer(spec["schema_version"]) != 1
            or selected.spec.key != prefix + "/spec.json"
            or selected.state.key != prefix + "/state.json"
            or state["cohort_sha256"] != selected.spec.sha256
        ):
            raise EtudeError("invalid", 422)
        entries = [obj(v) for v in seq(spec["entries"])]
        if state.get("insertion") is not None:
            insertion = obj(state["insertion"])
            original_ids = [string(e["job_id"]) for e in entries]
            index = original_ids.index(string(insertion["after_job"])) + 1
            entries[index:index] = [obj(v) for v in seq(insertion["entries"])]
        frozen = {definition.source_key: definition for definition in definitions}
        jobs = tuple(bind_definition(cohort_entry(e), frozen) for e in entries)
        ids = {j.source_key.source_id for j in jobs}
        if not jobs or len(jobs) > 100 or len(ids) != len(jobs):
            raise EtudeError("invalid", 422)
        observed = source_heartbeat(state.get("heartbeat_at"))
        observation = QueueObservation(
            source_key=key("cohort", identifier), jobs=jobs, acknowledged=True
        )
        self.receipt(selected.spec, "Cohort/1")
        self.retain(selected.state, observation, "CohortState", observed)
        if selected.report is not None:
            self.report(selected.report, selected.spec, observation)
        return observation

    def report(
        self, ref: ObjectRef, spec: ObjectRef, cohort: QueueObservation
    ) -> ReportObservation:
        report = self.reader.read(ref)
        if report["cohort_sha256"] != spec.sha256:
            raise EtudeError("invalid", 422)
        generated = source_heartbeat(report.get("observed_at"))
        rows = [obj(v) for v in seq(report["jobs"])]
        row_ids = [string(row["job_id"]) for row in rows]
        ids = {job.source_key.source_id for job in cohort.jobs}
        if len(set(row_ids)) != len(row_ids) or not set(row_ids) <= ids:
            raise EtudeError("invalid", 422)
        # These rows default missing counters to zero in Fantasia. Do not
        # turn them into JobSnapshots or completed evaluation games.
        clocks = [source_heartbeat(report.get("supervisor_heartbeat_at"))]
        clocks.extend(source_heartbeat(row.get("heartbeat_at")) for row in rows)
        dated = [clock for clock in clocks if clock is not None]
        watermark = (
            min(dated)
            if len(dated) == len(clocks) and not any(row.get("error") for row in rows)
            else None
        )
        if generated is not None and any(clock > generated for clock in dated):
            raise EtudeError("invalid", 422)
        observation = ReportObservation(
            source_key=key("report", cohort.source_key.source_id + ":projection"),
            title="Cohort monitoring report (not a scientific conclusion)",
            generated_at=generated,
            evidence_watermark=watermark,
        )
        self.retain(ref, observation, "CohortProjectionReport", generated)
        return observation


def source_heartbeat(value: object) -> datetime | None:
    """Native zero is the unpublished-clock default, not Unix-epoch evidence."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise EtudeError("invalid", 422)
    return timestamp(value) if value else None


def cohort_entry(entry: Object) -> JobSnapshot:
    """Project allocation/absolute target, retaining the full recipe in spec/state bytes."""
    plan = obj(entry["plan"])
    version = integer(plan["schema_version"])
    if version == 2:
        allocation = obj(plan["spec"])
        hours, cap = money(allocation["lifetime_hours"]), money(allocation["spending_limit"])
    elif version == 1:
        allocation = obj(plan["mix"])
        hours, cap = money(allocation["wall_seconds"]) / 3600, money(allocation["dollar_cap"])
    else:
        raise EtudeError("invalid", 422)
    stages = [obj(v) for v in seq(obj(plan["regime"])["stages"])]
    # Multiple stages have distinct coordinates; active-time recipes use updates
    # as a safety ceiling, not a scientific endpoint. Neither is flattened.
    target = None
    if len(stages) == 1:
        stage = stages[0]
        if stage.get("operation") == "train_self_play" and stage.get("active_seconds") is None:
            target = integer(stage["updates"]) if "updates" in stage else None
    origin = obj(entry["continuation"]) if entry.get("continuation") is not None else None
    return JobSnapshot(
        source_key=key("job", string(entry["job_id"])),
        target_step=target,
        lease_hours=hours,
        cap_usd=cap,
        source_revision=string(obj(plan["source"])["commit"]),
        scope="Native cohort entry; scientific role unspecified",
        continuation=Continuation(
            parent_job=key("job", string(origin["job_id"])),
            origin_step=integer(origin["iteration"]),
        )
        if origin is not None
        else None,
    )


def bind_definition(native: JobSnapshot, frozen: dict[SourceKey, JobSnapshot]) -> JobSnapshot:
    """Native operational receipts cannot replace historical scientific scope/ancestry.

    Compare fields actually supplied by this native projection. The existing
    ingestion transaction still checks the complete frozen definition, so a race
    fails closed rather than permitting a rewrite.
    """
    known = frozen.get(native.source_key)
    if known is None:
        return native
    if any(
        getattr(native, field) != getattr(known, field)
        for field in ("target_step", "lease_hours", "cap_usd", "source_revision")
    ):
        raise EtudeError("invalid", 422)
    if (native.continuation is None) != (known.continuation is None):
        raise EtudeError("invalid", 422)
    if native.continuation is not None and known.continuation is not None:
        if (
            native.continuation.parent_job != known.continuation.parent_job
            or native.continuation.origin_step != known.continuation.origin_step
        ):
            raise EtudeError("invalid", 422)
    return known


def manifest_run(manifest: Object) -> Object | None:
    """Validate the whole bundle, selecting only the run JSON for download."""
    files = [obj(v) for v in seq(obj(manifest["bundle"])["files"])]
    artifacts = [obj(v) for v in seq(manifest["artifacts"])]
    if len(files) != len(artifacts) or len({string(f["relative_path"]) for f in files}) != len(
        files
    ):
        raise EtudeError("invalid", 422)
    run = None
    for item, reference in zip(files, artifacts, strict=True):
        if item["sha256"] != reference["sha256"] or integer(item["size"]) != integer(
            reference["bytes"]
        ):
            raise EtudeError("invalid", 422)
        # Logs, checkpoints, code, databases and arbitrary URLs are never followed.
        if item["relative_path"] == "run/run.json":
            run = reference
    return run


# One mapping defines both the supported optional origin fields and their wire names.
# New source fields must be represented explicitly rather than silently discarded.
ORIGIN_FIELDS = {
    "stage_id": "origin_stage",
    "games": "origin_games",
    "environment_decisions": "origin_environment_decisions",
    "learner_transitions": "origin_learner_transitions",
    "optimizer_updates": "origin_optimizer_updates",
    "optimizer_exposures": "origin_optimizer_exposures",
    "active_training_seconds": "origin_active_training_seconds",
    "cumulative_seconds": "origin_cumulative_seconds",
    "streams": "origin_streams",
}


def continuation(
    spec: Object, run: Object | None, absolute_step: int | None
) -> Continuation | None:
    """Retain input digests even before a run export establishes its single origin."""
    stages = [obj(v) for v in seq(run["stages"])] if run is not None else []
    origins = [
        obj(s["learning_state_origin"])
        for s in stages
        if s.get("learning_state_origin") is not None
    ]
    if len(origins) > 1:
        # The tracker has one continuation edge, not a stage ancestry DAG.
        raise EtudeError("invalid", 422)
    recovery_parent = run.get("parent_run_id") if run else None
    recovery = run.get("recovery_artifact") if run else None
    if (
        not origins
        and not spec.get("learning_inputs")
        and recovery_parent is None
        and recovery is None
    ):
        return None
    if origins and not spec.get("learning_inputs"):
        raise EtudeError("invalid", 422)
    origin = origins[0] if origins else None
    origin_step = integer(origin["iteration"]) if origin is not None else None
    if origin_step is not None and absolute_step is not None and absolute_step < origin_step:
        raise EtudeError("invalid", 422)
    inputs = obj(spec["learning_inputs"]) if spec.get("learning_inputs") is not None else None
    # Field presence survives older exports: absent counters are not zero.
    if origin is not None and origin.keys() - ORIGIN_FIELDS.keys() - {"run_id", "iteration"}:
        raise EtudeError("unsupported_origin_fields", 422)
    origin_fields = {
        target: origin[source]
        for source, target in ORIGIN_FIELDS.items()
        if origin is not None and source in origin
    }
    return Continuation.model_validate(
        {
            **origin_fields,
            "parent_run": key("run", string(origin["run_id"])) if origin else None,
            "origin_step": origin_step,
            "recovery_parent_run": key("run", string(recovery_parent))
            if recovery_parent is not None
            else None,
            "recovery_artifact": artifact(obj(recovery)) if recovery is not None else None,
            "checkpoint_sha256": artifact(obj(inputs["raw"])).sha256 if inputs else None,
            "ema_sha256": artifact(obj(inputs["ema"])).sha256 if inputs else None,
            "optimizer_sha256": artifact(obj(inputs["optimizer"])).sha256 if inputs else None,
            "source_run_sha256": artifact(obj(inputs["source_run"])).sha256 if inputs else None,
        }
    )


def refresh_native(
    client: Client,
    source: ReadSource | S3Source,
    experiment_id: UUID,
    selection: NativeSelection | None = None,
    name: str = "fantasia-native",
) -> None:
    """One atomic session for pinned HTTPS and current S3 selections."""

    def collect(deadline: float) -> ObservationBatch:
        try:
            reader = Reader(
                source,
                deadline=deadline,
                dependencies=selection.dependencies if selection else None,
            )
            selected = selection
            if selected is None and isinstance(source, S3Source):
                selected = NativeSelection(
                    jobs=tuple(
                        NativeJob(
                            spec=reader.select(prefix + "/spec.json"),
                            record=reader.select(prefix + "/runtime/record.json"),
                        )
                        for prefix in source.prefixes
                    )
                )
            if selected is None:
                raise EtudeError("invalid", 422)
            translation = NativeTranslation(reader)
            ids: set[SourceKey] | None = None
            if selected.cohort is not None:
                detail = client.get(experiment_id)
                definitions = tuple(
                    job
                    for fact in detail.facts
                    if isinstance(fact.observation, QueueObservation)
                    for job in fact.observation.jobs
                )
                cohort = translation.cohort(selected.cohort, definitions)
                if detail.registration.source_key != key("collection", cohort.source_key.source_id):
                    raise EtudeError("invalid", 422)
                ids = {job.source_key for job in cohort.jobs}
            for job in selected.jobs:
                observed = translation.job(job)
                if ids is not None and observed.source_key not in ids:
                    raise EtudeError("invalid", 422)
            return ObservationBatch(experiment_id=experiment_id, records=tuple(translation.records))
        except (KeyError, ValueError, TypeError, OverflowError, EtudeError) as error:
            if isinstance(error, EtudeError) and error.status != 422:
                raise
            raise EtudeError("invalid", 422) from None

    refresh_batch(
        client,
        experiment_id,
        name,
        collect,
        timeout_seconds=1800 if isinstance(source, S3Source) else 45,
    )
