"""Explicit, read-only imports of retained Fantasia receipt formats.

No Fantasia Python is imported or executed. JSON access is narrowed here; exact
raw bytes remain the authority for fields the projection does not understand.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from etude_client.errors import EtudeError
from etude_client.models import (
    MAX_BYTES,
    ArtifactRef,
    BudgetObservation,
    BudgetRequest,
    Continuation,
    EvaluationProgress,
    ExperimentRegistration,
    JobSnapshot,
    Observation,
    ObservationBatch,
    ObservationEnvelope,
    QueueObservation,
    ReportObservation,
    ResourceRequest,
    SourceSnapshot,
    TrainingProgress,
)

from .reader import Reader
from .transports import LocalSource
from .values import Object, integer, key, money, number, obj, seq, string, timestamp

CAPACITY = "etu103-recovery-v3-science"
CAPACITY_FILES = (
    "experiments/data/etu126/capacity-recovery-v3.json",
    "experiments/data/etu126/cohort-admission-20261008.json",
    "experiments/data/etu126/controller-launch-20261008.json",
    "experiments/data/etu126/pilot-20261008.json",
    "experiments/model-capacity.md",
)
CONTINUATION_FILES = (
    "experiments/data/etu126/continuation-admission-20261008.json",
    "experiments/data/etu126/continuation-restart-20261008.json",
)
POOLING_FILES = (
    "experiments/data/pooling-filter/evidence.json",
    "experiments/pooling-filter-followup.md",
)

REPORTS = (
    "exp-03-distillation",
    "exp-00c-seat-balanced-baselines",
    "exp-04-potential-shaping",
    "exp-06-newworld-training",
    "exp-11-curriculum-exploitability",
    "exp-07-expert-iteration",
    "ataraxos-technique-screen",
    "history-input",
    "value-token-screen",
)


class FantasiaImport:
    def __init__(
        self,
        root: Path,
        collection: str,
        *,
        continuations: bool = True,
        revision: str | None = None,
    ) -> None:
        source = LocalSource(root, revision)
        self.reader = Reader(source)
        self.collection = collection
        self.revision = source.revision
        self.files = (
            CAPACITY_FILES + (CONTINUATION_FILES if continuations else ())
            if collection == "capacity-recovery-v3"
            else POOLING_FILES
            if collection == "pooling-filter"
            else (f"experiments/{collection}.md",)
            if collection in REPORTS
            else ()
        )
        if not self.files:
            raise EtudeError("unsupported_collection", 422)
        self.refs = {path: self.reader.select(path) for path in self.files}

    def data(self, path: str) -> Object:
        try:
            return self.reader.read(self.refs[path])
        except (ValueError, KeyError):
            raise EtudeError("invalid_source_json", 422) from None

    def registration(self) -> ExperimentRegistration:
        if self.collection in REPORTS:
            title = next(
                (
                    line.lstrip("# ").strip()
                    for line in self.reader.bytes(self.refs[self.files[0]]).decode().splitlines()
                    if line.startswith("# ")
                ),
                self.collection,
            )
            return ExperimentRegistration(
                source_key=key("report", self.collection),
                origin="import",
                title=title,
                source=SourceSnapshot(repository="loopflowstudio/fantasia", revision=self.revision),
            )
        if self.collection == "capacity-recovery-v3":
            protocol = self.data(CAPACITY_FILES[0])
            return ExperimentRegistration(
                source_key=key("collection", CAPACITY),
                origin="import",
                title="Model capacity — recovery v3",
                question="When does model capacity improve learning?",
                comparison=string(obj(protocol["analysis"])["primary"]),
                source=SourceSnapshot(
                    repository="loopflowstudio/fantasia",
                    revision=string(obj(protocol["frozen_source"])["commit"]),
                    definition_id=string(protocol["protocol_id"]),
                    content_sha256=self.refs[CAPACITY_FILES[0]].sha256,
                ),
                budget=BudgetRequest(
                    ceiling_usd=money(obj(protocol["budget"])["overall_ceiling_dollars"])
                ),
                resources=(
                    ResourceRequest(name="RunPod", hardware="L4; 6 vCPU / 48 GiB"),
                    ResourceRequest(name="mini-heart.local", role="controller"),
                ),
            )
        data = self.data(POOLING_FILES[0])
        return ExperimentRegistration(
            source_key=key("report", "pooling-filter-followup"),
            origin="import",
            title="Pooling × advantage floor",
            question="Does removing the advantage floor change learning differently by pooling?",
            comparison="Masked mean/value token × min_advantage 0.01/0; three paired seeds",
            source=SourceSnapshot(
                repository="loopflowstudio/fantasia",
                revision=string(obj(data["supervisor"])["source_commit"]),
                definition_id=string(data["schema"]),
                content_sha256=self.refs[POOLING_FILES[0]].sha256,
            ),
            budget=BudgetRequest(lease_hours=Decimal(8)),
            resources=(ResourceRequest(name="local CPU", hardware="one CPU thread"),),
        )

    def batches(self, experiment_id: UUID) -> tuple[ObservationBatch, ...]:
        records = (
            self._capacity()
            if self.collection == "capacity-recovery-v3"
            else self._pooling()
            if self.collection == "pooling-filter"
            else [
                self.envelope(
                    self.files[0],
                    ReportObservation(
                        source_key=key("report", self.collection), title=self.registration().title
                    ),
                )
            ]
        )
        # Evidence repeats in the wire envelope. Consume and validate the entire
        # translation before returning bounded batches for publication.
        batches: list[ObservationBatch] = []
        chunk: list[ObservationEnvelope] = []
        size = 0
        for record in records:
            length = len(record.model_dump_json().encode()) + 2
            if chunk and size + length > MAX_BYTES - 1000:
                batches.append(ObservationBatch(experiment_id=experiment_id, records=tuple(chunk)))
                chunk, size = [], 0
            chunk.append(record)
            size += length
        if chunk:
            batches.append(ObservationBatch(experiment_id=experiment_id, records=tuple(chunk)))
        return tuple(batches)

    def envelope(
        self,
        path: str,
        observation: Observation,
        *,
        selector: str = "$",
        time: datetime | None = None,
        generation: int | None = None,
    ) -> ObservationEnvelope:
        return self.reader.envelope(
            self.refs[path],
            observation=observation,
            selector=selector,
            source_time=time,
            generation=generation,
        )

    def _capacity(self) -> Iterator[ObservationEnvelope]:
        protocol = self.data(CAPACITY_FILES[0])
        admission = self.data(CAPACITY_FILES[1])
        launch = self.data(CAPACITY_FILES[2])
        revision = string(obj(admission["source"])["commit"])
        jobs = tuple(
            JobSnapshot(
                source_key=key("job", string(j["id"])),
                target_step=integer(j["updates"]),
                lease_hours=money(j["lease_hours"]),
                cap_usd=money(j["cap_dollars"]),
                source_revision=revision,
            )
            for j in (obj(v) for v in seq(admission["jobs"]))
        )
        yield self.envelope(
            CAPACITY_FILES[0],
            ReportObservation(
                source_key=key("protocol", string(protocol["protocol_id"])),
                title="Frozen protocol / recipe",
            ),
        )
        yield self.envelope(
            CAPACITY_FILES[1],
            QueueObservation(source_key=key("cohort", CAPACITY), jobs=jobs, acknowledged=True),
            time=timestamp(admission["frozen_at"]),
        )
        yield self.envelope(
            CAPACITY_FILES[1],
            BudgetObservation(
                source_key=key("budget", CAPACITY + ":reservation"),
                basis="reservation",
                amount_usd=money(admission["inclusive_reserved_dollars"]),
                scope="Inclusive reserved total, including estimates and shared reserves; not additive",
            ),
            selector="$.inclusive_reserved_dollars",
            time=timestamp(admission["frozen_at"]),
        )
        initial = obj(launch["initialization"])
        for index, value in enumerate(seq(launch["advancing_observations"])):
            observation = obj(value)
            record = obj(observation["record"])
            yield self.envelope(
                CAPACITY_FILES[2],
                job_record(
                    string(launch["job_id"]),
                    record,
                    revision=revision,
                    evaluation=EvaluationProgress(
                        completed_games=integer(initial["attempted_games"])
                        - integer(initial["failed_games"]),
                        expected_games=integer(initial["expected_games"]),
                        failed_games=integer(initial["failed_games"]),
                        completed_cohorts=integer(record["evaluations_completed"]),
                        runtime_identity=string(record["pod_id"]),
                        checkpoint_sha256=string(obj(initial["artifact"])["sha256"]),
                    ),
                ),
                selector=f"$.advancing_observations[{index}]",
                generation=integer(record["generation"]),
                time=timestamp(record["heartbeat_at"]),
            )
        pilot = self.data(CAPACITY_FILES[3])
        pilot_record = obj(pilot["job_record"])
        training = obj(pilot["training"])
        pilot_obs = job_record(
            string(pilot["job_id"]),
            pilot_record,
            revision=string(pilot["source_commit"]),
            evaluation=EvaluationProgress(
                completed_games=integer(obj(pilot["initial_evaluation"])["games"]),
                completed_cohorts=integer(pilot_record["evaluations_completed"]),
                runtime_identity=string(pilot_record["pod_id"]),
            ),
        )
        pilot_obs = pilot_obs.model_copy(
            update={
                "progress": TrainingProgress(
                    absolute_step=integer(pilot["iterations"]),
                    learner_transitions=integer(training["learner_transitions"]),
                    optimizer_updates=integer(obj(pilot["numerical"])["optimizer_steps"]),
                    optimizer_exposures=integer(training["optimizer_exposures"]),
                    active_training_seconds=number(training["collection_seconds"])
                    + number(training["learning_seconds"]),
                ),
                "evidence_state": "partial",
                "missing_evidence": ("Original remote endpoint omitted",),
            }
        )
        yield self.envelope(
            CAPACITY_FILES[3],
            pilot_obs,
            selector="$.job_record",
            generation=integer(pilot_record["generation"]),
            time=timestamp(pilot_record["heartbeat_at"]),
        )
        yield self.envelope(
            CAPACITY_FILES[3],
            ReportObservation(
                source_key=key("report", "pilot-separate-cpu-endpoint"),
                title="Separate CPU endpoint recovery; does not repair the original remote curve",
                generated_at=timestamp(obj(pilot["endpoint_recovery"])["finished_at"]),
            ),
            selector="$.endpoint_recovery",
            time=timestamp(obj(pilot["endpoint_recovery"])["finished_at"]),
        )
        yield self.envelope(
            CAPACITY_FILES[4],
            ReportObservation(
                source_key=key("report", "model-capacity"),
                title="Model capacity original report",
            ),
        )
        if CONTINUATION_FILES[0] in self.refs:
            insertion = self.data(CONTINUATION_FILES[0])
            revision = string(obj(insertion["source"])["commit"])
            by_id = {j.source_key.source_id: j for j in jobs}
            allocations = [obj(v) for v in seq(insertion["allocations"])]
            for i, allocation in enumerate(allocations):
                identifier = string(allocation["job_id"])
                by_id[identifier] = JobSnapshot(
                    source_key=key("job", identifier),
                    target_step=integer(allocation["target"]),
                    lease_hours=money(allocation["hours"]),
                    cap_usd=money(allocation["dollars"]),
                    source_revision=revision,
                    scope="exploratory continuation; not a comparison arm",
                    continuation=Continuation(
                        parent_run=key("run", string(insertion["parent_run"])) if i == 0 else None,
                        parent_job=key(
                            "job",
                            jobs[0].source_key.source_id
                            if i == 0
                            else string(allocations[i - 1]["job_id"]),
                        ),
                        origin_step=integer(insertion["parent_iteration"])
                        if i == 0
                        else integer(allocations[i - 1]["target"]),
                    ),
                )
            yield self.envelope(
                CONTINUATION_FILES[0],
                QueueObservation(
                    source_key=key("cohort", CAPACITY),
                    jobs=tuple(by_id[string(v)] for v in seq(insertion["order"])),
                    acknowledged=True,
                ),
                time=timestamp(insertion["observed_at"]),
            )
            for field in (
                "reservation_with_historical_full_caps",
                "charged_and_remaining_reservations",
            ):
                yield self.envelope(
                    CONTINUATION_FILES[0],
                    BudgetObservation(
                        source_key=key("budget", CAPACITY + ":" + field),
                        basis="estimate",
                        amount_usd=money(insertion[field]),
                        scope=field + "; alternative bases, not additive",
                    ),
                    selector="$." + field,
                    time=timestamp(insertion["observed_at"]),
                )
            restart = self.data(CONTINUATION_FILES[1])
            yield self.envelope(
                CONTINUATION_FILES[1],
                JobSnapshot(
                    source_key=key("job", string(insertion["active_job"])),
                    phase=string(restart["phase"]),
                    source_revision=jobs[1].source_revision,
                    resources=(
                        ResourceRequest(
                            name=string(restart["same_active_pod"]), hardware="RunPod L4"
                        ),
                    ),
                    progress=TrainingProgress(absolute_step=integer(restart["job1_updates"])),
                ),
                time=timestamp(restart["observed_at"]),
            )

    def _pooling(self) -> Iterator[ObservationEnvelope]:
        data = self.data(POOLING_FILES[0])
        if data["schema"] != "etu106-pooling-filter-evidence-v1":
            raise EtudeError("unsupported_source_schema", 422)
        supervisor = obj(data["supervisor"])
        revision = string(supervisor["source_commit"])
        yield self.envelope(
            POOLING_FILES[0],
            ReportObservation(
                source_key=key("report", "etu106-pooling-filter-evidence"),
                title="Compact evidence and exact resolved recipes; $.resolved_plan",
            ),
        )
        yield self.envelope(
            POOLING_FILES[1],
            ReportObservation(
                source_key=key("report", "etu106-pooling-filter-report"),
                title="Original pooling × floor report",
            ),
        )
        # No independent execution job ID is exported. Preserve run ID in a scoped
        # retained-attempt key, without claiming a provider allocation identity.
        for index, value in enumerate(seq(data["runs"])):
            run = obj(value)
            stages = [obj(v) for v in seq(run["stages"])]
            yield self.envelope(
                POOLING_FILES[0],
                JobSnapshot(
                    source_key=key("retained-attempt", string(run["id"])),
                    run=key("run", string(run["id"])),
                    source_revision=revision,
                    phase=string(run["status"]),
                    progress=TrainingProgress(
                        absolute_step=integer(supervisor["updates_per_run"]),
                        learner_transitions=sum(integer(s["learner_transitions"]) for s in stages),
                        optimizer_exposures=sum(integer(s["optimizer_exposures"]) for s in stages),
                        active_training_seconds=sum(
                            number(s["collection_seconds"]) + number(s["learning_seconds"])
                            for s in stages
                        ),
                    ),
                    resources=(ResourceRequest(name="local CPU", hardware="one CPU thread"),),
                    artifacts=tuple(
                        artifact(obj(v)) for s in stages for v in obj(s["artifacts"]).values()
                    ),
                    missing_evidence=(
                        "Full private archives linked by compact evidence; not fetched",
                    ),
                ),
                selector=f"$.runs[{index}]",
            )


def artifact(value: Object) -> ArtifactRef:
    return ArtifactRef(
        uri=string(value.get("uri", value.get("path"))),
        sha256=string(value["sha256"]) if "sha256" in value else None,
        size_bytes=integer(value["bytes"]) if "bytes" in value else None,
        version=string(value["version_id"]) if "version_id" in value else None,
    )


def job_record(
    job_id: str,
    record: Object,
    *,
    revision: str | None = None,
    evaluation: EvaluationProgress | None = None,
) -> JobSnapshot:
    """Missing counters stay None, including historical pre-coordinate records."""
    return JobSnapshot(
        source_key=key("job", job_id),
        phase=string(record["phase"]),
        run=key("run", string(record["run_id"])) if record.get("run_id") else None,
        source_revision=revision,
        progress=TrainingProgress(
            absolute_step=integer(record["updates"]) if "updates" in record else None
        ),
        evaluation=evaluation
        or EvaluationProgress(
            completed_cohorts=(
                integer(record["evaluations_completed"])
                if "evaluations_completed" in record
                else None
            )
        ),
        resources=(ResourceRequest(name=string(record["pod_id"]), hardware="remote worker"),)
        if record.get("pod_id")
        else (),
        artifacts=(artifact(obj(record["manifest"])),) if record.get("manifest") else (),
        failure="Source reports a failure; inspect original evidence"
        if record.get("error")
        else None,
        missing_evidence=("Linked manifest and full run exports not verified",),
    )
