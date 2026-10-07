"""Submit and reconnect to one durable rental without a client-owned finalizer.

Every provider create is preceded by a permanent S3 claim. A claimant that dies
before receiving the response can only discover the uniquely named resource; it
cannot rent a replacement. Observer disconnects never cancel remote work.
"""

import json
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING, Callable, Literal

from manabot.infra.artifacts import S3ArtifactStore
from manabot.training.checkpoint_queue import MonitoringBudget

from .deploy import current_source, verify_public_source
from .job_store import JobStore, S3JobStore, cancellation_requested, worker_credentials
from .jobs import (
    DEFAULT_JOBS,
    Cancellation,
    CreateClaim,
    Deletion,
    Job,
    JobRecord,
    JobStatus,
    Resource,
)
from .plan import DeploymentPlan
from .provider import ProviderError, RunPod
from .transport import job_startup, startup

if TYPE_CHECKING:
    from manabot.training.experiments import ResolvedCase

    from .plan import JobSpec, Source


def _provision(
    spec: Job,
    store: JobStore,
    provider: RunPod,
    purpose: Literal["guardian", "training"],
    deadline: float,
    script: str,
    environment: dict[str, str],
) -> Resource:
    key = f"{purpose}.json"
    stored = store.read(key)
    if stored is not None:
        resource = Resource.model_validate_json(stored.data)
        resource.validate_for(spec)
        return resource
    claim_key = f"{purpose}-claim.json"
    stored_claim = store.read(claim_key)
    if stored_claim is not None:
        claim = CreateClaim.model_validate_json(stored_claim.data)
    else:
        claim = CreateClaim(
            spec_sha256=spec.identity,
            name=f"manabot-{spec.identity[:32]}-{purpose}",
            purpose=purpose,
            intent_time=time.time(),
            deadline=deadline,
        )
    prices = provider.prices() if stored_claim is None else {}
    machine = spec.plan.spec.machine
    candidates = [
        g
        for g in machine.gpu_types
        if 0 < prices.get(g, float("inf")) <= machine.hourly_ceiling
    ]
    if stored_claim is None and not candidates:
        raise ValueError("no declared GPU has an admitted live price")
    winner = stored_claim is None and store.create(
        claim_key, claim.model_dump_json().encode()
    )
    if winner:
        if time.time() + 20 >= deadline:
            raise TimeoutError(
                "creation deadline expired; retained claim will not be retried"
            )
        # The fence is intentionally permanent even if this process dies here.
        pod = provider.create(
            {
                "name": claim.name,
                "imageName": machine.image,
                "cloudType": "SECURE",
                "computeType": "GPU",
                "gpuCount": 1,
                "gpuTypeIds": candidates,
                "gpuTypePriority": "custom",
                "minVCPUPerGPU": machine.vcpus,
                "minRAMPerGPU": machine.memory_gb,
                "containerDiskInGb": machine.container_gb,
                "volumeInGb": machine.volume_gb,
                "volumeMountPath": "/workspace",
                "dockerEntrypoint": ["/bin/bash", "-c"],
                "dockerStartCmd": [script],
                "env": environment,
            }
        )
    else:
        # A loser must load the winning claim (its timestamp may differ).
        existing = store.read(claim_key)
        if existing is None:
            raise RuntimeError(
                "creation claim unavailable; reconcile, do not resubmit a new ID"
            )
        claim = CreateClaim.model_validate_json(existing.data)
        matches = [p for p in provider.list() if p.name == claim.name]
        if len(matches) != 1:
            raise RuntimeError(
                "provider creation ambiguous; inspect this job ID, never retry with a new ID"
            )
        pod = matches[0]
    resource = Resource(claim=claim, pod=pod)
    # Retain bad assignments too, so explicit cancellation knows exact ownership.
    if not store.create(key, resource.model_dump_json().encode()):
        observed = store.read(key)
        if observed is None:
            raise RuntimeError("resource receipt unavailable")
        resource = Resource.model_validate_json(observed.data)
        if resource.pod.id != pod.id:
            raise ValueError(
                "conflicting provider resources; manual reconciliation required"
            )
    try:
        resource.validate_for(spec)
    except ValueError:
        # Failed price/resource admission is before remote acceptance. Do not
        # leave a rejected allocation billing through the full setup reserve.
        if resource.pod.name == resource.claim.name:
            provider.delete(resource.pod.id)
        raise
    return resource


def prepare_job(
    plan: DeploymentPlan,
    job_id: str,
    *,
    destination: str | None = None,
    monitoring: MonitoringBudget | None = None,
    checkpoint_seconds: float = 60,
    publish_seconds: float = 30,
    experiment_json: str | None = None,
    store: JobStore | None = None,
) -> Job:
    """Persist intent before renting; repeated IDs retain the first absolute deadline."""
    plan.allocation_at(time.time())
    now = time.time()
    spec = Job(
        job_id=job_id,
        plan=plan,
        created_at=now,
        deadline=plan.deadline_at(now),
        destination=destination
        if destination is not None
        else plan.spec.access.destination,
        monitoring=monitoring,
        checkpoint_seconds=checkpoint_seconds,
        publish_seconds=publish_seconds,
        experiment_json=experiment_json,
    )
    store = store or S3JobStore(spec.prefix)
    if not store.create("spec.json", spec.model_dump_json().encode()):
        existing = store.read("spec.json")
        if existing is None:
            raise RuntimeError("job intent unavailable; reconcile by ID")
        previous = Job.model_validate_json(existing.data)
        if previous.model_dump(exclude={"created_at", "deadline"}) != spec.model_dump(
            exclude={"created_at", "deadline"}
        ):
            raise ValueError("job ID already binds different immutable content")
        spec = previous
    # S3 returns 403, not 404, for missing keys to a role without ListBucket.
    # Provision the readable mailbox before delegation; no bucket-list privilege
    # is needed just to distinguish "no cancellation requested".
    store.create("cancel.json", Cancellation().model_dump_json().encode())
    return spec


def submit_job(
    spec: Job,
    *,
    store: JobStore | None = None,
    provider: RunPod | None = None,
    credentials: Callable[[Job], dict[str, str]] = worker_credentials,
) -> JobStatus:
    """Provision up to acceptance; interruption is reconciled with the same spec.

    The worker starts from the provider's startup command, not a connected SSH
    process. The separate startup guardian bounds billing before any bootstrap.
    """
    spec.plan.allocation_at(spec.created_at)
    store = store or S3JobStore(spec.prefix)
    spec = Job.model_validate_json(spec.model_dump_json())
    intent = store.read("spec.json")
    if intent is None or Job.model_validate_json(intent.data).identity != spec.identity:
        raise ValueError("submission differs from durable job intent")
    provider = provider or RunPod()
    if store.read("runtime/record.json") is not None:
        return job_status(spec, store=store, provider=provider)
    collection_cutoff = (
        spec.allocation.pause_at if spec.allocation is not None else spec.work_deadline
    )
    if time.time() >= collection_cutoff or cancellation_requested(store) is not None:
        raise ValueError(
            "job deadline expired or cancellation requested; inspect/cancel this ID"
        )
    if (
        store.read("training-claim.json") is None
        and current_source(Path.cwd()) != spec.plan.source
    ):
        raise ValueError(
            "new submission requires the exact clean source of the compiled plan"
        )
    verify_public_source(spec.plan.source)
    # Resolve credentials before acquiring the provider-create fence or spending.
    environment = credentials(spec)
    probe_key = "guardian-proof.json"
    if store.read(probe_key) is None:
        deadline = min(spec.created_at + 90, spec.work_deadline)
        # No SSH connection or private key is needed to prove self-deletion.
        probe = _provision(
            spec,
            store,
            provider,
            "guardian",
            deadline,
            startup(int(deadline), "ssh-ed25519 AAAA manabot-probe"),
            {},
        )
        while time.time() < deadline + 30:
            if provider.get(probe.pod.id) is None:
                if time.time() < deadline - 2:
                    raise ValueError("guardian probe disappeared before its deadline")
                store.create(
                    probe_key, json.dumps({"confirmed_at": time.time()}).encode()
                )
                break
            time.sleep(2)
        else:
            raise RuntimeError(
                "guardian deletion unconfirmed; cancel this job before further rental"
            )
    environment.update({"MANABOT_JOB_PREFIX": spec.prefix})
    _provision(
        spec, store, provider, "training", spec.deadline, job_startup(spec), environment
    )
    # Once launched, all setup/training/upload/deletion is remote-owned. A lost
    # acknowledgement is resolved by this same read; no finally block cancels it.
    while time.time() < spec.created_at + spec.plan.spec.setup_seconds:
        status = job_status(spec, store=store, provider=provider)
        if status.record is not None:
            return status
        if status.provider_state == "absent":
            raise RuntimeError(
                "rental disappeared before remote acceptance; evidence may be incomplete"
            )
        time.sleep(2)
    raise TimeoutError(
        f"remote acceptance not observed; reconnect to job {spec.job_id}"
    )


def load_job(job_id: str, destination: str = DEFAULT_JOBS) -> Job:
    # Validate path input before constructing an S3 key.
    import re

    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", job_id) is None:
        raise ValueError("invalid job ID")
    value = S3JobStore(f"{destination.rstrip('/')}/{job_id}").read("spec.json")
    if value is None:
        raise ValueError("remote job ID is unknown")
    spec = Job.model_validate_json(value.data)
    if spec.job_id != job_id or spec.destination != destination:
        raise ValueError("stored job location differs")
    return spec


def job_status(
    spec: Job,
    *,
    store: JobStore | None = None,
    provider: RunPod | None = None,
) -> JobStatus:
    store = store or S3JobStore(spec.prefix)
    raw = store.read("runtime/record.json")
    record = JobRecord.model_validate_json(raw.data) if raw else None
    if record is not None and record.spec_sha256 != spec.identity:
        raise ValueError("supervisor record belongs to a different job")
    state = "not-created"
    unsettled = False
    costs = 0.0
    for purpose in ("guardian", "training"):
        claim_raw, resource_raw = (
            store.read(f"{purpose}-claim.json"),
            store.read(f"{purpose}.json"),
        )
        if claim_raw is None:
            continue
        claim = CreateClaim.model_validate_json(claim_raw.data)
        resource = (
            Resource.model_validate_json(resource_raw.data) if resource_raw else None
        )
        if resource is None:
            unsettled, state = True, "ambiguous"
            continue
        deletion_raw = store.read(f"{purpose}-deletion.json")
        if deletion_raw is not None:
            costs += Deletion.model_validate_json(deletion_raw.data).estimated_dollars
            state = "absent"
            continue
        try:
            provider = provider or RunPod()
            pod = provider.get(resource.pod.id)
            if pod is not None:
                if pod.name != claim.name:
                    raise ProviderError("resource ownership differs")
                unsettled, state = True, "present"
                continue
            ended = time.time()
            deletion = Deletion(
                confirmed_at=ended,
                estimated_dollars=(ended - claim.intent_time)
                * (resource.pod.rate + spec.plan.spec.machine.storage_hourly_allowance)
                / 3600,
            )
            store.create(
                f"{purpose}-deletion.json", deletion.model_dump_json().encode()
            )
            saved = store.read(f"{purpose}-deletion.json")
            if saved is None:
                raise RuntimeError("deletion receipt unavailable")
            costs += Deletion.model_validate_json(saved.data).estimated_dollars
            state = "absent"
        except ProviderError:
            unsettled, state = True, "unknown"
    cleanup = (
        None
        if unsettled or state == "not-created"
        else Deletion(confirmed_at=time.time(), estimated_dollars=costs)
    )
    return JobStatus(
        spec=spec,
        record=record,
        provider_state=state,
        cleanup=cleanup,
        cancel_requested_at=cancellation_requested(store),
    )


def cancel_job(spec: Job, *, store: JobStore | None = None) -> None:
    """Request cancellation once; concurrent callers preserve its first timestamp."""
    store = store or S3JobStore(spec.prefix)
    request = Cancellation(requested_at=time.time()).model_dump_json().encode()
    while True:
        current = store.read("cancel.json")
        if current is None:
            if store.create("cancel.json", request):
                return
        elif Cancellation.model_validate_json(current.data).requested_at is not None:
            return
        elif store.replace("cancel.json", request, current.etag):
            return


def fetch_job(spec: Job, output: Path) -> Path:
    from .snapshots import JobManifest

    raw = S3JobStore(spec.prefix).read("runtime/record.json")
    if raw is None:
        raise ValueError("job has no published supervisor record")
    record = JobRecord.model_validate_json(raw.data)
    if record.spec_sha256 != spec.identity or record.manifest is None:
        raise ValueError("job has no committed artifact generation")
    artifacts = S3ArtifactStore()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache = output / "cache"
    manifest = JobManifest.model_validate_json(
        artifacts.fetch(record.manifest, cache).read_bytes()
    )
    if (
        manifest.spec_sha256 != spec.identity
        or manifest.generation != record.generation
    ):
        raise ValueError("artifact generation differs from supervisor record")
    destination = output / f"generation-{manifest.generation:06d}"
    destination.mkdir(exist_ok=False, mode=0o700)
    for entry, reference in zip(manifest.bundle.files, manifest.artifacts, strict=True):
        target = entry.destination(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artifacts.fetch(reference, cache), target)
    manifest.bundle.verify(destination)
    (destination / "bundle.json").write_text(manifest.bundle.model_dump_json(indent=2))
    (destination / "job.json").write_text(record.model_dump_json(indent=2))
    return destination


def prepare_experiment_job(
    case: "ResolvedCase",
    spec: "JobSpec",
    source: "Source",
    seed: int,
    job_id: str,
    *,
    monitoring: MonitoringBudget,
    destination: str | None = None,
    checkpoint_seconds: float = 60,
) -> Job:
    """Compile one resolved Experiment case through the ordinary job lifecycle.

    The caller owns case/seed/cohort selection and cumulative campaign allocation;
    this API never starts a hidden experiment scheduler or revises the declaration.
    """
    from .plan import compile_plan

    plan = compile_plan(case.configuration, spec, source, seed)
    return prepare_job(
        plan,
        job_id,
        destination=destination
        if destination is not None
        else plan.spec.access.destination,
        monitoring=monitoring,
        checkpoint_seconds=checkpoint_seconds,
        experiment_json=json.dumps(case.receipt(), sort_keys=True),
    )


def reconcile_job(spec: Job, *, delete: bool = False) -> JobStatus:
    """Discover uncertain creates; optionally delete only exact job-owned resources.

    Force deletion can discard an unpublished final generation. Normal cancel
    leaves finalization to the worker; this path settles failed bootstrap/outages.
    """
    store, provider = S3JobStore(spec.prefix), RunPod()
    for purpose in ("guardian", "training"):
        raw = store.read(f"{purpose}-claim.json")
        if raw is None:
            continue
        claim = CreateClaim.model_validate_json(raw.data)
        if claim.spec_sha256 != spec.identity:
            raise ValueError("creation claim belongs to another job")
        pods = [p for p in provider.list() if p.name == claim.name]
        if len(pods) > 1:
            raise ValueError(
                "multiple rentals share this job claim; manual reconciliation required"
            )
        if pods:
            resource = Resource(claim=claim, pod=pods[0])
            existing = store.read(f"{purpose}.json")
            if (
                existing is not None
                and Resource.model_validate_json(existing.data).pod.id
                != resource.pod.id
            ):
                raise ValueError("conflicting rental identity")
            store.create(f"{purpose}.json", resource.model_dump_json().encode())
            if delete:
                provider.delete(resource.pod.id)
    return job_status(spec, store=store, provider=provider)
