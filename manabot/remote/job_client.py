"""Submit and reconnect to one durable rental without a client-owned finalizer.

Every provider create is preceded by a permanent S3 claim. A claimant that dies
before receiving the response can only discover the uniquely named resource; it
cannot rent a replacement. Observer disconnects never cancel remote work.
"""

import fcntl
import json
from pathlib import Path
import shutil
import tempfile
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
from .plan import DeploymentPlan, compile_plan
from .provider import ProviderError, RunPod
from .transport import job_startup, startup

if TYPE_CHECKING:
    from manabot.training.experiments import ResolvedCase

    from .plan import JobSpec, Source
    from .snapshots import JobManifest


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
    checkpoint_seconds: float | None = None,
    checkpoint_updates: int | None = None,
    publish_seconds: float = 30,
    experiment_json: str | None = None,
    validate_numerics: bool = False,
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
        checkpoint_updates=checkpoint_updates,
        publish_seconds=publish_seconds,
        experiment_json=experiment_json,
        validate_numerics=validate_numerics,
    )
    return persist_job(spec, store=store, retain_existing_deadline=True)


def persist_job(
    spec: Job,
    *,
    store: JobStore | None = None,
    retain_existing_deadline: bool = False,
) -> Job:
    """Persist a pre-admitted intent; cohorts require its exact original deadline."""
    spec.plan.allocation_at(spec.created_at)
    store = store or S3JobStore(spec.prefix)
    if not store.create("spec.json", spec.model_dump_json().encode()):
        existing = store.read("spec.json")
        if existing is None:
            raise RuntimeError("job intent unavailable; reconcile by ID")
        previous = Job.model_validate_json(existing.data)
        excluded = {"created_at", "deadline"} if retain_existing_deadline else set()
        if previous.model_dump(exclude=excluded) != spec.model_dump(exclude=excluded):
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
    source_root: Path | None = None,
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
        and current_source(source_root or Path.cwd()) != spec.plan.source
    ):
        raise ValueError(
            "new submission requires the exact clean source of the compiled plan"
        )
    verify_public_source(spec.plan.source)
    # Resolve credentials before acquiring the provider-create fence or spending.
    environment = credentials(spec)
    probe_key = "guardian-proof.json"
    if store.read(probe_key) is None:
        deadline = min(time.time() + 90, spec.work_deadline)
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
        # A reconnect observes the original probe, never a renewed deadline.
        deadline = probe.claim.deadline
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
    training = _provision(
        spec, store, provider, "training", spec.deadline, job_startup(spec), environment
    )
    # Once launched, all setup/training/upload/deletion is remote-owned. A lost
    # acknowledgement is resolved by this same read; no finally block cancels it.
    acceptance_deadline = min(
        training.claim.intent_time + spec.plan.spec.setup_seconds, spec.work_deadline
    )
    while time.time() < acceptance_deadline:
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
    training_claimed = False
    costs = 0.0
    for purpose in ("guardian", "training"):
        claim_raw, resource_raw = (
            store.read(f"{purpose}-claim.json"),
            store.read(f"{purpose}.json"),
        )
        if claim_raw is None:
            continue
        claim = CreateClaim.model_validate_json(claim_raw.data)
        if claim.spec_sha256 != spec.identity or claim.purpose != purpose:
            raise ValueError("creation claim belongs to a different job or purpose")
        training_claimed |= purpose == "training"
        resource = (
            Resource.model_validate_json(resource_raw.data) if resource_raw else None
        )
        if resource is None:
            unsettled, state = True, "ambiguous"
            continue
        if resource.claim != claim or resource.pod.name != claim.name:
            raise ValueError("resource receipt differs from creation claim")
        if (
            purpose == "training"
            and record is not None
            and record.pod_id != resource.pod.id
        ):
            raise ValueError("supervisor record belongs to a different rental")
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
        if unsettled or not training_claimed or state == "not-created"
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


def job_manifest(
    spec: Job,
    record: JobRecord,
    cache: Path,
    *,
    artifacts: S3ArtifactStore | None = None,
) -> "JobManifest":
    """Read one pinned manifest, without fetching its referenced model bundles."""
    from .snapshots import JobManifest

    if record.spec_sha256 != spec.identity or record.manifest is None:
        raise ValueError("job has no committed artifact generation")
    if record.manifest.bytes > 16 * 1024**2:
        raise ValueError("job manifest exceeds control-object limit")
    artifacts = artifacts or S3ArtifactStore()
    manifest = JobManifest.model_validate_json(
        artifacts.fetch(record.manifest, cache).read_bytes()
    )
    if (
        manifest.spec_sha256 != spec.identity
        or manifest.generation != record.generation
    ):
        raise ValueError("artifact generation differs from supervisor record")
    return manifest


def fetch_job_file(
    spec: Job,
    record: JobRecord,
    relative_path: str,
    cache: Path,
) -> Path | None:
    """Return verified bytes for one manifest entry; None means it was not published.

    The caller supplies the observed record, so a newer remote generation cannot
    change what is fetched. This never claims to materialize a complete Bundle.
    """
    artifacts = S3ArtifactStore()
    manifest = job_manifest(spec, record, cache, artifacts=artifacts)
    for entry, reference in zip(manifest.bundle.files, manifest.artifacts, strict=True):
        if entry.relative_path == relative_path:
            entry.destination(cache)  # Retain ordinary Bundle path admission.
            return artifacts.fetch(reference, cache)
    return None


def fetch_job(spec: Job, output: Path) -> Path:
    """Materialize a verified generation; retry resumes a partial fetch in place.

    Complete files are installed atomically, and the bundle marker is written
    last. Existing evidence is hash-checked, never overwritten on mismatch.
    A local lock serializes clients sharing this output directory.
    """
    from .bundle import Bundle

    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / ".fetch.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = S3JobStore(spec.prefix).read("runtime/record.json")
        if raw is None:
            raise ValueError("job has no published supervisor record")
        record = JobRecord.model_validate_json(raw.data)
        artifacts = S3ArtifactStore()
        cache = output / "cache"
        manifest = job_manifest(spec, record, cache, artifacts=artifacts)
        destination = output / f"generation-{manifest.generation:06d}"
        if destination.is_symlink():
            raise ValueError("bundle root is a symlink")
        destination.mkdir(exist_ok=True, mode=0o700)
        metadata = {"bundle.json", "job.json"}
        if any(entry.relative_path in metadata for entry in manifest.bundle.files):
            raise ValueError("snapshot contains reserved transport metadata")
        bundle_path, record_path = destination / "bundle.json", destination / "job.json"
        if bundle_path.is_symlink() or record_path.is_symlink():
            raise ValueError("bundle metadata is a symlink")
        if bundle_path.exists():
            if Bundle.model_validate_json(bundle_path.read_bytes()) != manifest.bundle:
                raise ValueError("existing generation binds a different bundle")
        if record_path.exists():
            saved = JobRecord.model_validate_json(record_path.read_bytes())
            if (
                saved.spec_sha256 != spec.identity
                or saved.generation != record.generation
                or saved.manifest != record.manifest
            ):
                raise ValueError("existing generation binds a different job record")
        for entry, reference in zip(
            manifest.bundle.files, manifest.artifacts, strict=True
        ):
            target = entry.destination(destination)
            if target.exists():
                entry.verify(destination)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = artifacts.fetch(reference, cache)
            # A killed copy leaves only a private temporary file, never a truncated
            # destination that could masquerade as an immutable completed entry.
            with tempfile.TemporaryDirectory(
                dir=target.parent, prefix=".fetch-"
            ) as temporary:
                staged = Path(temporary) / "bytes"
                shutil.copyfile(source, staged)
                staged.replace(target)
        manifest.bundle.verify(destination)
        for path, data in (
            (record_path, record.model_dump_json(indent=2)),
            (bundle_path, manifest.bundle.model_dump_json(indent=2)),
        ):
            if path.is_symlink():
                raise ValueError("bundle metadata is a symlink")
            if not path.exists():
                with tempfile.TemporaryDirectory(
                    dir=destination, prefix=".fetch-"
                ) as temporary:
                    staged = Path(temporary) / "metadata.json"
                    staged.write_text(data)
                    staged.replace(path)
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
    checkpoint_seconds: float | None = None,
    checkpoint_updates: int | None = None,
    validate_numerics: bool = False,
) -> Job:
    """Compile one resolved Experiment case through the ordinary job lifecycle.

    The caller owns case/seed/cohort selection and cumulative campaign allocation;
    this API never starts a hidden experiment scheduler or revises the declaration.
    """
    plan = compile_plan(case.configuration, spec, source, seed)
    return prepare_job(
        plan,
        job_id,
        destination=destination
        if destination is not None
        else plan.spec.access.destination,
        monitoring=monitoring,
        checkpoint_seconds=checkpoint_seconds,
        checkpoint_updates=checkpoint_updates,
        experiment_json=json.dumps(case.receipt(), sort_keys=True),
        validate_numerics=validate_numerics,
    )


def reconcile_job(
    spec: Job,
    *,
    delete: bool = False,
    store: JobStore | None = None,
    provider: RunPod | None = None,
) -> JobStatus:
    """Discover uncertain creates; optionally delete only exact job-owned resources.

    Force deletion can discard an unpublished final generation. Normal cancel
    leaves finalization to the worker; this path settles failed bootstrap/outages.
    """
    store, provider = store or S3JobStore(spec.prefix), provider or RunPod()
    for purpose in ("guardian", "training"):
        raw = store.read(f"{purpose}-claim.json")
        if raw is None:
            continue
        claim = CreateClaim.model_validate_json(raw.data)
        if claim.spec_sha256 != spec.identity:
            raise ValueError("creation claim belongs to another job")
        saved = store.read(f"{purpose}.json")
        if saved is not None and not delete:
            continue
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
