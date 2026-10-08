"""Subprocess fault fixture for the real deploy cohort service entry point.

Only storage/provider adapters and final-manifest verification are replaced. The
first provider create persists its resource then waits to be killed, emulating a
lost response; a fresh worker reconciles it and completes two fake jobs.
"""

from pathlib import Path
import sys
import time

from manabot.cli import main
from manabot.infra.artifacts import StoredArtifact
from manabot.remote import cohort_service, job_client
from manabot.remote.cohort import Cohort, CohortSupervisor
from manabot.remote.jobs import Job, JobRecord, JobStatus
from manabot.remote.provider import Pod, RunPod
from tests.remote.job_fixtures import FileStore

ROOT = Path(sys.argv[1])


class DiskProvider(RunPod):
    def __init__(self) -> None:
        self.store = FileStore(ROOT / "provider.sqlite")

    def prices(self) -> dict[str, float]:
        return {"NVIDIA L4": 0.49}

    def create(self, payload: dict[str, object]) -> Pod:
        pods = self.list()
        ordinal = len(pods) + 1
        pod = Pod(
            id=f"pod{ordinal}",
            name=str(payload["name"]),
            rate=0.49,
            vcpus=6,
            memory_gb=48,
            gpu_count=1,
        )
        self.store.create(f"pod{ordinal}", pod.model_dump_json().encode())
        if ordinal == 1:
            (ROOT / "created-before-lost-response").touch()
            while not (ROOT / "restart-allowed").exists():
                time.sleep(0.05)
        return pod

    def list(self) -> list[Pod]:
        pods: list[Pod] = []
        for ordinal in range(1, 5):
            value = self.store.read(f"pod{ordinal}")
            if value is not None:
                pods.append(Pod.model_validate_json(value.data))
        return pods

    def get(self, pod_id: str) -> Pod | None:
        # Retain immutable provider history while reporting completed fake pods absent.
        return None


def jobs(spec: Job) -> FileStore:
    return FileStore(ROOT / f"{spec.job_id}.sqlite")


def persist(spec: Job) -> Job:
    saved = job_client.persist_job(spec, store=jobs(spec))
    jobs(spec).create("guardian-proof.json", b"{}")
    return saved


def observe(spec: Job) -> JobStatus:
    store = jobs(spec)
    pods = DiskProvider().list()
    matching = [p for p in pods if p.name == f"manabot-{spec.identity[:32]}-training"]
    if matching:
        record = JobRecord(
            spec_sha256=spec.identity,
            pod_id=matching[0].id,
            phase="completed",
            accepted_at=time.time(),
            heartbeat_at=time.time(),
            generation=1,
            artifacts_complete=True,
            manifest=StoredArtifact(
                uri=f"{spec.prefix}/runtime/manifest", sha256="f" * 64, bytes=1
            ),
        )
        store.create("runtime/record.json", record.model_dump_json().encode())
    return job_client.reconcile_job(spec, store=store, provider=DiskProvider())


def submit(spec: Job) -> JobStatus:
    return job_client.submit_job(
        spec, store=jobs(spec), provider=DiskProvider(), credentials=lambda spec: {}
    )


def cancel(spec: Job) -> None:
    job_client.cancel_job(spec, store=jobs(spec))


def verify(status: JobStatus, cache: Path) -> str:
    assert status.record is not None and status.record.manifest is not None
    return status.record.manifest.sha256


def factory(
    cohort: Cohort, owner: str, cache: Path, **kwargs: object
) -> CohortSupervisor:
    return CohortSupervisor(
        cohort,
        owner,
        cache,
        store=FileStore(ROOT / "cohort.sqlite"),
        observe=observe,
        persist=persist,
        submit=submit,
        cancel=cancel,
        verify=verify,
    )


def run() -> None:
    cohort = Cohort.model_validate_json((ROOT / "cohort.json").read_bytes())
    job_client.current_source = lambda root: cohort.entries[0].plan.source
    job_client.verify_public_source = lambda source: None
    cohort_service.CohortSupervisor = factory  # type: ignore[assignment]
    original = cohort_service.supervise_cohort

    def fast(plan: Path, directory: Path, *, interval: float = 30) -> None:
        original(plan, directory, interval=0.05)

    from manabot.remote import cohort_cli

    cohort_cli.supervise_cohort = fast
    sys.argv = [
        "manabot",
        "deploy",
        "cohort",
        "supervise",
        "--plan",
        str(ROOT / "cohort.json"),
        "--state-dir",
        str(ROOT / "service"),
    ]
    main()


if __name__ == "__main__":
    run()
