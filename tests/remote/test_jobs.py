"""Durable identity, uncertain creation and independent runtime behavior, offline."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from manabot.infra.artifacts import StoredArtifact
from manabot.remote import job_client, supervisor
from manabot.remote.job_store import cancellation_requested
from manabot.remote.jobs import CreateClaim, RemoteJobRecord, RemoteJobSpec, Resource
from manabot.remote.plan import HardwareMix, compile_plan
from manabot.remote.provider import ProviderError
from manabot.verify.store import VerifyStore
from tests.remote.job_fixtures import FileStore
from tests.remote.test_compile import ROOT, SOURCE
from tests.remote.test_lifecycle import Clock, Provider
from tests.training.test_checkpoint_queue import run_fixture


@pytest.fixture
def store(tmp_path: Path) -> FileStore:
    return FileStore(tmp_path / "control.sqlite")


def specification(store: FileStore, *, job_id: str = "offline-test") -> RemoteJobSpec:
    plan = compile_plan(
        (ROOT / "experiments/regimes/direct-self-play.json").read_text(),
        HardwareMix.model_validate_json(
            (ROOT / "ops/mixes/runpod-small.json").read_text()
        ),
        SOURCE,
        197,
    )
    return job_client.prepare_job(plan, job_id, store=store)


def admitted(spec: RemoteJobSpec, store: FileStore) -> Resource:
    provider = Provider(Clock())
    pod = provider.create({"name": "owned"})
    resource = Resource(
        claim=CreateClaim(
            spec_sha256=spec.identity,
            name=pod.name,
            purpose="training",
            intent_time=time.time(),
            deadline=spec.deadline,
        ),
        pod=pod,
    )
    store.create("training.json", resource.model_dump_json().encode())
    store.create("training-claim.json", resource.claim.model_dump_json().encode())
    return resource


def published(
    spec: RemoteJobSpec, root: Path, generation: int, *, complete: bool
) -> StoredArtifact:
    return StoredArtifact(
        uri=f"{spec.prefix}/runtime/{generation}", sha256="f" * 64, bytes=0
    )


def test_repeated_and_concurrent_intent_retains_deadline(store: FileStore) -> None:
    with ThreadPoolExecutor(max_workers=4) as executor:
        specs = list(executor.map(lambda _: specification(store), range(8)))
    assert len({s.identity for s in specs}) == 1
    assert len({s.deadline for s in specs}) == 1
    plan = specs[0].plan.model_copy(update={"seed": 198})
    with pytest.raises(ValueError, match="different immutable"):
        job_client.prepare_job(plan, specs[0].job_id, store=store)


@pytest.mark.parametrize("lost_response", [False, True])
def test_provider_create_is_never_repeated(
    store: FileStore, lost_response: bool
) -> None:
    spec = specification(store)
    provider = Provider(Clock(), failure="ambiguous" if lost_response else "")
    if lost_response:
        with pytest.raises(ProviderError):
            job_client._provision(
                spec, store, provider, "training", spec.deadline, "script", {}
            )
    else:
        job_client._provision(
            spec, store, provider, "training", spec.deadline, "script", {}
        )
    result = job_client._provision(
        spec, store, provider, "training", spec.deadline, "script", {}
    )
    assert result.pod.id == "pod1"
    assert provider.created == 1


def test_create_claim_before_provider_call_stays_ambiguous(store: FileStore) -> None:
    spec = specification(store)
    claim = CreateClaim(
        spec_sha256=spec.identity,
        name="owned",
        purpose="training",
        intent_time=time.time(),
        deadline=spec.deadline,
    )
    store.create("training-claim.json", claim.model_dump_json().encode())
    provider = Provider(Clock())
    with pytest.raises(RuntimeError, match="ambiguous"):
        job_client._provision(
            spec, store, provider, "training", spec.deadline, "script", {}
        )
    assert provider.created == 0
    assert (
        job_client.job_status(spec, store=store, provider=provider).provider_state
        == "ambiguous"
    )


def test_stale_heartbeat_does_not_mean_completed(store: FileStore) -> None:
    spec = specification(store)
    record = RemoteJobRecord(
        spec_sha256=spec.identity,
        pod_id="pod1",
        phase="running",
        accepted_at=1,
        heartbeat_at=1,
    )
    store.create("runtime/record.json", record.model_dump_json().encode())
    result = job_client.job_status(spec, store=store, provider=Provider(Clock()))
    assert result.stale and result.record is not None and not result.record.terminal
    assert not result.record.artifacts_complete


@pytest.mark.parametrize("reason", ["cancel", "deadline", "upload", "crash"])
def test_supervisor_failures_retain_honest_terminal_state(
    store: FileStore,
    tmp_path: Path,
    reason: str,
) -> None:
    spec = specification(store)
    if reason == "deadline":
        spec = spec.model_copy(
            update={"created_at": 1, "deadline": 1 + spec.plan.mix.wall_seconds}
        )
    resource = admitted(spec, store)
    if reason == "cancel":
        job_client.cancel_job(spec, store=store)

    def publish(
        spec: RemoteJobSpec, root: Path, generation: int, *, complete: bool
    ) -> StoredArtifact:
        if reason == "upload":
            raise OSError("storage unavailable")
        return published(spec, root, generation, complete=complete)

    result = supervisor.supervise(
        spec,
        store,
        tmp_path / "evidence",
        resource.pod.id,
        command=[sys.executable, "-c", "raise SystemExit(3)"],
        publish=publish,
    )
    assert (
        result.phase
        == {
            "cancel": "cancelled",
            "deadline": "deadline",
            "upload": "failed",
            "crash": "failed",
        }[reason]
    )
    assert result.artifacts_complete is (reason != "upload")
    if reason == "cancel":
        assert result.cancel_acknowledged_at is not None
    with pytest.raises(ValueError, match="already started"):
        supervisor.supervise(
            spec, store, tmp_path / "evidence", resource.pod.id, publish=publish
        )


def test_submitter_death_does_not_stop_accepted_supervisor(
    store: FileStore, tmp_path: Path
) -> None:
    spec = specification(store)
    admitted(spec, store)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json())
    evidence = tmp_path / "evidence"
    # This fixture uses the real supervisor and real process groups. A short
    # synthetic learner establishes lifecycle survival, not CUDA learning.
    worker_code = """
import sys
from pathlib import Path
from manabot.remote.jobs import RemoteJobSpec
from manabot.remote.supervisor import supervise
from tests.remote.job_fixtures import FileStore
from tests.remote.test_jobs import published
spec=RemoteJobSpec.model_validate_json(Path(sys.argv[1]).read_text())
command=[sys.executable, '-u', '-c', 'import time\\nfor i in range(8): print(i, flush=True); time.sleep(.5)']
supervise(spec, FileStore(Path(sys.argv[2])), Path(sys.argv[3]), 'pod1', command=command, publish=published)
"""
    parent_code = """
import subprocess, sys, time
child=subprocess.Popen([sys.executable, '-c', sys.argv[1], *sys.argv[2:]], start_new_session=True)
print(child.pid, flush=True)
time.sleep(30)
"""
    with (tmp_path / "process.log").open("wb") as log:
        parent = subprocess.Popen(
            [
                sys.executable,
                "-c",
                parent_code,
                worker_code,
                str(spec_path),
                str(store.path),
                str(evidence),
            ],
            stdout=subprocess.PIPE,
            stderr=log,
        )
        assert parent.stdout is not None
        worker_pid = int(parent.stdout.readline())
        try:
            until = time.time() + 15
            while store.read("runtime/record.json") is None and time.time() < until:
                time.sleep(0.05)
            assert store.read("runtime/record.json") is not None
            parent.kill()
            parent.wait(timeout=5)
            before = (
                (evidence / "training.log").read_bytes()
                if (evidence / "training.log").exists()
                else b""
            )
            while time.time() < until:
                current = store.read("runtime/record.json")
                assert current is not None
                record = RemoteJobRecord.model_validate_json(current.data)
                if record.terminal:
                    break
                time.sleep(0.1)
            assert record.terminal
            after = (evidence / "training.log").read_bytes()
            assert len(after) > len(before) and b"7" in after
            assert record.artifacts_complete
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            try:
                os.killpg(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_running_learner_is_stopped_at_original_deadline(
    store: FileStore, tmp_path: Path
) -> None:
    original = specification(store)
    reserves = original.plan.mix.transfer_seconds + original.plan.mix.cleanup_seconds
    start = time.time() + 2 + reserves - original.plan.mix.wall_seconds
    spec = original.model_copy(
        update={"created_at": start, "deadline": start + original.plan.mix.wall_seconds}
    )
    resource = admitted(spec, store)
    began = time.time()
    result = supervisor.supervise(
        spec,
        store,
        tmp_path / "evidence",
        resource.pod.id,
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        publish=published,
    )
    assert result.phase == "deadline" and time.time() - began < 10
    assert (tmp_path / "evidence/training-exit.txt").read_text() != "0"


def test_guardian_deletes_without_supervisor(tmp_path: Path) -> None:
    executable = tmp_path / "runpodctl"
    receipt = tmp_path / "deleted"
    executable.write_text(f'#!/bin/sh\nprintf "%s" "$*" > "{receipt}"\n')
    executable.chmod(0o700)
    # macOS has no GNU timeout; emulate only its command dispatch for this fixture.
    timeout = tmp_path / "timeout"
    timeout.write_text('#!/bin/sh\nshift\nexec "$@"\n')
    timeout.chmod(0o700)
    process = subprocess.Popen(
        ["bash", str(ROOT / "manabot/remote/guardian.sh")],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "MANABOT_DEADLINE": str(int(time.time()) + 1),
            "RUNPOD_POD_ID": "owned-fixture",
        },
        start_new_session=True,
    )
    try:
        until = time.time() + 5
        while not receipt.exists() and time.time() < until:
            time.sleep(0.05)
        assert receipt.read_text() == "remove pod owned-fixture"
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def test_worker_cancellation_mailbox_exists_before_rental(store: FileStore) -> None:
    spec = specification(store)
    # A scoped S3 reader can read existing keys but lacks ListBucket, so a missing
    # cancel key would return AccessDenied rather than an absent result.
    assert store.read("cancel.json") is not None
    assert cancellation_requested(store) is None
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: job_client.cancel_job(spec, store=store), range(4)))
    first = cancellation_requested(store)
    assert first is not None
    assert specification(store).deadline == spec.deadline
    job_client.cancel_job(spec, store=store)
    assert cancellation_requested(store) == first


def test_rejected_rental_is_deleted_before_setup(store: FileStore) -> None:
    spec = specification(store)
    provider = Provider(Clock(), failure="price")
    with pytest.raises(ValueError, match="price/resource"):
        job_client._provision(
            spec, store, provider, "training", spec.deadline, "script", {}
        )
    assert not provider.pods
    assert store.read("training.json") is not None
    status = job_client.job_status(spec, store=store, provider=provider)
    assert status.cleanup is not None and status.cleanup.estimated_dollars > 0


def test_explicit_cancellation_stops_an_accepted_process(
    store: FileStore, tmp_path: Path
) -> None:
    spec = specification(store)
    resource = admitted(spec, store)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            supervisor.supervise,
            spec,
            store,
            tmp_path / "evidence",
            resource.pod.id,
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
            publish=published,
        )
        until = time.time() + 10
        while time.time() < until:
            raw = store.read("runtime/record.json")
            if (
                raw is not None
                and RemoteJobRecord.model_validate_json(raw.data).phase == "running"
            ):
                break
            time.sleep(0.02)
        assert raw is not None
        job_client.cancel_job(spec, store=store)
        result = future.result(timeout=10)
    assert result.phase == "cancelled" and result.cancel_acknowledged_at is not None
    assert result.artifacts_complete


def test_failed_final_upload_keeps_previous_generation(
    store: FileStore, tmp_path: Path
) -> None:
    spec = specification(store)
    resource = admitted(spec, store)

    def publish(
        spec: RemoteJobSpec, root: Path, generation: int, *, complete: bool
    ) -> StoredArtifact:
        if complete:
            raise OSError("unavailable")
        return published(spec, root, generation, complete=False)

    result = supervisor.supervise(
        spec,
        store,
        tmp_path / "evidence",
        resource.pod.id,
        command=[
            sys.executable,
            "-c",
            "import time; time.sleep(.2); raise SystemExit(3)",
        ],
        publish=publish,
    )
    assert result.generation == 1 and result.manifest is not None
    assert not result.artifacts_complete and result.error is not None
    assert "final upload" in result.error


def test_terminal_progress_uses_database_not_lagging_export(
    store: FileStore, tmp_path: Path
) -> None:
    spec = specification(store)
    resource = admitted(spec, store)
    root = tmp_path / "evidence"
    (root / "run").mkdir(parents=True)
    run = run_fixture(root / "run/run.json")
    run.status = "completed"
    run.stages[0].diagnostics = [{} for _ in range(160)]
    with VerifyStore(root / "training.sqlite") as database:
        database.save_training_run(run)
    result = supervisor.supervise(
        spec,
        store,
        root,
        resource.pod.id,
        command=[sys.executable, "-c", "pass"],
        publish=published,
    )
    assert result.phase == "completed"
    assert result.updates == run.updates_through() == 160
