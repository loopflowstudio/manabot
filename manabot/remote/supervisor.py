"""The rental owns one learner, bounded checkpoint evaluation and durable uploads.

The provider starts this after pinned-source bootstrap. An immutable execution claim
prevents container restart from silently restarting training. The shell guardian is
independent of this process and retains the original absolute billing deadline.
"""

from functools import partial
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
from typing import Protocol

from manabot.infra.artifacts import S3ArtifactStore, StoredArtifact, verify_file
from manabot.training.checkpoint_queue import CheckpointQueue
from manabot.training.models import TrainingRun
from manabot.verify.store import VerifyStore

from .job_store import JobStore, S3JobStore, StoredValue, cancellation_requested
from .jobs import Job, JobRecord, Resource
from .snapshots import publish_snapshot
from .validation import validation_command


class SnapshotPublisher(Protocol):
    def __call__(
        self,
        spec: Job,
        root: Path,
        generation: int,
        *,
        complete: bool,
    ) -> StoredArtifact: ...


def stop_process(process: subprocess.Popen[bytes]) -> None:
    """Stop the learner group, including descendants of an already exited parent."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _training_run(root: Path) -> TrainingRun | None:
    """The JSON identifies the run; its current coordinates come from VerifyStore."""
    path = root / "run/run.json"
    if not path.exists():
        return None
    exported = TrainingRun.model_validate_json(path.read_bytes())
    with VerifyStore(root / "training.sqlite", read_only=True) as store:
        return store.training_run(exported.id)


def supervise(
    spec: Job,
    store: JobStore,
    root: Path,
    pod_id: str,
    *,
    command: list[str] | None = None,
    publish: SnapshotPublisher = publish_snapshot,
) -> JobRecord:
    """Execute once. Injection points support real-process offline lifecycle tests."""
    if publish is publish_snapshot:
        # Only this process's successful readbacks can skip repeat remote reads.
        # Final publication rechecks every artifact before declaring completion.
        publish = partial(publish_snapshot, verified={})
    resource_value = store.read("training.json")
    if resource_value is None:
        raise ValueError("client has not admitted this rental; reconcile submission")
    resource = Resource.model_validate_json(resource_value.data)
    resource.validate_for(spec)
    if resource.pod.id != pod_id:
        raise ValueError("supervisor is running on the wrong rental")
    if not store.create("runtime/execution-claim.json", spec.identity.encode()):
        raise ValueError("job already started; CUDA process recovery is unsupported")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    now = time.time()
    record = JobRecord(
        spec_sha256=spec.identity,
        pod_id=pod_id,
        phase="accepted",
        accepted_at=now,
        heartbeat_at=now,
    )
    previous: StoredValue | None = None

    def persist() -> None:
        nonlocal previous, record
        record = record.model_copy(update={"heartbeat_at": time.time()})
        data = record.model_dump_json().encode()
        okay = (
            store.create("runtime/record.json", data)
            if previous is None
            else store.replace("runtime/record.json", data, previous.etag)
        )
        if not okay:
            raise RuntimeError("remote job record ownership changed")
        previous = store.read("runtime/record.json")
        if previous is None or previous.data != data:
            raise RuntimeError("remote job record acknowledgement unavailable")

    persist()
    learner: subprocess.Popen[bytes] | None = None
    queue: CheckpointQueue | None = None
    recipe = root.parent / "regime.json"
    recipe.write_text(spec.plan.regime.model_dump_json(indent=2))
    allocation_path = root.parent / "allocation.json"
    if spec.allocation is not None:
        allocation_path.write_text(spec.allocation.model_dump_json())
    run_path = root / "run/run.json"
    admission = root / "initial-evaluation-admitted"
    require_initial = bool(
        spec.monitoring and spec.monitoring.require_initial_admission
    )
    try:
        if cancellation_requested(store) is not None:
            record = record.model_copy(
                update={"phase": "cancelled", "cancel_acknowledged_at": time.time()}
            )
        elif time.time() >= spec.work_deadline:
            record = record.model_copy(update={"phase": "deadline"})
        else:
            if spec.learning_inputs:
                artifacts = S3ArtifactStore()
                inputs = Path("/opt/manabot/learning-inputs")
                inputs.mkdir(parents=True, exist_ok=True, mode=0o700)
                for role, reference in spec.learning_inputs.items():
                    source = artifacts.fetch(reference, inputs / "cache")
                    target = inputs / role
                    shutil.copyfile(source, target)
                    verify_file(target, reference.sha256, reference.bytes)
                # Downloads consume the original allocation, never learner time.
                if time.time() >= spec.work_deadline:
                    raise TimeoutError("learning-state download exhausted allocation")
            if spec.monitoring is not None:
                queue = CheckpointQueue(root / "monitoring", spec.monitoring)
            training_command = command or [
                "uv",
                "run",
                "--no-sync",
                "manabot",
                "train",
                "--regime",
                str(recipe),
                "--seed",
                str(spec.plan.seed),
                "--out",
                str(root / "run"),
                *(
                    ["--checkpoint-updates", str(spec.checkpoint_updates)]
                    if spec.checkpoint_updates is not None
                    else ["--checkpoint-seconds", str(spec.checkpoint_seconds)]
                ),
                *(
                    ["--allocation", str(allocation_path)]
                    if spec.allocation is not None
                    else []
                ),
                *(["--initial-admission", str(admission)] if require_initial else []),
            ]
            if spec.validate_numerics:
                training_command = validation_command(spec, root, training_command)
            with (root / "training.log").open("ab") as log:
                learner = subprocess.Popen(
                    training_command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            record = record.model_copy(update={"phase": "running"})
            persist()
            last_upload = float("-inf")
            while True:
                if cancellation_requested(store) is not None:
                    record = record.model_copy(
                        update={
                            "phase": "cancelled",
                            "cancel_acknowledged_at": time.time(),
                        }
                    )
                    break
                if time.time() >= spec.work_deadline:
                    record = record.model_copy(update={"phase": "deadline"})
                    break
                sources = [run_path] if run_path.exists() else []
                if queue is not None:
                    queue.tick(
                        sources,
                        launch=time.time() + queue.config.attempt_seconds
                        < spec.work_deadline,
                    )
                if sources:
                    run = _training_run(root)
                    assert run is not None
                    record = record.model_copy(
                        update={"run_id": run.id, "updates": run.updates_through()}
                    )
                waiting_initial = require_initial and not admission.exists()
                if (
                    waiting_initial
                    and queue is not None
                    and any(
                        a.status in {"failed", "interrupted"} for a in queue.attempts
                    )
                ):
                    raise RuntimeError(
                        "initial evaluation failed; learning was not admitted"
                    )
                if time.time() - last_upload >= spec.publish_seconds:
                    uploaded = False
                    try:
                        manifest = publish(
                            spec, root, record.generation + 1, complete=False
                        )
                        record = record.model_copy(
                            update={
                                "generation": record.generation + 1,
                                "manifest": manifest,
                            }
                        )
                        uploaded = True
                    except Exception as error:
                        record = record.model_copy(
                            update={
                                "error": f"snapshot upload failed ({type(error).__name__})"
                            }
                        )
                    if uploaded and waiting_initial and queue is not None:
                        if queue.attempts and queue.attempts[0].status == "completed":
                            initial_run = _training_run(root)
                            assert initial_run is not None
                            inherited = sum(
                                stage.learning_state_origin.iteration
                                for stage in initial_run.stages
                                if stage.learning_state_origin is not None
                            )
                            if initial_run.updates_through() != inherited:
                                raise RuntimeError(
                                    "learning preceded initial admission"
                                )
                            checkpoint = initial_run.stages[0].artifacts["initial_raw"]
                            # Publish the durable manifest pointer before releasing
                            # this exact checkpoint; no client must stay connected.
                            persist()
                            admission.write_text(checkpoint["sha256"] + "\n")
                    last_upload = time.time()
                if queue is not None:
                    record = record.model_copy(
                        update={
                            "evaluations_completed": sum(
                                a.status == "completed" for a in queue.attempts
                            ),
                            "evaluator_seconds": queue.charged_seconds,
                        }
                    )
                persist()
                if learner.poll() is not None:
                    if learner.returncode != 0:
                        record = record.model_copy(
                            update={
                                "phase": "failed",
                                "error": (
                                    f"job process exited {learner.returncode}"
                                    if spec.validate_numerics
                                    else f"learner exited {learner.returncode}"
                                ),
                            }
                        )
                        break
                    if queue is not None:
                        # A learner can finish during publication, after the
                        # preceding queue scan. Discover its final exports from
                        # the now-closed run before deciding the queue is empty.
                        queue.tick(
                            [run_path] if run_path.exists() else [],
                            launch=time.time() + queue.config.attempt_seconds
                            < spec.work_deadline,
                        )
                    if queue is None or (
                        queue.process is None
                        and (
                            queue.pending == 0
                            or queue.config.seconds - queue.charged_seconds
                            < queue.config.attempt_seconds
                        )
                    ):
                        final_run = _training_run(root)
                        if final_run is not None:
                            record = record.model_copy(
                                update={
                                    "run_id": final_run.id,
                                    "updates": final_run.updates_through(),
                                }
                            )
                        if final_run is not None and final_run.status == "paused":
                            record = record.model_copy(update={"phase": "paused"})
                        elif final_run is None or final_run.status != "completed":
                            record = record.model_copy(
                                update={
                                    "phase": "failed",
                                    "error": "learner exited without completed TrainingRun",
                                }
                            )
                        elif queue is not None and (
                            queue.pending
                            or any(a.status != "completed" for a in queue.attempts)
                        ):
                            record = record.model_copy(
                                update={
                                    "phase": "failed",
                                    "error": "milestone evaluation incomplete",
                                }
                            )
                        else:
                            record = record.model_copy(update={"phase": "completed"})
                        break
                time.sleep(1)
    except Exception as error:
        # Capture locations, never locals, provider payloads or credential-bearing
        # exception messages. This remains useful when the learner never started.
        (root / "supervisor-error.json").write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "frames": [
                        {
                            "file": frame.filename,
                            "line": frame.lineno,
                            "function": frame.name,
                        }
                        for frame in traceback.extract_tb(error.__traceback__)
                    ],
                },
                indent=2,
            )
        )
        record = record.model_copy(
            update={
                "phase": "failed",
                "error": f"supervisor failed ({type(error).__name__})",
            }
        )
    finally:
        if learner is not None:
            stop_process(learner)
            (root / "training-exit.txt").write_text(str(learner.returncode))
        if queue is not None:
            queue.close()
            record = record.model_copy(
                update={
                    "evaluations_completed": sum(
                        a.status == "completed" for a in queue.attempts
                    ),
                    "evaluator_seconds": queue.charged_seconds,
                }
            )
        # The learner may advance or fail during a slow periodic upload. Refresh
        # every terminal path after stopping writers, from the same database that
        # the final snapshot exports; failure must not freeze an earlier heartbeat.
        try:
            final_run = _training_run(root)
            if final_run is not None:
                record = record.model_copy(
                    update={
                        "run_id": final_run.id,
                        "updates": final_run.updates_through(),
                    }
                )
        except Exception as error:
            (root / "terminal-progress-error.json").write_text(
                json.dumps({"error_type": type(error).__name__})
            )
        terminal = record.phase
        record = record.model_copy(update={"phase": "finalizing"})
        try:
            persist()
            manifest = publish(spec, root, record.generation + 1, complete=True)
            record = record.model_copy(
                update={
                    "generation": record.generation + 1,
                    "manifest": manifest,
                    "artifacts_complete": True,
                }
            )
        except Exception as error:
            record = record.model_copy(
                update={
                    "error": f"final upload incomplete ({type(error).__name__})",
                    "artifacts_complete": False,
                }
            )
        record = record.model_copy(update={"phase": terminal})
        try:
            persist()
        except Exception:
            # Last durable generation remains explicit. Guardian still deletes.
            pass
    return record


def main() -> None:
    store = S3JobStore(os.environ["MANABOT_JOB_PREFIX"])
    raw = store.read("spec.json")
    if raw is None:
        raise ValueError("remote job intent missing")
    spec = Job.model_validate_json(raw.data)
    pod_id = os.environ["RUNPOD_POD_ID"]
    # A lost create response can delay admission; a fresh submitter can reconcile
    # it without starting another rental. Never train before price admission.
    while time.time() < spec.work_deadline:
        try:
            if store.read("training.json") is not None:
                break
        except RuntimeError:
            # A missing resource receipt is deliberately unreadable to the
            # scoped role until the client reconciles it (no ListBucket grant).
            pass
        time.sleep(2)
    try:
        supervise(spec, store, Path("/workspace/evidence"), pod_id)
    finally:
        # No account key is forwarded. The same pod-scoped identity as guardian.
        subprocess.run(
            ["runpodctl", "remove", "pod", pod_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )


if __name__ == "__main__":
    main()
