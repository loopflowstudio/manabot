"""One owned rental at a time, with fixed deadlines and explicit deletion evidence.

Private receipts retain intent before create, including ambiguous create failures.
The account key stays in the laptop process. No training begins until a separate
short-lived pod has demonstrated its startup guardian's scoped self-deletion.
"""

from contextlib import contextmanager
import fcntl
import json
import math
import shutil
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import TYPE_CHECKING, Callable, Iterator, Literal

if TYPE_CHECKING:
    from manabot.training.checkpoint_queue import MonitoringBudget
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

from pydantic import BaseModel, ConfigDict, Field

from .bundle import Bundle, verify_training_bundle
from .plan import DeploymentPlan, Source, digest
from .provider import Pod, ProviderError, RunPod
from .transport import Transport, startup


class Attempt(BaseModel):
    name: str
    purpose: Literal["guardian", "training"]
    intent_time: float
    deadline: float
    pod_id: str | None = None
    hourly_rate: float | None = None
    assigned_vcpus: int | None = None
    assigned_memory_gb: float | None = None
    deleted_time: float | None = None
    guardian_proven: bool = False
    # An interrupted create cannot be proved absent by an immediate empty list.
    create_ambiguous: bool = False
    # Multiple pods or conflicting rates cannot be priced as one rental.
    cost_ambiguous: bool = False


class PhaseStamp(BaseModel):
    phase: str
    unix: float


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    job_destination: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    plan_sha256: str
    started: float
    deadline: float
    phase: str = "prepared"
    attempts: list[Attempt] = Field(default_factory=list)
    error: str | None = None
    complete: bool = False
    policies: list[str] = Field(default_factory=list)
    phase_events: list[PhaseStamp] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    estimated_dollars: float | None = None


def save(path: Path, receipt: Receipt) -> None:
    if not receipt.phase_events or receipt.phase_events[-1].phase != receipt.phase:
        receipt.phase_events.append(PhaseStamp(phase=receipt.phase, unix=time.time()))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(receipt.model_dump_json(indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def current_source(root: Path) -> Source:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("commit tracked source edits before compiling or deploying")
    # Untracked source would not reach the rented checkout. Scratch evidence is
    # deliberately excluded from this source-admission boundary.
    if git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "manabot",
        "managym",
        "pyproject.toml",
        "uv.lock",
    ):
        raise ValueError("commit untracked runtime source before compiling")
    return Source(
        commit=git("rev-parse", "HEAD"),
        tree=git("rev-parse", "HEAD^{tree}"),
        lock_sha256=digest((root / "uv.lock").read_bytes()),
    )


def verify_public_source(source: Source) -> None:
    request = Request(
        f"https://api.github.com/repos/loopflowstudio/fantasia/git/commits/{source.commit}",
        headers={"User-Agent": "manabot/0.1"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            value = json.load(response)
        if value["sha"] != source.commit or value["tree"]["sha"] != source.tree:
            raise ValueError("public source identity differs")
    except HTTPError as error:
        if error.code != 403:
            raise ValueError("committed source is not publicly fetchable") from None
        gh = shutil.which("gh")
        if gh is not None:
            # Reuse the host login without extracting or forwarding its token.
            result = subprocess.run(
                [
                    gh,
                    "api",
                    f"repos/loopflowstudio/fantasia/git/commits/{source.commit}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                value = json.loads(result.stdout)
                if value["sha"] != source.commit or value["tree"]["sha"] != source.tree:
                    raise ValueError("authenticated public source identity differs")
                return
        _verify_public_source_git(source)
    except (URLError, TimeoutError, KeyError):
        raise ValueError(
            "committed source is not publicly fetchable; publish before rental"
        ) from None


def _verify_public_source_git(source: Source) -> None:
    """Verify public commit/tree/lock bytes when the unauthenticated API is limited."""
    with TemporaryDirectory(prefix="manabot-public-source-") as directory:

        def git(*args: str) -> str:
            result = subprocess.run(
                ["git", "-C", directory, *args],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            if result.returncode:
                raise ValueError("public Git source verification failed")
            return result.stdout.strip()

        git("init", "--bare", "--quiet")
        git(
            "fetch",
            "--quiet",
            "--depth=1",
            "https://github.com/loopflowstudio/fantasia.git",
            source.commit,
        )
        if (
            git("rev-parse", "FETCH_HEAD") != source.commit
            or git("rev-parse", "FETCH_HEAD^{tree}") != source.tree
        ):
            raise ValueError("public Git source identity differs")
        lock = subprocess.run(
            ["git", "-C", directory, "show", "FETCH_HEAD:uv.lock"],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if lock.returncode or digest(lock.stdout) != source.lock_sha256:
            raise ValueError("public Git lock identity differs")


def _save_cleanup(path: Path, receipt: Receipt) -> None:
    try:
        save(path, receipt)
    except OSError:
        # Disk failure must never prevent the control-plane deletion attempt.
        print(f"Could not persist cleanup receipt at {path}", file=sys.stderr)


@contextmanager
def deployment_lock(directory: Path) -> Iterator[None]:
    """Serialize this laptop's deployments; remote inventory is checked separately."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / "remote.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another remote deployment is active") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _terminate_signal(signum: int, frame: object) -> None:
    raise KeyboardInterrupt("deployment interrupted")


def confirm_delete(provider: RunPod, attempt: Attempt, until: float) -> bool:
    """Delete only the recorded pod or exact unique name; failed reads are unknown."""
    while time.time() < until:
        try:
            pods = [p for p in provider.list() if p.name == attempt.name]
            if attempt.pod_id:
                recorded = provider.get(attempt.pod_id)
                if recorded is not None:
                    if recorded.name != attempt.name:
                        return False
                    if not any(p.id == recorded.id for p in pods):
                        pods.append(recorded)
            if not pods:
                if attempt.create_ambiguous:
                    # Keep watching the entire reserve for delayed provisioning.
                    time.sleep(min(2, max(0, until - time.time())))
                    continue
                if attempt.deleted_time is None:
                    attempt.deleted_time = time.time()
                return True
            if len(pods) != 1:
                attempt.cost_ambiguous = True
            for pod in pods:
                if attempt.pod_id not in (None, pod.id) or (
                    attempt.hourly_rate is not None and attempt.hourly_rate != pod.rate
                ):
                    attempt.cost_ambiguous = True
            if len(pods) == 1 and not attempt.cost_ambiguous:
                pod = pods[0]
                attempt.pod_id = pod.id
                attempt.hourly_rate = pod.rate
                attempt.assigned_vcpus = pod.vcpus
                attempt.assigned_memory_gb = pod.memory_gb
            for pod in pods:
                provider.delete(pod.id)
            # Once a delayed create is observed, subsequent absence is evidence.
            attempt.create_ambiguous = False
        except ProviderError:
            pass
        time.sleep(min(2, max(0, until - time.time())))
    return False


def _estimate_cost(receipt: Receipt, plan: DeploymentPlan) -> float | None:
    """Price intent-to-confirmed-absence time, never infer missing rental rates."""
    if digest(plan.model_dump_json().encode()) != receipt.plan_sha256:
        return None
    total = 0.0
    for attempt in receipt.attempts:
        if (
            attempt.create_ambiguous
            or attempt.cost_ambiguous
            or attempt.hourly_rate is None
            or attempt.deleted_time is None
            or not all(
                math.isfinite(v)
                for v in (
                    attempt.hourly_rate,
                    attempt.intent_time,
                    attempt.deleted_time,
                )
            )
            or attempt.hourly_rate < 0
            or attempt.deleted_time < attempt.intent_time
        ):
            return None
        total += (
            (attempt.deleted_time - attempt.intent_time)
            * (attempt.hourly_rate + plan.spec.machine.storage_hourly_allowance)
            / 3600
        )
    return total


def _create(
    provider: RunPod,
    plan: DeploymentPlan,
    receipt: Receipt,
    path: Path,
    purpose: Literal["guardian", "training"],
    deadline: float,
    public_key: str,
) -> Pod:
    prices = provider.prices()
    candidates = [
        gpu
        for gpu in plan.spec.machine.gpu_types
        if 0 < prices.get(gpu, float("inf")) <= plan.spec.machine.hourly_ceiling
    ]
    if not candidates:
        raise ValueError("no declared GPU has an admitted price")
    if time.time() + 20 >= deadline:
        raise TimeoutError("insufficient time for provisioning before deadline")
    # A failed create may still provision. Do not retry it speculatively.
    attempt = Attempt(
        name=f"manabot-{uuid.uuid4().hex}",
        purpose=purpose,
        intent_time=time.time(),
        deadline=deadline,
        create_ambiguous=True,
    )
    receipt.attempts.append(attempt)
    save(path, receipt)
    pod = provider.create(
        {
            "name": attempt.name,
            "imageName": plan.spec.machine.image,
            "cloudType": "SECURE",
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": candidates,
            "gpuTypePriority": "custom",
            "minVCPUPerGPU": plan.spec.machine.vcpus,
            "minRAMPerGPU": plan.spec.machine.memory_gb,
            "containerDiskInGb": plan.spec.machine.container_gb,
            "volumeInGb": plan.spec.machine.volume_gb,
            "volumeMountPath": "/workspace",
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "dockerEntrypoint": ["/bin/bash", "-c"],
            "dockerStartCmd": [startup(int(deadline), public_key)],
        }
    )
    attempt.pod_id = pod.id
    attempt.hourly_rate = pod.rate
    attempt.assigned_vcpus = pod.vcpus
    attempt.assigned_memory_gb = pod.memory_gb
    attempt.create_ambiguous = False
    save(path, receipt)
    if pod.name != attempt.name:
        raise ValueError("provider returned unexpected pod ownership")
    if (
        pod.gpu_count != plan.spec.machine.gpu_count
        or pod.rate > plan.spec.machine.hourly_ceiling
        or pod.vcpus < plan.spec.machine.vcpus
        or pod.memory_gb < plan.spec.machine.memory_gb
    ):
        raise ValueError(
            "assigned rental exceeds price or undersupplies declared resources"
        )
    return pod


def _ready(
    provider: RunPod, pod: Pod, directory: Path, deadline: float, identity: Path
) -> Transport:
    while time.time() < deadline:
        current = provider.get(pod.id)
        if current is None:
            raise ValueError("pod disappeared before SSH became ready")
        if current.public_ip and "22" in current.ports:
            transport = Transport(
                current, directory, min(deadline, time.time() + 15), identity
            )
            try:
                transport.shell("true")
                transport.deadline = deadline
                return transport
            except (RuntimeError, TimeoutError):
                pass
        time.sleep(2)
    raise TimeoutError("pod bootstrap deadline exceeded")


def deploy(
    plan: DeploymentPlan,
    out: Path,
    root: Path,
    *,
    job_id: str | None = None,
    checkpoint_seconds: float | None = None,
    checkpoint_updates: int | None = None,
    monitoring: "MonitoringBudget | None" = None,
    destination: str = "s3://etudefantasia/manabot/jobs",
    observe: Callable[[Transport], None] | None = None,
    after_training: Callable[[Transport], None] | None = None,
    bulk_return: bool = False,
) -> Receipt:
    """Submit and observe the shared durable lifecycle; disconnect never cancels.

    Arbitrary client callbacks cannot be transferred to a remote execution owner.
    Retained calibration scripts using those callbacks must use their pinned source;
    new callers supply the serializable monitoring contract instead.
    """
    from .job_client import fetch_job, job_status, prepare_job, submit_job

    if observe is not None or after_training is not None:
        raise ValueError(
            "client execution callbacks are unsupported; declare remote monitoring"
        )
    if current_source(root) != plan.source:
        raise ValueError("deployment source differs from compiled plan")
    out = out.resolve()
    if ".runs" not in out.parts or out.exists():
        raise ValueError(
            "use a new private .runs output directory, or reconnect by job ID"
        )
    out.mkdir(parents=True, mode=0o700)
    spec = prepare_job(
        plan,
        job_id or f"run-{uuid.uuid4().hex}",
        destination=destination,
        monitoring=monitoring,
        checkpoint_seconds=checkpoint_seconds,
        checkpoint_updates=checkpoint_updates,
    )
    (out / "job.json").write_text(spec.model_dump_json(indent=2))
    (out / "plan.json").write_text(plan.model_dump_json(indent=2))
    print(
        f"Job {spec.job_id}; reconnect: uv run manabot deploy status --job-id {spec.job_id}",
        flush=True,
    )
    status = submit_job(spec)
    while status.cleanup is None:
        if time.time() > spec.deadline + plan.spec.cleanup_seconds:
            raise RuntimeError(f"CLEANUP UNCONFIRMED; reconcile job {spec.job_id}")
        time.sleep(5)
        status = job_status(spec)
    record = status.record
    receipt = Receipt(
        job_id=spec.job_id,
        job_destination=spec.destination,
        plan_sha256=digest(plan.model_dump_json().encode()),
        started=spec.created_at,
        deadline=spec.deadline,
        phase="deleted",
        estimated_dollars=status.cleanup.estimated_dollars,
    )
    if record is not None and record.manifest is not None:
        generation = fetch_job(spec, out / "retrieved")
        evidence = out / "evidence"
        generation.replace(evidence)
        (out / "bundle.json").write_bytes((evidence / "bundle.json").read_bytes())
        bundle = Bundle.model_validate_json((evidence / "bundle.json").read_text())
        if record.phase == "completed" and record.artifacts_complete:
            receipt.policies = [
                str(p) for p in verify_training_bundle(evidence, bundle)
            ]
            receipt.complete = True
        receipt.error = record.error
    save(out / "deployment.json", receipt)
    if not receipt.complete:
        raise RuntimeError(
            f"job {spec.job_id} did not return complete evidence; inspect by ID"
        )
    return receipt


def cleanup(path: Path) -> Receipt:
    receipt = Receipt.model_validate_json(path.read_text())
    if receipt.job_id is not None:
        from .job_client import load_job, reconcile_job

        status = reconcile_job(
            load_job(
                receipt.job_id,
                receipt.job_destination or "s3://etudefantasia/manabot/jobs",
            ),
            delete=True,
        )
        if status.cleanup is None:
            raise RuntimeError("CLEANUP UNCONFIRMED; inspect the durable job ID")
        receipt.estimated_dollars = status.cleanup.estimated_dollars
        receipt.phase = "deleted"
        save(path, receipt)
        return receipt
    provider = RunPod()
    with deployment_lock(Path.home() / ".cache/manabot"):
        for attempt in receipt.attempts:
            if attempt.deleted_time is None:
                if not confirm_delete(provider, attempt, time.time() + 120):
                    save(path, receipt)
                    raise RuntimeError(
                        "CLEANUP UNCONFIRMED; provider absence could not be established"
                    )
                save(path, receipt)
        receipt.phase = "deleted"
        # A missing or changed plan must not block deletion or supply a guessed
        # storage allowance. Leave cost unresolved for the all-attempt gate.
        receipt.estimated_dollars = None
        try:
            plan = DeploymentPlan.model_validate_json(
                path.with_name("plan.json").read_text()
            )
        except (OSError, ValueError):
            pass
        else:
            receipt.estimated_dollars = _estimate_cost(receipt, plan)
        save(path, receipt)
        if receipt.estimated_dollars is None:
            print(
                "BILLING UNRESOLVED; deletion confirmed but cost evidence is incomplete",
                file=sys.stderr,
            )
    return receipt
