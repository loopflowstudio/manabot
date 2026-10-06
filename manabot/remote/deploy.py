"""One owned rental at a time, with fixed deadlines and explicit deletion evidence.

Private receipts retain intent before create, including ambiguous create failures.
The account key stays in the laptop process. No training begins until a separate
short-lived pod has demonstrated its startup guardian's scoped self-deletion.
"""

from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Iterator, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

from pydantic import BaseModel, ConfigDict, Field

from .bundle import Bundle, verify_training_bundle
from .plan import DeploymentPlan, Source, digest
from .provider import Pod, ProviderError, RunPod
from .transport import REPO_DIR, Transport, bootstrap, startup


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


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_sha256: str
    started: float
    deadline: float
    phase: str = "prepared"
    attempts: list[Attempt] = Field(default_factory=list)
    error: str | None = None
    complete: bool = False
    policies: list[str] = Field(default_factory=list)
    estimated_dollars: float | None = None


def save(path: Path, receipt: Receipt) -> None:
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
        f"https://api.github.com/repos/loopflowstudio/etude/git/commits/{source.commit}",
        headers={"User-Agent": "manabot/0.1"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            value = json.load(response)
        if value["sha"] != source.commit or value["tree"]["sha"] != source.tree:
            raise ValueError("public source identity differs")
    except (HTTPError, URLError, TimeoutError, KeyError):
        raise ValueError(
            "committed source is not publicly fetchable; publish before rental"
        ) from None


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
            * (attempt.hourly_rate + plan.mix.storage_hourly_allowance)
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
        for gpu in plan.mix.gpu_types
        if 0 < prices.get(gpu, float("inf")) <= plan.mix.hourly_ceiling
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
            "imageName": plan.mix.image,
            "cloudType": "SECURE",
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": candidates,
            "gpuTypePriority": "custom",
            "minVCPUPerGPU": plan.mix.vcpus,
            "minRAMPerGPU": plan.mix.memory_gb,
            "containerDiskInGb": plan.mix.container_gb,
            "volumeInGb": plan.mix.volume_gb,
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
        pod.gpu_count != plan.mix.gpu_count
        or pod.rate > plan.mix.hourly_ceiling
        or pod.vcpus < plan.mix.vcpus
        or pod.memory_gb < plan.mix.memory_gb
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


def deploy(plan: DeploymentPlan, out: Path, root: Path) -> Receipt:
    # Revalidate nested mutable Pydantic values before any provider operation.
    plan = DeploymentPlan.model_validate_json(plan.model_dump_json())
    if current_source(root) != plan.source:
        raise ValueError("deployment source differs from compiled plan")
    out = out.resolve()
    if ".runs" not in out.parts:
        raise ValueError(
            "private deployment output must be under an ignored .runs directory"
        )
    if out.exists():
        raise ValueError(
            "deployment directory already exists; use cleanup for prior attempts"
        )
    verify_public_source(plan.source)
    provider = RunPod()
    with deployment_lock(Path.home() / ".cache/manabot"):
        if any(p.name.startswith("manabot-") for p in provider.list()):
            raise ValueError(
                "an owned pod already exists; reconcile it before another rental"
            )
        out.mkdir(parents=True, mode=0o700)
        path = out / "deployment.json"
        start = time.time()
        receipt = Receipt(
            plan_sha256=digest(plan.model_dump_json().encode()),
            started=start,
            deadline=start + plan.mix.wall_seconds,
        )
        save(path, receipt)
        (out / "plan.json").write_text(plan.model_dump_json(indent=2))
        identity = out / "ssh_key"
        subprocess.run(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                "manabot-deployment",
                "-f",
                str(identity),
            ],
            check=True,
            timeout=15,
            capture_output=True,
        )
        public_key = identity.with_suffix(".pub").read_text().strip()
        previous_signal = signal.signal(signal.SIGTERM, _terminate_signal)
        try:
            receipt.phase = "guardian-probe"
            # Both probe and training are charged against this same deadline/cap.
            probe_deadline = min(
                start + 90, receipt.deadline - plan.mix.cleanup_seconds
            )
            probe = _create(
                provider, plan, receipt, path, "guardian", probe_deadline, public_key
            )
            transport = _ready(provider, probe, out, probe_deadline, identity)
            transport.shell(
                "test -f /tmp/manabot-guardian.sh; command -v runpodctl >/dev/null"
            )
            while time.time() < probe_deadline + 30:
                if provider.get(probe.id) is None:
                    if time.time() < probe_deadline - 2:
                        raise ValueError(
                            "probe disappeared before its guardian deadline"
                        )
                    receipt.attempts[-1].deleted_time = time.time()
                    receipt.attempts[-1].guardian_proven = True
                    save(path, receipt)
                    break
                time.sleep(2)
            else:
                raise ValueError("pod-scoped deadline deletion was not proven")
            receipt.phase = "provisioning"
            train_deadline = receipt.deadline - plan.mix.cleanup_seconds
            pod = _create(
                provider, plan, receipt, path, "training", train_deadline, public_key
            )
            transport = _ready(
                provider, pod, out, start + plan.mix.setup_seconds, identity
            )
            receipt.phase = "bootstrap"
            save(path, receipt)
            transport.shell(bootstrap(plan))
            recipe = out / "regime.json"
            recipe.write_text(plan.regime.model_dump_json(indent=2))
            transport.put(recipe, "/workspace/regime.json")
            receipt.phase = "training"
            save(path, receipt)
            transport.deadline = train_deadline - plan.mix.transfer_seconds
            # Training errors are retained, then the closed output is bundled.
            transport.shell(f"""export PATH=/root/.local/bin:/root/.cargo/bin:$PATH
cd {REPO_DIR}
uv run manabot train --regime /workspace/regime.json --seed {plan.seed} --out /workspace/evidence/run > /workspace/evidence/training.log 2>&1
status=$?
printf '%s\\n' "$status" > /workspace/evidence/training-exit.txt
uv run python -m manabot.remote.bundle /workspace/evidence
exit 0
""")
            receipt.phase = "transfer"
            save(path, receipt)
            transport.deadline = train_deadline
            transport.get("/workspace/bundle.json", out / "bundle.json")
            bundle = Bundle.model_validate_json((out / "bundle.json").read_text())
            destination = out / "evidence"
            destination.mkdir()
            for item in bundle.files:
                target = item.destination(destination)
                target.parent.mkdir(parents=True, exist_ok=True)
                transport.get(f"/workspace/evidence/{item.relative_path}", target)
            policies = verify_training_bundle(destination, bundle)
            receipt.policies = [str(p) for p in policies]
            receipt.complete = True
        except BaseException as error:
            # Do not serialize arbitrary exceptions carrying provider payloads.
            receipt.error = type(error).__name__
            raise
        finally:
            receipt.phase = "cleanup"
            _save_cleanup(path, receipt)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
            cleanup_until = time.time() + plan.mix.cleanup_seconds
            unsettled = False
            for attempt in receipt.attempts:
                if attempt.deleted_time is None:
                    if not confirm_delete(provider, attempt, cleanup_until):
                        unsettled = True
                    _save_cleanup(path, receipt)
            receipt.phase = "cleanup-unconfirmed" if unsettled else "deleted"
            receipt.complete = receipt.complete and not unsettled
            receipt.estimated_dollars = _estimate_cost(receipt, plan)
            _save_cleanup(path, receipt)
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_signal)
            if unsettled:
                raise RuntimeError(
                    f"CLEANUP UNCONFIRMED; possible billing for {[(a.pod_id or a.name, a.hourly_rate) for a in receipt.attempts if a.deleted_time is None]}. Run: uv run manabot remote cleanup --deployment {path}"
                )
        return receipt


def cleanup(path: Path) -> Receipt:
    receipt = Receipt.model_validate_json(path.read_text())
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
