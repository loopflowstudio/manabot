"""Bounded SSH transport and pinned-source bootstrap, without credential forwarding."""

import ipaddress
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .jobs import Job

from .plan import DeploymentPlan
from .provider import Pod

REPOSITORY = "https://github.com/loopflowstudio/fantasia.git"
REPO_DIR = "/opt/manabot/repo"
UV_CACHE_DIR = "/opt/manabot/uv-cache"


class Transport:
    def __init__(
        self, pod: Pod, directory: Path, deadline: float, identity: Path
    ) -> None:
        if pod.public_ip is None or "22" not in pod.ports:
            raise ValueError("pod SSH endpoint is not ready")
        ipaddress.ip_address(pod.public_ip)
        if not 1 <= pod.ports["22"] <= 65535:
            raise ValueError("invalid SSH port")
        self.deadline = deadline
        self.target = f"root@{pod.public_ip}"
        self.port = pod.ports["22"]
        self.options = [
            "-i",
            str(identity),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={directory / 'known_hosts'}",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=2",
        ]

    def _run(self, args: list[str], input_bytes: bytes | None = None) -> bytes:
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("remote phase deadline exceeded")
        try:
            result = subprocess.run(
                args,
                input=input_bytes,
                capture_output=True,
                timeout=remaining,
                check=False,
                env={
                    key: os.environ[key]
                    for key in ("PATH", "HOME", "SSH_AUTH_SOCK")
                    if key in os.environ
                },
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError("remote phase deadline exceeded") from None
        if result.returncode:
            # Remote output can include provider environment values. Keep it out
            # of exceptions/logs; the closed bundle contains only training logs.
            raise RuntimeError(f"remote command failed with exit {result.returncode}")
        return result.stdout

    def shell(self, script: str, *, observe: Callable[[], None] | None = None) -> bytes:
        args = ["ssh", *self.options, "-p", str(self.port), self.target, "bash -s"]
        if observe is None:
            return self._run(args, script.encode())
        # communicate drains pipes while callbacks retrieve immutable exports on
        # separate SSH connections. Neither polling nor callbacks extend the lease.
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                key: os.environ[key]
                for key in ("PATH", "HOME", "SSH_AUTH_SOCK")
                if key in os.environ
            },
        )
        payload: bytes | None = script.encode()
        try:
            while True:
                remaining = self.deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("remote phase deadline exceeded")
                try:
                    stdout, _ = process.communicate(payload, timeout=min(15, remaining))
                    if process.returncode:
                        raise RuntimeError(
                            f"remote command failed with exit {process.returncode}"
                        )
                    observe()
                    return stdout
                except subprocess.TimeoutExpired:
                    payload = None
                    observe()
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()

    def put(self, local: Path, remote: str) -> None:
        self._copy(str(local), f"{self.target}:{remote}")

    def get(self, remote: str, local: Path) -> None:
        self._copy(f"{self.target}:{remote}", str(local))

    def _copy(self, source: str, destination: str) -> None:
        self._run(
            [
                "scp",
                *self.options,
                "-P",
                str(self.port),
                source,
                destination,
            ]
        )


def startup(deadline: int, public_key: str) -> str:
    guardian = Path(__file__).with_name("guardian.sh").read_text()
    # Only a public SSH key and a deadline are transferred at provisioning.
    if not public_key.startswith("ssh-ed25519 ") or "\n" in public_key:
        raise ValueError("expected one ed25519 public key")
    return f"""set -eu
umask 077
export MANABOT_DEADLINE={deadline}
mkdir -p /root/.ssh /run/sshd
printf '%s\\n' {shlex.quote(public_key)} > /root/.ssh/authorized_keys
cat > /tmp/manabot-guardian.sh <<'MANABOT_GUARDIAN'
{guardian}
MANABOT_GUARDIAN
nohup bash /tmp/manabot-guardian.sh >/tmp/manabot-guardian.log 2>&1 </dev/null &
ssh-keygen -A
exec /usr/sbin/sshd -D
"""


def bootstrap(plan: DeploymentPlan) -> str:
    return f"""set -eu
export PATH=/root/.local/bin:/root/.cargo/bin:$PATH
bootstrap_start=$SECONDS
export UV_CACHE_DIR={UV_CACHE_DIR}
mkdir -p /opt/manabot /workspace/evidence
git clone --quiet {REPOSITORY} {REPO_DIR}
cd {REPO_DIR}
git checkout --quiet {plan.source.commit}
test "$(git rev-parse HEAD)" = {plan.source.commit}
test "$(git rev-parse 'HEAD^{{tree}}')" = {plan.source.tree}
test "$(sha256sum uv.lock | cut -d ' ' -f 1)" = {plan.source.lock_sha256}
curl -LsSf https://astral.sh/uv/0.8.22/install.sh | sh
command -v cargo >/dev/null || {{ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.98.1; }}
rustup toolchain install 1.98.1 --profile minimal
export RUSTUP_TOOLCHAIN=1.98.1
sync_start=$SECONDS
uv sync --locked --python 3.12 --extra play
sync_seconds=$((SECONDS - sync_start))
build_start=$SECONDS
uv run maturin develop --release --features python --manifest-path managym/Cargo.toml
build_seconds=$((SECONDS - build_start))
uv --version > /workspace/evidence/toolchain.txt
rustc --version >> /workspace/evidence/toolchain.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >> /workspace/evidence/toolchain.txt
# Whole seconds; the mtime interval matches the retained pre-change observation.
env_start=$(stat -c %Y .venv/pyvenv.cfg)
env_end=$(stat -c %Y /workspace/evidence/toolchain.txt)
printf '{{"bootstrap_seconds":%s,"uv_sync_seconds":%s,"native_build_seconds":%s,"environment_to_toolchain_seconds":%s}}\\n' "$((SECONDS - bootstrap_start))" "$sync_seconds" "$build_seconds" "$((env_end - env_start))" > /workspace/evidence/bootstrap-timing.json
"""


def job_startup(spec: "Job") -> str:
    """Provider-owned bootstrap; neither SSH nor the submitter owns its lifetime."""
    guardian = Path(__file__).with_name("guardian.sh").read_text()
    setup = bootstrap(spec.plan).replace(
        "--python 3.12 --extra play", "--python 3.12 --extra play --extra artifacts"
    )
    if spec.validate_numerics:
        setup = setup.replace("--extra artifacts", "--extra artifacts --extra dev")
    # Queue delay consumes the allocation, not the bootstrap allowance. The
    # absolute worker cutoff still bounds setup and cannot move on reconnect.
    setup_deadline = min(time.time() + spec.plan.spec.setup_seconds, spec.work_deadline)
    return f"""set -eu
umask 077
export MANABOT_DEADLINE={int(spec.deadline)}
mkdir -p /workspace/evidence
cat > /tmp/manabot-guardian.sh <<'MANABOT_GUARDIAN'
{guardian}
MANABOT_GUARDIAN
nohup bash /tmp/manabot-guardian.sh >/tmp/manabot-guardian.log 2>&1 </dev/null &
cat > /tmp/manabot-setup.sh <<'MANABOT_SETUP'
{setup}
MANABOT_SETUP
# Keep PID 1 alive even on setup failure so the independent guardian can delete.
setup_remaining=$(({int(setup_deadline)} - $(date +%s)))
if [ "$setup_remaining" -gt 0 ] && timeout "$setup_remaining" bash /tmp/manabot-setup.sh >/workspace/evidence/bootstrap.log 2>&1; then
  export PATH=/root/.local/bin:/root/.cargo/bin:$PATH
  cd {REPO_DIR}
  uv run --no-sync python -m manabot.remote.supervisor >>/workspace/evidence/supervisor.log 2>&1 || true
fi
while true; do sleep 5; done
"""
