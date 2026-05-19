"""
sandbox.py
CLI for creating/resuming a tagged GPU sandbox and opening SSH.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from ops.aws import AWSProvider, choose_single_machine
from ops.bootstrap import BOOTSTRAP_MARKER, DEFAULT_REPO, sandbox_user_data
from ops.provider import (
    Machine,
    MachineSpec,
    Provider,
    RuntimeSpec,
    load_machine_spec,
    load_runtime_spec,
)


class SandboxManager:
    """State machine for sandbox lifecycle operations."""

    def __init__(
        self,
        provider: Provider,
        spec: MachineSpec,
        runtime: RuntimeSpec,
        *,
        repo_url: str = DEFAULT_REPO,
        repo_ref: str = "main",
        ssh_user: str = "ubuntu",
        ssh_key_path: str | None = None,
        ssh_command: str | None = None,
        ssh_host_alias: str | None = "manabox",
        update_ssh_config: bool = True,
        sync_repo: bool = True,
        no_ssh: bool = False,
    ):
        self.provider = provider
        self.spec = spec
        self.runtime = runtime
        self.repo_url = repo_url
        self.repo_ref = repo_ref
        self.ssh_user = ssh_user
        self.ssh_key_path = ssh_key_path
        self.ssh_command = ssh_command
        self.ssh_host_alias = ssh_host_alias
        self.update_ssh_config = update_ssh_config
        self.sync_repo = sync_repo
        self.no_ssh = no_ssh

    def up(self) -> Machine:
        """Create/resume sandbox, verify bootstrap, optionally SSH in."""

        machine = self._existing()
        if machine is None:
            machine = self._create()
        elif machine.status in {"stopped", "stopping"}:
            print(f"Starting stopped instance {machine.id}...")
            machine = self.provider.start(machine)
        else:
            print(f"Found existing instance {machine.id} ({machine.status})")

        return self._ready_for_use(machine)

    def status(self) -> Machine | None:
        """Return current sandbox machine, if any."""

        return self._existing()

    def start(self) -> Machine:
        """Start an existing stopped sandbox and SSH in."""

        machine = self._existing()
        if machine is None:
            raise RuntimeError("No sandbox found. Run without flags to create one.")
        if machine.status != "running":
            machine = self.provider.start(machine)
        return self._ready_for_use(machine)

    def stop(self) -> None:
        """Stop an existing sandbox machine."""

        machine = self._existing()
        if machine is None:
            return
        if machine.status in {"stopped", "stopping", "terminated"}:
            return
        self.provider.stop(machine)

    def terminate(self) -> None:
        """Terminate sandbox machine."""

        machine = self._existing()
        if machine is None:
            return
        self.provider.terminate(machine)

    def _create(self) -> Machine:
        print(f"Creating {self.spec.instance_type} in {self.spec.region}...")
        user_data = sandbox_user_data(
            self.runtime,
            repo_url=self.repo_url,
            repo_ref=self.repo_ref,
        )
        machine = self.provider.create(
            self.spec,
            tags=self._tags(),
            user_data=user_data,
        )
        print(f"Created instance {machine.id}")
        return machine

    def _verify_bootstrap(self, machine: Machine, timeout: int = 600) -> None:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                result = self.provider.run_command(
                    machine,
                    f"test -f {BOOTSTRAP_MARKER} && echo READY || echo MISSING",
                    timeout=30,
                )
                if "READY" in result.stdout:
                    return
            except (TimeoutError, OSError):
                pass
            print("  waiting for bootstrap to finish...", flush=True)
            time.sleep(10)

        raise RuntimeError(
            f"Bootstrap did not complete within {timeout}s ({BOOTSTRAP_MARKER})."
        )

    def _ready_for_use(self, machine: Machine) -> Machine:
        print(f"Waiting for {machine.id} to be running...")
        machine = self.provider.wait_until_ready(machine, timeout=600)
        print(f"Instance running, ip={machine.public_ip or 'pending'}")
        print("Verifying bootstrap...")
        self._verify_bootstrap(machine)
        print("Bootstrap verified.")
        if self.sync_repo:
            print(f"Syncing /opt/manabot/repo to origin/{self.repo_ref}...")
            self._sync_repo_and_env(machine)
            print("Repo + Python environment synced.")
        if not self.no_ssh:
            self._open_ssh(machine)
        return machine

    def _existing(self) -> Machine | None:
        machines = self.provider.list(self._tags())
        return choose_single_machine(machines)

    def _tags(self) -> dict[str, str]:
        user = getattr(self.provider, "user", os.getenv("USER", "unknown"))
        return {
            "manabot:user": user,
            "manabot:role": "sandbox",
            "manabot:managed": "true",
            "manabot:region": self.spec.region,
        }

    def _sync_repo_and_env(self, machine: Machine) -> None:
        sync_command = f"""set -euo pipefail
REPO_URL={shlex.quote(self.repo_url)}
REPO_REF={shlex.quote(self.repo_ref)}
IMAGE={shlex.quote(self.runtime.image)}
FALLBACK_BUILD={"1" if self.runtime.fallback_build else "0"}
PYTHON_VERSION={shlex.quote(self.runtime.python_version)}

if [ ! -d /opt/manabot/repo/.git ]; then
  sudo git clone "$REPO_URL" /opt/manabot/repo
fi

sudo git -C /opt/manabot/repo remote set-url origin "$REPO_URL"
sudo git -C /opt/manabot/repo fetch --all --prune
if sudo git -C /opt/manabot/repo show-ref --verify --quiet "refs/remotes/origin/$REPO_REF"; then
  sudo git -C /opt/manabot/repo checkout -B "$REPO_REF" "origin/$REPO_REF"
  sudo git -C /opt/manabot/repo reset --hard "origin/$REPO_REF"
else
  sudo git -C /opt/manabot/repo checkout -B main origin/main
  sudo git -C /opt/manabot/repo reset --hard origin/main
fi

sudo chown -R ubuntu:ubuntu /opt/manabot/repo
sudo apt-get update
sudo apt-get install -y build-essential curl git pkg-config libssl-dev
if ! sudo -u ubuntu test -x /home/ubuntu/.local/bin/uv; then
  sudo -u ubuntu -H sh -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'
fi
if ! sudo -u ubuntu test -x /home/ubuntu/.cargo/bin/cargo; then
  sudo -u ubuntu -H sh -lc 'curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal'
fi
if ! sudo docker pull "$IMAGE"; then
  if [ "$FALLBACK_BUILD" != "1" ]; then
    echo "Failed to pull runtime image and fallback build is disabled." >&2
    exit 1
  fi
fi
sudo docker build --build-arg PYTHON_VERSION="$PYTHON_VERSION" -t "$IMAGE" -f /opt/manabot/repo/ops/Dockerfile /opt/manabot/repo
sudo -u ubuntu -H bash -lc 'export PATH=/home/ubuntu/.local/bin:/home/ubuntu/.cargo/bin:"$PATH"; cd /opt/manabot/repo && uv python install "$PYTHON_VERSION" && uv sync --frozen --extra dev --extra ops --python "$PYTHON_VERSION" && uv pip install --python .venv/bin/python --no-deps -e managym && .venv/bin/python -c '"'"'"'"'"'"'"'"'import sys, torch; import managym; print(f"python={{sys.version.split()[0]}} torch={{torch.__version__}} cuda={{torch.cuda.is_available()}} managym=ok")'"'"'"'"'"'"'"'"''
sudo docker run --rm --gpus all "$IMAGE" python -c 'import sys, torch; import managym; print(f"python={{sys.version.split()[0]}} torch={{torch.__version__}} cuda={{torch.cuda.is_available()}} managym=ok")'
"""
        result = self.provider.run_command(machine, sync_command, timeout=1800)
        if result.status != "Success":
            raise RuntimeError(
                f"Sandbox repo/env sync failed ({result.status}) for {machine.id}. "
                f"stderr={result.stderr}"
            )

    def _open_ssh(self, machine: Machine) -> None:
        if not machine.public_ip:
            raise RuntimeError(
                f"Sandbox {machine.id} has no public IP; cannot open SSH session"
            )

        ssh_target = f"{self.ssh_user}@{machine.public_ip}"
        if self.update_ssh_config and self.ssh_host_alias:
            self._write_ssh_config(machine)
            ssh_target = self.ssh_host_alias
            print(f"SSH shortcut ready: ssh {self.ssh_host_alias}")

        command = ["ssh"]
        if self.ssh_key_path:
            command.extend(["-i", self.ssh_key_path, "-o", "IdentitiesOnly=yes"])
        command.append(ssh_target)
        if self.ssh_command:
            command.append(self.ssh_command)

        subprocess.run(command, check=False)

    def _write_ssh_config(self, machine: Machine) -> None:
        if not machine.public_ip:
            return
        if not self.ssh_host_alias:
            return

        marker_start = f"# >>> manabot sandbox {self.ssh_host_alias} >>>"
        marker_end = f"# <<< manabot sandbox {self.ssh_host_alias} <<<"
        block_lines = [
            marker_start,
            f"Host {self.ssh_host_alias}",
            f"  HostName {machine.public_ip}",
            f"  User {self.ssh_user}",
            "  BatchMode yes",
            "  StrictHostKeyChecking accept-new",
            "  ConnectTimeout 10",
            "  ServerAliveInterval 30",
            "  ServerAliveCountMax 3",
        ]
        if self.ssh_key_path:
            block_lines.extend(
                [
                    f"  IdentityFile {self.ssh_key_path}",
                    "  IdentitiesOnly yes",
                ]
            )
        block_lines.append(marker_end)

        block = "\n".join(block_lines)
        config_path = self._ssh_config_path()
        current = (
            config_path.read_text(encoding="utf-8") if config_path.exists() else ""
        )
        updated = self._replace_or_append_block(
            current, marker_start=marker_start, marker_end=marker_end, block=block
        )
        config_path.write_text(updated, encoding="utf-8")

    def _ssh_config_path(self) -> Path:
        ssh_dir = Path.home() / ".ssh"
        ssh_dir.mkdir(parents=True, exist_ok=True)
        return ssh_dir / "config"

    @staticmethod
    def _replace_or_append_block(
        content: str, *, marker_start: str, marker_end: str, block: str
    ) -> str:
        start = content.find(marker_start)
        if start != -1:
            end = content.find(marker_end, start)
            if end != -1:
                end += len(marker_end)
                before = content[:start].rstrip("\n")
                after = content[end:].lstrip("\n")
                parts = [part for part in [before, block, after] if part]
                return "\n\n".join(parts) + "\n"

        body = content.rstrip("\n")
        parts = [part for part in [body, block] if part]
        return "\n\n".join(parts) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage manabot GPU sandbox")
    parser.add_argument("--status", action="store_true", help="Show sandbox state")
    parser.add_argument("--start", action="store_true", help="Start stopped sandbox")
    parser.add_argument("--stop", action="store_true", help="Stop sandbox")
    parser.add_argument(
        "--terminate",
        action="store_true",
        help="Terminate sandbox instance",
    )
    parser.add_argument(
        "--no-ssh",
        action="store_true",
        help="Prepare sandbox but do not launch ssh command",
    )
    parser.add_argument(
        "--ssh-user",
        default=os.getenv("MANABOT_SSH_USER", "ubuntu"),
        help="SSH username (default: ubuntu)",
    )
    parser.add_argument(
        "--ssh-key-path",
        default=os.getenv("MANABOT_SSH_KEY_PATH"),
        help="Path to SSH private key passed to ssh -i",
    )
    parser.add_argument(
        "--ssh-command",
        default=None,
        help="Optional non-interactive command to run over SSH",
    )
    parser.add_argument(
        "--ssh-host-alias",
        default=os.getenv("MANABOT_SSH_HOST_ALIAS", "manabox"),
        help="Alias written to ~/.ssh/config for quick reconnects",
    )
    parser.add_argument(
        "--no-ssh-config",
        action="store_true",
        help="Do not update ~/.ssh/config alias entry",
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO,
        help="Repository URL used during bootstrap",
    )
    parser.add_argument(
        "--repo-ref",
        default=None,
        help="Git branch/ref to sync on the sandbox (default: current local branch)",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip repo/env sync step after bootstrap",
    )
    parser.add_argument(
        "--iam-instance-profile",
        default=os.getenv("MANABOT_IAM_INSTANCE_PROFILE"),
        help="Use pre-provisioned IAM instance profile instead of creating one",
    )
    parser.add_argument(
        "--key-name",
        default=os.getenv("MANABOT_KEY_NAME"),
        help="EC2 keypair name override",
    )
    parser.add_argument(
        "--spec",
        default="sandbox",
        help="Machine spec name in ops/specs (default: sandbox)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    spec = load_machine_spec(args.spec)
    runtime = load_runtime_spec()

    provider = AWSProvider(
        region=spec.region,
        iam_instance_profile=args.iam_instance_profile,
        key_name=args.key_name,
        log_group_prefix=runtime.log_group_prefix,
    )
    repo_ref = args.repo_ref or detect_local_repo_ref()
    manager = SandboxManager(
        provider=provider,
        spec=spec,
        runtime=runtime,
        repo_url=args.repo_url,
        repo_ref=repo_ref,
        ssh_user=args.ssh_user,
        ssh_key_path=args.ssh_key_path,
        ssh_command=args.ssh_command,
        ssh_host_alias=args.ssh_host_alias,
        update_ssh_config=not args.no_ssh_config,
        sync_repo=not args.no_sync,
        no_ssh=args.no_ssh,
    )

    if args.status:
        machine = manager.status()
        if machine is None:
            print("sandbox: missing")
        else:
            print(
                f"sandbox: {machine.id} status={machine.status} ip={machine.public_ip or '-'}"
            )
        return 0

    if args.stop:
        manager.stop()
        print("sandbox: stop requested")
        return 0

    if args.terminate:
        manager.terminate()
        print("sandbox: terminate requested")
        return 0

    if args.start:
        machine = manager.start()
        print(f"sandbox: started {machine.id} ip={machine.public_ip or '-'}")
        return 0

    machine = manager.up()
    print(f"sandbox: ready {machine.id} ip={machine.public_ip or '-'}")
    return 0


def detect_local_repo_ref(default: str = "main") -> str:
    if os.getenv("MANABOT_REPO_REF"):
        return os.environ["MANABOT_REPO_REF"]

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return default

    ref = result.stdout.strip()
    if not ref or ref == "HEAD":
        return default
    return ref


if __name__ == "__main__":
    raise SystemExit(main())
