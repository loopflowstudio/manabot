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
        ssh_user: str = "ubuntu",
        no_ssh: bool = False,
    ):
        self.provider = provider
        self.spec = spec
        self.runtime = runtime
        self.repo_url = repo_url
        self.ssh_user = ssh_user
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
        if machine.status == "running":
            if not self.no_ssh:
                self._open_ssh(machine)
            return machine

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
            result = self.provider.run_command(
                machine,
                f"test -f {BOOTSTRAP_MARKER} && echo READY || echo MISSING",
                timeout=120,
            )
            if "READY" in result.stdout:
                return
            print("  waiting for bootstrap to finish...", flush=True)
            time.sleep(10)

        raise RuntimeError(
            f"Bootstrap did not complete within {timeout}s ({BOOTSTRAP_MARKER}). "
            f"Last command {result.command_id} status={result.status} stderr={result.stderr}"
        )

    def _ready_for_use(self, machine: Machine) -> Machine:
        print(f"Waiting for {machine.id} to be running...")
        machine = self.provider.wait_until_ready(machine, timeout=600)
        print(f"Instance running, ip={machine.public_ip or 'pending'}")
        print("Waiting for SSM agent...")
        self.provider.wait_for_ssm(machine, timeout=600)
        print("SSM online. Verifying bootstrap...")
        self._verify_bootstrap(machine)
        print("Bootstrap verified.")
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

    def _open_ssh(self, machine: Machine) -> None:
        if not machine.public_ip:
            raise RuntimeError(
                f"Sandbox {machine.id} has no public IP; cannot open SSH session"
            )

        subprocess.run(
            [
                "ssh",
                f"{self.ssh_user}@{machine.public_ip}",
            ],
            check=False,
        )


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
        "--repo-url",
        default=DEFAULT_REPO,
        help="Repository URL used during bootstrap",
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
    manager = SandboxManager(
        provider=provider,
        spec=spec,
        runtime=runtime,
        repo_url=args.repo_url,
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


if __name__ == "__main__":
    raise SystemExit(main())
