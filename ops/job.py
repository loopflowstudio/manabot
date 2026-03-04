"""
job.py
CLI for launching, observing, cancelling, and resuming spot GPU training jobs.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import secrets

from ops.aws import AWSProvider
from ops.bootstrap import BOOTSTRAP_MARKER, job_user_data
from ops.provider import (
    Machine,
    MachineSpec,
    Provider,
    RuntimeSpec,
    load_machine_spec,
    load_runtime_spec,
)


@dataclass
class JobRecord:
    """Persistent metadata needed to resume spot jobs."""

    job_id: str
    wandb_run_id: str
    config_name: str
    region: str


class JobStore:
    """Tiny local metadata store for resumable job information."""

    def __init__(self, path: Path | None = None):
        self.path = path or Path.home() / ".manabot" / "jobs.json"

    def get(self, job_id: str) -> JobRecord | None:
        return self._records().get(job_id)

    def upsert(self, record: JobRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records = self._records()
        records[record.job_id] = record
        payload = {key: asdict(value) for key, value in records.items()}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _records(self) -> dict[str, JobRecord]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        records: dict[str, JobRecord] = {}
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            records[key] = JobRecord(**value)
        return records


class JobManager:
    """Lifecycle orchestration for one-shot and resumable training jobs."""

    def __init__(
        self,
        provider: Provider,
        spec: MachineSpec,
        runtime: RuntimeSpec,
        *,
        store: JobStore | None = None,
    ):
        self.provider = provider
        self.spec = spec
        self.runtime = runtime
        self.store = store or JobStore()

    def launch(self, config_name: str, *, record: JobRecord | None = None) -> tuple[JobRecord, Machine]:
        """Launch a new or resumed spot machine and start training via systemd."""

        active_record = record or JobRecord(
            job_id=f"job-{secrets.token_hex(4)}",
            wandb_run_id=secrets.token_hex(8),
            config_name=config_name,
            region=self.spec.region,
        )

        log_group = self._log_group()
        user_data = job_user_data(
            self.runtime,
            config_name=active_record.config_name,
            job_id=active_record.job_id,
            wandb_run_id=active_record.wandb_run_id,
            region=self.spec.region,
            log_group=log_group,
        )

        machine = self.provider.create(
            self.spec,
            tags=self._tags(active_record),
            user_data=user_data,
        )
        machine = self.provider.wait_until_ready(machine, timeout=900)
        self.provider.wait_for_ssm(machine, timeout=900)

        env = self._job_environment(active_record, log_group)
        write_env_command = self._write_env_and_start_command(env)
        result = self.provider.run_command(machine, write_env_command, timeout=300)
        if result.status != "Success":
            raise RuntimeError(
                f"Failed to start manabot-job.service for {active_record.job_id}: "
                f"{result.status}\n{result.stderr}"
            )

        verify = self.provider.run_command(
            machine,
            f"test -f {BOOTSTRAP_MARKER} && echo READY || echo MISSING",
            timeout=120,
        )
        if "READY" not in verify.stdout:
            raise RuntimeError(
                f"Job bootstrap marker missing on {machine.id}. stderr={verify.stderr}"
            )

        self.store.upsert(active_record)
        return active_record, machine

    def list_jobs(self) -> list[Machine]:
        """List active managed job machines."""

        machines = self.provider.list(self._base_tags())
        return sorted(machines, key=lambda machine: machine.id)

    def logs(self, job_id: str, limit: int = 200) -> list[str]:
        """Fetch CloudWatch lines for a specific job."""

        return self.provider.logs(
            group=self._log_group(),
            stream_prefix=f"{job_id}/",
            limit=limit,
        )

    def cancel(self, job_id: str) -> int:
        """Terminate active instances matching a job id."""

        machines = self.provider.list(
            {
                **self._base_tags(),
                "manabot:job-id": job_id,
            }
        )
        for machine in machines:
            self.provider.terminate(machine)
        return len(machines)

    def resume(self, job_id: str) -> tuple[JobRecord, Machine]:
        """Launch a new spot machine using saved W&B metadata for the same job id."""

        record = self.store.get(job_id)
        if record is None:
            raise RuntimeError(
                f"No local record for job {job_id}. Launch it once before using --resume."
            )
        return self.launch(record.config_name, record=record)

    def _job_environment(self, record: JobRecord, log_group: str) -> dict[str, str]:
        env = {
            "JOB_ID": record.job_id,
            "WANDB_RUN_ID": record.wandb_run_id,
            "WANDB_RESUME": "allow",
            "CONFIG_NAME": record.config_name,
            "AWS_REGION": self.spec.region,
            "LOG_GROUP": log_group,
            "IMAGE": self.runtime.image,
        }
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            env["WANDB_API_KEY"] = wandb_api_key
        return env

    def _write_env_and_start_command(self, env: dict[str, str]) -> str:
        lines = [
            "sudo mkdir -p /opt/manabot",
            "sudo tee /opt/manabot/job.env >/dev/null <<'EOF'",
        ]
        for key, value in env.items():
            escaped = str(value).replace("'", "'\"'\"'")
            lines.append(f"{key}='{escaped}'")
        lines.append("EOF")
        lines.append("sudo systemctl daemon-reload")
        lines.append("sudo systemctl start manabot-job.service")
        return "\n".join(lines)

    def _user(self) -> str:
        return getattr(self.provider, "user", os.getenv("USER", "unknown"))

    def _base_tags(self) -> dict[str, str]:
        return {
            "manabot:user": self._user(),
            "manabot:role": "job",
            "manabot:managed": "true",
            "manabot:region": self.spec.region,
        }

    def _tags(self, record: JobRecord) -> dict[str, str]:
        return {
            **self._base_tags(),
            "manabot:job-id": record.job_id,
            "manabot:wandb-run-id": record.wandb_run_id,
            "manabot:config": record.config_name,
        }

    def _log_group(self) -> str:
        return f"{self.runtime.log_group_prefix.rstrip('/')}/job/{self._user()}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage manabot GPU spot jobs")
    parser.add_argument(
        "--config",
        default="simple",
        help="Hydra config passed to manabot/model/train.py --config-name",
    )
    parser.add_argument("--list", action="store_true", help="List active jobs")
    parser.add_argument("--logs", metavar="JOB_ID", help="Show CloudWatch logs for job")
    parser.add_argument("--cancel", metavar="JOB_ID", help="Terminate active job instances")
    parser.add_argument("--resume", metavar="JOB_ID", help="Resume job from saved metadata")
    parser.add_argument(
        "--iam-instance-profile",
        default=os.getenv("MANABOT_IAM_INSTANCE_PROFILE"),
        help="Use pre-provisioned IAM instance profile",
    )
    parser.add_argument(
        "--key-name",
        default=os.getenv("MANABOT_KEY_NAME"),
        help="EC2 keypair name override",
    )
    parser.add_argument(
        "--spec",
        default="job",
        help="Machine spec name in ops/specs (default: job)",
    )
    parser.add_argument(
        "--store",
        default=None,
        help="Path to local job metadata store (default: ~/.manabot/jobs.json)",
    )
    return parser


def _validate_mode(args: argparse.Namespace) -> None:
    enabled = [
        bool(args.list),
        bool(args.logs),
        bool(args.cancel),
        bool(args.resume),
    ]
    if sum(enabled) > 1:
        raise SystemExit("Use only one of --list/--logs/--cancel/--resume per invocation")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_mode(args)

    spec = load_machine_spec(args.spec)
    runtime = load_runtime_spec()

    provider = AWSProvider(
        region=spec.region,
        iam_instance_profile=args.iam_instance_profile,
        key_name=args.key_name,
        log_group_prefix=runtime.log_group_prefix,
    )
    store = JobStore(Path(args.store).expanduser()) if args.store else JobStore()
    manager = JobManager(provider, spec, runtime, store=store)

    if args.list:
        machines = manager.list_jobs()
        if not machines:
            print("jobs: none")
            return 0

        for machine in machines:
            print(
                " ".join(
                    [
                        f"id={machine.id}",
                        f"status={machine.status}",
                        f"job_id={machine.tags.get('manabot:job-id', '-')}",
                        f"run_id={machine.tags.get('manabot:wandb-run-id', '-')}",
                        f"ip={machine.public_ip or '-'}",
                    ]
                )
            )
        return 0

    if args.logs:
        lines = manager.logs(args.logs)
        for line in lines:
            print(line)
        return 0

    if args.cancel:
        count = manager.cancel(args.cancel)
        print(f"jobs: cancelled {count} machine(s) for {args.cancel}")
        return 0

    if args.resume:
        record, machine = manager.resume(args.resume)
        print(
            f"jobs: resumed {record.job_id} run_id={record.wandb_run_id} "
            f"instance={machine.id}"
        )
        return 0

    record, machine = manager.launch(args.config)
    print(
        f"jobs: launched {record.job_id} run_id={record.wandb_run_id} instance={machine.id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
