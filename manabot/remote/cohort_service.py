"""OS-managed cohort ownership on an explicitly selected, continuously online host.

launchd/systemd restart the deploy worker independently of terminal/agent lifetime.
A permanent S3 host/path binding and a local flock exclude a second supervisor.
Credentials come from the service account's normal renewable provider chain; no
secret is embedded in the plan, unit, plist or worker environment.
"""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Iterator, TypedDict

from manabot.training.execution import atomic_json

from .cohort import Cohort, CohortState, CohortSupervisor
from .deploy import current_source
from .job_client import submit_job
from .jobs import Job, JobStatus
from .plan import digest

if TYPE_CHECKING:
    from .cohort_projection import ProjectionConfig


class ErrorLocation(TypedDict):
    file: str
    function: str
    line: int


def _error_locations(error: Exception) -> list[ErrorLocation]:
    """Retain code coordinates, never exception text, locals or source lines."""
    locations: list[ErrorLocation] = []
    trace = error.__traceback__
    while trace is not None:
        code = trace.tb_frame.f_code
        locations.append(
            {
                "file": Path(code.co_filename).name,
                "function": code.co_name,
                "line": trace.tb_lineno,
            }
        )
        trace = trace.tb_next
    return locations


def _machine_identity() -> str:
    """Stable OS identity; hostnames can collide or change with the network."""
    if sys.platform == "linux":
        identity = Path("/etc/machine-id").read_text().strip()
    elif sys.platform == "darwin":
        payload = subprocess.check_output(
            ["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            text=True,
            timeout=10,
        )
        found = re.search(r'"IOPlatformUUID" = "([A-Fa-f0-9-]+)"', payload)
        identity = found.group(1) if found is not None else ""
    else:
        raise ValueError("unsupported controller host")
    if not identity:
        raise ValueError("stable controller machine identity unavailable")
    return identity


@contextmanager
def owner_lock(directory: Path) -> Iterator[str]:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / "owner.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("cohort supervisor is already running") from None
        # Canonical directory is part of ownership. Another host/directory cannot
        # take over by copying credentials or waiting for an expired heartbeat.
        owner = digest(
            f"{_machine_identity()}:{os.getuid()}:{directory.resolve()}".encode()
        )
        yield owner


def supervise_cohort(plan: Path, directory: Path, *, interval: float = 30) -> None:
    cohort = Cohort.model_validate_json(plan.read_bytes())
    with owner_lock(directory) as owner:
        atomic_json(
            directory / "driver.json",
            {
                "pid": os.getpid(),
                "started_at": time.time(),
                "owner": owner,
                "cohort_sha256": cohort.identity,
            },
        )
        observation_deadline = cohort.deadline + 1800
        retained = directory / "status.json"
        if retained.exists():
            state = CohortState.model_validate_json(retained.read_bytes())
            if state.cohort_sha256 != cohort.identity or state.owner != owner:
                raise ValueError("retained state belongs to a different cohort owner")
            observation_deadline = state.effective_deadline(cohort) + 1800

        def submit(spec: Job) -> JobStatus:
            # A controller upgrade does not rewrite frozen worker sources. Paths
            # are local placement, revalidated against each exact Source at submit.
            registry = directory / "sources.json"
            roots = json.loads(registry.read_text()) if registry.exists() else {}
            source_root = Path(roots.get(spec.plan.source.commit, str(Path.cwd())))
            return submit_job(spec, source_root=source_root)

        while time.time() < observation_deadline:
            supervisor: CohortSupervisor | None = None
            try:
                # Reload after every uncertain network write; CAS and the host lock
                # remain authoritative, never an in-memory guess at remote state.
                supervisor = CohortSupervisor(
                    cohort, owner, directory / "cache", submit=submit
                )
                state = supervisor.tick()
                observation_deadline = state.effective_deadline(cohort) + 1800
                atomic_json(directory / "status.json", state.model_dump(mode="json"))
                (directory / "error.json").unlink(missing_ok=True)
                if state.settled:
                    return
            except Exception as error:
                if supervisor is not None:
                    try:
                        supervisor.observation_failed(error)
                    except Exception:
                        pass  # Preserve local evidence when S3 is unavailable too.
                # Exception text can contain credential_process output. Retain only
                # code coordinates and freshness; keep the last successful observation.
                atomic_json(
                    directory / "error.json",
                    {
                        "at": time.time(),
                        "error_type": type(error).__name__,
                        "error_location": _error_locations(error),
                        "state": "uncertain",
                        "pid": os.getpid(),
                    },
                )
            time.sleep(min(interval, max(0, observation_deadline - time.time())))


def _systemd_quote(value: str, *, command: bool = False) -> str:
    if any(c in value for c in ("\n", "\r", "\x00")):
        raise ValueError("invalid service path")
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if command:
        value = value.replace("$", "$$")
    return '"' + value + '"'


def install_service(
    cohort: Cohort,
    directory: Path,
    *,
    doppler: bool = False,
    projection: "ProjectionConfig | None" = None,
    source_roots: tuple[Path, ...] = (),
    replace_service: bool = False,
) -> str:
    """Install/start the current user's service on this host, without renting.

    Linux requires an existing lingering user manager. A macOS LaunchAgent survives
    launcher exit but runs only while that account is logged in. Neither promises
    operation while this host sleeps/offlines; choose an always-on controller host.
    """
    if sys.platform not in {"darwin", "linux"}:
        raise ValueError("cohort services require launchd or systemd")
    repo = Path.cwd().resolve()
    sources = {
        current_source(root): str(root.resolve()) for root in (repo, *source_roots)
    }
    if any(entry.plan.source not in sources for entry in cohort.entries):
        raise ValueError("service requires the cohort's exact clean source checkout")
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("uv is required by the cohort service")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    plan = directory / "cohort.json"
    data = cohort.model_dump_json(indent=2)
    if plan.exists() and plan.read_text() != data:
        raise ValueError("service directory already contains another cohort")
    plan.write_text(data)
    registry = directory / "sources.json"
    previous_roots = json.loads(registry.read_text()) if registry.exists() else {}
    for source, path in sources.items():
        if source.commit in previous_roots and previous_roots[source.commit] != path:
            raise ValueError("registered source placement differs")
        previous_roots[source.commit] = path
    atomic_json(registry, previous_roots)
    command = [
        uv,
        "run",
        "--no-sync",
        "manabot",
        "deploy",
        "cohort",
        "supervise",
        "--plan",
        str(plan),
        "--state-dir",
        str(directory),
    ]
    if projection is not None:
        configuration = directory / "projection-config.json"
        value = projection.model_dump_json(indent=2)
        if configuration.exists() and configuration.read_text() != value:
            raise ValueError("projection service already binds another allocation")
        configuration.write_text(value)
        command[6] = "project"
        command.extend(["--config", str(configuration), "--follow"])
    if doppler:
        executable = shutil.which("doppler")
        if executable is None:
            raise ValueError("Doppler is not installed on the controller host")
        command = [
            executable,
            "run",
            "--project",
            "etude",
            "--config",
            "prd",
            "--",
            *command,
        ]
    # Only nonsecret profile selectors survive installation. Account credentials
    # are resolved by the service account/provider chain when the worker runs.
    environment = {
        key: os.environ[key]
        for key in (
            "AWS_PROFILE",
            "AWS_DEFAULT_REGION",
            "MANABOT_REMOTE_ISSUER_PROFILE",
            "MANABOT_REMOTE_ROLE_ARN",
        )
        if key in os.environ
    }
    environment["PATH"] = (
        f"{Path(uv).parent}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    )
    label = f"manabot.cohort.{cohort.cohort_id}" + (
        ".projection" if projection is not None else ""
    )
    if sys.platform == "darwin":
        target = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = plistlib.dumps(
            {
                "Label": label,
                "ProgramArguments": command,
                "WorkingDirectory": str(repo),
                "EnvironmentVariables": environment,
                "RunAtLoad": True,
                "KeepAlive": {"SuccessfulExit": False},
                "ThrottleInterval": 30,
                "StandardOutPath": str(directory / "service.log"),
                "StandardErrorPath": str(directory / "service.log"),
            }
        )
        if target.exists() and target.read_bytes() != payload:
            if not replace_service:
                raise ValueError("existing service differs; no replacement performed")
            # Stop only the controller/companion, never its independently running
            # rental. The permanent owner directory and all remote claims survive.
            domain = f"gui/{os.getuid()}"
            loaded = subprocess.run(
                ["launchctl", "print", f"{domain}/{label}"], capture_output=True
            )
            if loaded.returncode == 0:
                subprocess.run(
                    ["launchctl", "bootout", f"{domain}/{label}"], check=True
                )
            # bootout acknowledges removal before the Python child necessarily
            # exits. Retrying a previously interrupted handoff may find no job.
            lock_directory = (
                directory / "service" if projection is not None else directory
            )
            until = time.monotonic() + 15
            while True:
                try:
                    with owner_lock(lock_directory):
                        break
                except RuntimeError:
                    if time.monotonic() >= until:
                        raise
                    time.sleep(0.1)
        target.write_bytes(payload)
        domain = f"gui/{os.getuid()}"
        loaded = subprocess.run(
            ["launchctl", "print", f"{domain}/{label}"],
            capture_output=True,
            check=False,
        )
        if loaded.returncode != 0:
            subprocess.run(["launchctl", "bootstrap", domain, str(target)], check=True)
    else:
        if replace_service:
            raise ValueError("service replacement currently requires macOS launchd")
        lingering = subprocess.run(
            ["loginctl", "show-user", str(os.getuid()), "--property=Linger", "--value"],
            capture_output=True,
            text=True,
            check=True,
        )
        if lingering.stdout.strip() != "yes":
            raise ValueError("systemd user manager needs preconfigured lingering")
        target = Path.home() / ".config/systemd/user" / f"{label}.service"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload_text = (
            "[Unit]\nDescription=manabot experiment cohort\n"
            "[Service]\nType=simple\n"
            f"WorkingDirectory={_systemd_quote(str(repo))}\n"
            f"ExecStart={' '.join(_systemd_quote(v, command=True) for v in command)}\n"
            + "".join(
                f"Environment={_systemd_quote(key + '=' + value)}\n"
                for key, value in environment.items()
            )
            + "Restart=on-failure\nRestartSec=30\nUMask=0077\n"
            "[Install]\nWantedBy=default.target\n"
        )
        if target.exists() and target.read_text() != payload_text:
            raise ValueError("existing service differs; no replacement performed")
        target.write_text(payload_text)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", target.name], check=True
        )
    return label
