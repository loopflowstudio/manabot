"""Independent process lifetime and OS service installation, without live rentals."""

import json
import os
from pathlib import Path
import plistlib
import shutil
import signal
import subprocess
import sys
import time
import uuid

import pytest
from typer.testing import CliRunner

from manabot.cli import app
from manabot.remote import cohort_cli, cohort_service
from manabot.remote.cohort import Cohort, CohortEntry, CohortState
from manabot.remote.plan import JobSpec, compile_plan
from tests.remote.job_fixtures import FileStore
from tests.remote.test_compile import ROOT, SOURCE


def plan() -> Cohort:
    deployment = compile_plan(
        (ROOT / "ops/examples/step-target.json").read_text(),
        JobSpec.model_validate_json((ROOT / "ops/jobs/runpod-small.json").read_text()),
        SOURCE,
        197,
    )
    return Cohort(
        cohort_id="process-proof",
        deadline=time.time() + 86400,
        spending_limit=12,
        prior_dollars=1,
        controller_dollars=0,
        entries=tuple(
            CohortEntry(job_id=f"process-proof-{i}", plan=deployment) for i in range(2)
        ),
    )


def wait_path(path: Path, timeout: float = 20) -> None:
    end = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > end:
            pytest.fail(f"timed out waiting for {path.name}")
        time.sleep(0.05)


def test_launcher_exit_and_forced_supervisor_restart(tmp_path: Path) -> None:
    (tmp_path / "cohort.json").write_text(plan().model_dump_json())
    pid: int | None = None
    try:
        # The launcher exits immediately. The worker uses the real deploy command
        # with durable fake-provider adapters, not a mocked state transition.
        launcher = """import pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1])
with (root/'service.log').open('ab') as log:
 p=subprocess.Popen([sys.executable,'-m','tests.remote.cohort_worker',str(root)],stdout=log,stderr=log,start_new_session=True)
(root/'pid').write_text(str(p.pid))
"""
        subprocess.run(
            [sys.executable, "-c", launcher, str(tmp_path)], check=True, timeout=15
        )
        pid = int((tmp_path / "pid").read_text())
        wait_path(tmp_path / "created-before-lost-response")
        state_raw = FileStore(tmp_path / "cohort.sqlite").read("state.json")
        assert state_raw is not None
        before = CohortState.model_validate_json(state_raw.data)
        assert len(before.attempts) == 1
        os.killpg(pid, signal.SIGKILL)
        pid = None
        (tmp_path / "restart-allowed").touch()
        completed = subprocess.run(
            [sys.executable, "-m", "tests.remote.cohort_worker", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert completed.returncode == 0, completed.stderr
        raw = FileStore(tmp_path / "cohort.sqlite").read("state.json")
        assert raw is not None
        after = CohortState.model_validate_json(raw.data)
        assert after.phase == "completed" and len(after.attempts) == 2
        assert after.attempts[0].spec == before.attempts[0].spec
        provider = FileStore(tmp_path / "provider.sqlite")
        assert provider.read("pod1") and provider.read("pod2")
        assert provider.read("pod3") is None
    finally:
        if pid is not None:
            os.killpg(pid, signal.SIGKILL)


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_start_cli_installs_restartable_service_without_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    value = plan()
    file = tmp_path / "input.json"
    file.write_text(value.model_dump_json())
    monkeypatch.setattr(cohort_cli, "prepare_cohort", lambda cohort: None)
    monkeypatch.setattr(cohort_service, "current_source", lambda root: SOURCE)
    monkeypatch.setattr(cohort_service.sys, "platform", platform)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-copy")
    monkeypatch.setenv("RUNPOD_API_KEY", "do-not-copy")
    monkeypatch.setenv("AWS_PROFILE", "named-profile")
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 1 if command[:2] == ["launchctl", "print"] else 0, stdout="yes\n"
        )

    monkeypatch.setattr(cohort_service.subprocess, "run", run)
    result = CliRunner().invoke(
        app,
        [
            "deploy",
            "cohort",
            "start",
            "--plan",
            str(file),
            "--state-dir",
            str(tmp_path / "state"),
        ],
    )
    assert result.exit_code == 0, result.output
    if platform == "darwin":
        target = next(tmp_path.glob("Library/LaunchAgents/*.plist"))
        unit = plistlib.loads(target.read_bytes())
        assert unit["KeepAlive"] == {"SuccessfulExit": False}
        assert "supervise" in unit["ProgramArguments"]
        assert commands[-1][:2] == ["launchctl", "bootstrap"]
    else:
        target = next(tmp_path.glob(".config/systemd/user/*.service"))
        assert "Restart=on-failure" in target.read_text()
        assert commands[-1][:4] == ["systemctl", "--user", "enable", "--now"]
    assert "do-not-copy" not in target.read_text()
    assert "named-profile" in target.read_text()


@pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("MANABOT_TEST_LAUNCHD") != "1",
    reason="explicit host-service proof; ordinary CI uses subprocess fault injection",
)
def test_launchd_restarts_killed_cohort_without_launcher(tmp_path: Path) -> None:
    """Use the real OS service manager, retaining only fake storage/provider state."""
    (tmp_path / "cohort.json").write_text(plan().model_dump_json())
    label = f"manabot.test.cohort.{uuid.uuid4().hex}"
    domain = f"gui/{os.getuid()}"
    target = tmp_path / "test.plist"
    target.write_bytes(
        plistlib.dumps(
            {
                "Label": label,
                "ProgramArguments": [
                    shutil.which("uv"),
                    "run",
                    "--no-sync",
                    "python",
                    "-m",
                    "tests.remote.cohort_worker",
                    str(tmp_path),
                ],
                "WorkingDirectory": str(ROOT),
                "RunAtLoad": True,
                "KeepAlive": {"SuccessfulExit": False},
                "ThrottleInterval": 1,
                "StandardOutPath": str(tmp_path / "service.log"),
                "StandardErrorPath": str(tmp_path / "service.log"),
            }
        )
    )
    try:
        subprocess.run(
            ["launchctl", "bootstrap", domain, str(target)], check=True, timeout=10
        )
        wait_path(tmp_path / "created-before-lost-response")
        driver = json.loads((tmp_path / "service/driver.json").read_text())
        original_pid = driver["pid"]
        os.kill(original_pid, signal.SIGKILL)
        (tmp_path / "restart-allowed").touch()
        end = time.monotonic() + 30
        while time.monotonic() < end:
            raw = FileStore(tmp_path / "cohort.sqlite").read("state.json")
            if raw is not None:
                state = CohortState.model_validate_json(raw.data)
                if state.phase == "completed":
                    break
            time.sleep(0.1)
        else:
            pytest.fail((tmp_path / "service.log").read_text())
        restarted = json.loads((tmp_path / "service/driver.json").read_text())
        assert restarted["pid"] != original_pid
        assert len(state.attempts) == 2
        assert FileStore(tmp_path / "provider.sqlite").read("pod3") is None
    finally:
        subprocess.run(
            ["launchctl", "bootout", f"{domain}/{label}"],
            capture_output=True,
            timeout=10,
        )
