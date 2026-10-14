"""Independent command supervision keeps one deadline and retains stopped attempts."""

from pathlib import Path
import subprocess
import sys
import time

import pytest

from manabot.training.local_service import (
    LocalAllocation,
    LocalCommand,
    _stop,
    supervise,
)


def test_expired_allocation_cannot_start_or_extend(tmp_path: Path) -> None:
    allocation = LocalAllocation(
        id="expired",
        started_unix=time.time() - 10,
        deadline_unix=time.time() - 5,
        seconds=5,
        cpu_threads=1,
        max_evaluators=1,
    )
    path = tmp_path / "allocation.json"
    path.write_text(allocation.model_dump_json())
    command = LocalCommand(
        allocation=str(path),
        cwd=str(tmp_path),
        argv=[sys.executable, "-c", "raise RuntimeError('must not start')"],
    )
    job = tmp_path / "command.json"
    job.write_text(command.model_dump_json())
    with pytest.raises(ValueError, match="expired"):
        supervise(job)
    assert not (tmp_path / "worker.log").exists()


def test_teardown_reaps_subprocess() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    _stop(process)
    assert process.poll() is not None
