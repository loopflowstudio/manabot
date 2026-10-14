"""Worker caps remain effective without a supervising Experiment process."""

import signal
import subprocess
import sys

import pytest


@pytest.mark.parametrize("active_runtime", [False, True])
def test_isolated_worker_expires(active_runtime: bool) -> None:
    # A past calendar deadline must not override an awake-time allowance.
    command = (
        "from manabot.training.clock import arm_worker_deadline\n"
        "import time\n"
        f"arm_worker_deadline(active_runtime={active_runtime!r}, "
        "allowance_seconds=0.1, deadline_unix=time.time() - 1)\n"
        "time.sleep(10)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", command], start_new_session=True)
    try:
        assert process.wait(timeout=5) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
