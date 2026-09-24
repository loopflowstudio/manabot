"""Exercise verifier scheduling and cleanup without a native build or browser."""

import os
from pathlib import Path
import shutil
import signal
import subprocess

import pytest


@pytest.mark.parametrize("outcome", ["ready", "launch_failure", "timeout"])
def test_browser_prepares_before_readiness_and_is_reaped(tmp_path, outcome):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (tmp_path / "managym").mkdir()
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "scripts/verify-clean-machine",
        scripts / "verify-clean-machine",
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()

    def executable(path, text):
        path.write_text("#!/bin/sh\n" + text)
        path.chmod(0o755)

    executable(binaries / "uv", "echo 123456789\n")
    executable(
        binaries / "sleep",
        'if [ "$1" = 60 ]; then exec /bin/sleep 1; fi\n'
        'exec /bin/sleep "$@"\n',
    )
    executable(
        scripts / "play",
        "mkdir -p frontend/node_modules\n"
        "touch frontend/node_modules/.etude-package-lock.sha256\n"
        "while [ ! -f browser-started ]; do sleep 0.01; done\n"
        'if [ "$PROOF_OUTCOME" = launch_failure ]; then exit 7; fi\n'
        'if [ "$PROOF_OUTCOME" = ready ]; then echo \'ETUDE_PLAY_READY {}\'; fi\n'
        "while :; do sleep 0.1; done\n",
    )
    executable(
        binaries / "npm",
        'echo "$$" > browser-pid\n'
        "trap 'touch browser-stopped; exit 0' TERM\n"
        "touch browser-started\n"
        'while ! grep -q "^ETUDE_PLAY_READY " "$ETUDE_LAUNCH_LOG"; do sleep 0.01; done\n'
        'echo \'{"result":"pass"}\' > "$ETUDE_CLEAN_RECEIPT"\n',
    )
    process = subprocess.Popen(
        ["sh", str(scripts / "verify-clean-machine")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "PROOF_OUTCOME": outcome,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()

    assert (tmp_path / "browser-started").exists()
    if outcome == "ready":
        assert process.returncode == 0, stderr
        assert '"result":"pass"' in stdout
    else:
        assert process.returncode == 2, stderr
        expected_code = "proof.launch" if outcome == "launch_failure" else "proof.timeout"
        assert expected_code in stderr
        assert (tmp_path / "browser-stopped").exists()
    browser_pid = int((tmp_path / "browser-pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(browser_pid, 0)
