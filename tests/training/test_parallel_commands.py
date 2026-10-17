"""Real children prove concurrency, cancellation and retained failure evidence."""

from pathlib import Path
import sys
import time

import pytest

from experiments.runners.gpu_job import Deadlines, Worker
from experiments.runners.parallel_commands import Command, run_parallel


def test_children_can_progress_concurrently(tmp_path: Path) -> None:
    script = tmp_path / "barrier.py"
    script.write_text(
        "from pathlib import Path\nimport sys,time\n"
        "root=Path(sys.argv[1]); own=sys.argv[2]; other=sys.argv[3]\n"
        "(root/own).write_text('ready')\n"
        "while not (root/other).exists(): time.sleep(.01)\n"
        "(root/(own+'-complete')).write_text('done')\n"
    )
    commands = tuple(
        Command(name, (sys.executable, str(script), str(tmp_path), name, other), 3)
        for name, other in (("a", "b"), ("b", "a"))
    )
    worker = Worker(tmp_path, Deadlines(time.time() + 10, retention_seconds=1))
    assert run_parallel(worker, commands, 2) == 0
    assert (tmp_path / "a-complete").read_text() == "done"
    assert (tmp_path / "b-complete").read_text() == "done"


def test_failure_stops_sibling_preserves_output_and_skips_pending(
    tmp_path: Path,
) -> None:
    script = tmp_path / "failure.py"
    script.write_text(
        "from pathlib import Path\nimport sys,time\nroot=Path(sys.argv[1])\n"
        "if sys.argv[2]=='failure':\n"
        " while not (root/'checkpoint').exists(): time.sleep(.01)\n"
        " print('original failure',flush=True)\n sys.exit(7)\n"
        "if sys.argv[2]=='sibling':\n"
        " (root/'checkpoint').write_text('retained')\n time.sleep(30)\n"
        "(root/'forbidden').write_text('unexpected late execution')\n"
    )
    commands = tuple(
        Command(name, (sys.executable, str(script), str(tmp_path), name), 40)
        for name in ("failure", "sibling", "pending")
    )
    worker = Worker(tmp_path, Deadlines(time.time() + 60, retention_seconds=1))
    assert run_parallel(worker, commands, 2) == 7
    assert (tmp_path / "checkpoint").read_text() == "retained"
    assert not (tmp_path / "forbidden").exists()
    assert not (tmp_path / "commands/pending").exists()
    assert "original failure" in (tmp_path / "commands/failure/run.log").read_text()
    assert (
        "interrupted" in (tmp_path / "commands/sibling/worker-status.json").read_text()
    )


def test_expired_outer_deadline_prevents_execution(tmp_path: Path) -> None:
    worker = Worker(tmp_path, Deadlines(time.time(), retention_seconds=1))
    command = Command("expired", (sys.executable, "-c", "raise AssertionError"), 5)
    with pytest.raises(TimeoutError, match="retention reserve"):
        run_parallel(worker, (command,), 1)
    assert not (tmp_path / "commands/expired/run.log").exists()
