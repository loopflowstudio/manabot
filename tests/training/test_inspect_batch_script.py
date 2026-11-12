"""The batch inspection script in docs/learning.md must keep working."""

from pathlib import Path
import subprocess
import sys


def test_inspect_batch_reports_a_real_batch() -> None:
    script = Path(__file__).parents[2] / "scripts" / "inspect_batch.py"
    result = subprocess.run(
        [sys.executable, str(script), "--rows", "3"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "seat trajectories" in result.stdout
    assert "bootstrapped tail fraction: 0" in result.stdout
