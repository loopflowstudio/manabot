"""The retired training command directs existing invocations to Etude."""

from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from manabot.cli import app


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--list"],
        ["--recipe", "local", "--device", "mps", "--out", "old-run"],
        ["--regime", "saved.json", "--resume-from", "checkpoint.pt"],
    ],
)
def test_retired_train_names_the_replacement_without_starting_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["train", *options])
    assert result.exit_code == 2, result.output
    assert "manabot train is removed" in result.output
    assert "uv run etude train FILE" in result.output
    assert "--resume" in result.output
    assert "Traceback" not in result.output
    assert list(tmp_path.iterdir()) == []


def test_migration_message_is_available_without_the_training_sdk() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "sys.modules['etude.stopping'] = None; "
            "sys.modules['manabot.training.loops.self_play'] = None; "
            "sys.argv = ['manabot', 'train']; "
            "from manabot.cli import main; main()",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "uv run etude train FILE" in result.stderr
    assert "Traceback" not in result.stderr
