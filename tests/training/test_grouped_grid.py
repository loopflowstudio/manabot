"""Grouping must stop at failures and reject incomplete or altered cell evidence."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

from experiments.runners import grouped_grid


def retained_cell(root: Path) -> None:
    (root / "training/run").mkdir(parents=True)
    (root / "training/run/run.json").write_text(
        json.dumps({"status": "completed", "seed": 13571})
    )
    (root / "summary.json").write_text(
        json.dumps(
            {"checkpoint_finite": True, "learner_transitions": 131072, "updates": 8}
        )
    )
    checkpoint = root / "policy.pt"
    checkpoint.write_bytes(b"retained policy")
    (root / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "path": checkpoint.name,
                    "bytes": checkpoint.stat().st_size,
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                }
            ]
        )
    )


def test_group_rejects_changed_or_incomplete_evidence(tmp_path: Path) -> None:
    retained_cell(tmp_path)
    cell = grouped_grid.Cell.parse("small-deep:256")
    grouped_grid.verify_completed(tmp_path, cell)
    with pytest.raises(ValueError, match="did not finish"):
        grouped_grid.verify_completed(
            tmp_path, grouped_grid.Cell.parse("small-deep:128")
        )
    (tmp_path / "policy.pt").write_bytes(b"substituted")
    with pytest.raises(ValueError, match="artifact differs"):
        grouped_grid.verify_completed(tmp_path, cell)


def test_failed_cell_does_not_launch_following_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv(
        "ETUDE_DEADLINE", datetime.fromtimestamp(time.time() + 7200, UTC).isoformat()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["grouped-grid", "--cell", "small-deep:256", "--cell", "small-deep:128"],
    )
    calls: list[tuple[str, float]] = []

    def command(
        self: grouped_grid.Worker, name: str, args: list[str], seconds: float
    ) -> int:
        calls.append((name, seconds))
        return 124 if name == "small-deep-s256" else 0

    monkeypatch.setattr(grouped_grid.Worker, "command", command)
    monkeypatch.setattr(grouped_grid.signal, "signal", lambda *args: None)
    with pytest.raises(SystemExit) as stopped:
        grouped_grid.main()
    assert stopped.value.code == 124
    assert calls == [
        ("nvidia-smi", 15),
        ("setup", 1200),
        ("cuda-admission", 60),
        ("small-deep-s256", 1860),
    ]
    receipt = json.loads((tmp_path / "worker-status.json").read_text())
    assert receipt["phase"] == "incomplete"
    assert receipt["completed"] == []
    assert receipt["remaining"] == ["small-deep-s256", "small-deep-s128"]


def test_group_progress_keeps_prior_cell_clocks_and_checkpoint_paths(
    tmp_path: Path,
) -> None:
    cells = [
        grouped_grid.Cell.parse("small-deep:256"),
        grouped_grid.Cell.parse("small-deep:128"),
    ]
    for index, cell in enumerate(cells):
        root = tmp_path / cell.name
        root.mkdir()
        (root / "policy.pt").write_bytes(b"retained")
        (root / "etude-result.json").write_text(
            json.dumps(
                {
                    "clocks": {
                        "updates": 8 if index == 0 else 1,
                        "learner_transitions": 131072 if index == 0 else 8192,
                        "training_seconds": 100 if index == 0 else 5,
                    },
                    "checkpoint": "policy.pt",
                }
            )
        )
    grouped_grid.publish_progress(tmp_path, cells, 1)
    progress = json.loads((tmp_path / "etude-result.json").read_text())
    assert progress["clocks"]["learner_transitions"] == 139264
    assert progress["clocks"]["cells_completed"] == 1
    assert progress["clocks"]["cell_0_updates"] == 8
    assert progress["clocks"]["cell_1_updates"] == 1
    assert progress["checkpoint"] == "small-deep-s128/policy.pt"
