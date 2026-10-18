"""A learning allocation must retain its exact seed, horizon and checkpoints."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Literal

import pytest

from experiments.runners import grouped_learning


def _retained_study(root: Path) -> None:
    run_dir = root / "training/run"
    run_dir.mkdir(parents=True)
    artifacts: dict[str, dict[str, object]] = {}
    for name in ("initial_raw", "raw", "optimizer", "milestone"):
        path = run_dir / f"{name}.pt"
        path.write_bytes(name.encode())
        artifacts[name] = {
            "path": f"/remote/run/{path.name}",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    run = {
        "status": "completed",
        "seed": 13591,
        "parent_run_id": None,
        "stages": [{"learner_transitions": 32768, "artifacts": artifacts}],
        "monitoring_checkpoints": [
            {
                "learner_transitions": 16384,
                "artifact": artifacts["milestone"],
                "error": None,
            }
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(run))
    (root / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint_finite": True,
                "learner_transitions": 32768,
                "updates": 2,
            }
        )
    )
    manifest = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_completed_cell_requires_its_seed_and_manifested_checkpoints(
    tmp_path: Path,
) -> None:
    _retained_study(tmp_path)
    planned = grouped_learning.PlannedCell.parse("small:256:300")
    grouped_learning.verify_completed(
        tmp_path, planned, seed=13591, transitions=32768, checkpoints=(16384, 32768)
    )
    with pytest.raises(ValueError, match="declared trajectory"):
        grouped_learning.verify_completed(
            tmp_path, planned, seed=13592, transitions=32768, checkpoints=(16384, 32768)
        )
    with pytest.raises(ValueError, match="unique retained checkpoint"):
        grouped_learning.verify_completed(
            tmp_path, planned, seed=13591, transitions=32768, checkpoints=(8192, 32768)
        )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(
        json.dumps(
            [row for row in manifest if row["path"] != "training/run/initial_raw.pt"]
        )
    )
    with pytest.raises(ValueError, match="not present in the retained manifest"):
        grouped_learning.verify_completed(
            tmp_path, planned, seed=13591, transitions=32768, checkpoints=(16384, 32768)
        )


@pytest.mark.parametrize("value", ("small:64:0", "small:64:nan", "small:64:inf"))
def test_nonfinite_or_nonpositive_learner_limits_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        grouped_learning.PlannedCell.parse(value)


def test_plan_rejects_rounded_or_missing_endpoints() -> None:
    cell = grouped_learning.PlannedCell.parse("small:256:300")
    with pytest.raises(ValueError, match="exact iteration boundary"):
        grouped_learning.validate_plan([cell], 13591, 32768, (8192, 32768))
    with pytest.raises(ValueError, match="include the transition endpoint"):
        grouped_learning.validate_plan([cell], 13591, 32768, (16384,))
    with pytest.raises(ValueError, match="only once"):
        grouped_learning.validate_plan([cell, cell], 13591, 32768, (16384, 32768))


@pytest.mark.parametrize("failure", ("child", "checkpoint", "stopped"))
def test_failure_stops_later_cells_and_retains_the_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Literal["child", "checkpoint", "stopped"],
) -> None:
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv(
        "ETUDE_DEADLINE", datetime.fromtimestamp(time.time() + 7200, UTC).isoformat()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grouped-learning",
            "--cell",
            "small:256:300",
            "--cell",
            "middle:64:600",
            "--seed",
            "13591",
            "--transitions",
            "32768",
            "--checkpoint-transitions",
            "16384",
            "32768",
            "--locked-setup",
            str(tmp_path / "setup.py"),
        ],
    )
    calls: list[tuple[str, list[str], float]] = []

    def command(
        self: grouped_learning.Worker, name: str, args: list[str], seconds: float
    ) -> int:
        calls.append((name, args, seconds))
        if name == "cuda-admission" and failure == "stopped":
            self.stopping = True
        return 124 if name == "small-s256" and failure == "child" else 0

    def verify(
        root: Path,
        planned: grouped_learning.PlannedCell,
        *,
        seed: int,
        transitions: int,
        checkpoints: tuple[int, ...],
    ) -> None:
        raise ValueError("missing checkpoint")

    monkeypatch.setattr(grouped_learning.Worker, "command", command)
    monkeypatch.setattr(grouped_learning, "verify_completed", verify)
    monkeypatch.setattr(grouped_learning.signal, "signal", lambda *args: None)
    expected = ValueError if failure == "checkpoint" else SystemExit
    with pytest.raises(expected) as caught:
        grouped_learning.main()
    expected_calls = ["nvidia-smi", "setup", "cuda-admission"]
    if failure != "stopped":
        expected_calls.append("small-s256")
        _, args, deadline = calls[-1]
        assert deadline == 360
        assert args[args.index("--seconds") + 1] == "300.0"
        assert args[args.index("--seed") + 1] == "13591"
        assert args[args.index("--transitions") + 1] == "32768"
        point_index = args.index("--checkpoint-transitions")
        assert args[point_index + 1 : point_index + 3] == ["16384", "32768"]
    assert [name for name, _, _ in calls] == expected_calls
    receipt = json.loads((tmp_path / "worker-status.json").read_text())
    assert receipt["phase"] == "incomplete" and receipt["completed"] == []
    assert receipt["remaining"] == ["small-s256", "middle-s64"]
    assert receipt["exit_code"] == (1 if failure == "checkpoint" else 124)
    assert receipt["error"] == (
        "ValueError: missing checkpoint" if failure == "checkpoint" else None
    )
    if isinstance(caught.value, SystemExit):
        assert caught.value.code == receipt["exit_code"]
