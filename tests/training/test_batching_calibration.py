"""Calibration rejects partial panels and keeps evidence when orchestration fails."""

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import pytest

from experiments.runners import batching_calibration as calibration
from experiments.runners.gpu_job import Worker


def test_cost_reserve_accounts_for_slowest_game_and_coordinator(tmp_path: Path) -> None:
    path = tmp_path / "monitor.json"
    record = {
        "status": "completed",
        "expected_games": 8,
        "evaluation_seconds": 16,
        "rows": [{"game_seconds": n + 1} for n in range(8)],
    }
    path.write_text(json.dumps(record))
    assert (
        calibration.evaluation_timing(path, "large", "random").reserved_seconds_per_game
        == 12
    )
    record["evaluation_seconds"] = 160
    path.write_text(json.dumps(record))
    assert (
        calibration.evaluation_timing(path, "large", "random").reserved_seconds_per_game
        == 30
    )
    record["rows"] = [{"game_seconds": 1}]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="all eight"):
        calibration.evaluation_timing(path, "large", "random")


@pytest.mark.parametrize("fail", [False, True])
def test_worker_retains_panel_and_stops_on_command_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    calls: list[str] = []

    def command(self: Worker, name: str, args: list[str], seconds: float) -> int:
        calls.append(name)
        (tmp_path / f"{name}.log").write_text("retained child output\n")
        if name.startswith("eval-"):
            assert args[args.index("--transitions") + 1] == "0"
            panel = Path(args[args.index("--out") + 1]) / "transitions-0"
            panel.mkdir(parents=True)
            (panel / "monitor.json").write_text(
                json.dumps(
                    {
                        "status": "incomplete" if fail else "completed",
                        "expected_games": 8,
                        "evaluation_seconds": 8,
                        "rows": [{"game_seconds": 1}] * 8,
                    }
                )
            )
            if fail:
                return 124
        return 0

    monkeypatch.setattr(Worker, "command", command)
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv(
        "ETUDE_DEADLINE", (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    )
    if fail:
        with pytest.raises(RuntimeError, match="calibration incomplete"):
            calibration.main()
        assert calls[-1] == "eval-small-scripted_greedy"
        assert not (tmp_path / "timings.json").exists()
    else:
        calibration.main()
        timings = json.loads((tmp_path / "timings.json").read_text())
        assert timings["complete"]
        assert {(t["size"], t["anchor"]) for t in timings["timings"]} == {
            (size, anchor)
            for size in calibration.SIZES
            for anchor in calibration.ANCHORS
        }
    status = json.loads((tmp_path / "worker-status.json").read_text())
    assert status["phase"] == ("failed" if fail else "completed")
    files = json.loads((tmp_path / "manifest.json").read_text())["files"]
    assert any(f["path"] == "eval-small-scripted_greedy.log" for f in files)
    assert any(
        f["path"] == "small-scripted_greedy/transitions-0/monitor.json" for f in files
    )
