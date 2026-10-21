"""Etude publication preserves admitted arena estimates and failed attempts."""

import json
from pathlib import Path

from etude.track.read import read_all
from etude.track.records import Step
from etude.track.run import init
import pytest

from manabot.arena.models import file_sha256
from manabot.training.etude_evaluation import publish_saved_evaluation
from manabot.training.monitor_evaluation import ArenaRow, MonitorResult
from tests.training.test_monitor_evaluation import _manifest, _rows


def saved_result(tmp_path: Path, *, incomplete: bool = False) -> Path:
    result = _manifest()
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture artifact; no training or policy admission claim")
    result.artifact = {
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "sha256": file_sha256(checkpoint),
    }
    result.candidate = result.candidate.model_copy(
        update={
            "checkpoint_sha256": result.artifact["sha256"],
            "checkpoint_bytes": result.artifact["bytes"],
        }
    )
    result.rows = [ArenaRow.model_validate(row) for row in _rows(result)]
    if incomplete:
        result.rows.pop()
    result.validate_cohort()
    path = tmp_path / "monitor.json"
    path.write_text(result.model_dump_json())
    return path


@pytest.mark.parametrize("incomplete", [False, True])
def test_sdk_retains_cohort_state_and_deal_intervals(
    tmp_path: Path, incomplete: bool
) -> None:
    source = saved_result(tmp_path, incomplete=incomplete)
    tracker = init(
        experiment="arena-export-fixture",
        clocks=[Step("updates", "updates")],
        directory=tmp_path / "tracking",
        strict=True,
    )
    try:
        key = publish_saved_evaluation(tracker, source)
    finally:
        tracker.finish()
    data = read_all(tmp_path / "tracking")[0]
    assert len(data.evaluations) == 1
    evaluation = data.evaluations[0]
    assert evaluation.key == key
    assert evaluation.state == ("failed" if incomplete else "settled")
    assert evaluation.episodes == (7 if incomplete else 8)
    assert evaluation.expected_episodes == 8
    if incomplete:
        assert evaluation.estimates == ()
    else:
        score = next(e for e in evaluation.estimates if e.name == "score")
        assert (score.mean, score.lower, score.upper, score.n) == (0.5, 0, 1, 2)
    episodes = list((tmp_path / "tracking").rglob("*.episodes.jsonl"))
    assert len(episodes) == 1
    rows = [json.loads(line) for line in episodes[0].read_text().splitlines()]
    assert len(rows) == evaluation.episodes
    assert all("result_sha256" in row and "coordinates" in row for row in rows)
    assert all("trace_path" not in row and "command" not in row for row in rows)


def test_changed_checkpoint_and_fabricated_summary_fail_before_publication(
    tmp_path: Path,
) -> None:
    path = saved_result(tmp_path)
    tracker = init(
        experiment="rejection-fixture",
        clocks=[Step("updates", "updates")],
        directory=tmp_path / "tracking",
        strict=True,
    )
    try:
        original = path.read_text()
        result = MonitorResult.model_validate_json(original)
        assert result.score is not None
        result.score.mean = 1
        path.write_text(result.model_dump_json())
        with pytest.raises(ValueError, match="inconsistent score"):
            publish_saved_evaluation(tracker, path)
        path.write_text(original)
        (tmp_path / "fixture.pt").write_bytes(b"changed")
        with pytest.raises(ValueError, match="checkpoint bytes"):
            publish_saved_evaluation(tracker, path)
    finally:
        tracker.finish()
    assert not read_all(tmp_path / "tracking")[0].evaluations
