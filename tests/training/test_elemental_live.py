"""The isolated Etude owner must retain failures and wait for endpoint coverage.

Run with the exact evaluation SDK runtime; the training SDK intentionally lacks
these APIs and skips this module.
"""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("etude.eval.query")
from experiments.runners import elemental_live as live  # noqa: E402


def _config(tmp_path: Path) -> live.Config:
    return live.Config(
        root=tmp_path,
        directory=tmp_path,
        job_id="mtg174-test",
        revision="a" * 40,
        deadline=10**12,
        monitoring=(),
        scoring=(),
    )


def _observation() -> dict[str, Any]:
    # This fake is the validated JSON process boundary, not a second job model.
    return {
        "source": {"commit": "a" * 40},
        "observed_at": "2026-10-22T01:00:00Z",
        "status": {
            "record": {"phase": "paused", "updates": 501},
            "cleanup": {"estimated_dollars": 2.5},
        },
        "checkpoints": [
            dict(
                path=f"/{update}.pt",
                sha256=str(update),
                coordinates={"updates": update},
            )
            for update in (0, 500, 501)
        ],
    }


def test_endpoint_and_initialization_remain_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    ledger = live.Ledger(config_sha256="test")
    launched: list[dict[str, Any]] = []
    monkeypatch.setattr(live, "_remote", lambda *a, **kw: _observation())
    monkeypatch.setattr(live, "_report", lambda *a: None)
    monkeypatch.setattr(live.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(live, "launch_evaluation", lambda *a, **kw: launched.append(kw))

    def refresh(config: live.Config, point: live.Point) -> None:
        saved = live.Ledger.model_validate_json((tmp_path / "ledger.json").read_bytes())
        assert (
            next(p for p in saved.points if p.eval_id == point.eval_id).started
            is not None
        )
        point.phase, point.outcome = "done", "succeeded"
        point.games = point.valid = 200
        point.score = 0.6

    monkeypatch.setattr(live, "_refresh", refresh)
    live.operate(config, ledger, tmp_path / "ledger.json")
    assert ledger.settled and ledger.cleanup_confirmed
    assert ledger.inclusive_cost == 2.5
    assert len(launched) == 6
    assert len({p.eval_id for p in ledger.points}) == 6
    assert {(p.cohort, p.update) for p in ledger.points} == {
        ("monitor", 0),
        ("monitor", 500),
        ("monitor", 501),
        ("final-initial", 501),
        ("final-greedy", 501),
        ("initial-greedy", 0),
    }


def test_invalid_evaluation_retains_failure_and_cancels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    ledger = live.Ledger(config_sha256="test")
    cancellations: list[bool] = []

    def remote(config: live.Config, *, cancel: bool = False) -> dict[str, Any]:
        cancellations.append(cancel)
        return _observation()

    def refresh(config: live.Config, point: live.Point) -> None:
        point.phase, point.outcome = "done", "succeeded"
        point.games, point.valid = 4, 3

    monkeypatch.setattr(live, "_remote", remote)
    monkeypatch.setattr(live, "_refresh", refresh)
    monkeypatch.setattr(live, "_report", lambda *a: None)
    monkeypatch.setattr(live, "launch_evaluation", lambda *a, **kw: "test")
    with pytest.raises(RuntimeError, match="invalid games"):
        live.operate(config, ledger, tmp_path / "ledger.json")
    retained = live.Ledger.model_validate_json((tmp_path / "ledger.json").read_bytes())
    assert not retained.settled
    assert retained.points[0].valid == 3
    assert retained.error is not None
    assert cancellations == [False, True]
