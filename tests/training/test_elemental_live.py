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


def _observation(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "0.pt").write_bytes(b"initial")
    # This fake is the validated JSON process boundary, not a second job model.
    return {
        "run": {"id": "training-174"},
        "source": {"commit": "a" * 40},
        "observed_at": "2026-10-22T01:00:00Z",
        "status": {
            "record": {"phase": "paused", "updates": 501},
            "cleanup": {"estimated_dollars": 2.5},
        },
        "checkpoints": [
            dict(
                path=str(tmp_path / f"{update}.pt"),
                sha256=str(update),
                coordinates={"updates": update},
            )
            for update in (0, 25, 100, 499, 500, 501)
        ],
    }


def test_endpoint_and_initialization_remain_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        live,
        "status",
        lambda run: (_ for _ in ()).throw(live.LauncherError(f"no run {run!r}")),
    )
    ledger = live.Ledger(config_sha256="test")
    launched: list[dict[str, Any]] = []
    monkeypatch.setattr(live, "_remote", lambda *a, **kw: _observation(tmp_path))
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
    monkeypatch.setattr(
        live,
        "status",
        lambda run: (_ for _ in ()).throw(live.LauncherError(f"no run {run!r}")),
    )
    ledger = live.Ledger(config_sha256="test")
    cancellations: list[bool] = []

    def remote(config: live.Config, *, cancel: bool = False) -> dict[str, Any]:
        cancellations.append(cancel)
        return _observation(tmp_path)

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
    live.operate(config, retained, tmp_path / "ledger.json")
    assert cancellations == [False, True, True]
    assert not retained.settled
    assert retained.error is not None


@pytest.mark.parametrize("accepted", [False, True])
def test_restart_reconciles_interruption_around_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accepted: bool
) -> None:
    config = _config(tmp_path)
    point = live.Point(
        key="request",
        training_run="training-174",
        update=500,
        checkpoint=tmp_path / "model.pt",
        sha256="a" * 64,
        cohort="monitor",
        eval_id="174000000004",
        started=123,
    )
    ledger = live.Ledger(config_sha256="test", points=[point])
    path = tmp_path / "ledger.json"
    live._save(path, ledger)
    # A durable accepted job exists independently of the owner's ledger.
    jobs = {point.eval_id} if accepted else set()
    submissions = 0

    def status(run: str) -> None:
        if run not in jobs:
            raise live.LauncherError(f"no run {run!r}")

    def launch(*args: object, **kwargs: Any) -> str:
        nonlocal submissions
        run = str(kwargs["run_id"])
        if run in jobs:
            raise AssertionError("duplicate submission")
        jobs.add(run)
        submissions += 1
        raise KeyboardInterrupt  # Lose the reply after durable acceptance.

    monkeypatch.setattr(live, "status", status)
    monkeypatch.setattr(live, "launch_evaluation", launch)
    if not accepted:
        with pytest.raises(KeyboardInterrupt):
            live._ensure_launched(config, point, ledger, path)
    retained = live.Ledger.model_validate_json(path.read_bytes())
    live._ensure_launched(config, retained.points[0], retained, path)
    assert jobs == {point.eval_id}
    assert submissions == (0 if accepted else 1)
    assert retained.points[0].phase == "pending"


def test_uncertain_submission_lookup_stops_without_resubmission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    point = live.Point(
        key="request",
        training_run="training-174",
        update=0,
        checkpoint=tmp_path / "model",
        sha256="a" * 64,
        cohort="monitor",
        eval_id="174000000005",
    )

    def unavailable(run: str) -> None:
        raise live.LauncherError("controller connection lost")

    monkeypatch.setattr(live, "status", unavailable)
    with pytest.raises(live.LauncherError, match="connection lost"):
        live._ensure_launched(
            config, point, live.Ledger(config_sha256="test"), tmp_path / "ledger.json"
        )
    assert not (tmp_path / "ledger.json").exists()
