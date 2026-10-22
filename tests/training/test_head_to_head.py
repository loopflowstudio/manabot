"""Direct evaluations bind both checkpoints and preserve every deck/seat leg."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("etude.eval.match")

from manabot.arena import duel as duel_eval
from manabot.arena.monitor import MonitorProtocol
from manabot.infra.files import atomic_json
from tests.training.test_etude_evaluation import _admission, _play_rows


def _duel(tmp_path: Path) -> duel_eval.Duel:
    first = _admission(tmp_path)
    first.protocol = MonitorProtocol(
        deal_seeds=tuple(range(101, 126)), opponent="heuristic", include_mirrors=True
    )
    first.evaluation_identities = {
        name: "x"
        for name in (
            "world",
            "engine_source_sha256",
            "content_manifest_sha256",
            "observation_abi_sha256",
            "action_abi_sha256",
            "matchup_sha256",
        )
    }
    return duel_eval.Duel(two_seat=first, one_seat=first.model_copy(deep=True))


def test_direct_etude_plays_two_checkpoint_players_and_balanced_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duel = _duel(tmp_path)
    target = tmp_path / "evaluations" / "1000"
    target.mkdir(parents=True)
    atomic_json(target / "duel.json", duel.model_dump(mode="json"))
    played: list[int] = []

    def play_cell(
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        assert (
            kwargs["player_a"].runner_kind
            == kwargs["player_b"].runner_kind
            == "checkpoint"
        )
        assert set(kwargs["checkpoint_paths"]) == {"one-seat", "two-seat"}
        seed = kwargs["deal_seeds"][0]
        played.append(seed)
        return _play_rows(duel.two_seat, seed, kwargs["out_dir"]), {}, {}

    monkeypatch.setattr(duel_eval, "play_cell", play_cell)
    duel_eval.evaluate(target / "duel.json")
    result = json.loads((target / "result.json").read_bytes())
    public = tmp_path / "source" / "public"
    public.mkdir(parents=True)
    duel_eval.report(
        tmp_path,
        public.parent,
        {
            "1000": duel_eval.DirectAttempt(
                phase="completed", started_at=1, two_seat="a" * 64, one_seat="b" * 64
            )
        },
    )
    assert "W/D/L" in (public / "head-to-head.html").read_text()
    assert len(played) == 25
    assert result["groups"]["all"]["games"] == 200
    assert [
        result["groups"][label]["games"] for label in ("LvA", "AvL", "LvL", "AvA")
    ] == [50] * 4
    duel_eval.evaluate(target / "duel.json")
    assert len(played) == 25
    assert json.loads((target / "result.json").read_bytes())["etude"]["skipped"] == 25


def test_duel_rejects_unequal_updates_runtime_and_bytes(tmp_path: Path) -> None:
    duel = _duel(tmp_path)
    duel.one_seat.coordinates.updates += 1
    with pytest.raises(ValueError, match="equal update"):
        duel.validate_inputs()
    duel.one_seat.coordinates.updates -= 1
    duel.one_seat.evaluation_identities["world"] = "other"
    with pytest.raises(ValueError, match="runtime differs"):
        duel.validate_inputs()
    duel.one_seat.evaluation_identities["world"] = "x"
    Path(duel.one_seat.artifact["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact bytes differ"):
        duel.validate_inputs()


def test_follower_selects_1000_and_retains_failure_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from manabot.remote.live_evaluation import Attempt, Ledger

    duel = _duel(tmp_path)
    duel.two_seat.coordinates.updates = duel.one_seat.coordinates.updates = 1000
    source = tmp_path / "source"
    (source / "public").mkdir(parents=True)
    controls = tmp_path / "controls"
    controls.mkdir()
    atomic_json(controls / "1000.json", duel.one_seat.model_dump(mode="json"))
    digest = duel.two_seat.artifact["sha256"]
    target = source / "evaluations" / f"{digest}-attempt-1"
    target.mkdir(parents=True)
    atomic_json(target / "admission.json", duel.two_seat.model_dump(mode="json"))
    ledger = Ledger(
        cohort_sha256="a" * 64,
        cohort_phase="completed",
        attempts={
            digest: Attempt(
                digest=digest,
                job="job",
                updates=1000,
                learner_decisions=1,
                training_seconds=1,
                phase="completed",
                started_at=1,
            )
        },
    )
    atomic_json(source / "ledger.json", ledger.model_dump(mode="json"))
    calls: list[object] = []

    class FailedChild:
        pid = 123
        returncode = 1

        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append(args)

        def poll(self) -> int:
            return 1

        def wait(self, timeout: float | None = None) -> int:
            return 1

    monkeypatch.setattr(duel_eval.subprocess, "Popen", FailedChild)
    for _ in range(2):
        duel_eval.follow(
            source, controls, tmp_path / "direct", deadline=duel_eval.time.time() + 10
        )
    assert len(calls) == 1
    ledger = json.loads((tmp_path / "direct/ledger.json").read_bytes())
    assert ledger["1000"]["phase"] == "failed"
    assert "failed" in (source / "public/head-to-head.html").read_text()
