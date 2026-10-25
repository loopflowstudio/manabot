"""Direct evaluations bind both checkpoints and preserve every deck/seat leg."""

import json
from pathlib import Path
import time
from typing import Any

import pytest

pytest.importorskip("etude.eval.match")

from manabot.eval import duel as duel_eval
from manabot.eval.endpoint import InferenceEndpoint
from manabot.eval.evaluation import EvaluationRequest, EvaluationResult
from manabot.eval.match import derive_seed
from manabot.eval.monitor import MonitorProtocol
from manabot.eval.results import EvaluationRow
from manabot.infra.files import atomic_json
from manabot.sim.execution import EvalExecution
from tests.eval.test_local_evaluation import _publish_blocks
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


@pytest.mark.parametrize("shared", [False, True])
def test_direct_etude_plays_two_checkpoint_players_and_balanced_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared: bool
) -> None:
    duel = _duel(tmp_path)
    candidate, opponent = duel_eval.registrations(duel)
    seed_key = duel.two_seat.key.model_copy(
        update={"anchor_cohort_sha256": opponent.identity_sha256}
    )
    execution = (
        EvalExecution(deadline_unix=time.time() + 60, max_active_games=1)
        if shared
        else None
    )
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
        pair = (candidate.player_id, opponent.player_id)
        assert derive_seed(
            kwargs["key"], pair, seed, candidate.player_id
        ) == derive_seed(seed_key, pair, seed, candidate.player_id)
        played.append(seed)
        manifest = duel.two_seat.model_copy(
            update={
                "key": kwargs["key"],
                "candidate": kwargs["player_a"],
                "opponent": kwargs["player_b"],
            }
        )
        rows = [
            EvaluationRow.model_validate(row)
            for row in _play_rows(manifest, seed, kwargs["out_dir"])
        ]
        _publish_blocks(kwargs["out_dir"], rows)
        return [row.model_dump(mode="json") for row in rows], {}, {}

    monkeypatch.setattr("manabot.eval.evaluation.play_cell", play_cell)
    monkeypatch.setattr(InferenceEndpoint, "play_cell", staticmethod(play_cell))
    duel_eval.evaluate(target / "duel.json", execution=execution)
    result = json.loads((target / "result.json").read_bytes())
    assert len(played) == 25
    assert result["groups"]["all"]["games"] == 200
    assert [
        result["groups"][label]["games"] for label in ("LvA", "AvL", "LvL", "AvA")
    ] == [50] * 4
    if execution is not None:
        execution = execution.model_copy(
            update={"deadline_unix": time.time() + 3600, "instrumentation": False}
        )
    duel_eval.evaluate(target / "duel.json", execution=execution)
    assert len(played) == 25
    if shared:
        assert len(list((target / "allocation").glob("*/execution.json"))) == 2
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


@pytest.mark.parametrize("historical", [False, True])
def test_summary_reads_continuation_after_empty_attempt_and_historical_rows(
    tmp_path: Path, historical: bool
) -> None:
    manifest = _admission(tmp_path)
    for seed in range(101, 126):
        panel = tmp_path / "arena" / str(seed)
        panel.mkdir(parents=True)
        rows = [
            EvaluationRow.model_validate(row)
            for row in _play_rows(manifest, seed, panel)
        ]
        if historical:
            atomic_json(
                panel / "rows.json", [row.model_dump(mode="json") for row in rows]
            )
            continue
        original = EvaluationResult(
            request=EvaluationRequest(
                key=manifest.key,
                candidate=manifest.candidate,
                opponent=manifest.opponent,
                checkpoint_paths={
                    manifest.candidate.player_id: manifest.artifact["path"]
                },
                deal_seeds=(seed,),
                matchup_mode="all",
            ),
            evaluation_id=str(seed),
            error="interrupted before publishing a block",
        )
        atomic_json(panel / "evaluation.json", original.model_dump(mode="json"))
        attempt = panel / "attempts/000001"
        attempt.mkdir(parents=True)
        _publish_blocks(attempt, rows)
        retained = original.model_copy(update={"rows": rows, "error": None})
        atomic_json(attempt / "evaluation.json", retained.model_dump(mode="json"))
        assert not (panel / "rows.json").exists()
    groups = duel_eval.summarize(tmp_path)
    assert groups["all"].games == 200
    assert [groups[label].games for label in ("LvA", "AvL", "LvL", "AvA")] == [50] * 4
