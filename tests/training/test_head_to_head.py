"""Direct evaluations bind both checkpoints and preserve every deck/seat leg."""

from pathlib import Path

import pytest

pytest.importorskip("etude.eval.match")

from manabot.eval import duel as duel_eval
from manabot.eval.evaluation import EvaluationRequest, EvaluationResult
from manabot.eval.monitor import MonitorProtocol
from manabot.eval.results import EvaluationRow
from manabot.infra.files import atomic_json
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
