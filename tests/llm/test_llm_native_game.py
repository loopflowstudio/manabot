"""The full-game harness must retain native replay and explicit incomplete runs."""

from pathlib import Path
import time

import pytest

from manabot.verify.llm_game import GameConfig, play


def test_native_random_control_completes_and_replays(tmp_path: Path) -> None:
    summary = play(GameConfig(), tmp_path / "game", None)
    assert summary.completed and summary.failure is None
    assert summary.replay_matches
    assert summary.decisions > 100
    assert summary.model_decisions == 0
    assert summary.origins == {"random-control": summary.decisions}
    assert (
        len(list((tmp_path / "game").glob("decision-*/transition.json")))
        == summary.decisions
    )


def test_decision_limit_is_not_a_completed_game(tmp_path: Path) -> None:
    summary = play(GameConfig(max_decisions=2), tmp_path / "game", None)
    assert not summary.completed
    assert summary.failure is not None and "decision limit" in summary.failure
    assert summary.decisions == 2
    assert (tmp_path / "game" / "summary.json").is_file()


def test_sequence_uses_one_session_and_retains_individual_game_summaries(
    tmp_path: Path,
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    import json
    from typing import cast

    from experiments.inference_baseline import Cell, LLMWorkload, Protocol, Recipe
    from experiments.inference_baselines import llm
    from manabot.llm.cache import CacheReceipt
    from manabot.llm.schema import InferenceConfig
    from manabot.llm.session import InferenceSession
    from manabot.verify import llm_game

    config_path = Path("configs/llm/qwen3.5-9b.json")
    config = InferenceConfig.model_validate_json(config_path.read_bytes())
    manifest = tmp_path / "cache-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_id": config.model_id,
                "revision": config.model_revision,
                "files": [],
            }
        )
    )
    paths = tuple(tmp_path / f"input-{i}.json" for i in range(2))
    for seed, path in enumerate(paths):
        path.write_text(GameConfig(seed=seed).model_dump_json())
    workload = LLMWorkload(
        id="sequence",
        kind="llm",
        config=config_path,
        cache_manifest=manifest,
        games=paths,
    )
    cell = Cell(id="resident", workload="sequence")
    protocol = Protocol(
        recipe=Recipe(workloads=(workload,), cells=(cell,), repetitions=1),
        source=(),
        inputs=(),
        runtime={},
    )
    out = tmp_path / "out"
    out.mkdir()

    class FixtureSession:
        load_seconds = 7.0
        closed = False
        games = 0

        def __init__(
            self, config: InferenceConfig, *, load_seconds: float, factory: object
        ) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    def cache(*args: object, **kwargs: object) -> CacheReceipt:
        return CacheReceipt(
            model_id=config.model_id,
            revision=config.model_revision,
            model_path="fixture",
            tokenizer_path="fixture",
            files=[],
            seconds=3,
            acquisition_seconds=1,
            verification_seconds=2,
            offline=True,
        )

    def game(
        config: GameConfig, out: Path, session: InferenceSession
    ) -> llm_game.GameSummary:
        fixture = cast(FixtureSession, session)
        assert not fixture.closed
        fixture.games += 1
        out.mkdir()
        summary = llm_game.GameSummary(
            mode="fixture",
            completed=True,
            model_load_seconds=fixture.load_seconds,
            game_seconds=2,
        )
        (out / "config.json").write_text(config.model_dump_json())
        (out / "summary.json").write_text(summary.model_dump_json())
        (out / "session-state.json").write_text(json.dumps({"games": fixture.games}))
        return summary

    monkeypatch.setattr(llm, "prepare", cache)
    monkeypatch.setattr(llm, "InferenceSession", FixtureSession)
    monkeypatch.setattr(llm_game, "play", game)
    result = llm.execute(protocol, workload, cell, out, False, time.time() + 30)
    assert result["completed_games"] == 2 and result["status"] == "completed"
    assert json.loads((out / "game-000/session-state.json").read_text())["games"] == 1
    assert json.loads((out / "game-001/session-state.json").read_text())["games"] == 2
    assert json.loads((out / "session-cleanup.json").read_text())["closed"]
    assert json.loads((out / "readiness.json").read_text())["seconds"] == 7


def test_reasoning_campaign_schedule_is_complete_and_balanced() -> None:
    from experiments.elemental_grid import DECKS
    from experiments.qwen_mtg183 import Campaign, schedule

    games = schedule(Campaign())
    assert len(games) == 100
    assert {(g.qwen_deck, g.policy_deck) for g in games} == {
        (a, b) for a in DECKS for b in DECKS
    }
    assert sum(g.config.model_seat for g in games) == 50
    assert len({g.config.seed for g in games}) == 100
    assert all(g.config.skip_trivial for g in games)
    assert [g.model_dump() for g in games] == [
        g.model_dump() for g in schedule(Campaign())
    ]


def test_campaign_archive_retains_completed_and_partial_games(tmp_path: Path) -> None:
    import tarfile

    from experiments.qwen_mtg183 import collect

    out = tmp_path / "outputs"
    out.mkdir()
    record = tmp_path / "game-records/game-000"
    record.mkdir(parents=True)
    (record / "progress.json").write_text('{"decisions": 1}')
    collect(out)
    (record / "summary.json").write_text('{"completed": true}')
    collect(out)
    with tarfile.open(out / "game-000.tar.gz") as archive:
        assert "game-000/summary.json" in archive.getnames()
    before = (out / "game-000.tar.gz").read_bytes()
    collect(out)
    assert (out / "game-000.tar.gz").read_bytes() == before
