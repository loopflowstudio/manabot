"""The developer's collector -> export -> policy-free replay workflow."""

import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import torch
from typer.testing import CliRunner

from manabot.env.failures import GameInputs, TrainingExposureReceipt
from manabot.infra.identity import canonical_sha256
from manabot.sim.recording import TrainingGames, record_actions, record_games
from manabot.sim.replay import read_trace, replay_games
from manabot.training.loops.minimal import train
from managym.academy.launch import read_games


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("recorded-games")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with record_games(directory, players=("learner", "learner")):
            train(1, streams=2, decisions=16, seed=3)
    finally:
        torch.set_num_threads(previous)
    return directory


def test_actual_training_games_export_and_replay_without_policy(
    captured: Path, tmp_path: Path
) -> None:
    games = TrainingGames(source=captured)
    assert len(games.games()) >= 2
    assert games.sample(seed=0) == games.sample(seed=0)
    for index, game in enumerate(games.games()):
        before = game.path.read_bytes()
        path = game.export(tmp_path / f"{index}.game.jsonl.gz")
        assert read_games(path)[0].game_trace_sha256
        exported = read_trace(path)[0]
        receipt = TrainingExposureReceipt.model_validate_json(before)
        assert exported["source"]["kind"] == "training"
        assert exported["source"]["receipt_sha256"] == game.receipt_sha256
        assert exported["seat_players"] == ["learner", "learner"]
        assert [
            int(row["command"]["offer_id"]) for row in exported["decisions"]
        ] == receipt.game.trace.actions
        assert exported["match_hypers"]["hero_deck"] == game.decks[0]
        assert exported["match_hypers"]["villain_deck"] == game.decks[1]
        assert len(exported["decisions"]) == game.decisions
        assert replay_games([exported]).passed
        assert game.path.read_bytes() == before
    # A fresh process blocks even attempted Torch imports, including via replay.
    script = """
import builtins
from pathlib import Path
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == 'torch':
        raise AssertionError('Torch imported during export')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from manabot.sim.recording import TrainingGames
from manabot.sim.replay import read_trace, replay_games
path = TrainingGames(Path(SOURCE)).sample(seed=0).export(Path(TARGET))
assert replay_games(read_trace(path)).passed
from manabot.cli import main
main(['export-game', SOURCE, '--out', TARGET, '--game', '0'])
""".replace("SOURCE", repr(str(captured))).replace(
        "TARGET", repr(str(tmp_path / "torch-free.gz"))
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)


def _write_receipt(directory: Path, payload: dict[str, Any]) -> Path:
    receipt = TrainingExposureReceipt.model_validate(payload)
    directory.mkdir(exist_ok=True)
    path = directory / f"{receipt.sha256}.jsonl"
    path.write_text(
        json.dumps(
            receipt.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    return path


def test_cut_off_edited_and_foreign_receipts_are_refused(
    captured: Path, tmp_path: Path
) -> None:
    game = TrainingGames(captured).sample(seed=0)
    payload = json.loads(game.path.read_text())
    payload["game"]["terminal"] = False
    cut = _write_receipt(tmp_path / "cut", payload)
    games = TrainingGames(cut.parent)
    assert games.games() == []
    assert games.cut_off == 1
    with pytest.raises(ValueError, match="no complete games.*1 cut off"):
        games.sample()
    edited = tmp_path / game.path.name
    edited.write_text(game.path.read_text().replace("learner", "edited"))
    with pytest.raises(ValueError, match="digest mismatch"):
        TrainingGames(source=edited).games()
    payload = json.loads(game.path.read_text())
    payload["identities"]["world"] = "foreign-world"
    foreign = _write_receipt(tmp_path / "foreign", payload)
    with pytest.raises(ValueError, match="receipt world foreign-world differs"):
        TrainingGames(foreign).sample().export(tmp_path / "bad.gz")
    payload = json.loads(game.path.read_text())
    payload["game"]["cards"]["capture"]["content_digest"] = "0" * 64
    foreign_cards = _write_receipt(tmp_path / "foreign-cards", payload)
    with pytest.raises(ValueError, match="retained cards 0000"):
        TrainingGames(foreign_cards).sample().export(tmp_path / "bad-cards.gz")
    payload = json.loads(game.path.read_text())
    payload["game"]["trace"]["actions"][0] = 100000
    with pytest.raises(ValueError, match="choice|selected"):
        _write_receipt(tmp_path / "action", payload)
    receipt = TrainingExposureReceipt.model_validate_json(game.path.read_bytes())
    inputs = GameInputs(
        seed=receipt.game.seed, players=receipt.game.trace.player_configs
    )
    with pytest.raises(ValueError, match="action not offered"):
        record_actions(inputs, [100000], players=("a", "b"), source={})
    with pytest.raises(ValueError, match="option count differs"):
        record_actions(
            inputs,
            receipt.game.trace.actions,
            players=("a", "b"),
            source={},
            option_counts=[0],
        )
    # Source bytes stay digest-bound after export as well.
    exported = read_trace(game.export(tmp_path / "valid.gz"))[0]
    changed = copy.deepcopy(exported)
    changed["world"] = {"world_version": "foreign"}
    changed["game_trace_sha256"] = canonical_sha256(
        {k: v for k, v in changed.items() if k != "game_trace_sha256"}
    )
    with pytest.raises(
        ValueError, match="recorded world.*foreign.*differs from running"
    ):
        replay_games([changed])
    exported["source"]["kind"] = "evaluation"
    assert not replay_games([exported]).passed


def test_export_game_cli_lists_selects_and_names_seats(
    captured: Path, tmp_path: Path
) -> None:
    from manabot.cli import app

    runner = CliRunner()
    output = tmp_path / "selected.game.jsonl.gz"
    result = runner.invoke(app, ["export-game", str(captured), "--out", str(output)])
    assert result.exit_code == 2, result.output
    assert "complete games" in result.output
    assert "receipt" in result.output
    result = runner.invoke(
        app,
        [
            "export-game",
            str(captured),
            "--out",
            str(output),
            "--game",
            "0",
            "--players",
            "alpha",
            "beta",
        ],
    )
    assert result.exit_code == 0, result.output
    assert read_trace(output)[0]["seat_players"] == ["alpha", "beta"]


def test_export_game_cli_refuses_a_directory_without_complete_games(
    tmp_path: Path,
) -> None:
    from manabot.cli import app

    empty = tmp_path / "games"
    empty.mkdir()
    result = CliRunner().invoke(
        app, ["export-game", str(empty), "--out", str(tmp_path / "out.gz")]
    )
    assert result.exit_code == 2, result.output
    assert "no complete game" in result.output
    assert "--game" not in result.output
