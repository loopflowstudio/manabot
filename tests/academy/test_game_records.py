"""Managym records open as closed Academy recordings without a bot provider."""

import json
from pathlib import Path

import pytest

import managym
from managym.academy import server, trace
from managym.academy.launch import read_games
from managym.academy.watch import import_arena, inspect_watch
from managym.recording import ScalarCapture, runtime_identities


def test_interrupted_record_import_is_closed_and_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path / "traces")
    players = [
        managym.PlayerConfig(name=str(i), decklist={"Mountain": 12}) for i in (0, 1)
    ]
    path = tmp_path / "game.jsonl"
    env = managym.Env(seed=7)
    capture = ScalarCapture(seed=7, games_path=path)
    capture.reset(env, players)
    capture.step(env, 0, annotation={"value": 0.31})
    capture.close()
    selected = read_games(path)[0]
    imported = import_arena(selected._record)
    assert imported["stop"] == "interrupted"
    assert imported["id"] not in server.SESSION_REGISTRY
    shown = inspect_watch(imported["id"], 0, imported["token"])
    assert shown["count"] == 1
    assert shown["outputs"]["value"] == 0.31
    assert shown["winner"] is None
    assert not server.SESSION_REGISTRY


def test_retained_exposure_receipt_opens_without_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path / "traces")
    vector = managym.VectorEnv(num_envs=1, seed=7)
    vector.enable_exposure_capture()
    vector.reset_all(
        [managym.PlayerConfig(name=str(i), decklist={"Mountain": 12}) for i in (0, 1)]
    )
    for _ in range(1000):
        result = vector.step([0])
        if result[0][2]:
            break
    else:
        raise AssertionError("bounded fixture did not complete")
    game = json.loads(vector.take_exposure()[0][0])
    receipt = {
        "schema": "training-exposure-v1",
        "phase": "training",
        "stream": 0,
        "game": game,
        "skip_trivial": True,
        "identities": runtime_identities(),
        "context": {},
        "seconds": 0,
    }
    path = tmp_path / "receipt.jsonl"
    path.write_text(json.dumps(receipt) + "\n")
    imported = import_arena(read_games(path)[0]._record)
    assert imported["stop"] == "complete"
    assert imported["id"] not in server.SESSION_REGISTRY


def test_diverged_transition_never_persists_its_consequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path / "traces")
    path = tmp_path / "game.jsonl"
    env = managym.Env(seed=7)
    capture = ScalarCapture(seed=7, games_path=path)
    capture.reset(
        env,
        [managym.PlayerConfig(name=str(i), decklist={"Mountain": 12}) for i in (0, 1)],
    )
    capture.step(env, 0)
    capture.step(env, 0)
    capture.step(env, 0)
    capture.close()
    from managym.academy.watch import import_record
    from managym.recording import read_record

    record = read_record(path)
    record.games[0].digests[2] = "edited-next-state"
    imported = import_record(record)
    assert imported["stop"] == "diverged"
    shown = inspect_watch(imported["id"], 0, imported["token"])
    assert shown["count"] == 1
    assert shown["winner"] is None
    assert imported["id"] not in server.SESSION_REGISTRY
