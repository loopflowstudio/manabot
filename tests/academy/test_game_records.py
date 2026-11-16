"""Managym records open as closed Academy recordings without a bot provider."""

import json
from pathlib import Path

import pytest

import managym
from managym.academy import server, trace
from managym.academy.launch import read_games
from managym.academy.watch import import_arena, inspect_watch
from managym.recording import ScalarCapture, runtime_identities


@pytest.mark.parametrize("vector", [False, True])
def test_mixed_disclosure_record_replays_and_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vector: bool
) -> None:
    from managym.recording import CapturedVector, read_records
    from managym.replay import walk

    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path / "traces")
    setup = managym.GameSetup(
        [managym.PlayerConfig(name=str(i), decklist={"Mountain": 12}) for i in (0, 1)],
        decklists="mixed",
    )
    path = tmp_path / "games.jsonl"
    if vector:
        capture = CapturedVector(num_envs=1, seed=7, games_path=path)
        capture.reset_all(setup)
        for _ in range(1000):
            if capture.step([0])[0][2]:
                break
        else:
            raise AssertionError("bounded fixture did not complete")
        capture.step([0])
        capture.close()
    else:
        scalar = ScalarCapture(seed=7, games_path=path)
        env = managym.Env(seed=7)
        scalar.reset(env, setup)
        scalar.step(env, 0)
        scalar.close()
    records = list(read_records(path))
    assert len(records) == (2 if vector else 1)
    for record in records:
        game = record.games[0]
        expected = managym.resolve_decklists(setup, game.seed)
        assert {player.disclosure for player in game.players} == {expected}
        replay = walk(record)
        assert len(list(replay)) == len(game.actions)
        assert replay.stop is not None
        assert game.end is not None
        assert replay.stop.status == game.end.status
        imported = import_arena(record.model_dump(mode="json", by_alias=True))
        assert imported["id"] not in server.SESSION_REGISTRY


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


def test_original_command_import_and_stale_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from managym.academy.watch import import_record
    from managym.decision import Command, DecisionFrame
    from managym.recording import read_record

    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path / "traces")
    path = tmp_path / "commands.jsonl"
    env = managym.Env(seed=23)
    capture = ScalarCapture(seed=23, games_path=path)
    deck = {"Mountain": 10, "Gray Ogre": 20, "Lightning Bolt": 4}
    capture.reset(env, [managym.PlayerConfig(str(seat), deck) for seat in (0, 1)])
    for _ in range(15):
        actions = env.observation_for_player(
            env.current_agent_index() or 0
        ).action_space.actions
        chosen = next(
            (
                i
                for i, action in enumerate(actions)
                if action.action_type == managym.ActionEnum.PRIORITY_PLAY_LAND
            ),
            next(
                (
                    i
                    for i, action in enumerate(actions)
                    if action.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
                ),
                0,
            ),
        )
        capture.step(env, chosen)
    frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
    assert frame.object_candidates
    command = Command(
        "original-command",
        frame.revision,
        frame.offers[0]["id"],
        object_preconditions=(frame.object_candidates[0],),
    )
    capture.command(env, command.to_json(), annotation={"value": 0.5})
    capture.close()
    record = read_record(path)
    imported = import_record(record)
    assert imported["stop"] == "interrupted"
    shown = inspect_watch(imported["id"], 15, imported["token"])
    assert shown["count"] == 16
    assert shown["outputs"]["value"] == 0.5
    retained = server.trace_store.load_trace(imported["id"])
    assert (
        retained["canonical_replay"]["decisions"][15]["native_command"]
        == command.to_json()
    )

    edited = json.loads(record.games[0].actions[-1])
    edited["expected_revision"] += 1
    record.games[0].actions[-1] = json.dumps(edited)
    rejected = import_record(record)
    assert rejected["stop"] == "diverged"
    assert (
        len(
            server.trace_store.load_trace(rejected["id"])["canonical_replay"][
                "decisions"
            ]
        )
        == 15
    )
    assert rejected["id"] not in server.SESSION_REGISTRY
