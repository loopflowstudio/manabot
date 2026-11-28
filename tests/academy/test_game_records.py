"""Managym records open as closed Academy recordings without a bot provider."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

import managym
from managym.academy import server, trace
from managym.academy.launch import read_games
from managym.academy.watch import import_game, import_record, inspect_watch
from managym.recording import (
    RecordedVectorEnv,
    ScalarCapture,
    read_record,
    runtime_identities,
)
from managym.replay import walk
from tests.fixtures.recorded_commands import record_until_object_candidate


@pytest.mark.parametrize("vector", [False, True])
def test_mixed_disclosure_record_replays_and_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vector: bool
) -> None:
    from managym.recording import read_records

    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path / "traces")
    setup = managym.GameSetup(
        [managym.PlayerConfig(name=str(i), decklist={"Mountain": 12}) for i in (0, 1)],
        decklists="mixed",
    )
    path = tmp_path / "games.jsonl"
    if vector:
        capture = RecordedVectorEnv(num_envs=1, seed=7, games_path=path)
        capture.reset_all(setup)
        for _ in range(1000):
            if capture.step([0])[0][2]:
                break
        else:
            raise AssertionError("bounded fixture did not complete")
        capture.reset_all(setup)
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
        imported = import_game(record.model_dump(mode="json", by_alias=True))
        assert imported["id"] not in server.LIVE_GAMES


def test_interrupted_record_import_is_closed_and_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path / "traces")
    players = [
        managym.PlayerConfig(name=str(i), decklist={"Mountain": 12}) for i in (0, 1)
    ]
    path = tmp_path / "game.jsonl"
    env = managym.Env(seed=7)
    capture = ScalarCapture(
        seed=7,
        games_path=path,
        context={
            "source_kind": "training",
            "players": ["private-label-0", "private-label-1"],
        },
    )
    capture.reset(env, players)
    capture.step(env, 0, annotation={"value": 0.31})
    capture.close()
    selected = read_games(path)[0]
    imported = import_game(selected._record)
    assert imported["stop"] == "interrupted"
    from managym.academy.attempts import AttemptStore

    retained = AttemptStore(trace.TRACES_DIR / "play.sqlite").load(imported["id"])
    assert retained["recording_import"]["stop"]["status"] == "interrupted"
    assert "stop" not in retained["config"]["opponent"]
    assert imported["id"] not in server.LIVE_GAMES
    shown = inspect_watch(imported["id"], 0)
    assert shown["count"] == 1
    assert shown["players"] == ["0", "1"]
    assert shown["source_kind"] == "training"
    assert shown["outputs"]["value"] == 0.31
    assert shown["winner"] is None
    assert imported["id"] not in server.LIVE_GAMES
    # The launcher opens the recording in a browser that holds no credential.
    anonymous = TestClient(server.app).get(f"/api/traces/{imported['id']}/recording")
    assert anonymous.status_code == 200, anonymous.text
    assert "private-label" not in anonymous.text
    assert "private-label" not in json.dumps(shown)


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
    game = json.loads(vector.current_exposure([True])[0])
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
    imported = import_game(read_games(path)[0]._record)
    assert imported["stop"] == "complete"
    assert imported["id"] not in server.LIVE_GAMES


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
    record = read_record(path)
    record.games[0].digests[2] = "edited-next-state"
    imported = import_record(record)
    assert imported["stop"] == "diverged"
    shown = inspect_watch(imported["id"], 0)
    assert shown["count"] == 1
    assert shown["winner"] is None
    assert imported["id"] not in server.LIVE_GAMES


@pytest.mark.parametrize(
    ("policies", "first_external"),
    [
        (("agent", "heuristic:64"), 0),
        (("heuristic:64", "agent"), 1),
        (("random", "agent"), 1),
    ],
)
def test_native_vector_game_replay_keeps_both_acting_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policies: tuple[str, str],
    first_external: int,
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path / "traces")
    path = tmp_path / "native.jsonl"
    setup = managym.GameSetup(
        [
            managym.PlayerConfig("Red", {"Mountain": 8, "Gray Ogre": 4}),
            managym.PlayerConfig("White", {"Plains": 8, "Savannah Lions": 4}),
        ],
        decklists="closed",
    )
    capture = RecordedVectorEnv(
        num_envs=1,
        seed=7,
        games_path=path,
        player_policies=policies,
        context={"source_kind": "evaluation"},
    )
    capture.reset_all(setup)
    for _ in range(100):
        _, _, done, _, _ = capture.step([0], annotations=[{"value": 0.25}])[0]
        if done:
            break
    else:
        pytest.fail("native-seat fixture did not end")
    capture.close()
    record = read_record(path)
    replay = walk(record, capture_cards=True)
    decisions = list(replay)
    exposure = replay.cards().capture.exposure
    assert exposure is not None
    client = TestClient(server.app)
    response = client.post(
        "/api/watch/import",
        content=path.read_bytes(),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    imported = response.json()
    trace_id = imported["id"]
    assert imported["stop"] == "complete"
    assert imported["count"] == len(exposure.choices) > len(decisions) > 2
    assert trace_id not in server.LIVE_GAMES

    recorded, _ = server._recording_read(trace_id)
    assert recorded.replay is not None
    for row, choice in zip(recorded.replay.decisions, exposure.choices, strict=True):
        assert (row.viewer, row.offer_id) == (choice.actor, choice.selected)
    end = record.games[0].end
    assert end is not None and recorded.payload["winner"] == end.winner

    first = inspect_watch(trace_id, 0)
    assert first["decision"]["viewer"] == 0
    actor_ordinal = next(
        row.ordinal for row in recorded.replay.decisions if row.viewer == first_external
    )
    assert inspect_watch(trace_id, actor_ordinal)["outputs"]["value"] == 0.25
    native_rows = [
        row for row in recorded.replay.decisions if row.viewer != first_external
    ]
    native = inspect_watch(trace_id, native_rows[-1].ordinal)
    assert native["source_kind"] == "evaluation"
    assert native["outputs"]["unavailable"] == "not retained"
    assert native["observation"]["agent"]["player_index"] == 1 - first_external
    assert native["observation"]["opponent"]["hand"] == []
    assert native["presentation"]
    assert all(
        event["seq"] < native["decision"]["presentation_cursor"]
        for event in native["presentation"]
    )
    spectator = inspect_watch(trace_id, native_rows[-1].ordinal, reveal=True)
    assert spectator["decision"] == native["decision"]
    assert spectator["presentation"] == native["presentation"]
    assert (
        client.get(
            f"/api/traces/{trace_id}/decisions/{native['decision']['address']}"
        ).json()["viewer"]
        == 1 - first_external
    )

    # Corrupt the second transition's target boundary. Its automatic reply
    # must disappear with the unverified external action.
    record.games[0].digests[2] = "edited-next-state"
    rejected = import_record(record)
    assert rejected["stop"] == "diverged"
    external_ordinals = [
        i for i, choice in enumerate(exposure.choices) if choice.actor == first_external
    ]
    assert rejected["count"] == external_ordinals[1]


def test_native_random_replay_preserves_rng_and_forced_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path / "traces")
    env = managym.Env(seed=42, player_policies=("random", "heuristic"))
    env.reset(
        [managym.PlayerConfig(str(i), {"Mountain": 8, "Gray Ogre": 4}) for i in (0, 1)]
    )
    env.step(None)
    branch = env.snapshot().restore(managym.RestoreOptions.exact(forced_first_action=0))
    imported = import_game(json.loads(branch.snapshot().to_replay_json()))
    assert imported["stop"] == "interrupted"
    assert imported["count"] == 2
    assert [inspect_watch(imported["id"], i)["decision"]["viewer"] for i in (0, 1)] == [
        0,
        1,
    ]


def test_original_command_import_and_stale_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from managym.decision import Command

    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path / "traces")
    path = tmp_path / "commands.jsonl"
    env = managym.Env(seed=23)
    capture = ScalarCapture(seed=23, games_path=path)
    frame, ordinal = record_until_object_candidate(env, capture)
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
    shown = inspect_watch(imported["id"], ordinal)
    assert shown["count"] == ordinal + 1
    assert shown["outputs"]["value"] == 0.5
    retained = server.trace_store.load_trace(imported["id"])
    assert (
        retained["canonical_replay"]["decisions"][ordinal]["native_command"]
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
        == ordinal
    )
    assert rejected["id"] not in server.LIVE_GAMES


def test_journal_listing_projects_the_complete_file(tmp_path: Path) -> None:
    from managym.recording import records_from_source

    path = tmp_path / "journal.jsonl"
    capture = ScalarCapture(seed=7, journal_path=path)
    env = managym.Env(seed=7)
    capture.reset(
        env,
        [managym.PlayerConfig(name=str(i), decklist={"Mountain": 12}) for i in (0, 1)],
    )
    capture.step(env, 0)
    capture.close()
    rows = read_games(path)
    assert len(rows) == 1
    row = rows[0]
    assert (row.seed, row.stream, row.players, row.decisions, row.ended) == (
        7,
        0,
        ("0", "1"),
        1,
        "interrupted",
    )
    assert row.digest == read_record(path).sha256
    with pytest.warns(DeprecationWarning, match="removal in 0.3.0"):
        assert row.deal_seed == row.seed
    with pytest.warns(DeprecationWarning, match="removal in 0.3.0"):
        assert row.game_trace_sha256 == row.digest
    checkpoint = json.loads(path.read_text().splitlines()[0])
    with pytest.raises(ValueError, match="complete file"):
        records_from_source(checkpoint)


def test_malformed_native_row_never_falls_back_to_bot_import() -> None:
    with pytest.raises(ValueError):
        import_game({"schema": "managym-game-v1", "decisions": []})


def test_imported_state_edit_marks_the_verified_replay_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path / "traces")
    env = managym.Env(seed=7)
    obs, _ = env.reset(
        [managym.PlayerConfig(str(seat), {"Mountain": 12}) for seat in (0, 1)]
    )
    for _ in range(8):
        if any(
            action.action_type == managym.ActionEnum.PRIORITY_PLAY_LAND
            for action in obs.action_space.actions
        ):
            break
        obs, *_ = env.step(0)
    else:
        pytest.fail("seeded game did not reach a land play")
    env.edit_game_state([managym.GameStateEdit.set_life(obs.agent.player_index, 13)])
    env.step(0)

    imported = import_game(json.loads(env.snapshot().to_replay_json()))
    assert imported["stop"] == "interrupted"
    recording, _ = server._recording_read(imported["id"])
    marked = [
        kinds
        for position in recording.positions
        if position.kind == "choice"
        and (kinds := recording.metadata(position)["branched"])
    ]
    assert marked == [["game_state_edits"]]


def test_search_outputs_open_without_private_runner_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path / "traces")
    path = tmp_path / "evaluation.jsonl"
    env = managym.Env(seed=7)
    capture = ScalarCapture(
        seed=7,
        games_path=path,
        context={
            "source_kind": "evaluation",
            "evaluation": {"token": "private-runner-context"},
        },
    )
    observation, _ = capture.reset(
        env, [managym.PlayerConfig(str(i), {"Mountain": 12}) for i in (0, 1)]
    )
    count = len(observation.action_space.actions)
    capture.step(
        env,
        0,
        annotation={
            "probabilities": [1 / count] * count,
            "search_scores": [0.25] * count,
            "belief": {
                "model": "physical",
                "digest": "public-summary",
                "condition_mass": 1.0,
                "sampled_worlds": 2,
            },
            "extra": {"evaluation": {"local_update_receipt": "private-search-receipt"}},
        },
    )
    capture.close()
    imported = import_game(read_record(path).model_dump(mode="json", by_alias=True))
    shown = inspect_watch(imported["id"], 0)
    assert shown["outputs"]["search_scores"] == [0.25] * count
    assert shown["outputs"]["belief"]["sampled_worlds"] == 2
    assert shown["source_kind"] == "evaluation"
    public = json.dumps(shown)
    assert "private-search-receipt" not in public
    assert "private-runner-context" not in public
