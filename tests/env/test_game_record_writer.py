"""Recorded games retain acknowledged choices without changing play."""

from pathlib import Path

import pytest

import managym
from managym.recording import ScalarCapture, read_record, read_records
from managym.replay import walk


def players() -> list[managym.PlayerConfig]:
    return [
        managym.PlayerConfig(name=str(seat), decklist={"Mountain": 12})
        for seat in (0, 1)
    ]


def test_complete_and_interrupted_records(tmp_path: Path) -> None:
    path = tmp_path / "games.jsonl"
    capture = ScalarCapture(seed=7, games_path=path)
    env = managym.Env(seed=7)
    capture.reset(env, players())
    for _ in range(1000):
        capture.step(env, 0, annotation={"value": 0.31})
        if env.is_game_over():
            break
    assert env.is_game_over()
    capture.reset(env, players())
    capture.step(env, 0)
    capture.close()
    records = list(read_records(path))
    assert len(records) == 2
    complete = walk(records[0])
    decisions = list(complete)
    assert complete.stop is not None and complete.stop.status == "complete"
    assert decisions[0].annotation is not None
    assert decisions[0].annotation.value == 0.31
    interrupted = walk(records[1])
    assert len(list(interrupted)) == 1
    assert interrupted.stop is not None and interrupted.stop.status == "interrupted"
    assert path.stat().st_mode & 0o777 == 0o600


def test_off_has_no_digest_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = ScalarCapture(
        seed=7, games_path=tmp_path / "off.jsonl", record_fraction=0
    )
    env = managym.Env(seed=7)
    capture.reset(env, players())

    # PyO3 instances do not permit replacing methods: a transparent boundary
    # fixture rejects only the read whose off-path absence is the contract.
    class NoDigest:
        def state_digest(self) -> str:
            raise AssertionError("digest read with recording off")

        def current_agent_index(self) -> int:
            return env.current_agent_index()

        def semantic_decision_frame_json(self) -> str:
            return env.semantic_decision_frame_json()

        def step(self, action: int) -> object:
            return env.step(action)

        def is_game_over(self) -> bool:
            return env.is_game_over()

    capture.step(NoDigest(), 0)  # type: ignore[arg-type] — rejects a forbidden native read
    capture.close()
    assert not (tmp_path / "off.jsonl").exists()


def test_divergence_and_world_admission(tmp_path: Path) -> None:
    path = tmp_path / "game.jsonl"
    env = managym.Env(seed=3)
    capture = ScalarCapture(seed=3, games_path=path)
    capture.reset(env, players())
    capture.step(env, 0)
    capture.close()
    record = read_record(path)
    record.games[0].digests[0] = "wrong"
    replay = walk(record)
    assert list(replay) == []
    assert replay.stop is not None and replay.stop.status == "diverged"
    record.identities["world"] = "another-world"
    with pytest.raises(ValueError, match="recorded managym.*running managym"):
        walk(record)


def test_vector_record_replays_scalar(tmp_path: Path) -> None:
    from managym.recording import CapturedVector

    path = tmp_path / "vector.jsonl"
    vector = CapturedVector(num_envs=1, seed=7, games_path=path)
    vector.reset_all(players())
    vector.step([0])
    vector.close()
    replay = walk(read_record(path))
    assert len(list(replay)) == 1
    assert replay.stop is not None and replay.stop.status == "interrupted"


def test_evidence_warns_and_returns_the_cached_snapshot() -> None:
    import pytest

    capture = ScalarCapture(seed=7)
    with pytest.warns(DeprecationWarning, match="snapshot.*0.4.0"):
        assert capture.evidence().model_dump(
            exclude={"seconds"}
        ) == capture.snapshot().model_dump(exclude={"seconds"})


def test_snapshot_admission_keeps_each_stream_boundary(tmp_path: Path) -> None:
    from managym.academy.launch import read_games
    from managym.recording import CapturedVector, NativeBoundary, records_from_source

    vector = CapturedVector(num_envs=2, seed=7)
    vector.reset_all(players())
    vector.step([0, 0])
    snapshot = vector.snapshot()
    vector.close()
    snapshot.boundary = NativeBoundary(
        operation="step",
        phase="executing",
        sequence=2,
        timestamp=1,
        active=[False, True],
        pending_actions=[0, 0],
        committed_counts=[1, 0],
        proposed_seeds=[7, 8],
        last_known_frames=[None, "pending"],
        frame_prefix_lengths=[1, 0],
    )
    path = tmp_path / "snapshot.jsonl"
    path.write_text(snapshot.model_dump_json(by_alias=True) + "\n")
    assert len(read_record(path).games) == 2
    admitted = list(read_records(path))
    from_row = records_from_source(snapshot.model_dump(mode="json", by_alias=True))
    assert admitted == from_row
    assert [record.games[0].stream for record in admitted] == [0, 1]
    assert [record.boundary.committed_counts for record in admitted] == [[1], [0]]
    assert admitted[1].boundary.last_known_frames == ["pending"]
    assert [len(list(walk(record))) for record in admitted] == [1, 0]
    listed = read_games(path)
    assert [game.digest for game in listed] == [record.sha256 for record in admitted]
    assert len(list(read_records(path))) == 2  # reading never changes the snapshot


@pytest.mark.parametrize("boundary_ready", [False, True])
def test_cli_views_prefix_on_changed_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    boundary_ready: bool,
) -> None:
    import json
    import sys

    from managym.recording import NativeBoundary
    from managym.replay import main

    path = tmp_path / "prefix.jsonl"
    capture = ScalarCapture(seed=7, games_path=path)
    env = managym.Env(seed=7)
    capture.reset(env, players())
    capture.step(env, 0)
    capture.close()
    record = read_record(path)
    record.games[0].end = None
    record.identities["engine_extension_sha256"] = "another-platform"
    record.identities["engine_recording_source_sha256"] = "another-build"
    if boundary_ready:
        record.boundary = NativeBoundary(
            operation="step",
            phase="ready",
            sequence=1,
            timestamp=1,
            committed_counts=[1],
            returned=True,
        )
    path.write_text(record.model_dump_json(by_alias=True) + "\n")
    monkeypatch.setattr(sys, "argv", ["replay", str(path), "--worker", "--statistics"])
    main()
    result = json.loads(capsys.readouterr().out)["results"][0]
    assert result["status"] == "interrupted"
    assert result["index"] == 1
    assert result["build_differs"]
    assert result["statistics"] is not None


def test_cli_failure_requires_runtime_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import sys

    from managym.replay import main

    path = tmp_path / "failure.jsonl"
    capture = ScalarCapture(seed=7, games_path=path)
    env = managym.Env(seed=7)
    capture.reset(env, players())
    capture.close()
    record = read_record(path)
    record.error_class, record.error = "AgentError", "retained error"
    record.identities["engine_extension_sha256"] = "another-platform"
    path.write_text(record.model_dump_json(by_alias=True) + "\n")
    monkeypatch.setattr(sys, "argv", ["replay", str(path), "--worker"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--allow-changed-runtime required" in capsys.readouterr().err
