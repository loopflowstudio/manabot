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
