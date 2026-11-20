"""Custom questions cover vector streams without coupling analysis to gameplay."""

from pathlib import Path
from typing import Any

import pytest

from manabot.env.observation import ObservationSpace, allocate_vector_buffers
import managym
from managym.card_capture import CapturedCards
from managym.recording import CapturedVector, PlayerSetup, ScalarCapture, read_records
from managym.replay import RecordedDecision, ReplayStop, walk
from managym.statistics import (
    AnalysisFailure,
    Behaviour,
    CardPerformance,
    Reducer,
    reduce,
)


class History(Reducer[tuple[int, int, str, int]]):
    def start(self, setup: list[PlayerSetup]) -> None:
        self.count = 0
        self.annotated = 0

    def decision(self, decision: RecordedDecision) -> None:
        assert decision.index == self.count
        self.count += 1
        self.annotated += int(decision.annotation is not None)

    def end(
        self, result: ReplayStop, capture: CapturedCards
    ) -> list[tuple[int, int, str, int]]:
        assert result.index == self.count
        return [(self.count, self.annotated, result.status, capture.turn)]


def players() -> list[managym.PlayerConfig]:
    return [managym.PlayerConfig(str(seat), {"Mountain": 12}) for seat in (0, 1)]


def buffers(vector: CapturedVector, count: int) -> None:
    space = ObservationSpace()
    arrays = allocate_vector_buffers(space, count)
    vector.set_buffers(arrays, layout=space.layout)


def test_vector_games_and_masked_history_equal_offline_analysis(tmp_path: Path) -> None:
    path = tmp_path / "games.jsonl"
    vector = CapturedVector(
        num_envs=2,
        seed=7,
        games_path=path,
        reducers=[CardPerformance(), Behaviour(), History()],
    )
    buffers(vector, 2)
    vector.reset_all_into_buffers(players())
    completed = [0, 0]
    for _ in range(1000):
        before = vector.current_game_generations()
        vector.step_into_buffers(
            [0, 0],
            [n < 2 for n in completed],
            annotations=[{"extra": {"update": 42}}, None],
        )
        after = vector.current_game_generations()
        completed = [
            n + int(a != b) for n, a, b in zip(completed, before, after, strict=True)
        ]
        if min(completed) == 2:
            break
    assert completed == [2, 2]
    vector.close()
    assert vector.analysis_error is None
    expected = reduce(read_records(path), [CardPerformance(), Behaviour(), History()])
    actual = vector.rows()
    assert actual[:2] == expected[:2]
    # Streams can complete in another order; table content must still match.
    assert sorted(actual[2]) == sorted(expected[2])
    assert sum(row[2] == "complete" for row in actual[2]) == 4
    assert sum(row[1] for row in actual[2]) > 0


def test_unrecorded_vector_questions_follow_resets_and_active_mask() -> None:
    vector = CapturedVector(num_envs=2, seed=7, record_fraction=0, reducers=[History()])
    buffers(vector, 2)
    vector.reset_all_into_buffers(players())
    vector.step_into_buffers([0, 0], [True, False])
    vector.reset_seeds_into_buffers([7, 8])
    vector.step_into_buffers([0, 0], [False, True], annotations=[None, {"value": 0.5}])
    vector.close()
    assert sorted(vector.rows()[0]) == sorted(
        [
            (1, 0, "interrupted", 2),
            (0, 0, "interrupted", 1),
            (0, 0, "interrupted", 1),
            (1, 1, "interrupted", 2),
        ]
    )


def test_unused_vector_hooks_do_not_read_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_type = managym.VectorEnv

    class NoAnalysisReads:
        def __init__(self, **kwargs: Any) -> None:
            self.native = native_type(**kwargs)

        def __getattr__(self, name: str) -> Any:
            assert name not in {
                "decision_observations",
                "current_exposure",
                "take_exposure",
                "enable_exposure_capture",
            }
            return getattr(self.native, name)

    monkeypatch.setattr(managym, "VectorEnv", NoAnalysisReads)
    vector = CapturedVector(num_envs=1)
    vector.reset_all(players())
    vector.step([0])
    vector.close()
    assert vector.rows() == []


@pytest.mark.parametrize("phase", ["start", "decision", "end", "finish"])
def test_live_analysis_errors_preserve_scalar_and_vector_records(
    tmp_path: Path, phase: str
) -> None:
    class Broken(History):
        def start(self, setup: list[PlayerSetup]) -> None:
            if phase == "start":
                raise ValueError("broken analysis")
            super().start(setup)

        def decision(self, decision: RecordedDecision) -> None:
            if phase == "decision":
                raise ValueError("broken analysis")
            super().decision(decision)

        def end(
            self, result: ReplayStop, capture: CapturedCards
        ) -> list[tuple[int, int, str, int]]:
            if phase == "end":
                raise ValueError("broken analysis")
            return super().end(result, capture)

        def finish(
            self, rows: list[tuple[int, int, str, int]]
        ) -> list[tuple[int, int, str, int]]:
            raise ValueError("broken analysis")

    path = tmp_path / "vector.jsonl"
    vector = CapturedVector(num_envs=1, seed=7, games_path=path, reducers=[Broken()])
    vector.reset_all(players())
    vector.step([0])
    vector.close()
    scalar_path = tmp_path / "scalar.jsonl"
    env = managym.Env(seed=7)
    scalar = ScalarCapture(seed=7, games_path=scalar_path, reducers=[Broken()])
    scalar.reset(env, players())
    scalar.step(env, 0)
    scalar.close()
    for capture, source in [(vector, path), (scalar, scalar_path)]:
        with pytest.raises(AnalysisFailure, match="broken analysis"):
            capture.rows()
        assert capture.analysis_error and "broken analysis" in capture.analysis_error
        record = next(read_records(source))
        replay = walk(record)
        assert len(list(replay)) == 1
        assert record.error is None
        assert replay.stop.status == "interrupted"


def test_isolated_failure_never_reaches_vector_hook() -> None:
    vector = CapturedVector(num_envs=2, seed=7, reducers=[History()])
    buffers(vector, 2)
    vector.reset_all_into_buffers(players())
    vector.step_isolated_into_buffers([9999, 0], [True, True])
    assert vector.get_last_info()[0]["game_error"]
    vector.close()
    assert sorted(row[0] for row in vector.rows()[0]) == [0, 1]


def test_root_reassignment_ends_old_reducer_before_starting_new_game() -> None:
    vector = CapturedVector(num_envs=2, seed=7, reducers=[History()])
    buffers(vector, 2)
    vector.reset_all_into_buffers(players())
    vector.reassign_roots_into_buffers([players(), None])
    vector.step_into_buffers([0, 0])
    vector.close()
    assert sorted(row[0] for row in vector.rows()[0]) == [0, 1, 1]


def test_strict_batch_failure_marks_analysis_coverage_unknown() -> None:
    vector = CapturedVector(num_envs=2, seed=7, reducers=[History()])
    vector.reset_all(players())
    with pytest.raises(managym.AgentError):
        vector.step([9999, 0])
    vector.close()
    with pytest.raises(AnalysisFailure, match="coverage is unknown"):
        vector.rows()
