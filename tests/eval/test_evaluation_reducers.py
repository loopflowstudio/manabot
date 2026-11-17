"""Custom evaluation statistics share verified replay and retain private tables."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any

from pydantic import TypeAdapter
import pytest

from manabot.eval.match import play_cell
from manabot.sim.card_evidence import GameTrace, capture_verified_game
from manabot.sim.replay import read_trace, replay_game
from manabot.sim.supervision import replay_retained
from managym.recording import read_records
from managym.replay import RecordedDecision, RecordWalker
from managym.statistics import Behaviour, CardPerformance, Reducer, Reduction, reduce
from tests.eval.arena_fixture import arena_fixture
from tests.eval.reducer_fixture import DecisionCount, DecisionDistribution


@pytest.fixture(scope="module")
def evaluation(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    fixture = arena_fixture()
    output = tmp_path_factory.mktemp("evaluation-reducers")
    _, trace, replay = play_cell(
        key=fixture.key,
        player_a=fixture.anchors[0],
        player_b=fixture.anchors[1],
        deal_seeds=(3, 4),
        legs=(0, 1),
        out_dir=output,
        card_evidence_dir=output / "card-evidence",
        reducers={
            "cards": CardPerformance(),
            "behaviour": Behaviour(),
            "choices": DecisionDistribution(),
        },
    )
    return output, trace, replay


def test_evaluation_tables_equal_one_offline_reduction(
    evaluation: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    output, trace, replay = evaluation
    assert replay["passed"] and replay["games"] == 4, replay
    tables = json.loads(Path(trace["statistics_path"]).read_text())
    records = list(read_records(output / "games"))
    expected = reduce(records, [CardPerformance(), Behaviour(), DecisionDistribution()])
    expected_json = TypeAdapter(list[list[Any]]).dump_python(expected, mode="json")
    assert tables["tables"] == dict(
        zip(("cards", "behaviour", "choices"), expected_json, strict=True)
    )
    assert tables["tables"]["choices"] == [
        {"decisions": 4},
        {"decisions": replay["decisions"]},
    ]
    assert tables["replay_passed"] and tables["error"] is None
    assert (output / "statistics.json").stat().st_mode & 0o777 == 0o600
    assert all("tables" not in record.context for record in records)
    assert len(trace["card_evidence"]) == 4


def test_card_evidence_and_reducers_use_one_walker(
    evaluation: tuple[Path, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, trace, _ = evaluation
    game = next(iter(read_records(Path(trace["path"]))))
    starts = 0
    initialize = RecordWalker.__init__

    def counted(self: RecordWalker, *args: Any, **kwargs: Any) -> None:
        nonlocal starts
        starts += 1
        initialize(self, *args, **kwargs)

    monkeypatch.setattr(RecordWalker, "__init__", counted)
    reduction = Reduction([CardPerformance(), DecisionDistribution()])
    evidence = capture_verified_game(
        GameTrace(data=game.model_dump(mode="json", by_alias=True)), reduction=reduction
    )
    assert starts == 1 and evidence.replay["passed"]
    assert reduction.rows[1] == [DecisionCount(evidence.replay["decisions"])]
    assert reduction.finish()[0]


def test_divergent_game_contributes_no_statistics(
    evaluation: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    _, trace, _ = evaluation
    record = next(iter(read_records(Path(trace["path"]))))
    record.games[0].digests[0] = "f" * 64
    reduction = Reduction([DecisionDistribution()])
    receipt, _ = replay_game(
        record.model_dump(mode="json", by_alias=True), reduction=reduction
    )
    assert not receipt.passed and reduction.rows == [[]]


def test_script_local_reducers_and_rows_survive_supervision(
    evaluation: tuple[Path, dict[str, Any], dict[str, Any]], tmp_path: Path
) -> None:
    _, trace, _ = evaluation
    script = tmp_path / "custom_reducer.py"
    script.write_text(
        textwrap.dedent("""
        from dataclasses import dataclass
        from pathlib import Path
        import sys
        from managym.statistics import Reducer
        from managym import ActionEnum
        LAND = ActionEnum.PRIORITY_PLAY_LAND
        from managym.recording import PlayerSetup
        from managym.replay import RecordedDecision, ReplayStop
        from managym.card_capture import CapturedCards
        from manabot.sim.replay import read_trace
        from manabot.sim.supervision import replay_retained

        @dataclass
        class LocalRow:
            count: int

        class LocalReducer(Reducer[LocalRow]):
            def start(self, setup: list[PlayerSetup]) -> None: self.count = 0
            def decision(self, decision: RecordedDecision) -> None:
                assert LAND == ActionEnum.PRIORITY_PLAY_LAND
                self.count += 1
            def end(self, result: ReplayStop, capture: CapturedCards) -> list[LocalRow]: return [LocalRow(self.count)]

        outcome = replay_retained(read_trace(Path(sys.argv[1])), 60, reducers=[LocalReducer()])
        assert outcome.error is None, outcome.error
        assert outcome.process["torch_loaded/replay"] == 0
        assert all(isinstance(row, LocalRow) and row.count > 0 for row in outcome.statistics_rows[0])
        print("local reducer and dataclass rows returned")
    """)
    )
    repo = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(script), trace["path"]],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "local reducer and dataclass rows returned" in completed.stdout


def test_reducer_failure_is_reported_without_rewriting_game(
    evaluation: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    class Broken(Reducer[int]):
        def decision(self, decision: RecordedDecision) -> None:
            raise ValueError("my statistic failed")

    _, trace, _ = evaluation
    path = Path(trace["path"])
    original = path.read_bytes()
    outcome = replay_retained(read_trace(path), 60, reducers=[Broken()])
    assert outcome.error is not None and "my statistic failed" in outcome.error
    assert outcome.statistics_rows == []
    assert path.read_bytes() == original
