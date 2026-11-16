"""The same questions produce the same answers live and from saved games."""

from pathlib import Path

import managym
from managym.card_capture import CapturedCards
from managym.recording import PlayerSetup, ScalarCapture, read_records
from managym.replay import ReplayStop
from managym.statistics import Behaviour, CardPerformance, Reducer, reduce


class Choices(Reducer[int]):
    def start(self, setup: list[PlayerSetup]) -> None:
        self.count = 0

    def decision(self, observation: managym.Observation, chosen: int) -> None:
        self.count += 1

    def end(self, result: ReplayStop, capture: CapturedCards) -> list[int]:
        return [self.count]


def test_live_reducers_equal_replayed_tables(tmp_path: Path) -> None:
    path = tmp_path / "game.jsonl"
    env = managym.Env(seed=7)
    capture = ScalarCapture(
        seed=7, games_path=path, reducers=[CardPerformance(), Behaviour(), Choices()]
    )
    players = [
        managym.PlayerConfig(name=str(seat), decklist={"Mountain": 12})
        for seat in (0, 1)
    ]
    capture.reset(env, players)
    decisions = 0
    while not env.is_game_over():
        capture.step(env, 0)
        decisions += 1
        assert decisions < 1000
    capture.close()
    replayed = reduce(read_records(path), [CardPerformance(), Behaviour(), Choices()])
    assert replayed == capture.rows()
    cards, behaviour, custom = replayed
    assert cards and all(row.games == 1 for row in cards)
    assert any(row.land_opportunities for row in behaviour)
    assert custom == [decisions]
