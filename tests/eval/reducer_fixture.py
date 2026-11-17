"""Importable custom reducer shared with supervised evaluation fixtures."""

from dataclasses import dataclass

from managym.card_capture import CapturedCards
from managym.recording import PlayerSetup
from managym.replay import RecordedDecision, ReplayStop
from managym.statistics import Reducer


@dataclass(frozen=True)
class DecisionCount:
    decisions: int


class DecisionDistribution(Reducer[DecisionCount]):
    def start(self, setup: list[PlayerSetup]) -> None:
        self.decisions = 0

    def decision(self, decision: RecordedDecision) -> None:
        self.decisions += 1

    def end(self, result: ReplayStop, capture: CapturedCards) -> list[DecisionCount]:
        return [DecisionCount(self.decisions)]

    def finish(self, rows: list[DecisionCount]) -> list[DecisionCount]:
        # Non-associative: aggregating in the block workers and again in the
        # parent would lose the number of original games.
        return [
            DecisionCount(len(rows)),
            DecisionCount(sum(row.decisions for row in rows)),
        ]
