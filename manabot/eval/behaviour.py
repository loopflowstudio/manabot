"""Count observed play choices in the existing verification replay.

Play emits two native-seat rows for each completed game. Pool their integer
counts before dividing so games contribute in proportion to opportunities.
"""

from dataclasses import dataclass, replace

from managym import ActionSpaceEnum as Space
from managym.card_capture import CapturedCards
from managym.enums import ActionEnum as Action
from managym.recording import PlayerSetup
from managym.replay import RecordedDecision, ReplayStop
from managym.statistics import Reducer


@dataclass
class PlayCounts:
    seat: int
    land_turns: int = 0
    lands_played: int = 0
    cast_turns: int = 0
    casts_made: int = 0
    attackers_offered: int = 0
    attackers_declared: int = 0
    blockers_offered: int = 0
    blockers_declared: int = 0
    casts: int = 0
    casts_off_turn: int = 0
    games: int = 0
    turns: int = 0


class Play(Reducer[PlayCounts]):
    """Deduplicate own-turn opportunities across priority windows."""

    def start(self, setup: list[PlayerSetup]) -> None:
        self.seats = (PlayCounts(0), PlayCounts(1))
        self.turn = 0
        self.offered: set[int] = set()
        self.taken: set[int] = set()

    def decision(self, decision: RecordedDecision) -> None:
        obs = decision.observation
        space = obs.action_space
        kind = space.action_space_type
        if kind not in (Space.PRIORITY, Space.DECLARE_ATTACKER, Space.DECLARE_BLOCKER):
            return
        actions = space.actions
        if len(actions) < 2:
            return
        index = decision.chosen
        if index < 0:
            raise ValueError("play counts require a recorded legal choice")
        selected = actions[index]
        row = self.seats[obs.agent.player_index]
        if kind == Space.DECLARE_ATTACKER:
            row.attackers_offered += 1
            row.attackers_declared += int(selected.declared is True)
            return
        if kind == Space.DECLARE_BLOCKER:
            row.blockers_offered += 1
            row.blockers_declared += int(selected.declared is True)
            return

        chosen = int(selected.action_type)
        if chosen == Action.PRIORITY_CAST_SPELL:
            row.casts += 1
            row.casts_off_turn += int(not obs.agent.is_active)
        if not obs.agent.is_active:
            return
        if self.turn != obs.turn.turn_number:
            self.turn = obs.turn.turn_number
            self.offered.clear()
            self.taken.clear()

        offers = {int(action.action_type) for action in actions}
        new = offers - self.offered
        row.land_turns += Action.PRIORITY_PLAY_LAND in new
        row.cast_turns += Action.PRIORITY_CAST_SPELL in new
        self.offered.update(offers)
        if chosen not in self.taken:
            row.lands_played += chosen == Action.PRIORITY_PLAY_LAND
            row.casts_made += chosen == Action.PRIORITY_CAST_SPELL
            self.taken.add(chosen)

    def end(self, result: ReplayStop, capture: CapturedCards) -> list[PlayCounts]:
        if result.status != "complete":
            return []
        return [replace(row, games=1, turns=capture.turn) for row in self.seats]
