"""Count observed play choices in the existing verification replay.

Play emits two native-seat rows for each completed game. Pool their
counts before dividing so games contribute in proportion to opportunities.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

from managym import ActionSpaceEnum as Space
from managym.card_capture import CapturedCards, TurnMana
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
    attackers_lost_for_nothing: int = 0
    blockers_lost_for_nothing: int = 0
    mana_turns_early: int = 0
    mana_use_early: float = 0.0
    mana_turns_mid: int = 0
    mana_use_mid: float = 0.0
    mana_turns_late: int = 0
    mana_use_late: float = 0.0
    casts: int = 0
    casts_off_turn: int = 0
    games: int = 0
    turns: int = 0


def checkpoint_counts(
    pair: tuple[PlayCounts, PlayCounts], game_seat: int
) -> dict[str, int | float]:
    """Select the native seat and label it as Etude's checkpoint participant."""
    return {**asdict(pair[game_seat]), "seat": 0, "game_seat": game_seat}


def mana_use(turns: Sequence[TurnMana]) -> tuple[int, float]:
    """Count funded own turns and sum their individual paid fractions."""
    fractions = [turn.paid / turn.available for turn in turns if turn.available]
    return len(fractions), sum(fractions, 0.0)


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
        facts = capture.capture.exposure
        if (
            facts is None
            or facts.mana_turns is None
            or facts.attackers_lost_for_nothing is None
            or facts.blockers_lost_for_nothing is None
        ):
            raise ValueError("play counts require combat and turn mana exposure")
        rows = []
        for counts in self.seats:
            seat = counts.seat
            turns = facts.mana_turns[seat]
            row = replace(
                counts,
                games=1,
                turns=capture.turn,
                attackers_lost_for_nothing=facts.attackers_lost_for_nothing[seat],
                blockers_lost_for_nothing=facts.blockers_lost_for_nothing[seat],
            )
            row.mana_turns_early, row.mana_use_early = mana_use(turns[:3])
            row.mana_turns_mid, row.mana_use_mid = mana_use(turns[3:6])
            row.mana_turns_late, row.mana_use_late = mana_use(turns[6:])
            rows.append(row)
        return rows
