"""Typed optional game counters; native capture owns all counting formulas.

Live summaries are partial until terminal. Retained capture reduction validates
copy continuity, but says nothing about experiment eligibility or replay success.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, TypeAdapter

import managym
from managym.card_capture import CapturedCards


class CardCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    seat: int
    definition: int
    name: str
    deck_copies: int
    opening_copies: int
    drawn_copies: int
    opportunities: int | None
    selections: int | None
    resolutions: int | None


class GameStatistics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    seed: int
    generation: int
    stream: int
    terminal: bool
    winner: int | None
    cards: tuple[CardCounts, ...]


def summary(raw: str) -> GameStatistics | None:
    return TypeAdapter(GameStatistics | None).validate_json(raw)


def summaries(raw: str) -> list[GameStatistics | None]:
    return TypeAdapter(list[GameStatistics | None]).validate_json(raw)


def reduce_cards(cards: CapturedCards) -> list[CardCounts]:
    return TypeAdapter(list[CardCounts]).validate_json(
        managym._managym.reduce_card_capture_json(
            cards.capture.model_dump_json(by_alias=True)
        )
    )


# Python reducers run on sampled records or explicitly instrumented scalar play.
# They never add an audited evaluation eligibility claim to the native facts.

if TYPE_CHECKING:
    from managym.recording import GameRecord, PlayerSetup
    from managym.replay import RecordedDecision, ReplayStop


class Reducer[T]:
    """One question over setup, chooser decisions, movements and resolutions.

    end receives the complete private card capture, including movements and
    resolution facts. finish may combine per-game rows into a table.
    """

    def start(self, setup: list["PlayerSetup"]) -> None:
        pass

    def decision(self, decision: "RecordedDecision") -> None:
        pass

    def end(self, result: "ReplayStop", capture: CapturedCards) -> list[T]:
        return []

    def finish(self, rows: list[T]) -> list[T]:
        return rows


@dataclass(frozen=True)
class BehaviourRow:
    seat: int
    land_opportunities: int = 0
    lands_played: int = 0
    attack_opportunities: int = 0
    attacks: int = 0

    @property
    def land_played_when_able(self) -> float | None:
        return (
            self.lands_played / self.land_opportunities
            if self.land_opportunities
            else None
        )

    @property
    def attacked_when_able(self) -> float | None:
        return (
            self.attacks / self.attack_opportunities
            if self.attack_opportunities
            else None
        )


class Behaviour(Reducer[BehaviourRow]):
    def start(self, setup: list["PlayerSetup"]) -> None:
        self._seats = [BehaviourRow(0), BehaviourRow(1)]

    def decision(self, decision: "RecordedDecision") -> None:
        observation, chosen = decision.observation, decision.chosen
        actions = observation.action_space.actions
        if chosen < 0 or len(actions) < 2:
            return
        seat = int(observation.agent.player_index)
        row = self._seats[seat]
        land = managym.ActionEnum.PRIORITY_PLAY_LAND
        attack = managym.ActionEnum.DECLARE_ATTACKER
        self._seats[seat] = replace(
            row,
            land_opportunities=row.land_opportunities
            + any(a.action_type == land for a in actions),
            lands_played=row.lands_played + (actions[chosen].action_type == land),
            attack_opportunities=row.attack_opportunities
            + any(a.action_type == attack for a in actions),
            attacks=row.attacks + (actions[chosen].action_type == attack),
        )

    def end(self, result: "ReplayStop", capture: CapturedCards) -> list[BehaviourRow]:
        return self._seats

    def finish(self, rows: list[BehaviourRow]) -> list[BehaviourRow]:
        return [
            BehaviourRow(
                seat,
                sum(r.land_opportunities for r in rows if r.seat == seat),
                sum(r.lands_played for r in rows if r.seat == seat),
                sum(r.attack_opportunities for r in rows if r.seat == seat),
                sum(r.attacks for r in rows if r.seat == seat),
            )
            for seat in (0, 1)
        ]


@dataclass(frozen=True)
class CardRow:
    seat: int
    card: str
    name: str
    games: int = 0
    wins: int = 0
    opening_hand: int = 0
    opening_won: int = 0
    drawn: int = 0
    drawn_won: int = 0
    gih: int = 0
    gih_won: int = 0
    not_seen: int = 0
    not_seen_won: int = 0

    @property
    def gih_win_rate(self) -> float | None:
        return self.gih_won / self.gih if self.gih else None

    @property
    def not_seen_win_rate(self) -> float | None:
        return self.not_seen_won / self.not_seen if self.not_seen else None


class CardPerformance(Reducer[CardRow]):
    """gih-copies-v1: OH/GD/GIH count copies; GP/GNS count games."""

    def __init__(self) -> None:
        self.excluded_games = 0

    def end(self, result: "ReplayStop", capture: CapturedCards) -> list[CardRow]:
        if result.status != "complete" or result.winner is None:
            self.excluded_games += 1
            return []
        rows = []
        for count in reduce_cards(capture):
            if not (count.deck_copies or count.opening_copies or count.drawn_copies):
                continue
            win = int(result.winner == count.seat)
            gih = count.opening_copies + count.drawn_copies
            gp = int(count.deck_copies > 0)
            gns = int(gp and gih == 0)
            rows.append(
                CardRow(
                    count.seat,
                    f"{capture.capture.content_digest}:{count.definition}",
                    count.name,
                    games=gp,
                    wins=gp * win,
                    opening_hand=count.opening_copies,
                    opening_won=count.opening_copies * win,
                    drawn=count.drawn_copies,
                    drawn_won=count.drawn_copies * win,
                    gih=gih,
                    gih_won=gih * win,
                    not_seen=gns,
                    not_seen_won=gns * win,
                )
            )
        return rows

    def finish(self, rows: list[CardRow]) -> list[CardRow]:
        merged: dict[tuple[int, str], CardRow] = {}
        for row in rows:
            key = row.seat, row.card
            old = merged.get(key)
            if old is None:
                merged[key] = row
            else:
                merged[key] = CardRow(
                    row.seat,
                    row.card,
                    row.name,
                    old.games + row.games,
                    old.wins + row.wins,
                    old.opening_hand + row.opening_hand,
                    old.opening_won + row.opening_won,
                    old.drawn + row.drawn,
                    old.drawn_won + row.drawn_won,
                    old.gih + row.gih,
                    old.gih_won + row.gih_won,
                    old.not_seen + row.not_seen,
                    old.not_seen_won + row.not_seen_won,
                )
        return sorted(merged.values(), key=lambda row: (row.seat, row.name, row.card))


def reduce(
    records: Iterable["GameRecord"], reducers: Sequence[Reducer[Any]]
) -> list[list[Any]]:
    """Traverse each record once for all questions; rows describe this sample.

    Reducers may return their own row types, so the heterogeneous result is the
    intentionally dynamic boundary of the extension API.
    """
    from managym.replay import walk

    rows: list[list[Any]] = [[] for _ in reducers]
    for record in records:
        for game in record.games:
            replay = walk(
                record.model_copy(update={"games": [game]}), capture_cards=True
            )
            for reducer in reducers:
                reducer.start(game.players)
            for decision in replay:
                for reducer in reducers:
                    reducer.decision(decision)
            assert replay.stop is not None
            captured = CapturedCards.model_validate_json(replay.env.card_capture_json())
            for reducer, output in zip(reducers, rows, strict=True):
                output.extend(reducer.end(replay.stop, captured))
    return [
        reducer.finish(output) for reducer, output in zip(reducers, rows, strict=True)
    ]
