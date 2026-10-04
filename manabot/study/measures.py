"""
measures.py
Questions about how a player behaves, answered from recorded games

Every measure is a pure function of `GameRecord`s and a player label, and
returns `Rate`s: how often something happened out of the times it could have.
A new question should be a few lines built on `own_turns` or a filter over
decisions. Keep judgments about what was *possible* tied to what the record
shows the acting player was offered or could see.
"""

from __future__ import annotations

# Standard library
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from statistics import median
from typing import Callable, Iterable, Iterator

# Local imports
from .record import ATTACK, BLOCK, CAST_SPELL, PLAY_LAND, Decision, GameRecord


@dataclass(frozen=True)
class Example:
    game_id: str
    turn: int
    detail: str


@dataclass(frozen=True)
class Rate:
    """`hits` out of `total` opportunities, with examples of the misses."""

    hits: float
    total: int
    misses: tuple[Example, ...] = ()

    @property
    def value(self) -> float | None:
        return self.hits / self.total if self.total else None

    def interval(self, z: float = 1.96) -> tuple[float, float] | None:
        """Wilson score interval; 95% by default."""
        if not self.total:
            return None
        n = self.total
        p = self.hits / n
        centre = p + z * z / (2 * n)
        spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        scale = 1 + z * z / n
        return max(0.0, (centre - spread) / scale), min(1.0, (centre + spread) / scale)


def seats(games: Iterable[GameRecord], label: str) -> Iterator[tuple[GameRecord, int]]:
    """Every (game, seat) the labelled player occupied."""
    for game in games:
        for seat in game.seats(label):
            yield game, seat


def own_turns(game: GameRecord, seat: int) -> dict[int, list[Decision]]:
    """The player's decisions during each of its own turns."""
    turns: dict[int, list[Decision]] = defaultdict(list)
    for decision in game.by(seat):
        if decision.own_turn:
            turns[decision.turn].append(decision)
    return turns


def rate_of(outcomes: Iterable[tuple[bool, Example]], keep: int = 12) -> Rate:
    hits = total = 0
    misses = []
    for hit, example in outcomes:
        total += 1
        hits += hit
        if not hit and len(misses) < keep:
            misses.append(example)
    return Rate(hits, total, tuple(misses))


def position(game: GameRecord, seat: int) -> str:
    return "play" if seat == game.first_player else "draw"


def win_rates(
    games: Iterable[GameRecord],
    label: str,
    by: Callable[[GameRecord, int], object] = lambda game, seat: (
        game.decks[seat],
        position(game, seat),
    ),
) -> dict[object, Rate]:
    """Wins per group over completed games. A draw counts as half a win."""
    wins: Counter = Counter()
    played: Counter = Counter()
    for game, seat in seats(games, label):
        if not game.completed:
            continue
        key = by(game, seat)
        played[key] += 1
        wins[key] += 0.5 if game.winner is None else float(game.winner == seat)
    return {key: Rate(wins[key], played[key]) for key in sorted(played, key=str)}


def land_drops(games: Iterable[GameRecord], label: str) -> Rate:
    """Own turns where a land play was offered: was a land played that turn?"""

    def outcomes():
        for game, seat in seats(games, label):
            for turn, decisions in own_turns(game, seat).items():
                if any(PLAY_LAND in d.offered for d in decisions):
                    played = any(d.chosen == PLAY_LAND for d in decisions)
                    lands = sum(p.is_land for p in decisions[-1].mine)
                    yield (
                        played,
                        Example(
                            game.game_id, turn, f"held a land with {lands} in play"
                        ),
                    )

    return rate_of(outcomes())


def land_drops_by_lands(games: Iterable[GameRecord], label: str) -> dict[str, Rate]:
    """Land drops split by how many lands the player already had in play."""
    buckets = {"0 to 2 lands in play": (0, 2), "3 to 4": (3, 4), "5 or more": (5, 99)}
    hits: Counter = Counter()
    total: Counter = Counter()
    for game, seat in seats(games, label):
        for decisions in own_turns(game, seat).values():
            if any(PLAY_LAND in d.offered for d in decisions):
                first = next(d for d in decisions if PLAY_LAND in d.offered)
                lands = sum(p.is_land for p in first.mine)
                name = next(
                    k for k, (low, high) in buckets.items() if low <= lands <= high
                )
                total[name] += 1
                hits[name] += any(d.chosen == PLAY_LAND for d in decisions)
    return {name: Rate(hits[name], total[name]) for name in buckets if total[name]}


def held_back(games: Iterable[GameRecord], label: str) -> dict[str, Rate]:
    """Open attacks per creature name: how often each one attacked."""
    hits: Counter = Counter()
    total: Counter = Counter()
    for game, seat in seats(games, label):
        for decision in game.by(seat):
            if decision.kind == ATTACK and _blockers(decision) == 0:
                total[decision.subject or "unknown"] += 1
                hits[decision.subject or "unknown"] += bool(decision.declared)
    return {name: Rate(hits[name], count) for name, count in total.most_common()}


def casting(games: Iterable[GameRecord], label: str) -> Rate:
    """Own turns where a spell could be cast: was at least one cast?"""

    def outcomes():
        for game, seat in seats(games, label):
            for turn, decisions in own_turns(game, seat).items():
                options = [d for d in decisions if CAST_SPELL in d.offered]
                if options:
                    cast = any(d.chosen == CAST_SPELL for d in decisions)
                    offered = sorted(
                        {
                            text
                            for d in options
                            for text in d.offer_labels
                            if text.startswith("Cast")
                        }
                    )
                    yield cast, Example(game.game_id, turn, "; ".join(offered[:4]))

    return rate_of(outcomes())


def _blockers(decision: Decision) -> int:
    """Untapped opposing creatures able to block the deciding attacker."""
    attacker = next((p for p in decision.mine if p.name == decision.subject), None)
    flying = attacker is not None and "flying" in attacker.keywords
    return sum(
        p.is_creature
        and not p.tapped
        and (not flying or "flying" in p.keywords or "reach" in p.keywords)
        for p in decision.theirs
    )


def _declarations(
    games: Iterable[GameRecord],
    label: str,
    kind: str,
    keep: Callable[[Decision], bool],
) -> Rate:
    def outcomes():
        for game, seat in seats(games, label):
            for decision in game.by(seat):
                if decision.kind == kind and keep(decision):
                    yield (
                        bool(decision.declared),
                        Example(
                            game.game_id,
                            decision.turn,
                            f"{decision.subject or 'creature'} held back; "
                            f"opponent at {decision.life[1]} life, "
                            f"{_blockers(decision)} untapped blockers",
                        ),
                    )

    return rate_of(outcomes())


def attacks(games: Iterable[GameRecord], label: str) -> Rate:
    """Each creature that could attack: did it?"""
    return _declarations(games, label, ATTACK, lambda decision: True)


def open_attacks(games: Iterable[GameRecord], label: str) -> Rate:
    """Attack decisions while the opponent showed no untapped creature."""
    return _declarations(games, label, ATTACK, lambda d: _blockers(d) == 0)


def contested_attacks(games: Iterable[GameRecord], label: str) -> Rate:
    """Attack decisions while the opponent had an untapped creature."""
    return _declarations(games, label, ATTACK, lambda d: _blockers(d) > 0)


def blocks(games: Iterable[GameRecord], label: str) -> Rate:
    """Each offered block: was it made?"""
    return _declarations(games, label, BLOCK, lambda decision: True)


def choices(games: Iterable[GameRecord], label: str, kind: str) -> Counter:
    """What the player chose at every decision of one kind."""
    return Counter(
        decision.chosen
        for game, seat in seats(games, label)
        for decision in game.by(seat)
        if decision.kind == kind
    )


def game_lengths(games: Iterable[GameRecord]) -> dict[str, float]:
    finished = [game for game in games if game.completed]
    if not finished:
        return {"games": 0}
    return {
        "games": len(finished),
        "median_turns": median(game.turns for game in finished),
        "median_decisions": median(len(game.decisions) for game in finished),
        "median_seconds": median(game.seconds for game in finished),
    }


def outcomes(games: Iterable[GameRecord]) -> Counter:
    """How games ended, including the ones that did not finish."""
    return Counter(game.end for game in games)
