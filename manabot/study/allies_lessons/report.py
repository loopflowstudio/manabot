"""
report.py
Compose the Allies versus Lessons behavior report from recorded games

Each section asks one question a player would ask, answers it in a sentence
computed from the records, and shows the table behind the answer. The subject's
rates sit beside a random player's so a reader can tell skill from the rules.
"""

from __future__ import annotations

# Standard library
from typing import Iterable

# Local imports
from .. import measures as m, report as r
from ..record import GameRecord
from .matchup import DECKS, LEARN, learn_choices

NAMES = {"search": "Demo Search-64", "random": "Random", "greedy": "Scripted greedy"}


def name(label: str) -> str:
    return NAMES.get(label, label)


def pairing(games: Iterable[GameRecord], first: str, second: str) -> list[GameRecord]:
    return [game for game in games if sorted(game.labels) == sorted((first, second))]


def _deck_tables(games: list[GameRecord], label: str) -> str:
    cells = m.win_rates(games, label)
    by_deck = m.win_rates(games, label, by=lambda game, seat: game.decks[seat])
    by_position = m.win_rates(
        games, label, by=lambda game, seat: m.position(game, seat)
    )
    return r.rate_table(
        [
            (f"{DECKS[deck]}, on the {where}", rate)
            for (deck, where), rate in cells.items()
        ],
        what="Deck and position",
        hit="Win rate",
        of="Wins",
    ) + r.rate_table(
        [(DECKS[deck], rate) for deck, rate in by_deck.items()]
        + [(f"On the {where}", rate) for where, rate in by_position.items()],
        what="Overall",
        hit="Win rate",
        of="Wins",
    )


def _sentence(rate: m.Rate, did: str, chance: str) -> str:
    if not rate.total:
        return f"No {chance} were recorded."
    return (
        f"It {did} in {rate.hits:g} of {rate.total} {chance} ({r.percent(rate.value)})."
    )


def _behavior(
    games: list[GameRecord],
    subject: str,
    baseline: str,
    measure,
    *,
    did: str,
    chance: str,
    what: str,
    missed: str,
) -> tuple[str, str]:
    rate = measure(games, subject)
    rows = [(name(subject), rate)]
    if baseline != subject and any(game.seats(baseline) for game in games):
        rows.append((name(baseline), measure(games, baseline)))
    return (
        _sentence(rate, did, chance),
        r.rate_table(rows, what=what, hit="Rate", of="Count") + r.misses(missed, rate),
    )


def build_report(
    games: Iterable[GameRecord],
    *,
    subject: str = "search",
    baseline: str = "random",
    fragment: bool = False,
) -> str:
    games = list(games)
    mirror = pairing(games, subject, subject)
    random_mirror = pairing(games, baseline, baseline)
    versus = pairing(games, subject, baseline)
    played = [game for game in games if game.seats(subject)]
    sections = []

    mirror_done = [game for game in mirror if game.completed]
    on_play = m.win_rates(mirror, subject, by=lambda g, s: m.position(g, s)).get("play")
    ur = m.win_rates(mirror, subject, by=lambda g, s: g.decks[s]).get("ur_lessons")
    answer = "No mirror games were recorded."
    if on_play and ur:
        answer = (
            f"When {name(subject)} plays itself, UR Lessons wins {r.percent(ur.value)} "
            f"of {ur.total} games and the player on the play wins "
            f"{r.percent(on_play.value)}."
        )
    sections.append(
        r.section(
            "Which deck wins, on the play and on the draw?",
            answer,
            r.group(f"{name(subject)} against itself", _deck_tables(mirror, subject)),
            r.group(
                f"{name(baseline)} against itself",
                _deck_tables(random_mirror, baseline),
            ),
            r.note(
                "Both seats are the same player, so these rates describe the decks and "
                "the starting seat, not the player. Every deal seed is "
                f"played once in each deck order. {len(mirror_done)} of {len(mirror)} "
                f"{name(subject)} mirror games finished."
            ),
        )
    )

    rows = [
        (f"{DECKS[deck]}, on the {where}", rate)
        for (deck, where), rate in m.win_rates(versus, subject).items()
    ]
    total = m.win_rates(versus, subject, by=lambda g, s: "all")
    overall = total.get("all")
    sections.append(
        r.section(
            f"Does it beat {name(baseline)}?",
            f"{name(subject)} won {overall.hits:g} of {overall.total} games against "
            f"{name(baseline)} ({r.percent(overall.value)})."
            if overall
            else "No games between the two were recorded.",
            r.rate_table(rows, what="Its deck and position", hit="Win rate", of="Wins"),
        )
    )

    questions = [
        (
            "Does it play a land every turn it can?",
            m.land_drops,
            dict(
                did="played a land",
                chance="own turns where a land play was offered",
                what="Land played when one was offered",
                missed="Turns where it kept the land",
            ),
            "A chance is one of the player's own turns in which a land play appeared "
            "among its offers at least once.",
        ),
        (
            "Does it attack when nothing can block?",
            m.open_attacks,
            dict(
                did="attacked",
                chance="attack decisions with no untapped opposing creature",
                what="Attacked with no blocker showing",
                missed="Creatures held back with no blocker showing",
            ),
            "Each creature is one decision. A blocker is an untapped creature the "
            "opponent controls, and only a flying or reach creature can block a "
            "flyer. Tricks, other evasion and what the attacker would leave "
            "undefended are not considered.",
        ),
        (
            "Does it attack into blockers?",
            m.contested_attacks,
            dict(
                did="attacked",
                chance="attack decisions with an untapped opposing creature",
                what="Attacked with a blocker showing",
                missed="Creatures held back with a blocker showing",
            ),
            "Holding back here is often correct, so this rate describes style more "
            "than mistakes.",
        ),
        (
            "Does it block?",
            m.blocks,
            dict(
                did="blocked",
                chance="offered blocks",
                what="Block made when offered",
                missed="Blocks it declined",
            ),
            "Each possible blocker against each attacker is one decision.",
        ),
        (
            "Does it cast spells when it can?",
            m.casting,
            dict(
                did="cast at least one spell",
                chance="own turns where a spell was castable",
                what="Cast a spell when one was castable",
                missed="Turns where it cast nothing",
            ),
            "A chance is one of the player's own turns in which a cast appeared among "
            "its offers. Holding a reactive spell counts as a miss.",
        ),
    ]
    extras = {
        m.land_drops: lambda: r.group(
            f"{name(subject)}, by lands already in play",
            r.rate_table(
                m.land_drops_by_lands(played, subject).items(),
                what="When the land was offered",
                hit="Rate",
                of="Count",
            ),
        ),
        m.open_attacks: lambda: r.group(
            f"{name(subject)}, by creature",
            r.rate_table(
                list(m.held_back(played, subject).items())[:10],
                what="Creature (ten most frequent)",
                hit="Rate",
                of="Count",
            ),
        ),
    }
    for question, measure, words, definition in questions:
        answer, table = _behavior(played, subject, baseline, measure, **words)
        extra = extras[measure]() if measure in extras else ""
        sections.append(r.section(question, answer, table, extra, r.note(definition)))

    learn = learn_choices(played, subject)
    other = learn_choices(games, baseline) if baseline != subject else {}
    kinds = sorted(set(learn) | set(other))
    sections.append(
        r.section(
            "What does it do when it resolves Learn?",
            f"{name(subject)} resolved Learn {sum(learn.values())} times."
            if learn
            else f"{name(subject)} never resolved Learn in these games.",
            r.count_table(
                [
                    (LEARN.get(kind, kind), (learn.get(kind, 0), other.get(kind, 0)))
                    for kind in kinds
                ],
                ("Choice", name(subject), name(baseline)),
            ),
        )
    )

    pairs = sorted({tuple(sorted(game.labels)) for game in games})
    rows = []
    for first, second in pairs:
        group = pairing(games, first, second)
        lengths = m.game_lengths(group)
        ends = m.outcomes(group)
        rows.append(
            (
                f"{name(first)} against {name(second)}",
                (
                    len(group),
                    ends.get("terminal", 0),
                    len(group) - ends.get("terminal", 0),
                    lengths.get("median_turns", "n/a"),
                    lengths.get("median_decisions", "n/a"),
                    f"{lengths.get('median_seconds', 0):.1f}",
                ),
            )
        )
    failures = [game for game in games if not game.completed]
    sections.append(
        r.section(
            "What was recorded?",
            f"{len(games)} games; {len(failures)} did not finish.",
            r.count_table(
                rows,
                (
                    "Pairing",
                    "Games",
                    "Finished",
                    "Unfinished",
                    "Median turns",
                    "Median decisions",
                    "Median seconds",
                ),
            ),
            r.note(
                "Intervals are 95% Wilson score intervals. They treat games as "
                "independent; games that share a deal seed are not fully independent, "
                "so the true uncertainty is somewhat wider. Unfinished games are kept "
                "in the counts above and left out of win rates. This report describes "
                "behavior. It makes no strength, rating or promotion claim."
            ),
        )
    )

    return r.page(
        "Allies versus Lessons Play Study",
        f"How {name(subject)} plays the corrected Allies versus Lessons matchup, read "
        f"from {len(games)} recorded games.",
        sections,
        fragment=fragment,
    )
