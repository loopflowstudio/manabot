"""
balance.py
Measure how decklist changes move the Allies versus Lessons win rate

    uv run python -m manabot.study.allies_lessons.balance \\
        --ur "+2 Tiger-Seal, -2 Pop Quiz" --deals 100
    uv run python -m manabot.study.allies_lessons.balance --sweep --deals 50

An edit is a signed count and a card name. Every candidate plays the same deal
seeds in both deck orders, so differences between candidates are paired. The
rate reported is always UR Lessons' share of wins in a mirror of one player.
"""

from __future__ import annotations

# Standard library
import argparse
import json
from pathlib import Path
import re
from typing import Iterable

# Local imports
from ..measures import Rate, win_rates
from ..record import GameRecord, record_games
from .matchup import DECKS, PLAYERS, Lists, authored_lists, schedule

LANDS = {"Island", "Mountain", "Plains", "Forest"}
FILLER = {"ur_lessons": "Island", "gw_allies": "Plains"}


def parse_edits(text: str) -> list[tuple[int, str]]:
    """'+2 Tiger-Seal, -2 Pop Quiz' -> [(2, 'Tiger-Seal'), (-2, 'Pop Quiz')]"""
    edits = []
    for part in filter(None, (piece.strip() for piece in text.split(","))):
        found = re.fullmatch(r"([+-]\d+)\s+(.+)", part)
        if not found:
            raise ValueError(
                f"cannot read edit {part!r}; write it like '+2 Tiger-Seal'"
            )
        edits.append((int(found.group(1)), found.group(2)))
    return edits


def edited(lists: Lists, deck: str, edits: Iterable[tuple[int, str]]) -> Lists:
    """A copy of `lists` with the edits applied to one deck."""
    cards = dict(lists[deck])
    for change, card in edits:
        count = cards.get(card, 0) + change
        if count < 0:
            raise ValueError(f"{DECKS[deck]} has fewer than {-change} {card}")
        if count:
            cards[card] = count
        else:
            cards.pop(card, None)
    return {**lists, deck: cards}


def describe(lists: Lists) -> dict[str, str]:
    """Each deck's difference from the authored list, in edit notation."""
    authored = authored_lists()
    out = {}
    for deck in DECKS:
        names = sorted(set(authored[deck]) | set(lists[deck]))
        changes = [
            f"{lists[deck].get(name, 0) - authored[deck].get(name, 0):+d} {name}"
            for name in names
            if lists[deck].get(name, 0) != authored[deck].get(name, 0)
        ]
        out[deck] = ", ".join(changes) or "authored"
    return out


def lessons_win_rate(games: Iterable[GameRecord], player: str) -> Rate:
    rates = win_rates(games, player, by=lambda game, seat: game.decks[seat])
    return rates.get("ur_lessons", Rate(0, 0))


def evaluate(
    lists: Lists,
    *,
    player: str = "search",
    deals: int = 50,
    seed: int = 95_000,
    workers: int = 8,
    tag: str = "",
) -> tuple[Rate, list[GameRecord]]:
    """UR Lessons' win rate over `2 * deals` mirror games of `player`."""
    specs = schedule(
        player,
        player,
        deals=deals,
        seed=seed,
        lists=lists,
        keep_decisions=False,
        tag=tag,
    )
    games = list(record_games(specs, workers=workers))
    return lessons_win_rate(games, player), games


def removals(lists: Lists) -> list[tuple[str, Lists]]:
    """One candidate per nonland card: every copy replaced by a basic land.

    The drop in a deck's win rate when a card is removed estimates what that
    card contributes, which says where to cut or add copies.
    """
    out = [("authored", lists)]
    for deck in DECKS:
        for card, count in sorted(lists[deck].items()):
            if card in LANDS:
                continue
            swap = [(-count, card), (count, FILLER[deck])]
            out.append((f"{DECKS[deck]}: no {card}", edited(lists, deck, swap)))
    return out


def line(name: str, rate: Rate) -> str:
    low, high = rate.interval() or (0.0, 0.0)
    value = rate.value or 0.0
    return (
        f"{value * 100:5.1f}%  {rate.hits:g} of {rate.total}  "
        f"[{low * 100:.0f}% to {high * 100:.0f}%]  {name}"
    )


def results_page(results: list[dict], *, fragment: bool = False) -> str:
    """Saved candidate results as one HTML page, closest to even first."""
    from .. import report as r

    rows = sorted(
        results, key=lambda row: abs(row["lessons_wins"] / row["games"] - 0.5)
    )
    table = r.rate_table(
        [(row["name"], Rate(row["lessons_wins"], row["games"])) for row in rows],
        what="Candidate",
        hit="UR Lessons wins",
        of="Games",
    )
    changes = r.count_table(
        [
            (row["name"], (row["changes"]["ur_lessons"], row["changes"]["gw_allies"]))
            for row in rows
        ],
        ("Candidate", "UR Lessons changes", "GW Allies changes"),
    )
    players = sorted({row["player"] for row in results})
    best = rows[0] if rows else None
    answer = (
        f"Closest to even: {best['name']}, where UR Lessons won "
        f"{best['lessons_wins']:g} of {best['games']} games."
        if best
        else "No candidates were evaluated."
    )
    return r.page(
        "Allies versus Lessons Balance",
        f"UR Lessons' share of wins for each pair of decklists, with {', '.join(players)} "
        "playing both sides.",
        [
            r.section("Which lists play closest to even?", answer, table),
            r.section(
                "What changed in each candidate?",
                "Changes are counted from the authored main decks.",
                changes,
                r.note(
                    "Candidates evaluated with the same seed share their deals, so "
                    "they can be compared with each other more tightly than the "
                    "intervals suggest. The balance holds for the player named "
                    "above; a different player may shift it."
                ),
            ),
        ],
        fragment=fragment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--ur", default="", help="edits to UR Lessons")
    parser.add_argument("--gw", default="", help="edits to GW Allies")
    parser.add_argument("--sweep", action="store_true", help="remove each card in turn")
    parser.add_argument(
        "--candidates",
        type=Path,
        help='JSON list of {"name", "ur", "gw"} edits to evaluate in order',
    )
    parser.add_argument("--player", default="search", choices=sorted(PLAYERS))
    parser.add_argument("--deals", type=int, default=50)
    parser.add_argument("--seed", type=int, default=95_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, help="append one JSON result per candidate")
    parser.add_argument("--html", type=Path, help="write results as an HTML page")
    parser.add_argument(
        "--results", type=Path, nargs="+", help="render saved results; plays nothing"
    )
    args = parser.parse_args()

    if args.results:
        rows = [
            json.loads(line)
            for path in args.results
            for line in path.read_text().splitlines()
        ]
        (args.html or Path("balance.html")).write_text(results_page(rows))
        print(f"report: {args.html or Path('balance.html')}")
        return

    lists = edited(authored_lists(), "ur_lessons", parse_edits(args.ur))
    lists = edited(lists, "gw_allies", parse_edits(args.gw))
    if args.candidates:
        candidates = [
            (
                entry["name"],
                edited(
                    edited(lists, "ur_lessons", parse_edits(entry.get("ur", ""))),
                    "gw_allies",
                    parse_edits(entry.get("gw", "")),
                ),
            )
            for entry in json.loads(args.candidates.read_text())
        ]
    elif args.sweep:
        candidates = removals(lists)
    else:
        candidates = [("candidate", lists)]

    rows = []
    print("UR Lessons win rate, 95% interval, candidate", flush=True)
    for index, (name, candidate) in enumerate(candidates):
        rate, games = evaluate(
            candidate,
            player=args.player,
            deals=args.deals,
            seed=args.seed,
            workers=args.workers,
            tag=f"c{index}:",
        )
        unfinished = sum(not game.completed for game in games)
        print(
            line(name, rate) + (f"  ({unfinished} unfinished)" if unfinished else ""),
            flush=True,
        )
        rows.append(
            {
                "name": name,
                "changes": describe(candidate),
                "lists": candidate,
                "player": args.player,
                "deals": args.deals,
                "seed": args.seed,
                "lessons_wins": rate.hits,
                "games": rate.total,
                "unfinished": unfinished,
            }
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("a") as handle:
                handle.write(json.dumps(rows[-1]) + "\n")
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(results_page(rows))
        print(f"report: {args.html}")


if __name__ == "__main__":
    main()
