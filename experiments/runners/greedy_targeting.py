"""Audit retained Study targeting metadata without launching games.

uv run python -m experiments.runners.greedy_targeting --games games.jsonl.gz

Counts include unfinished records. Unknown sources, unresolved targets, declines
and old unmeasured prompts remain explicit. This is not an error-rate estimator
or a strength evaluation; a source-specific card review must interpret targets.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from manabot.study.record import read_games
from manabot.study.targeting import TargetCount, target_counts

# Cell name -> (labelled player's deck, opponent's deck).
CELLS: dict[str, tuple[str, str]] = {
    "LvA": ("ur_lessons", "gw_allies"),
    "AvL": ("gw_allies", "ur_lessons"),
    "LvL": ("ur_lessons", "ur_lessons"),
    "AvA": ("gw_allies", "gw_allies"),
}


@dataclass(frozen=True)
class TargetingCell:
    cell: str
    player_seats: int
    unfinished_player_seats: int
    counts: list[TargetCount]


@dataclass(frozen=True)
class TargetingAudit:
    records_sha256: str
    label: str
    games: int
    unfinished_games: int
    cells: list[TargetingCell]
    schema: str = "manabot.targeting-audit.v1"


def audit(path: Path, label: str) -> TargetingAudit:
    games = [game for game in read_games(path) if label in game.labels]
    if not games:
        raise ValueError(f"no records for player label {label!r}")
    # One entry per labelled seat; mirrors can include a game twice.
    seats = [
        ((game.decks[seat], game.decks[1 - seat]), game)
        for game in games
        for seat in game.seats(label)
    ]
    if any(decks not in CELLS.values() for decks, _ in seats):
        raise ValueError("target audit requires Allies/Lessons deck keys")
    counts = target_counts(games, label)
    cells = [
        TargetingCell(
            cell=name,
            player_seats=sum(d == decks for d, _ in seats),
            unfinished_player_seats=sum(
                d == decks and not game.completed for d, game in seats
            ),
            counts=[row for row in counts if (row.deck, row.opponent_deck) == decks],
        )
        for name, decks in CELLS.items()
    ]
    return TargetingAudit(
        records_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        label=label,
        games=len(games),
        unfinished_games=sum(not game.completed for game in games),
        cells=cells,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--label", default="greedy")
    args = parser.parse_args()
    print(json.dumps(asdict(audit(args.games, args.label)), indent=2))


if __name__ == "__main__":
    main()
