"""The 10×10 Elemental deck grid: one checkpoint against the heuristic.

Jack Heart's evaluation design (MTG-181, 2026-10-23): the model plays every
Elemental deck (x axis) against the heuristic playing every Elemental deck
(y axis), mirrors included, 50 games per cell. Each game's seat, and so whether
the model is on the play (seat 0 takes the first turn), comes from one recorded
random seed; deals are fixed per game. Every checkpoint therefore plays the same
5,000 games, so checkpoints and training seeds are compared on paired deals.

`schedule` is the frozen game list and `SCHEDULE_SHA256` pins it. Etude's
`elemental_play` callback plays one (cell, seat) group per match through the
replay-checked arena. `summarize` reduces retained rows to cell, row, column
and overall scores with 95% Wilson intervals; `grid_table` renders the report.

Scores count a win as 1, a draw as 0.5 and a loss as 0. Only valid games count:
terminated, no failure, and the trace replayed. Invalid games are reported
beside the score, never replaced or replayed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
import random
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from manabot.infra.identity import canonical_sha256

PACK = "elemental-world-v2"
DECKS = (
    "elemental_bg",
    "elemental_gr",
    "elemental_gw",
    "elemental_rb",
    "elemental_ub",
    "elemental_ug",
    "elemental_ur",
    "elemental_uw",
    "elemental_wb",
    "elemental_wr",
)
GRID_SEED = 18_190_001
GAMES_PER_CELL = 50
DEAL_BASE = 1_819_000_000
GAME_SECONDS = 600.0
MAX_COMMANDS = 10_000
WORKERS = 10


@dataclass(frozen=True)
class Game:
    model_deck: str
    heuristic_deck: str
    index: int
    deal: int
    model_seat: Literal[0, 1]


def schedule() -> list[Game]:
    """All 5,000 games in cell order; one RNG stream assigns every seat."""
    rng = random.Random(GRID_SEED)
    games: list[Game] = []
    for x, model_deck in enumerate(DECKS):
        for y, heuristic_deck in enumerate(DECKS):
            cell = x * len(DECKS) + y
            for index in range(GAMES_PER_CELL):
                seat: Literal[0, 1] = 0 if rng.random() < 0.5 else 1
                games.append(
                    Game(
                        model_deck,
                        heuristic_deck,
                        index,
                        DEAL_BASE + cell * 1_000 + index,
                        seat,
                    )
                )
    return games


def schedule_sha256() -> str:
    return canonical_sha256([vars(game) for game in schedule()])


SCHEDULE_SHA256 = "ecc487e4708a8565b9c787068ff676f62b0c35420b179204fbbf72056f436264"


# ---------------------------------------------------------------------------
# Worker: one (model deck, heuristic deck, model seat) group in one process


class Group(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_deck: str
    heuristic_deck: str
    model_seat: Literal[0, 1]
    deals: tuple[int, ...]


class GameRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_deck: str
    heuristic_deck: str
    model_seat: Literal[0, 1]
    deal: int
    score: float | None
    valid: bool
    terminated: bool
    truncated: bool
    failure: str | None
    decisions: int
    model_decisions: int
    seconds: float


def groups(games: Iterable[Game]) -> list[Group]:
    found: dict[tuple[str, str, int], list[int]] = {}
    for game in games:
        key = (game.model_deck, game.heuristic_deck, game.model_seat)
        found.setdefault(key, []).append(game.deal)
    return [
        Group(model_deck=x, heuristic_deck=y, model_seat=seat, deals=tuple(deals))  # type: ignore[arg-type]
        for (x, y, seat), deals in found.items()
    ]


def _legs(group: Group) -> tuple[str, str, int]:
    """Authored pair and arena leg that put the model's deck in its seat.

    Cross legs 0 and 3 give player A (the model) role A's deck in seats 0 and
    1; mirror legs 4 and 5 give both seats role A's deck. A mirror needs some
    distinct role B to form the authored pair; it never plays.
    """
    x, y = group.model_deck, group.heuristic_deck
    if x != y:
        return x, y, (0, 3)[group.model_seat]
    partner = DECKS[(DECKS.index(x) + 1) % len(DECKS)]
    return x, partner, (4, 5)[group.model_seat]


# ---------------------------------------------------------------------------
# One checkpoint: every group, durable per group, bounded workers


# A machine-wide shared allocation's endpoint (`endpoint_environment`): when set,
# every group plays its cell through it, so all games share one model batch.
Cell = tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]


def _group_name(group: Group) -> str:
    return f"{group.model_deck}--{group.heuristic_deck}--seat{group.model_seat}"


def collect(directory: Path, games: Sequence[Game] | None = None) -> list[GameRow]:
    """Retained rows for every scheduled game; failed groups become invalid rows."""
    rows: list[GameRow] = []
    for group in groups(schedule() if games is None else games):
        target = directory / _group_name(group)
        if (target / "games.json").exists():
            found = [
                GameRow.model_validate(r)
                for r in json.loads((target / "games.json").read_text())
            ]
            if sorted(r.deal for r in found) != sorted(group.deals):
                raise ValueError(f"{target} differs from the frozen schedule")
            rows.extend(found)
        elif (target / "failed.json").exists():
            error = json.loads((target / "failed.json").read_text())["error"]
            rows.extend(
                GameRow(
                    model_deck=group.model_deck,
                    heuristic_deck=group.heuristic_deck,
                    model_seat=group.model_seat,
                    deal=deal,
                    score=None,
                    valid=False,
                    terminated=False,
                    truncated=False,
                    failure=f"group failed: {error[-200:]}",
                    decisions=0,
                    model_decisions=0,
                    seconds=0.0,
                )
                for deal in group.deals
            )
    return rows


def _atomic(path: Path, value: object) -> None:
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(value))
    partial.replace(path)


# ---------------------------------------------------------------------------
# Summary


class Score(BaseModel):
    score: float | None
    low: float | None
    high: float | None
    valid: int
    invalid: int


class Summary(BaseModel):
    cells: dict[str, dict[str, Score]]
    model_decks: dict[str, Score]
    heuristic_decks: dict[str, Score]
    model_seats: dict[str, Score]
    overall: Score
    games: int
    decisions: int
    model_decisions: int
    seconds: float


def wilson(rows: Sequence[GameRow]) -> Score:
    """Mean score with a 95% Wilson interval, treating a draw as half a win."""
    valid = [r.score for r in rows if r.valid and r.score is not None]
    n = len(valid)
    invalid = len(rows) - n
    if n == 0:
        return Score(score=None, low=None, high=None, valid=0, invalid=invalid)
    p = sum(valid) / n
    z = 1.959963984540054
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return Score(
        score=p,
        low=max(0.0, centre - half),
        high=min(1.0, centre + half),
        valid=n,
        invalid=invalid,
    )


def summarize(rows: Sequence[GameRow]) -> Summary:
    def where(**match: object) -> list[GameRow]:
        return [r for r in rows if all(getattr(r, k) == v for k, v in match.items())]

    return Summary(
        cells={
            x: {y: wilson(where(model_deck=x, heuristic_deck=y)) for y in DECKS}
            for x in DECKS
        },
        model_decks={x: wilson(where(model_deck=x)) for x in DECKS},
        heuristic_decks={y: wilson(where(heuristic_deck=y)) for y in DECKS},
        model_seats={
            "play": wilson(where(model_seat=0)),
            "draw": wilson(where(model_seat=1)),
        },
        overall=wilson(rows),
        games=len(rows),
        decisions=sum(r.decisions for r in rows),
        model_decisions=sum(r.model_decisions for r in rows),
        seconds=sum(r.seconds for r in rows),
    )


def _short(deck: str) -> str:
    return deck.removeprefix("elemental_").upper()


def _cell(score: Score, *, strong: bool = False) -> str:
    if score.score is None:
        return '<td class="na">—</td>'
    shade = round(255 - 90 * score.score)
    text = f"{score.score:.0%}"
    tag = "strong" if strong else "span"
    invalid = f" · {score.invalid} invalid" if score.invalid else ""
    return (
        f'<td style="background:rgb({shade},{shade + (255 - shade) // 3},255)" '
        f'title="{score.score:.3f} [{score.low:.3f}, {score.high:.3f}], '
        f'{score.valid} valid{invalid}"><{tag}>{text}</{tag}>'
        f"<small>{score.low:.0%}–{score.high:.0%}</small></td>"
    )


def grid_table(summary: Summary, caption: str) -> str:
    """Rows are the model's deck, columns the heuristic's deck; margins aggregate."""
    head = "".join(f"<th>{_short(y)}</th>" for y in DECKS)
    body = "".join(
        f"<tr><th>{_short(x)}</th>"
        + "".join(_cell(summary.cells[x][y]) for y in DECKS)
        + _cell(summary.model_decks[x], strong=True)
        + "</tr>"
        for x in DECKS
    )
    foot = (
        "<tr><th>All</th>"
        + "".join(_cell(summary.heuristic_decks[y], strong=True) for y in DECKS)
        + _cell(summary.overall, strong=True)
        + "</tr>"
    )
    return (
        f'<table class="grid"><caption>{escape(caption)}</caption>'
        f"<tr><th>Model ↓ / Heuristic →</th>{head}<th>All</th></tr>{body}{foot}</table>"
    )
