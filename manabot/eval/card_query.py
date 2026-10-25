"""Card statistics over an experiment's kept database, with shared-deal uncertainty.

`card_data` selects games from every eval run of one experiment (Etude
`statsdb`) and reports 17Lands-style columns per card. Players come from
Etude's `seats` and `agents` tables, so subjects and opponents are joins, never
fields copied into card rows. Queries trust the rows admitted when each match
was played; `audit` re-reads and re-verifies the retained evidence on request.

Counts are copies: a game with two copies of a card in hand counts twice. `GP`
and `GNS` count games. Only eligible games (a winner, a passed replay) carry a
win; the rest are reported by outcome. A deal is the bootstrap unit across every
run, opponent and seat/deck leg, including deals with zero card opportunities.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import duckdb
from etude.eval.query import estimate
from etude.eval.records import Match, TournamentRun, assemble
from etude.run.listing import as_storage, places
from etude.run.storage import Storage

from manabot.eval.card_metrics import (
    COUNT_FIELDS,
    METRIC_VERSION,
    PLAY_FIELDS,
    normalize,
    reduce_cards,
)
from manabot.eval.card_schedule import admit_card_game, admit_card_runtime
from manabot.infra.identity import StrictModel
from manabot.sim.card_evidence import (
    CAPTURE_VERSION,
    EvidenceRef,
    GameEvidence,
    read_evidence,
)
from manabot.sim.replay import replay_games


class CardSelection(StrictModel):
    """Which games to pool. Empty tuples select everything in the experiment.

    A subject or opponent is an agent identity, its name, or a prefix (at least
    eight characters) of its checkpoint digest.
    """

    experiment: str
    subjects: tuple[str, ...] = ()
    opponents: tuple[str, ...] = ()
    runs: tuple[str, ...] = ()
    commits: tuple[str, ...] = ()
    on_play: bool | None = None
    deck: str | None = None
    opponent_deck: str | None = None


class Rate(StrictModel):
    count: int
    rate: float | None


class CardStats(StrictModel):
    card: str
    name: str
    gp: Rate
    oh: Rate
    gd: Rate
    gih: Rate
    gih_interval: tuple[float, float] | None
    gns: Rate
    # Copy-weighted GIH WR minus GNS WR, in percentage points.
    iwd: float | None


class CardData(StrictModel):
    """`games` counts player-games: without a subject, each game counts once per side."""

    selection: CardSelection
    metric_version: str = METRIC_VERSION
    capture_version: int = CAPTURE_VERSION
    subjects: dict[str, str]
    opponents: dict[str, int]
    runs: tuple[str, ...]
    games: int
    deals: int
    win_rate: float | None
    excluded: dict[str, int]
    failed_matches: int
    cards: tuple[CardStats, ...]


def _agents(db: duckdb.DuckDBPyConnection, values: Sequence[str]) -> list[str]:
    known = db.execute("select agent, name, artifact from agents").fetchall()
    found: list[str] = []
    for value in values:
        matches = [
            agent
            for agent, name, artifact in known
            if value in (agent, name)
            or (len(value) >= 8 and (artifact or "").startswith(value))
        ]
        if not matches:
            raise ValueError(f"no agent in this experiment matches {value!r}")
        found.extend(matches)
    return found


def selected_games(
    db: duckdb.DuckDBPyConnection, selection: CardSelection
) -> tuple[str, list[object]]:
    """The selected (run, match, player) games as SQL, with its parameters."""
    where = ["true"]
    parameters: list[object] = []

    def among(column: str, values: Sequence[object]) -> None:
        where.append(f"{column} in ({', '.join('?' for _ in values)})")
        parameters.extend(values)

    if selection.subjects:
        among("s.agent", _agents(db, selection.subjects))
    if selection.opponents:
        among("o.agent", _agents(db, selection.opponents))
    if selection.runs:
        among("p.run", selection.runs)
    if selection.commits:
        where.append(
            "(" + " or ".join("p.source_commit like ?" for _ in selection.commits) + ")"
        )
        parameters.extend(f"{commit}%" for commit in selection.commits)
    for column, value in (
        ("p.on_play", selection.on_play),
        ("p.deck", selection.deck),
        ("p.opponent_deck", selection.opponent_deck),
    ):
        if value is not None:
            where.append(f"{column} = ?")
            parameters.append(value)
    return (
        "select p.*, s.agent as subject, o.agent as opponent from card_plays p"
        " join seats s on s.run = p.run and s.match = p.match and s.player = p.player"
        " join seats o on o.run = p.run and o.match = p.match"
        " and o.player = 1 - p.player"
        f" where {' and '.join(where)}"
    ), parameters


def _has_cards(db: duckdb.DuckDBPyConnection) -> bool:
    tables = {row[0] for row in db.execute("show tables").fetchall()}
    return {"card_plays", "card_counts", "cards"} <= tables


def card_data(db: duckdb.DuckDBPyConnection, selection: CardSelection) -> CardData:
    """17Lands-style card columns for the selected games of one experiment."""
    if not _has_cards(db):
        raise ValueError("this experiment has no card evaluations")
    sql, parameters = selected_games(db, selection)
    db.execute(f"create or replace temp table selected as {sql}", parameters)
    names = dict(
        db.execute(
            "select agent, name from agents where agent in"
            " (select distinct subject from selected)"
        ).fetchall()
    )
    opponents = dict(
        db.execute(
            "select a.name, count(*) from selected s"
            " join agents a on a.agent = s.opponent group by a.name"
        ).fetchall()
    )
    runs = tuple(
        r
        for (r,) in db.execute(
            "select distinct run from selected order by run"
        ).fetchall()
    )
    excluded = dict(
        db.execute(
            "select outcome, count(distinct run || '/' || game) from selected"
            " where not eligible group by outcome"
        ).fetchall()
    )
    games, wins = db.execute(
        "select count(*), sum(win) from selected where eligible"
    ).fetchone() or (0, None)
    deals = [
        d
        for (d,) in db.execute(
            "select distinct deal from selected where eligible"
        ).fetchall()
    ]
    failed = db.execute(
        "select count(*) from matches m where state = 'failed'"
        + (f" and run in ({', '.join('?' for _ in runs)})" if runs else " and false"),
        list(runs),
    ).fetchone()
    # Per card and deal: copies and winning copies, so whole deals resample.
    blocks = db.execute(
        "select c.card, s.deal,"
        " count(*) filter (where c.deck_copies > 0),"
        " coalesce(sum(s.win) filter (where c.deck_copies > 0), 0),"
        " sum(c.opening_copies), sum(c.opening_copies * s.win),"
        " sum(c.drawn_copies), sum(c.drawn_copies * s.win),"
        " sum(c.gih_copies), sum(c.gih_copies * s.win),"
        " count(*) filter (where c.deck_copies > 0 and c.gih_copies = 0),"
        " coalesce(sum(s.win) filter (where c.deck_copies > 0 and c.gih_copies = 0), 0)"
        " from card_counts c join selected s"
        " on c.run = s.run and c.match = s.match and c.player = s.player"
        " and c.game = s.game"
        " where s.eligible group by c.card, s.deal"
    ).fetchall()
    by_card: dict[str, dict[int, tuple[float, ...]]] = defaultdict(dict)
    for card, deal, *sums in blocks:
        by_card[card][deal] = tuple(float(v or 0) for v in sums)
    labels = dict(db.execute("select distinct card, name from cards").fetchall())
    stats = []
    for card, per_deal in by_card.items():
        totals = [sum(values[i] for values in per_deal.values()) for i in range(10)]

        def rate(count: float, won: float) -> Rate:
            return Rate(count=int(count), rate=won / count if count else None)

        gih = estimate(
            [
                (per_deal.get(d, (0.0,) * 10)[7], per_deal.get(d, (0.0,) * 10)[6])
                for d in deals
            ]
        )
        gns = rate(totals[8], totals[9])
        gih_rate = gih[0] if gih else None
        stats.append(
            CardStats(
                card=card,
                name=labels.get(card, card),
                gp=rate(totals[0], totals[1]),
                oh=rate(totals[2], totals[3]),
                gd=rate(totals[4], totals[5]),
                gih=Rate(count=int(totals[6]), rate=gih_rate),
                gih_interval=gih[1] if gih else None,
                gns=gns,
                iwd=None
                if gih_rate is None or gns.rate is None
                else 100 * (gih_rate - gns.rate),
            )
        )
    stats.sort(key=lambda s: (s.gih.rate is None, -(s.gih.rate or 0), s.name))
    return CardData(
        selection=selection,
        subjects=names,
        opponents=opponents,
        runs=runs,
        games=int(games or 0),
        deals=len(deals),
        win_rate=float(wins) / games if games else None,
        excluded=excluded,
        failed_matches=int(failed[0]) if failed else 0,
        cards=tuple(stats),
    )


# ---- Evidence: export and audit ----------------------------------------------


def _folder(store: Storage, run: str, key: str) -> str:
    """The folder of `run` that holds `key`; replicas hold the same bytes."""
    for folder in places(store).get(run, []):
        if folder + key in store.keys(folder):
            return folder
    raise ValueError(f"evidence unavailable for run {run}: {key}")


def load_game(
    db: duckdb.DuckDBPyConnection, source: Storage | Path | str, run: str, game: str
) -> GameEvidence:
    """One game's retained evidence, verified against its admitted digest."""
    store = as_storage(source)
    found = db.execute(
        "select distinct evidence, evidence_sha256, evidence_bytes from card_plays"
        " where run = ? and game = ?",
        [run, game],
    ).fetchall()
    if len(found) != 1:
        raise ValueError("run and game must identify exactly one retained game")
    path, digest, size = found[0]
    ref = EvidenceRef(path=path, sha256=digest, bytes=size, game=game)
    return read_evidence(store.get(_folder(store, run, path) + path), ref)


class AuditResult(StrictModel):
    games: int
    replayed: int
    problems: tuple[str, ...]


def audit(
    db: duckdb.DuckDBPyConnection, source: Storage | Path | str, *, replay: bool = False
) -> AuditResult:
    """Re-read every admitted game's evidence and rebuild its card rows.

    Checks bytes, the game's schedule binding and declared runtime, and that the
    stored rows equal the reducer's. `replay` also re-runs each trace.
    """
    store = as_storage(source)
    plays = db.execute(
        "select run, match, game, evidence, evidence_sha256, evidence_bytes"
        " from card_plays group by all order by run, match, game"
    ).fetchall()
    held: dict[str, tuple[dict[str, Match], TournamentRun]] = {}
    problems: list[str] = []
    replayed = 0
    for run, match_key, game_id, path, digest, size in plays:
        try:
            if run not in held:
                folder = _folder(store, run, "tournament.json")
                tournament = TournamentRun.model_validate_json(
                    store.get(folder + "tournament.json")
                )
                matches = {
                    s.match.key: s.match for s in assemble([tournament]).schedule()
                }
                held[run] = (matches, tournament)
            matches, tournament = held[run]
            ref = EvidenceRef(path=path, sha256=digest, bytes=size, game=game_id)
            game = read_evidence(store.get(_folder(store, run, path) + path), ref)
            participants = admit_card_game(game, matches[match_key])
            admit_card_runtime(game, tournament.tournament.environment)
            expected = normalize(reduce_cards(game, participants))
            stored_plays = _rows(db, "card_plays", PLAY_FIELDS, run, match_key, game_id)
            stored_counts = _rows(
                db, "card_counts", COUNT_FIELDS, run, match_key, game_id
            )
            if stored_plays != _sorted(expected.plays) or stored_counts != _sorted(
                expected.counts
            ):
                raise ValueError("stored card rows differ from retained evidence")
            if replay:
                if not replay_games([game.trace.data]).passed:
                    raise ValueError("retained trace no longer replays")
                replayed += 1
        except (ValueError, KeyError) as problem:
            problems.append(f"{run}/{game_id}: {problem}")
    return AuditResult(games=len(plays), replayed=replayed, problems=tuple(problems))


def _sorted(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda r: tuple(str(v) for v in r.values()))


def _rows(
    db: duckdb.DuckDBPyConnection,
    table: str,
    fields: Sequence[str],
    run: str,
    match: str,
    game: str,
) -> list[dict[str, object]]:
    columns = ", ".join(f'"{f}"' for f in fields)
    found = db.execute(
        f"select {columns} from {table} where run = ? and match = ? and game = ?",
        [run, match, game],
    ).fetchall()
    return _sorted([dict(zip(fields, row, strict=True)) for row in found])
