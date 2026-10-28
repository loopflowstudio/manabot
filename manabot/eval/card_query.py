"""Card statistics over an experiment's kept database, with shared-deal uncertainty.

`card_data` reports 17Lands-style columns per card for one selection of the
games in every eval run of one experiment; `card_variants` reports them for
every combination of the selection's filters in one pass. Both read only the
per-deal totals of `CARD_ROLLUPS` (`card_deals`, `game_deals`), which Etude
keeps as each match is stored, so no card row is read again. Open the database
with `card_db` so runs stored without those rollups are summed once. Players
are the rollups' `player`/`opponents` keys, named by Etude's `players` table,
never fields copied into card rows. Queries trust the rows admitted when each
match was played; `audit` re-reads and re-verifies the retained evidence blobs
on request.

Counts are copies: a game with two copies of a card in hand counts twice. `GP`
and `GNS` count games. Only eligible games (a winner, a passed replay) carry a
win; the rest are reported by outcome. A deal is the bootstrap unit across every
run, opponent and seat/deck leg, including deals with zero card opportunities.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Literal

import duckdb
from etude.eval.query import estimate, read_blobs, statsdb
from etude.eval.records import Match, TournamentRun, assemble
from etude.run.errors import EtudeError
from etude.run.listing import as_storage, places
from etude.run.storage import Storage
import numpy as np
import numpy.typing as npt

from manabot.eval.card_metrics import (
    CARD_ROLLUPS,
    COUNT_FIELDS,
    METRIC_VERSION,
    PLAY_FIELDS,
    normalize,
    reduce_cards,
)
from manabot.eval.card_schedule import (
    CardConditions,
    admit_card_game,
    admit_card_runtime,
)
from manabot.infra.identity import StrictModel
from manabot.sim.card_evidence import (
    CAPTURE_VERSION,
    EvidenceRef,
    GameEvidence,
    evidence_key,
    read_evidence,
)
from manabot.sim.replay import replay_games


def card_db(
    experiment: str,
    source: Storage | Path | str | None = None,
    *,
    rebuild: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Etude's `statsdb` for `experiment`, holding the card rollups for every run."""
    return statsdb(experiment, source, rebuild=rebuild, rollups=CARD_ROLLUPS)


class CardSelection(StrictModel):
    """Which games to pool. Empty tuples select everything in the experiment.

    A subject or opponent is a player's key, its name, its checkpoint's key
    (such as `RUN/checkpoints/NAME`), or a prefix (at least eight characters)
    of its checkpoint digest.
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
    """`games` and `excluded` count player-games: without a subject, each game
    counts once per side."""

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


def _players(db: duckdb.DuckDBPyConnection, values: Sequence[str]) -> list[str]:
    known = db.execute(
        "select player, name, checkpoint, artifact from players"
    ).fetchall()
    found: list[str] = []
    for value in values:
        matches = [
            player
            for player, name, checkpoint, artifact in known
            if value in (player, name, checkpoint)
            or (len(value) >= 8 and (artifact or "").startswith(value))
        ]
        if not matches:
            raise ValueError(f"no player in this experiment matches {value!r}")
        found.extend(matches)
    return found


def _has_cards(db: duckdb.DuckDBPyConnection) -> bool:
    tables = {row[0] for row in db.execute("show tables").fetchall()}
    return {rollup.name for rollup in CARD_ROLLUPS} | {"cards"} <= tables


# ---- One pass over every filter combination ----------------------------------

# The filters `card_variants` can split by: player names, then card_plays columns.
type Dimension = Literal["subject", "opponent", "on_play", "deck", "opponent_deck"]
DIMENSIONS: tuple[Dimension, ...] = (
    "subject",
    "opponent",
    "on_play",
    "deck",
    "opponent_deck",
)
# A combination's value per split dimension, None for all of them.
type Variant = tuple[str | bool | None, ...]
# Per card and deal, the totals of these `card_deals` sums, in this order.
_SUMS = (
    "gp",
    "gp_won",
    "oh",
    "oh_won",
    "gd",
    "gd_won",
    "gih",
    "gih_won",
    "gns",
    "gns_won",
)


def _select(
    db: duckdb.DuckDBPyConnection, selection: CardSelection, rollup: str, table: str
) -> None:
    """Materialize the selection's rollup rows with player names in a temp table."""
    where = ["true"]
    parameters: list[object] = []

    def among(column: str, values: Sequence[object]) -> None:
        where.append(f"{column} in ({', '.join('?' for _ in values)})")
        parameters.extend(values)

    if selection.subjects:
        among("r.player", _players(db, selection.subjects))
    if selection.opponents:
        among("r.opponents", _players(db, selection.opponents))
    if selection.runs:
        among("r.run", selection.runs)
    if selection.commits:
        where.append(
            "(" + " or ".join("r.source_commit like ?" for _ in selection.commits) + ")"
        )
        parameters.extend(f"{commit}%" for commit in selection.commits)
    for column, value in (
        ("r.on_play", selection.on_play),
        ("r.deck", selection.deck),
        ("r.opponent_deck", selection.opponent_deck),
    ):
        if value is not None:
            where.append(f"{column} = ?")
            parameters.append(value)
    # A view cannot take parameters, so the selection is materialized.
    db.execute(
        f"create or replace temp table {table} as"
        " select r.*, s.name as subject, o.name as opponent"
        f" from {rollup} r join players s on s.player = r.player"
        " join players o on o.player = r.opponents"
        f" where {' and '.join(where)}",
        parameters,
    )


def _grouping(
    dimensions: Sequence[Dimension], within: Sequence[str]
) -> tuple[str, str]:
    """A `group by` over every subset of `dimensions` with `within`, and its mask column."""
    sets = [
        f"({', '.join([*chosen, *within])})"
        for size in range(len(dimensions) + 1)
        for chosen in combinations(dimensions, size)
    ]
    # Without a split there is one grouping set and `grouping()` needs a column.
    mask = f"grouping({', '.join(dimensions)})" if dimensions else "0"
    return f"group by grouping sets ({', '.join(sets)})", mask


def _variant(bits: int, values: Sequence[str | bool | None]) -> Variant:
    """The combination a grouping mask and its dimensions' values name."""
    width = len(values)
    # grouping() sets bit (width - 1 - i) when dimension i is rolled up.
    return tuple(
        None if bits >> (width - 1 - i) & 1 else values[i] for i in range(width)
    )


def _game_groups(
    db: duckdb.DuckDBPyConnection,
    dimensions: Sequence[Dimension],
    within: Sequence[str],
    values: str,
    where: str = "true",
) -> list[tuple[Variant, tuple[object, ...]]]:
    """Selected game totals per filter combination and `within` columns."""
    group_by, mask = _grouping(dimensions, within)
    picked = ", ".join([mask, *dimensions, *within, values])
    width = len(dimensions)
    return [
        (_variant(int(row[0]), row[1 : width + 1]), tuple(row[width + 1 :]))
        for row in db.execute(
            f"select {picked} from picked_games where {where} {group_by}"
        ).fetchall()
    ]


def _rate(count: float, won: float) -> Rate:
    return Rate(count=int(count), rate=won / count if count else None)


def card_variants(
    db: duckdb.DuckDBPyConnection,
    selection: CardSelection,
    dimensions: Sequence[Dimension] = (),
) -> dict[Variant, CardData]:
    """`CardData` for `selection` narrowed to every combination of `dimensions`.

    Each combination gives each dimension one value present in the selection's
    games, or all of them (None). Combinations without any game are absent;
    those without eligible games report exclusions only. Each statistic equals
    `card_data` of the narrowed selection; the empty combination is
    `card_data(db, selection)`.
    """
    if not _has_cards(db):
        raise ValueError("this experiment has no card evaluations")
    if unknown := set(dimensions) - set(DIMENSIONS):
        raise ValueError(f"unknown card dimensions: {sorted(unknown)}")
    _select(db, selection, "game_deals", "picked_games")
    # A dimension with one value in the selection's games is not split: that
    # value and all of them are the same games, computed once.
    distinct = db.execute(
        "select "
        + ", ".join(["count(*)", *(f"list(distinct {d})" for d in dimensions)])
        + " from picked_games"
    ).fetchone()
    assert distinct is not None
    if not distinct[0]:
        return {}
    _select(db, selection, "card_deals", "picked_cards")
    fixed = {
        d: values[0]
        for d, values in zip(dimensions, distinct[1:], strict=True)
        if len(values) == 1
    }
    dims = tuple(d for d in dimensions if d not in fixed)
    runs = {
        variant: tuple(run_list)
        for variant, (run_list,) in _game_groups(
            db, dims, (), "list(distinct run order by run)"
        )
    }
    opponents: dict[Variant, dict[str, int]] = defaultdict(dict)
    excluded: dict[Variant, dict[str, int]] = defaultdict(dict)
    subjects: dict[Variant, set[str]] = defaultdict(set)
    for variant, (player, opponent, games) in _game_groups(
        db, dims, ("player", "opponent"), "sum(rows)"
    ):
        subjects[variant].add(player)
        opponents[variant][opponent] = opponents[variant].get(opponent, 0) + int(games)
    for variant, (outcome, games) in _game_groups(
        db, dims, ("outcome",), "sum(rows)", "not eligible"
    ):
        excluded[variant][outcome] = int(games)
    # Eligible deals (the bootstrap blocks) with their player-games and wins.
    deals: dict[Variant, dict[int, tuple[float, float]]] = defaultdict(dict)
    for variant, (deal, games, wins) in _game_groups(
        db, dims, ("deal",), "sum(rows), sum(wins)", "eligible"
    ):
        deals[variant][int(deal)] = (float(games), float(wins or 0))
    cards = _card_blocks(db, dims)
    failed = dict(
        db.execute(
            "select run, count(*) from matches where state = 'failed' group by run"
        ).fetchall()
    )
    names = dict(db.execute("select player, name from players").fetchall())
    labels = dict(db.execute("select distinct card, name from cards").fetchall())
    found: dict[Variant, CardData] = {}
    for variant, variant_runs in runs.items():
        per_deal = deals.get(variant, {})
        games = sum(g for g, _ in per_deal.values())
        found[variant] = CardData(
            selection=selection,  # Narrowed once when fixed dimensions are expanded.
            subjects={p: names[p] for p in sorted(subjects[variant])},
            opponents=opponents[variant],
            runs=variant_runs,
            games=int(games),
            deals=len(per_deal),
            win_rate=sum(w for _, w in per_deal.values()) / games if games else None,
            excluded=excluded.get(variant, {}),
            failed_matches=sum(int(failed.get(run, 0)) for run in variant_runs),
            cards=_card_stats(sorted(per_deal), cards.get(variant), labels)
            if games
            else (),
        )
    return _expanded(found, selection, dims, tuple(dimensions), fixed)


def _expanded(
    found: dict[Variant, CardData],
    selection: CardSelection,
    split: tuple[Dimension, ...],
    dimensions: tuple[Dimension, ...],
    fixed: dict[Dimension, str | bool],
) -> dict[Variant, CardData]:
    """Results over `split` keyed by every dimension, each fixed one as its value or all."""
    expanded: dict[Variant, CardData] = {}
    for variant, data in found.items():
        given = dict(zip(split, variant, strict=True))
        choices = [(None, fixed[d]) if d in fixed else (given[d],) for d in dimensions]
        for key in product(*choices):
            expanded[key] = data.model_copy(
                update={"selection": _narrowed(selection, dimensions, key)}
            )
    return expanded


def _narrowed(
    selection: CardSelection, dimensions: Sequence[Dimension], variant: Variant
) -> CardSelection:
    update: dict[str, object] = {}
    for dimension, value in zip(dimensions, variant, strict=True):
        if value is None:
            continue
        if dimension in ("subject", "opponent"):
            update[f"{dimension}s"] = (value,)
        else:
            update[dimension] = value
    return selection.model_copy(update=update)


@dataclass(frozen=True)
class _CardBlocks:
    """Aligned columns for one variant: each row is one card/deal's metric totals."""

    cards: npt.NDArray[np.str_]
    deals: npt.NDArray[np.int64]
    sums: npt.NDArray[np.float64]


def _card_blocks(
    db: duckdb.DuckDBPyConnection, dimensions: Sequence[Dimension]
) -> dict[Variant, _CardBlocks]:
    """Per combination: card, deal and the `_SUMS` totals of each (card, deal) as arrays.

    Hundreds of thousands of rows arrive as columns, so they are never Python
    objects one by one; a combination is a run of rows with one `segment`.
    """
    group_by, mask = _grouping(dimensions, ("card", "deal"))
    keys = ", ".join(["mask", *dimensions])
    picked = ", ".join([f"{mask} as mask", *dimensions, "card", "deal"])
    db.execute(
        "create or replace temp table card_blocks as"
        f" select dense_rank() over (order by {keys}) - 1 as segment, *"
        f" from (select {picked}, "
        + ", ".join(f"sum({name})::DOUBLE as {name}" for name in _SUMS)
        + f" from picked_cards {group_by})"
    )
    variants = {
        int(segment): _variant(int(bits), values)
        for segment, bits, *values in db.execute(
            f"select distinct segment, {keys} from card_blocks"
        ).fetchall()
    }
    columns = db.execute(
        f"select segment, card, deal, {', '.join(_SUMS)} from card_blocks"
        " order by segment"
    ).fetchnumpy()
    segments = np.asarray(columns["segment"])
    if not len(segments):
        return {}
    cards = np.asarray(columns["card"]).astype(str)
    deals = np.asarray(columns["deal"]).astype(np.int64)
    sums = np.column_stack(
        [np.nan_to_num(np.asarray(columns[name], dtype=np.float64)) for name in _SUMS]
    )
    starts = np.flatnonzero(np.diff(segments, prepend=-1))
    ends = np.append(starts[1:], len(segments))
    return {
        variants[int(segments[start])]: _CardBlocks(
            cards=cards[start:end],
            deals=deals[start:end],
            sums=sums[start:end],
        )
        for start, end in zip(starts, ends, strict=True)
    }


def _card_stats(
    deals: Sequence[int], blocks: _CardBlocks | None, labels: dict[str, str]
) -> tuple[CardStats, ...]:
    """Per card, from one combination's (card, deal, sums) over its eligible deals."""
    if blocks is None:
        return ()
    order, column = np.unique(blocks.cards, return_inverse=True)
    row = np.searchsorted(np.asarray(deals), blocks.deals)
    # Only GIH needs deal-level matrices for uncertainty. All other metrics
    # need per-card sums, without expanding absent card/deal pairs to zeros.
    summed = np.zeros((len(order), len(_SUMS)))
    np.add.at(summed, column, blocks.sums)
    opportunities = np.zeros((len(deals), len(order)))
    wins = np.zeros_like(opportunities)
    gih_index = _SUMS.index("gih")
    opportunities[row, column] = blocks.sums[:, gih_index]
    wins[row, column] = blocks.sums[:, gih_index + 1]
    # One shared bootstrap resamples whole deals for every card at once.
    estimates = estimate(list(deals), wins, opportunities)
    stats = []
    for j, card in enumerate(order.tolist()):
        gp, gp_won, oh, oh_won, gd, gd_won, gih, _, gns_n, gns_won = summed[j]
        gih_estimate = estimates[j]
        gih_rate = gih_estimate.value if gih_estimate else None
        gns = _rate(gns_n, gns_won)
        stats.append(
            CardStats(
                card=card,
                name=labels.get(card, card),
                gp=_rate(gp, gp_won),
                oh=_rate(oh, oh_won),
                gd=_rate(gd, gd_won),
                gih=Rate(count=int(gih), rate=gih_rate),
                gih_interval=gih_estimate.interval if gih_estimate else None,
                gns=gns,
                iwd=None
                if gih_rate is None or gns.rate is None
                else 100 * (gih_rate - gns.rate),
            )
        )
    stats.sort(key=lambda s: (s.gih.rate is None, -(s.gih.rate or 0), s.name))
    return tuple(stats)


def card_data(db: duckdb.DuckDBPyConnection, selection: CardSelection) -> CardData:
    """17Lands-style card columns for the selected games of one experiment."""
    found = card_variants(db, selection)
    if () in found:
        return found[()]
    # No game matches: nothing to report, but the selection is still valid.
    return CardData(
        selection=selection,
        subjects={},
        opponents={},
        runs=(),
        games=0,
        deals=0,
        win_rate=None,
        excluded={},
        failed_matches=0,
        cards=(),
    )


# ---- Evidence: export and audit ----------------------------------------------


def _folder(store: Storage, run: str, key: str) -> str:
    """The folder of `run` that holds `key`; replicas hold the same bytes."""
    for folder in places(store).get(run, []):
        if folder + key in store.keys(folder):
            return folder
    raise ValueError(f"{key} unavailable for run {run}")


def _reference(digest: str, size: int, game: str) -> EvidenceRef:
    return EvidenceRef(path=evidence_key(digest), sha256=digest, bytes=size, game=game)


def load_game(
    db: duckdb.DuckDBPyConnection, source: Storage | Path | str, run: str, game: str
) -> GameEvidence:
    """One game's retained evidence blob, verified against its admitted digest.

    `source` is the store `db` was opened from.
    """
    found = db.execute(
        "select distinct evidence_sha256, evidence_bytes from card_plays"
        " where run = ? and game = ?",
        [run, game],
    ).fetchall()
    if len(found) != 1:
        raise ValueError("run and game must identify exactly one retained game")
    digest, size = found[0]
    [(_, body)] = read_blobs(db, source, [digest])
    return read_evidence(body, _reference(digest, size, game))


class AuditResult(StrictModel):
    games: int
    replayed: int
    problems: tuple[str, ...]


def audit(
    db: duckdb.DuckDBPyConnection, source: Storage | Path | str, *, replay: bool = False
) -> AuditResult:
    """Re-read every admitted game's evidence blob and rebuild its card rows.

    Checks bytes and that the stored rows equal the reducer's. A card
    evaluation's game is also bound to its scheduled match and to the runtime
    its eval run declared; another evaluation's game is rebuilt with the
    participants its rows name. `replay` also re-runs each trace. `source` is
    the store `db` was opened from; blobs are read in pack order, one at a time.
    """
    store = as_storage(source)
    columns = {r[0] for r in db.execute("describe card_plays").fetchall()}
    play_fields = tuple(
        field for field in PLAY_FIELDS if field != "game_seat" or field in columns
    )
    participant = "participant" if "participant" in columns else "null"
    plays = db.execute(
        "select run, match, game, evidence_sha256, evidence_bytes,"
        f" list({participant} order by seat)"
        " from card_plays group by all order by run, match, game"
    ).fetchall()
    by_digest: dict[str, list[tuple[str, str, str, int, list[str | None]]]] = (
        defaultdict(list)
    )
    for run, match_key, game_id, digest, size, named in plays:
        by_digest[digest].append((run, match_key, game_id, size, named))
    held: dict[str, tuple[dict[str, Match], TournamentRun] | None] = {}
    problems: list[str] = []
    replayed = 0

    def check(
        run: str,
        match_key: str,
        game_id: str,
        game: GameEvidence,
        named: list[str | None],
    ) -> None:
        if run not in held:
            held[run] = _card_schedule(store, run)
        schedule = held[run]
        if schedule is not None and match_key in schedule[0]:
            participants = admit_card_game(game, schedule[0][match_key])
            admit_card_runtime(game, schedule[1].tournament.environment)
        elif len(named) == 2 and named[0] and named[1]:
            participants = (named[0], named[1])
        else:
            raise ValueError("no schedule or participants to rebuild this game")
        if any(named) and tuple(named) != participants:
            raise ValueError("stored participants differ from the schedule")
        expected = normalize(reduce_cards(game, participants))
        stored_plays = _rows(db, "card_plays", play_fields, run, match_key, game_id)
        # Card tables already recorded native position as on_play. Older
        # rows lack the redundant game_seat, including nulls in mixed runs.
        # Preserve explicit values so corrupt new rows still fail the audit.
        for play in stored_plays:
            if play.get("game_seat") is None:
                play["game_seat"] = 0 if play["on_play"] else 1
        stored_counts = _rows(db, "card_counts", COUNT_FIELDS, run, match_key, game_id)
        if _sorted(stored_plays) != _sorted(expected.plays) or _sorted(
            stored_counts
        ) != _sorted(expected.counts):
            raise ValueError("stored card rows differ from retained evidence")

    try:
        for digest, body in read_blobs(db, store, list(by_digest)):
            for run, match_key, game_id, size, named in by_digest.pop(digest):
                try:
                    game = read_evidence(body, _reference(digest, size, game_id))
                    check(run, match_key, game_id, game, named)
                    if replay:
                        if not replay_games([game.trace.data]).passed:
                            raise ValueError("retained trace no longer replays")
                        replayed += 1
                except (ValueError, KeyError) as problem:
                    problems.append(f"{run}/{game_id}: {problem}")
    except EtudeError as unreadable:
        # Blobs the database does not know or cannot read leave their games unchecked.
        problems.extend(
            f"{run}/{game_id}: evidence unreadable ({unreadable.code})"
            for games in by_digest.values()
            for run, _, game_id, _, _ in games
        )
    return AuditResult(games=len(plays), replayed=replayed, problems=tuple(problems))


def _card_schedule(
    store: Storage, run: str
) -> tuple[dict[str, Match], TournamentRun] | None:
    """A card evaluation's scheduled matches by key; None for any other eval run."""
    try:
        folder = _folder(store, run, "tournament.json")
    except ValueError:
        return None
    tournament = TournamentRun.model_validate_json(
        store.get(folder + "tournament.json")
    )
    matches = {s.match.key: s.match for s in assemble([tournament]).schedule()}
    if not all(_is_card_match(m) for m in matches.values()):
        return None
    return matches, tournament


def _is_card_match(match: Match) -> bool:
    try:
        CardConditions.model_validate(match.conditions)
    except ValueError:
        return False
    return True


def _sorted(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    # Within one game, plays are keyed by seat and counts by (seat, card).
    return sorted(rows, key=lambda row: (str(row["seat"]), str(row.get("card", ""))))


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
    return [dict(zip(fields, row, strict=True)) for row in found]
