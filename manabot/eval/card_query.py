"""Card statistics over an experiment's kept database, with shared-deal uncertainty.

`card_data` reports 17Lands-style columns per card for one selection of the
games in every eval run of one experiment; `card_variants` reports them as
columns for every combination of the selection's filters in one pass, with
one batched bootstrap per deal count. Both read only the
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
from itertools import combinations, pairwise, product
from pathlib import Path
from typing import Literal

import duckdb
from etude.eval.query import REPLICATES, Estimate, estimate, read_blobs, statsdb
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


@dataclass(frozen=True)
class _Segments:
    """Grouped rows ordered by combination: `at[variant]` slices out its rows."""

    at: dict[Variant, slice]
    columns: dict[str, npt.NDArray[np.generic]]

    def rows(self, variant: Variant, column: str) -> npt.NDArray[np.generic]:
        """A combination's rows of `column`; none for a combination without any."""
        return self.columns[column][self.at.get(variant, slice(0))]


def _segments(
    db: duckdb.DuckDBPyConnection,
    table: str,
    dimensions: Sequence[Dimension],
    within: Sequence[str],
    values: dict[str, str],
    where: str = "true",
    *,
    ordered: bool = True,
) -> _Segments:
    """Totals per filter combination and `within` columns, as columns by combination.

    Millions of rows arrive as arrays, so they are never Python objects one
    by one; a combination is a run of rows with one `segment`, in `within`
    order when `ordered`. Combinations are numbered from their distinct keys,
    so the rows themselves are sorted once.
    """
    group_by, mask = _grouping(dimensions, within)
    keys = ["mask", *dimensions]
    picked = ", ".join(
        [
            f"{mask} as mask",
            *dimensions,
            *within,
            *(f"{sql} as {name}" for name, sql in values.items()),
        ]
    )
    db.execute(
        "create or replace temp table grouped as"
        f" select {picked} from {table} where {where} {group_by}"
    )
    db.execute(
        "create or replace temp table segment_keys as"
        " select (row_number() over () - 1)::INTEGER as segment, *"
        f" from (select distinct {', '.join(keys)} from grouped)"
    )
    variants = {
        int(segment): _variant(int(bits), found)
        for segment, bits, *found in db.execute(
            f"select segment, {', '.join(keys)} from segment_keys"
        ).fetchall()
    }
    # A rolled-up dimension is null in both tables.
    on = " and ".join(f"g.{key} is not distinct from k.{key}" for key in keys)
    columns = db.execute(
        f"select k.segment, {', '.join(f'g.{c}' for c in [*within, *values])}"
        f" from grouped g join segment_keys k on {on}"
        f" order by {', '.join(['k.segment', *(within if ordered else ())])}"
    ).fetchnumpy()
    segment = np.asarray(columns.pop("segment"))
    bounds = [*np.flatnonzero(np.diff(segment, prepend=-1)).tolist(), len(segment)]
    return _Segments(
        at={
            variants[int(segment[start])]: slice(start, end)
            for start, end in pairwise(bounds)
        },
        columns={name: np.asarray(column) for name, column in columns.items()},
    )


@dataclass(frozen=True)
class CardColumns:
    """One combination's per-card statistics as aligned columns, in display order.

    Rows are cards, by GIH WR descending, cards without one last, then name.
    `counts` and `rates` hold GP, OH, GD, GIH and GNS in that order; a rate,
    interval bound or IWD without opportunities is NaN.
    """

    cards: tuple[str, ...]
    names: tuple[str, ...]
    counts: npt.NDArray[np.int64]
    rates: npt.NDArray[np.float64]
    gih_interval: npt.NDArray[np.float64]
    # Copy-weighted GIH WR minus GNS WR, in percentage points.
    iwd: npt.NDArray[np.float64]

    def records(self) -> tuple[CardStats, ...]:
        def rate(row: int, metric: int) -> Rate:
            value = float(self.rates[row, metric])
            return Rate(
                count=int(self.counts[row, metric]),
                rate=None if np.isnan(value) else value,
            )

        stats = []
        for row, (card, name) in enumerate(zip(self.cards, self.names, strict=True)):
            low, high = self.gih_interval[row].tolist()
            iwd = float(self.iwd[row])
            stats.append(
                CardStats(
                    card=card,
                    name=name,
                    gp=rate(row, 0),
                    oh=rate(row, 1),
                    gd=rate(row, 2),
                    gih=rate(row, 3),
                    gih_interval=None if np.isnan(low) else (low, high),
                    gns=rate(row, 4),
                    iwd=None if np.isnan(iwd) else iwd,
                )
            )
        return tuple(stats)


_NO_CARDS = CardColumns(
    cards=(),
    names=(),
    counts=np.zeros((0, 5), dtype=np.int64),
    rates=np.zeros((0, 5)),
    gih_interval=np.zeros((0, 2)),
    iwd=np.zeros(0),
)


@dataclass(frozen=True)
class CardVariant:
    """One combination: its `CardData` without card records, and its cards as columns."""

    data: CardData
    columns: CardColumns

    def card_data(self) -> CardData:
        return self.data.model_copy(update={"cards": self.columns.records()})


def card_variants(
    db: duckdb.DuckDBPyConnection,
    selection: CardSelection,
    dimensions: Sequence[Dimension] = (),
) -> dict[Variant, CardVariant]:
    """A `CardVariant` for `selection` narrowed to every combination of `dimensions`.

    Each combination gives each dimension one value present in the selection's
    games, or all of them (None). Combinations without any game are absent;
    those without eligible games report exclusions only. Each one's
    `card_data()` equals `card_data` of the narrowed selection; the empty
    combination's is `card_data(db, selection)`.
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
    deals = _segments(
        db,
        "picked_games",
        dims,
        ("deal",),
        {"games": "sum(rows)::DOUBLE", "wins": "coalesce(sum(wins), 0)::DOUBLE"},
        "eligible",
    )
    # Cards are numbered in key order, so a card is an integer from here on.
    db.execute(
        "create or replace temp table card_codes as select card,"
        " (row_number() over (order by card) - 1)::INTEGER as code"
        " from (select distinct card from picked_cards)"
    )
    keys = [
        card
        for (card,) in db.execute(
            "select card from card_codes order by code"
        ).fetchall()
    ]
    db.execute(
        "create or replace temp view coded_cards as"
        " select * from picked_cards join card_codes using (card)"
    )
    summed = _segments(
        db,
        "coded_cards",
        dims,
        ("code",),
        {name: f"sum({name})::DOUBLE" for name in _SUMS},
    )
    # Only GIH needs deal-level totals, for its uncertainty; a zero adds nothing.
    gih = _segments(
        db,
        "coded_cards",
        dims,
        ("code", "deal"),
        {"opportunities": "sum(gih)::DOUBLE", "wins": "sum(gih_won)::DOUBLE"},
        "gih > 0",
        ordered=False,
    )
    failed = dict(
        db.execute(
            "select run, count(*) from matches where state = 'failed' group by run"
        ).fetchall()
    )
    names = dict(db.execute("select player, name from players").fetchall())
    labels = dict(db.execute("select distinct card, name from cards").fetchall())
    card_names = [labels.get(card, card) for card in keys]
    name_rank = np.argsort(
        np.argsort(np.array(card_names, dtype=object), kind="stable")
    )
    totals: dict[Variant, _CardTotals] = {}
    played: dict[Variant, tuple[int, float, float]] = {}
    for variant in deals.at:
        per_deal = deals.rows(variant, "deal")
        games = float(deals.rows(variant, "games").sum())
        played[variant] = (
            len(per_deal),
            games,
            float(deals.rows(variant, "wins").sum()),
        )
        if games and variant in summed.at:
            totals[variant] = _card_totals(variant, per_deal, summed, gih)
    estimates = dict(zip(totals, _gih_estimates(list(totals.values())), strict=True))
    found: dict[Variant, CardVariant] = {}
    for variant, variant_runs in runs.items():
        deal_count, games, wins = played.get(variant, (0, 0.0, 0.0))
        found[variant] = CardVariant(
            data=CardData(
                selection=selection,  # Narrowed once fixed dimensions are expanded.
                subjects={p: names[p] for p in sorted(subjects[variant])},
                opponents=opponents[variant],
                runs=variant_runs,
                games=int(games),
                deals=deal_count,
                win_rate=wins / games if games else None,
                excluded=excluded.get(variant, {}),
                failed_matches=sum(int(failed.get(run, 0)) for run in variant_runs),
                cards=(),
            ),
            columns=_card_columns(
                totals[variant], estimates[variant], keys, card_names, name_rank
            )
            if variant in totals
            else _NO_CARDS,
        )
    return _expanded(found, selection, dims, tuple(dimensions), fixed)


def _expanded(
    found: dict[Variant, CardVariant],
    selection: CardSelection,
    split: tuple[Dimension, ...],
    dimensions: tuple[Dimension, ...],
    fixed: dict[Dimension, str | bool],
) -> dict[Variant, CardVariant]:
    """Results over `split` keyed by every dimension, each fixed one as its value or all."""
    expanded: dict[Variant, CardVariant] = {}
    for variant, table in found.items():
        given = dict(zip(split, variant, strict=True))
        choices = [(None, fixed[d]) if d in fixed else (given[d],) for d in dimensions]
        for key in product(*choices):
            narrowed = _narrowed(selection, dimensions, key)
            expanded[key] = CardVariant(
                data=table.data.model_copy(update={"selection": narrowed}),
                columns=table.columns,
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
class _CardTotals:
    """One combination's per-card sums and its GIH opportunities and wins per deal.

    `codes` are the combination's cards in key order and `summed` their
    `_SUMS`. `deal` and `card` index a sparse deals × cards matrix: `deal` is
    the position among the combination's eligible deals in key order, the
    order `estimate` resamples them in, and `card` a row of `codes`.
    """

    deals: int
    codes: npt.NDArray[np.int32]
    summed: npt.NDArray[np.float64]
    deal: npt.NDArray[np.intp]
    card: npt.NDArray[np.intp]
    opportunities: npt.NDArray[np.float64]
    wins: npt.NDArray[np.float64]


def _card_totals(
    variant: Variant,
    deals: npt.NDArray[np.generic],
    summed: _Segments,
    gih: _Segments,
) -> _CardTotals:
    codes = summed.rows(variant, "code")
    return _CardTotals(
        deals=len(deals),
        codes=codes,
        summed=np.column_stack([summed.rows(variant, name) for name in _SUMS]),
        deal=np.searchsorted(deals, gih.rows(variant, "deal")),
        card=np.searchsorted(codes, gih.rows(variant, "code")),
        opportunities=gih.rows(variant, "opportunities"),
        wins=gih.rows(variant, "wins"),
    )


# Bounds each batched `estimate` call's matrices to about 32 MB apiece.
_BATCH_CELLS = 4_000_000


def _gih_estimates(totals: Sequence[_CardTotals]) -> list[list[Estimate | None]]:
    """Each combination's per-card GIH `estimate`, batched by deal count.

    `estimate`'s bootstrap weights depend only on the number of deals, and a
    column's result only on its own numerators and denominators. So the
    columns of every combination with `n` deals share calls, with the deals'
    positions as keys, and each equals the combination's own call. Inputs
    are integer counts, so resampled totals are exact in any summation order.
    Columns are ordered by how many deals hold opportunities, which groups
    the sparse ones whose replicates can draw none.
    """
    found: list[list[Estimate | None]] = [[None] * len(t.codes) for t in totals]
    by_count: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for index, total in enumerate(totals):
        held = np.bincount(total.card, minlength=len(total.codes))
        for column, count in enumerate(held.tolist()):
            by_count[total.deals].append((count, index, column))
    for deals, columns in by_count.items():
        columns.sort()
        width = max(1, min(_BATCH_CELLS // deals, _BATCH_CELLS // REPLICATES))
        for start in range(0, len(columns), width):
            batch = columns[start : start + width]
            wins = np.zeros((deals, len(batch)))
            opportunities = np.zeros_like(wins)
            positions: dict[int, npt.NDArray[np.intp]] = {}
            for position, (_, index, column) in enumerate(batch):
                if index not in positions:
                    positions[index] = np.full(len(totals[index].codes), -1)
                positions[index][column] = position
            for index, position_of in positions.items():
                total = totals[index]
                placed = position_of[total.card]
                rows = placed >= 0
                opportunities[total.deal[rows], placed[rows]] = total.opportunities[
                    rows
                ]
                wins[total.deal[rows], placed[rows]] = total.wins[rows]
            for (_, index, column), result in zip(
                batch, estimate(range(deals), wins, opportunities), strict=True
            ):
                found[index][column] = result
    return found


def _card_columns(
    totals: _CardTotals,
    gih: Sequence[Estimate | None],
    keys: Sequence[str],
    names: Sequence[str],
    name_rank: npt.NDArray[np.intp],
) -> CardColumns:
    counts = totals.summed[:, 0::2]
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(counts > 0, totals.summed[:, 1::2] / counts, np.nan)
    # GIH WR and its interval are the shared bootstrap's.
    rates[:, 3] = [np.nan if e is None else e.value for e in gih]
    interval = np.array(
        [
            (np.nan, np.nan) if e is None or e.interval is None else e.interval
            for e in gih
        ],
        dtype=np.float64,
    ).reshape(-1, 2)
    gih_rate = rates[:, 3]
    # Missing GIH WR last, then descending; ties by name, then card key.
    order = np.lexsort(
        (
            totals.codes,
            name_rank[totals.codes],
            -np.nan_to_num(gih_rate),
            np.isnan(gih_rate),
        )
    )
    codes = totals.codes[order].tolist()
    return CardColumns(
        cards=tuple(keys[code] for code in codes),
        names=tuple(names[code] for code in codes),
        counts=counts[order].astype(np.int64),
        rates=rates[order],
        gih_interval=interval[order],
        iwd=100 * (gih_rate - rates[:, 4])[order],
    )


def card_data(db: duckdb.DuckDBPyConnection, selection: CardSelection) -> CardData:
    """17Lands-style card columns for the selected games of one experiment."""
    found = card_variants(db, selection)
    if () in found:
        return found[()].card_data()
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
