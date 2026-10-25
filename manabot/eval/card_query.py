"""Explicit-cohort card queries over Etude tables, with shared-deal uncertainty.

Each query selects one run and subject agent. A deal is the bootstrap unit across
all opponents and seat/deck legs, including deals with zero card opportunities."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from operator import attrgetter
from pathlib import Path

from etude.eval.query import estimate, statsdb
from etude.eval.records import Match, TournamentRun, assemble
from etude.run.listing import as_storage, places
from etude.run.storage import Storage

from manabot.eval.card_metrics import (
    METRIC_VERSION,
    CardGame,
    MatchExclusions,
    reduce_cards,
)
from manabot.eval.card_schedule import admit_card_match
from manabot.infra.identity import StrictModel
from manabot.sim.card_evidence import (
    CAPTURE_VERSION,
    EvidenceRef,
    GameEvidence,
    read_evidence,
)


class CardQuery(StrictModel):
    run: str
    player: str
    opponents: tuple[str, ...] = ()
    deck: str | None = None
    opponent_deck: str | None = None
    on_play: bool | None = None
    partial: bool = False


class Completeness(StrictModel):
    scheduled: int
    completed: int
    failed: int
    pending: int
    scheduled_games: int
    failed_games: int

    @property
    def complete(self) -> bool:
        return self.scheduled == self.completed


class CardRank(StrictModel):
    card: str
    name: str
    rate: float | None
    interval: tuple[float, float] | None
    qualifying_copies: int
    winning_copies: int
    games: int
    excluded_games: int
    # Stable game IDs join card_evidence, never ephemeral local paths.
    evidence_games: tuple[str, ...]


class CardRanking(StrictModel):
    selection: CardQuery
    subject: str
    environment: str
    metric_version: str = METRIC_VERSION
    capture_version: int = CAPTURE_VERSION
    completeness: Completeness
    complete: bool
    opponent_mix: dict[str, int]
    excluded_outcomes: dict[str, int]
    cards: tuple[CardRank, ...]


def evidence_prefix(source: Storage, run: str) -> str:
    """Locate a run's evidence, rejecting replicas with conflicting schedules."""
    folders = places(source).get(run, [])
    if not folders:
        raise ValueError(f"evaluation run unavailable: {run}")
    # Etude allows replicated runs. Equal content-addressed evidence can be
    # read from any replica; conflicting manifests are never silently merged.
    manifests = {
        folder: source.get(folder + "tournament.json")
        for folder in folders
        if folder + "tournament.json" in source.keys(folder)
    }
    if len(set(manifests.values())) != 1:
        raise ValueError("card evidence unavailable or conflicting evaluation schedule")
    return next(iter(manifests))


def run_experiment(source: Storage, prefix: str) -> str:
    """The experiment an evaluation run declared, which names its database."""
    return json.loads(source.get(prefix + "run.json"))["experiment"]


def evidence_refs(source: Storage | Path | str, run: str) -> list[EvidenceRef]:
    store = as_storage(source)
    experiment = run_experiment(store, evidence_prefix(store, run))
    with statsdb(experiment, store) as db:
        if "card_evidence" not in {r[0] for r in db.execute("show tables").fetchall()}:
            raise ValueError("card evidence unavailable (outcome-only evaluation)")
        rows = db.execute(
            "select distinct path, sha256, bytes, game from card_evidence where run = ?",
            [run],
        ).fetchall()
        return [EvidenceRef(path=p, sha256=h, bytes=b, game=g) for p, h, b, g in rows]


def load_game(
    source: Storage | Path | str, run: str, reference: EvidenceRef
) -> GameEvidence:
    store = as_storage(source)
    prefix = evidence_prefix(store, run)
    return read_evidence(store.get(prefix + reference.path), reference)


def _selected_schedule(
    held: TournamentRun, query: CardQuery
) -> tuple[str, list[Match]]:
    all_matches = [scheduled.match for scheduled in assemble([held]).schedule()]
    agents = {a.identity: a for match in all_matches for a in match.agents}

    def resolve(value: str) -> str:
        found = [
            identity
            for identity, agent in agents.items()
            if value in (identity, agent.name)
        ]
        if len(found) != 1:
            raise ValueError(f"agent must identify exactly one entrant: {value}")
        return found[0]

    subject = resolve(query.player)
    opponents = {resolve(opponent) for opponent in query.opponents}
    if subject in opponents:
        raise ValueError("subject cannot be its own opponent")
    matches = [
        match
        for match in all_matches
        if subject in {a.identity for a in match.agents}
        and (not opponents or any(a.identity in opponents for a in match.agents))
    ]
    if not matches:
        raise ValueError("empty selected schedule")
    return subject, matches


def rank_cards(source: Storage | Path | str, query: CardQuery) -> CardRanking:
    store = as_storage(source)
    prefix = evidence_prefix(store, query.run)
    held = TournamentRun.model_validate_json(store.get(prefix + "tournament.json"))
    if held.tournament.environment.config.get("metric_version") != METRIC_VERSION:
        raise ValueError("card evidence unavailable or incompatible metric version")
    subject, schedule = _selected_schedule(held, query)
    with statsdb(run_experiment(store, prefix), store) as db:
        records = db.execute(
            "select match, state, error from matches where run = ?", [query.run]
        ).fetchall()
        states = {key: state for key, state, _ in records}
        errors = {key: error for key, state, error in records if state == "failed"}
        failed_outcomes: Counter[str] = Counter()
        for match in schedule:
            if states.get(match.key) == "failed":
                error = errors.get(match.key) or ""
                if error.startswith("CardMatchError: "):
                    failed_outcomes.update(
                        MatchExclusions.model_validate_json(
                            error.removeprefix("CardMatchError: ")
                        ).model_dump(exclude_defaults=True)
                    )
                else:
                    failed_outcomes["unobserved"] += match.games
        completeness = Completeness(
            scheduled=len(schedule),
            completed=sum(states.get(m.key) == "done" for m in schedule),
            failed=sum(states.get(m.key) == "failed" for m in schedule),
            pending=sum(m.key not in states for m in schedule),
            scheduled_games=sum(m.games for m in schedule),
            failed_games=sum(
                m.games for m in schedule if states.get(m.key) == "failed"
            ),
        )
        if not completeness.complete and not query.partial:
            raise ValueError(
                f"selected schedule incomplete: {completeness.model_dump()}; use partial explicitly"
            )
        tables = {r[0] for r in db.execute("show tables").fetchall()}
        if completeness.completed and not {"card_games", "card_evidence"} <= tables:
            raise ValueError("card evidence unavailable (outcome-only evaluation)")
        rows: list[CardGame] = []
        opponents_by_game: dict[str, str] = {}
        for match in schedule:
            if states.get(match.key) != "done":
                continue
            player = next(
                i for i, a in enumerate(match.agents) if a.identity == subject
            )
            raw_refs = db.execute(
                "select path, sha256, bytes, game from card_evidence where run = ? and match = ? and player = ?",
                [query.run, match.key, player],
            ).fetchall()
            games: list[GameEvidence] = []
            for path, digest, size, game_id in raw_refs:
                ref = EvidenceRef(path=path, sha256=digest, bytes=size, game=game_id)
                games.append(read_evidence(store.get(prefix + path), ref))
            names = admit_card_match(games, match, held.tournament.environment)
            expected = [
                row
                for game in games
                for row in reduce_cards(game, names)
                if row.player == player
            ]
            for game in games:
                opponents_by_game[game.game] = match.agents[1 - player].identity
            columns = list(CardGame.model_fields)
            raw = db.execute(
                "select "
                + ", ".join(f'"{c}"' for c in columns)
                + " from card_games where run = ? and match = ? and player = ?",
                [query.run, match.key, player],
            ).fetchall()
            actual = [
                CardGame.model_validate(dict(zip(columns, row, strict=True)))
                for row in raw
            ]

            key = attrgetter("game", "card")
            if sorted(actual, key=key) != sorted(expected, key=key):
                raise ValueError("card tables differ from retained evidence")
            rows.extend(actual)
    selected = [
        r
        for r in rows
        if (query.deck is None or r.deck == query.deck)
        and (query.opponent_deck is None or r.opponent_deck == query.opponent_deck)
        and (query.on_play is None or r.on_play == query.on_play)
    ]
    if rows and not selected:
        raise ValueError("filters select no games")
    return summarize_cards(
        selected,
        query,
        subject,
        held.tournament.environment.identity,
        completeness,
        opponents_by_game,
        dict(failed_outcomes),
    )


def summarize_cards(
    rows: list[CardGame],
    query: CardQuery,
    subject: str,
    environment: str,
    completeness: Completeness,
    opponents: dict[str, str],
    failed_outcomes: dict[str, int] | None = None,
) -> CardRanking:
    by_card: dict[str, list[CardGame]] = defaultdict(list)
    games: dict[str, CardGame] = {}
    for row in rows:
        if row.metric_version != METRIC_VERSION:
            raise ValueError("incompatible metric version")
        by_card[row.card].append(row)
        games[row.game] = row
    # Zeros are real blocks: dropping games where a card was not seen would
    # condition the uncertainty estimate on the observed denominator.
    deals = sorted({r.deal for r in rows})
    rankings = []
    for card, values in by_card.items():
        eligible = [r for r in values if r.eligible]
        copies: Counter[int] = Counter()
        wins: Counter[int] = Counter()
        for row in eligible:
            copies[row.deal] += row.gih_copies
            wins[row.deal] += (row.win or 0) * row.gih_copies
        result = estimate([(wins[deal], copies[deal]) for deal in deals])
        rankings.append(
            CardRank(
                card=card,
                name=values[0].name,
                rate=result[0] if result else None,
                interval=result[1] if result else None,
                qualifying_copies=copies.total(),
                winning_copies=wins.total(),
                games=sum(r.gih_copies > 0 for r in eligible),
                excluded_games=sum(not r.eligible for r in values),
                evidence_games=tuple(sorted(r.game for r in eligible if r.gih_copies)),
            )
        )
    excluded = Counter(r.outcome for r in games.values() if not r.eligible)
    excluded.update(failed_outcomes or {})
    return CardRanking(
        selection=query,
        subject=subject,
        environment=environment,
        completeness=completeness,
        complete=completeness.complete,
        opponent_mix=dict(Counter(opponents[g] for g in games)),
        excluded_outcomes=dict(excluded),
        cards=tuple(
            sorted(rankings, key=lambda r: (r.rate is None, -(r.rate or 0), r.card))
        ),
    )
