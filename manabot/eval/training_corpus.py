"""Query completed training games by update, ordered deck pair and card identity."""

from collections import Counter, defaultdict
from collections.abc import Iterator
from itertools import batched
import json
from pathlib import Path

import duckdb
from etude.checkpoints import CheckpointRecord
from etude.run import as_storage, located_runs
from etude.storage import Storage
from etude.train.read import RunData, read, read_all
from pydantic import TypeAdapter

from manabot.model.writer_catalog import WriterCatalog, archived_catalog, writer_catalog
from manabot.reports.update_metrics import scalar_updates
from manabot.training.corpus import DECK_TABLE, SETTINGS_TABLE
from manabot.training.decks import CorpusDeck, DeckCard, corpus_decks
from manabot.training.rollout.self_play import SelfPlayOutcome
from manabot.training.update_metrics import OUTCOME_TABLE, ROSTER_TABLE
from managym import Deck
from managym.statistics import read_statistics


class UnknownCoverage(ValueError):
    """The retained evidence cannot establish the corpus through this update."""


def bound_catalog(digest: str, location: str, current: WriterCatalog) -> WriterCatalog:
    return (
        current
        if current["world_digest"] == digest
        else archived_catalog(digest, location)
    )


def roster(data: RunData) -> list[CorpusDeck]:
    """Resolve all declared population entries, including decks never played."""
    rows = recorded_deck_cards(data)
    return historical_roster(data) if rows is None else roster_from_cards(rows)


def recorded_deck_cards(data: RunData) -> list[DeckCard] | None:
    for table in data.read_table(DECK_TABLE, "updates"):
        if table.x == 0:
            return TypeAdapter(list[DeckCard]).validate_python(table.rows)
    return None


def roster_from_cards(rows: list[DeckCard]) -> list[CorpusDeck]:
    cards: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in rows:
        cards[row["entry"], row["deck"]].add(row["card"])
    return [
        CorpusDeck(deck=deck, cards=sorted(keys)) for (_, deck), keys in cards.items()
    ]


def historical_roster(data: RunData) -> list[CorpusDeck]:
    for table in data.read_table(ROSTER_TABLE, "updates"):
        if table.x == 0:
            return TypeAdapter(list[CorpusDeck]).validate_python(table.rows)
    config = (
        json.loads(data.file.config_json) if data.file and data.file.config_json else {}
    )
    try:
        training = config["training"]
        decks = training["regime"]["deck_source"]["decks"]
        digest = training["identities"]["content_digest"]
    except (KeyError, TypeError) as error:
        raise UnknownCoverage(
            f"{data.name}: resolved training decks are unknown"
        ) from error
    catalog = bound_catalog(digest, data.name, writer_catalog())
    return corpus_decks((Deck.from_dict(deck) for deck in decks), catalog)


def games_by_update(data: RunData, through: int) -> dict[int, Counter[tuple[str, str]]]:
    """Count the play seat once in each recorded matchup, including mirrors."""
    games: dict[int, Counter[tuple[str, str]]] = {}
    # Historical writers logged each count as the scalar
    # `self_play_outcomes/<deck>/<opponent>/<seat>/<outcome>`; decode scalars
    # only when such a series exists, since current writers log the table.
    if any(
        c.name.startswith(OUTCOME_TABLE + "/")
        for b in data.blocks
        for c in b.header.series
    ):
        for update, row in scalar_updates(data, through).items():
            for name, count in row.items():
                parts = name.split("/")
                if update > 0 and len(parts) == 5 and parts[0] == OUTCOME_TABLE:
                    if parts[3] == "play":
                        games.setdefault(update, Counter())[parts[1], parts[2]] += int(
                            count
                        )
    outcomes = TypeAdapter(list[SelfPlayOutcome])
    for table in data.read_table(OUTCOME_TABLE, "updates"):
        if 0 < table.x <= through:
            counts: Counter[tuple[str, str]] = Counter()
            for outcome in outcomes.validate_python(list(table.rows), strict=True):
                if outcome["position"] == "play":
                    counts[outcome["deck"], outcome["opponent_deck"]] += (
                        outcome["wins"] + outcome["losses"] + outcome["draws"]
                    )
            games[int(table.x)] = counts
    for update in range(1, through + 1):
        if update not in games:
            raise UnknownCoverage(f"{data.name}: coverage unknown at update {update}")
    return games


def add_card_statistics(
    db: duckdb.DuckDBPyConnection,
    path: Path,
    games: dict[int, Counter[tuple[str, str]]],
    through: int,
) -> None:
    current = writer_catalog()
    catalogs: dict[str, WriterCatalog] = {}
    terminal: Counter[int] = Counter()

    def rows() -> Iterator[dict[str, object]]:
        seen = False
        for game in read_statistics(path) if path.exists() else ():
            seen = True
            if not game.terminal:
                continue
            finished = (game.context or {}).get("update")
            if type(finished) is not int or finished < 0:
                raise UnknownCoverage(
                    f"{path}: terminal statistics lack a collection update"
                )
            # Capture runs during collection, before the learner advances its clock.
            update = finished + 1
            if update > through:
                continue
            terminal[update] += 1
            digest = game.card_digest
            assert digest is not None  # read_statistics validates the digest
            if digest not in catalogs:
                catalogs[digest] = bound_catalog(digest, str(path), current)
            keys = catalogs[digest]["definition_keys"]
            for count in game.cards:
                if not 0 <= count.definition < len(keys):
                    raise UnknownCoverage(
                        f"{path}: missing definition {count.definition} in {digest}"
                    )
                yield {
                    "update_number": update,
                    "card": keys[count.definition],
                    "games": int(count.deck_copies > 0),
                    "wins": int(count.deck_copies > 0 and game.winner == count.seat),
                    **count.model_dump(exclude={"definition", "name"}),
                }
        if not seen:
            raise UnknownCoverage(f"{path}: no statistics rows")
        for update, pairs in games.items():
            if terminal[update] < pairs.total():
                raise UnknownCoverage(
                    f"{path}: update {update} has {terminal[update]} terminal statistics games "
                    f"for {pairs.total()} corpus games"
                )

    db.execute("""create table card_counts(update_number bigint, card varchar, games bigint,
        wins bigint, seat integer, deck_copies bigint, opening_copies bigint, drawn_copies bigint,
        opportunities bigint, selections bigint, resolutions bigint)""")
    # Reduce each batch before storing it. Only the grouped contributions to an
    # update are retained, even when the capture holds millions of game rows.
    totals = """select update_number, seat, card, sum(games) games, sum(wins) wins,
        sum(deck_copies) deck_copies, sum(opening_copies) opening_copies,
        sum(drawn_copies) drawn_copies,
        case when count(opportunities) = count(*) then sum(opportunities) end opportunities,
        case when count(selections) = count(*) then sum(selections) end selections,
        case when count(resolutions) = count(*) then sum(resolutions) end resolutions"""
    for batch in batched(rows(), 2048):
        db.execute(
            f"insert into card_counts by name {totals} from "
            "(select unnest(?, recursive := true)) group by all",
            [list(batch)],
        )
    db.execute(
        f"create table card_statistics as {totals} from card_counts group by all"
    )
    db.execute("drop table card_counts")


def training_corpus(
    run: RunData | CheckpointRecord,
    through: int | None = None,
    *,
    source: Storage | Path | str | None = None,
    statistics: Path | None = None,
) -> duckdb.DuckDBPyRelation:
    """Return one row per update and ordered matchup, inclusive of the cutoff.

    Supply a run and update, or a published Etude checkpoint and its storage.
    Card lists include both main deck and sideboard. Each completed game
    contributes once, even in mirrors. Missing update evidence raises.
    """
    if isinstance(run, CheckpointRecord):
        if source is None or through is not None:
            raise ValueError("supply a checkpoint and its source")
        coordinate = run.at.get("updates")
        if coordinate is None or not float(coordinate).is_integer():
            raise UnknownCoverage("checkpoint has no integral updates coordinate")
        through = int(coordinate)
        data = read(source, run.experiment, run.run)
    else:
        data = run
    if type(through) is not int or through < 0:
        raise ValueError("supply a nonnegative update")

    db = duckdb.connect()
    db.execute("""create table settings(experiment varchar, run varchar,
        decklists varchar, mana_taps boolean, max_actions bigint, action_overflow varchar,
        policy_input_version bigint, behavior varchar, opponent_mode varchar,
        learning_seats varchar, matchup_curriculum varchar, parent_run varchar,
        parent_update bigint, parent_sha256 varchar)""")
    db.execute("""create table deck_cards(experiment varchar, run varchar, entry bigint,
        deck varchar, card varchar, copies bigint, board varchar)""")
    db.execute(
        "create table decks(experiment varchar, run varchar, deck varchar, cards varchar[])"
    )
    db.execute("""create table matchups(experiment varchar, run varchar, update_number bigint,
        play varchar, draw varchar, games bigint)""")
    visited: set[tuple[str, str]] = set()
    while True:
        identity = data.experiment, data.name
        if identity in visited:
            raise UnknownCoverage(f"{identity}: cycle in checkpoint ancestry")
        visited.add(identity)
        parent = add_run(db, data, through, statistics)
        if parent is None:
            break
        if statistics is not None:
            raise UnknownCoverage(
                "one statistics directory cannot establish parent and child coverage"
            )
        storage = as_storage(source) if source is not None else data.blocks[0].storage
        name, update, digest = parent
        keys = sorted(storage.keys(""))
        candidates = located_runs(storage, keys, run=name)
        matches = [
            ancestor
            for ancestor in read_all(storage, keys=keys, declarations=candidates)
            if any(
                cp.at.get("updates") == update and cp.artifact.sha256 == digest
                for cp in ancestor.checkpoints
            )
        ]
        if len(matches) != 1:
            raise UnknownCoverage(
                f"cannot resolve parent checkpoint {name} at {update} with SHA-256 {digest}"
            )
        data, through = matches[0], update
    return db.sql("""
        select m.update_number, m.play, m.draw, m.games, p.cards as play_cards,
               d.cards as draw_cards, m.experiment, m.run
        from matchups m
        join decks p on (p.experiment, p.run, p.deck) = (m.experiment, m.run, m.play)
        join decks d on (d.experiment, d.run, d.deck) = (m.experiment, m.run, m.draw)
    """)


def add_run(
    db: duckdb.DuckDBPyConnection, data: RunData, through: int, statistics: Path | None
) -> tuple[str, int, str] | None:
    identity = [data.experiment, data.name]
    rows = recorded_deck_cards(data)
    if rows is not None:
        population = roster_from_cards(rows)
        if rows:
            db.execute(
                "insert into deck_cards by name select ? as experiment, ? as run, unnest(?, recursive := true)",
                [*identity, rows],
            )
    else:
        population = historical_roster(data)
    cards = {row["deck"]: row["cards"] for row in population}
    games = games_by_update(data, through)
    matchups = [
        (update, play, draw, count)
        for update, pairs in games.items()
        for (play, draw), count in pairs.items()
        if count
    ]
    if any(deck not in cards for _, play, draw, _ in matchups for deck in (play, draw)):
        raise UnknownCoverage(f"{data.name}: an outcome names an undeclared deck")
    db.execute(
        "insert into decks select ?, ?, unnest(?::varchar[]), unnest(?::varchar[][])",
        [*identity, list(cards), list(cards.values())],
    )
    # One columnar insert: frontier runs hold hundreds of matchups per update.
    columns = [list(column) for column in zip(*matchups)] or [[], [], [], []]
    db.execute(
        "insert into matchups select ?, ?, unnest(?::bigint[]), unnest(?::varchar[]), unnest(?::varchar[]), unnest(?::bigint[])",
        [*identity, *columns],
    )
    if statistics is not None:
        add_card_statistics(db, statistics, games, through)
    tables = data.read_table(SETTINGS_TABLE, "updates")
    if not tables:
        if rows is not None:
            raise UnknownCoverage(f"{data.name}: missing recorded training settings")
        db.execute("insert into settings(experiment, run) values (?, ?)", identity)
        return None
    (table,) = tables
    (settings,) = table.rows
    db.execute(
        "insert into settings by name select ? as experiment, ? as run, unnest(?)",
        [*identity, settings],
    )
    digest = settings["parent_sha256"]
    if digest is None:
        return None
    name, update = settings["parent_run"], settings["parent_update"]
    if (
        not isinstance(name, str)
        or type(update) is not int
        or update < 0
        or not isinstance(digest, str)
    ):
        raise UnknownCoverage(
            f"{data.name}: unknown source of parent checkpoint {digest}"
        )
    return name, update, digest
