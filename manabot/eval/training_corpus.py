"""Query completed training games by update, ordered deck pair and card identity."""

from collections import Counter
import json
from pathlib import Path

import duckdb
from etude.checkpoints import CheckpointRecord
from etude.storage import Storage
from etude.train.read import RunData, read
from pydantic import TypeAdapter

from manabot.model.writer_catalog import archived_catalog, writer_catalog
from manabot.reports.update_metrics import scalar_updates
from manabot.training.decks import CorpusDeck, corpus_decks
from manabot.training.rollout.self_play import SelfPlayOutcome
from manabot.training.update_metrics import OUTCOME_TABLE, ROSTER_TABLE
from managym import Deck


class UnknownCoverage(ValueError):
    """The retained evidence cannot establish the corpus through this update."""


def roster(data: RunData) -> list[CorpusDeck]:
    """Resolve all declared population entries, including decks never played."""
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
    catalog = writer_catalog()
    if catalog["world_digest"] != digest:
        catalog = archived_catalog(digest, data.name)
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


def training_corpus(
    run: RunData | CheckpointRecord,
    through: int | None = None,
    *,
    source: Storage | Path | str | None = None,
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

    cards = {row["deck"]: row["cards"] for row in roster(data)}
    matchups = [
        (update, play, draw, count)
        for update, pairs in games_by_update(data, through).items()
        for (play, draw), count in pairs.items()
        if count
    ]
    if any(deck not in cards for _, play, draw, _ in matchups for deck in (play, draw)):
        raise UnknownCoverage(f"{data.name}: an outcome names an undeclared deck")
    db = duckdb.connect()
    db.execute("create table decks(deck varchar, cards varchar[])")
    db.execute(
        "insert into decks select unnest(?::varchar[]), unnest(?::varchar[][])",
        [list(cards), list(cards.values())],
    )
    # One columnar insert: frontier runs hold hundreds of matchups per update.
    columns = [list(column) for column in zip(*matchups)] or [[], [], [], []]
    db.execute(
        "create table matchups(update_number bigint, play varchar, draw varchar, games bigint)"
    )
    db.execute(
        "insert into matchups select unnest(?::bigint[]), unnest(?::varchar[]), unnest(?::varchar[]), unnest(?::bigint[])",
        columns,
    )
    return db.sql("""
        select m.*, p.cards as play_cards, d.cards as draw_cards
        from matchups m join decks p on p.deck = m.play join decks d on d.deck = m.draw
    """)
