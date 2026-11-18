"""A training corpus follows the writer's cards, completed games and resume cuts."""

from pathlib import Path

from etude.train.read import read
import pytest

from manabot.eval.training_corpus import UnknownCoverage, training_corpus
from manabot.model.writer_catalog import writer_catalog
from manabot.training.decks import corpus_decks
from manabot.training.rollout.self_play import SelfPlayCounter
from manabot.training.update_metrics import OUTCOME_TABLE, ROSTER_TABLE, metric_row
from managym import Deck
from tests.fixtures.training_records import recording


@pytest.mark.parametrize("historical", [False, True])
def test_corpus_queries_use_completed_games_through_update(
    tmp_path: Path, historical: bool
) -> None:
    island, forest = Deck({"Island": 10}), Deck({"Forest": 10}, {"Island": 1})
    catalog = writer_catalog()
    decks = [island, forest]
    config = (
        {
            "training": {
                "identities": {"content_digest": catalog["world_digest"]},
                "regime": {
                    "deck_source": {"decks": [deck.to_dict() for deck in decks]}
                },
            }
        }
        if historical
        else {}
    )
    with recording(
        experiment="corpus", directory=tmp_path, clocks=["updates"], config=config
    ) as run:
        if not historical:
            run.log_table(ROSTER_TABLE, corpus_decks(decks, catalog), updates=0)
        counter = SelfPlayCounter()
        for update, pair in enumerate([(island, island), (forest, island)], 1):
            counter.record((pair[0].identity, pair[1].identity), None)
            rows = counter.drain()
            if historical:
                run.log(metric_row({OUTCOME_TABLE: rows}), updates=update)
            else:
                run.log_table(OUTCOME_TABLE, rows, updates=update)

        artifact = tmp_path / "policy.bin"
        artifact.write_bytes(b"checkpoint at update one")
        run.checkpoint(artifact, updates=1)
    (checkpoint,) = read(tmp_path, "corpus", run.name).checkpoints
    assert training_corpus(checkpoint, source=tmp_path).project("games").fetchall() == [
        (1,)
    ]

    def corpus_sql(update: int, sql: str) -> list[tuple[object, ...]]:
        corpus = training_corpus(read(tmp_path, "corpus", run.name), update)
        return corpus.query("corpus", sql).fetchall()

    assert corpus_sql(1, "select play, draw, games from corpus") == [
        (island.identity, island.identity, 1)
    ]
    assert corpus_sql(
        1,
        "select unnest(['lea.forest', 'lea.island']) except select unnest(play_cards || draw_cards) from corpus",
    ) == [("lea.forest",)]
    assert corpus_sql(
        2,
        "select sum(games) from corpus where list_contains(play_cards || draw_cards, 'lea.island')",
    ) == [(2,)]
    assert corpus_sql(
        2,
        """select update_number, deck, sum(games) from
        (select update_number, unnest([play, draw]) as deck, games from corpus)
        group by all order by update_number, deck""",
    ) == [
        (1, island.identity, 2),
        *((2, deck, 1) for deck in sorted([island.identity, forest.identity])),
    ]
    # An attempt can rewind to an update and omit its old outcomes.
    with recording(
        experiment="corpus",
        run=run.name,
        directory=tmp_path,
        clocks=["updates"],
        resume=True,
        config=config,
    ) as resumed:
        resumed.log({"loss": 1}, updates=2)
    with pytest.raises(UnknownCoverage, match="coverage unknown at update 2"):
        corpus_sql(2, "select count(*) from corpus")
    assert corpus_sql(1, "select sum(games) from corpus") == [(1,)]


def test_logged_empty_outcomes_are_distinct_from_missing_evidence(
    tmp_path: Path,
) -> None:
    with recording(experiment="empty", directory=tmp_path, clocks=["updates"]) as run:
        run.log_table(
            ROSTER_TABLE,
            corpus_decks([Deck({"Island": 10})], writer_catalog()),
            updates=0,
        )
        run.log_table(OUTCOME_TABLE, [], updates=1)
        run.log({"loss": 1}, updates=2)
    data = read(tmp_path, "empty", run.name)
    assert training_corpus(data, 1).fetchall() == []
    with pytest.raises(UnknownCoverage, match="coverage unknown at update 2"):
        training_corpus(data, 2)
