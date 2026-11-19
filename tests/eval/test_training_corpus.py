"""A training corpus follows the writer's cards, completed games and resume cuts."""

import json
from pathlib import Path

from etude.train.read import read
import pytest

from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.eval.training_corpus import UnknownCoverage, training_corpus
from manabot.infra.config import ModelConfig, RewardConfig
from manabot.model.writer_catalog import writer_catalog
from manabot.policy.checkpoint import write_safetensors
from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.corpus import log_corpus_outcomes, log_corpus_setup
from manabot.training.decks import UniformDecks, deck_cards
from manabot.training.loops.self_play import CLOCKS, self_play
from manabot.training.rollout.collection import close_collector, collect, open_collector
from manabot.training.rollout.self_play import GameSetting, SelfPlayCounter
from manabot.training.update_metrics import (
    DECK_TABLE,
    GAME_SETTINGS_TABLE,
    OUTCOME_TABLE,
    SEAT_LABELS,
    outcome_scalars,
)
from managym import WORLD_SETUP, Deck
from managym.statistics import CardCounts, GameStatistics
from tests.fixtures.training_records import recording

# The three questions the guide asks of a setting, as it prints them.
ALWAYS = """
    select setting, seat, min(value) from game_settings
    group by setting, seat having count(distinct value) = 1 order by all
"""
SHARE = """
    select setting, seat, value, sum(games) as games,
           sum(games) / sum(sum(games)) over (partition by setting, seat) as share
    from game_settings group by setting, seat, value order by all
"""
CHANGED = """
    select experiment, run, setting, seat, value,
           min(update_number) as first_update, max(update_number) as last_update,
           sum(games) as games
    from game_settings group by all order by setting, seat, experiment, run, first_update
"""
CLOSED = (
    GameSetting(None, "decklists", "closed"),
    GameSetting(None, "skip_trivial", "true"),
)


@pytest.mark.parametrize("stored", ["seat", "position", "scalar"])
def test_corpus_queries_use_completed_games_through_update(
    tmp_path: Path, stored: str
) -> None:
    """Every stored outcome form reads into the same relations: rows by native
    seat, earlier rows labelled `play` or `draw`, and scalar counts."""
    historical = stored == "scalar"
    island, forest = Deck({"Island": 10}), Deck({"Forest": 10}, {"Island": 1})
    catalog = writer_catalog()
    decks = [island, forest]
    config = (
        {
            "training": {
                "identities": {"content_digest": catalog["world_digest"]},
                "regime": {
                    "deck_source": {
                        "decks": [deck.to_dict() for deck in decks],
                        "decklists": "closed",
                    },
                    "observation": {"policy_input_version": 9, "max_actions": 64},
                    "stages": [
                        {"behavior": "current-self", "learning_seats": "one"},
                        {"behavior": "current-self", "learning_seats": "both"},
                    ],
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
            run.log_table(DECK_TABLE, deck_cards(decks, catalog), updates=0)
        counter = SelfPlayCounter()
        for update, pair in enumerate([(island, island), (forest, island)], 1):
            counter.record((pair[0].identity, pair[1].identity), None, CLOSED)
            games = counter.drain()
            if historical:
                run.log(outcome_scalars(games.outcomes), updates=update)
            else:
                rows = [
                    {**row, "position": SEAT_LABELS[row.pop("seat")]}
                    if stored == "position"
                    else row
                    for row in map(dict, games.outcomes)
                ]
                run.log_table(OUTCOME_TABLE, rows, updates=update)
                run.log_table(GAME_SETTINGS_TABLE, games.settings, updates=update)

        artifact = tmp_path / "policy.bin"
        artifact.write_bytes(b"checkpoint at update one")
        run.checkpoint(artifact, updates=1)
    (checkpoint,) = read(tmp_path, "corpus", run.name).checkpoints
    assert training_corpus(checkpoint, source=tmp_path).sql(
        "select games from corpus"
    ).fetchall() == [(1,)]

    def corpus_sql(update: int, sql: str) -> list[tuple[object, ...]]:
        corpus = training_corpus(read(tmp_path, "corpus", run.name), update)
        return corpus.sql(sql).fetchall()

    assert corpus_sql(
        1,
        "select policy_input_version, max_actions, behavior, learning_seats from runs",
    ) == [(9, 64, "current-self", None) if historical else (None,) * 4]
    # A fixed disclosure held in every game. An older run's declaration says
    # so for decklists only; what it never recorded has no rows.
    assert corpus_sql(2, ALWAYS) == [
        ("decklists", None, "closed"),
        *(() if historical else [("skip_trivial", None, "true")]),
    ]
    assert corpus_sql(
        2,
        "select update_number, games from game_settings where setting = 'decklists' order by all",
    ) == [(1, 1), (2, 1)]
    assert corpus_sql(1, "select board, card, copies from deck_cards order by all") == [
        ("main", "lea.forest", 10),
        ("main", "lea.island", 10),
        ("sideboard", "lea.island", 1),
    ]
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
            DECK_TABLE, deck_cards([Deck({"Island": 10})], writer_catalog()), updates=0
        )
        run.log_table(OUTCOME_TABLE, [], updates=1)
        run.log({"loss": 1}, updates=2)
    data = read(tmp_path, "empty", run.name)
    assert training_corpus(data, 1).sql("from corpus").fetchall() == []
    with pytest.raises(UnknownCoverage, match="coverage unknown at update 2"):
        training_corpus(data, 2)


def test_mixed_disclosure_reads_as_a_share_of_the_games(tmp_path: Path) -> None:
    source = UniformDecks(
        decks=tuple(
            Deck({name: 10}, {"Forest": 1}) for name in ("Plains", "Mountain", "Island")
        ),
        decklists="mixed",
    )
    space = ObservationSpace()
    collector = open_collector(
        space,
        Match(source.match_config()),
        Reward(RewardConfig()),
        num_envs=2,
        seed=17402,
        opponent_mode="self",
        games_per_stream=2,
        deck_source=source,
        numerics=CollectionSession(device="cpu", seed=17402, precision="float32"),
    )
    agent = ActorCritic(
        space, ModelConfig(hidden_dim=8, num_attention_heads=2, world=WORLD_SETUP)
    )
    try:
        with recording(
            experiment="mixed", directory=tmp_path, clocks=["updates"]
        ) as run:
            log_corpus_setup(
                run, collector, behavior="current-self", curriculum="fixed", updates=0
            )
            collect(collector, agent, 1)
            log_corpus_outcomes(run, collector, updates=1)
    finally:
        close_collector(collector)
    corpus = training_corpus(read(tmp_path, "mixed", run.name), 1)
    (games,) = corpus.sql("select sum(games) from corpus").fetchone()
    shares = corpus.sql(SHARE).fetchall()
    disclosure = {
        value: count for setting, _, value, count, _ in shares if setting == "decklists"
    }
    assert set(disclosure) == {"open", "closed"}
    assert sum(disclosure.values()) == games == collector.stats.games
    assert ("skip_trivial", None, "true", games, 1.0) in shares
    # Disclosure varied, so only the setting every game shared reads as always.
    assert corpus.sql(ALWAYS).fetchall() == [("skip_trivial", None, "true")]


def test_a_game_without_setting_counts_leaves_coverage_unknown(tmp_path: Path) -> None:
    island = Deck({"Island": 10})
    pair = island.identity, island.identity
    with recording(experiment="dropped", directory=tmp_path, clocks=["updates"]) as run:
        run.log_table(DECK_TABLE, deck_cards([island], writer_catalog()), updates=0)
        counter = SelfPlayCounter()
        counter.record(pair, 0, CLOSED)
        games = counter.drain()
        run.log_table(OUTCOME_TABLE, games.outcomes, updates=1)
        run.log_table(GAME_SETTINGS_TABLE, games.settings, updates=1)
        counter.record(pair, 0, CLOSED)
        counter.record(pair, 1)
        games = counter.drain()
        run.log_table(OUTCOME_TABLE, games.outcomes, updates=2)
        run.log_table(GAME_SETTINGS_TABLE, games.settings, updates=2)
    data = read(tmp_path, "dropped", run.name)
    assert training_corpus(data, 1).sql(ALWAYS).fetchall() == [
        ("decklists", None, "closed"),
        ("skip_trivial", None, "true"),
    ]
    with pytest.raises(UnknownCoverage, match="update 2 records .* for 1 of 2 games"):
        training_corpus(data, 2)


def test_card_statistics_count_terminal_games_at_the_outcome_update(
    tmp_path: Path,
) -> None:
    island = Deck({"Island": 10})
    catalog = writer_catalog()
    with recording(experiment="cards", directory=tmp_path, clocks=["updates"]) as run:
        run.log_table(DECK_TABLE, deck_cards([island], catalog), updates=0)
        counter = SelfPlayCounter()
        for update in (1, 2):
            counter.record((island.identity, island.identity), 0)
            run.log_table(OUTCOME_TABLE, counter.drain().outcomes, updates=update)
    data = read(tmp_path, "cards", run.name)
    path = tmp_path / "statistics"
    with pytest.raises(UnknownCoverage, match="no statistics rows"):
        training_corpus(data, 1, statistics=path)

    path.mkdir()

    def game(update: int, terminal: bool) -> str:
        return (
            GameStatistics(
                seed=1,
                generation=0,
                stream=0,
                terminal=terminal,
                winner=0,
                context={"update": update},
                card_digest=catalog["world_digest"],
                cards=(
                    CardCounts(
                        seat=0,
                        definition=catalog["definition_keys"].index("lea.island"),
                        name="Island",
                        deck_copies=10,
                        opening_copies=2,
                        drawn_copies=3,
                        opportunities=7,
                        selections=5,
                        resolutions=None,
                    ),
                ),
            ).model_dump_json()
            + "\n"
        )

    (path / "games.jsonl").write_text(game(0, True) * 2 + game(1, False))
    corpus = training_corpus(data, 1, statistics=path)
    assert corpus.sql(
        """
        select c.games, s.games, s.wins, s.drawn_copies, s.selections, s.resolutions
        from corpus c join card_statistics s using (update_number)
        where s.seat = 0 and list_contains(c.play_cards, s.card)
    """,
    ).fetchall() == [(1, 2, 2, 6, 10, None)]
    with pytest.raises(
        UnknownCoverage, match="update 2 has 0 terminal statistics games for 1 corpus"
    ):
        training_corpus(data, 2, statistics=path)


def test_pretrained_corpus_follows_the_published_parent_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = gray_ogre_self_play(horizon=2).model_copy(update={"checkpoint_every": 1})
    monkeypatch.delenv("ETUDE_EXPERIMENT", raising=False)
    monkeypatch.setenv("ETUDE_RUN", "parent")
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "parent"))
    with recording(
        experiment="first", run="parent", directory=tmp_path / "track", clocks=CLOCKS
    ) as run:
        final = self_play(settings, run, updates=2)
    parent = final.with_name("00000001-raw.safetensors")
    # The file names its run and update. This one was written with no
    # experiment in its environment, as files before the record were.
    track = tmp_path / "track"
    with pytest.raises(UnknownCoverage, match=r"records no Etude experiment \(pass"):
        training_corpus(parent, source=track)
    assert training_corpus(parent, source=track, experiment="first").sql(
        "select run, update_number, sum(games) from corpus group by all"
    ).fetchall() == [("parent", 1, 2)]
    # A string path reads as the file, and a storage without the run says so.
    with pytest.raises(UnknownCoverage, match="holds no run first/parent"):
        training_corpus(str(parent), source=tmp_path / "parent", experiment="first")
    unrecorded = tmp_path / "policy.safetensors"
    write_safetensors(
        unrecorded,
        {},
        {
            key: json.dumps({"commit": "7c1e9a2"} if key == "source" else {})
            for key in ("hypers", "world_binding", "architecture", "source")
        },
    )
    with pytest.raises(UnknownCoverage, match="records no Etude run, no update"):
        training_corpus(unrecorded, source=track, experiment="first")
    monkeypatch.setenv("ETUDE_EXPERIMENT", "second")
    monkeypatch.setenv("ETUDE_RUN", "child")
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "child"))
    with recording(
        experiment="second", run="child", directory=tmp_path / "track", clocks=CLOCKS
    ) as run:
        # The child plays with decklists closed; its parent played them open.
        closed = settings.match.model_copy(update={"decklists": "closed"})
        child = self_play(
            settings.model_copy(update={"pretrained": parent, "match": closed}),
            run,
            updates=1,
        )
    with pytest.raises(ValueError, match="records experiment 'second', not 'first'"):
        training_corpus(child, source=track, experiment="first")
    # A file written under Etude selects its whole corpus, parent included.
    corpus = training_corpus(child, source=track)
    assert corpus.sql(
        """select run, update_number, sum(games) from corpus
        group by all order by run""",
    ).fetchall() == [("child", 1, 2), ("parent", 1, 2)]
    assert corpus.sql(
        """select run, learning_seats, parent_run, parent_update from runs
        order by run""",
    ).fetchall() == [
        ("child", "both", "parent", 1),
        ("parent", "both", None, None),
    ]
    # Disclosure changed where the child took over from the parent's update 1.
    assert corpus.sql(CHANGED).fetchall() == [
        ("first", "parent", "decklists", None, "open", 1, 1, 2),
        ("second", "child", "decklists", None, "closed", 1, 1, 2),
        ("first", "parent", "skip_trivial", None, "true", 1, 1, 2),
        ("second", "child", "skip_trivial", None, "true", 1, 1, 2),
    ]
    assert corpus.sql(ALWAYS).fetchall() == [("skip_trivial", None, "true")]
    assert corpus.sql(
        """select distinct board, card from deck_cards where copies > 0
        order by card""",
    ).fetchall() == [
        ("main", key)
        for key in ("lea.forest", "lea.gray_ogre", "lea.llanowar_elves", "lea.mountain")
    ]
