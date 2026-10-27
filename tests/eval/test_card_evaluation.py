"""Card counts, per-game admission, experiment pooling and the card data page."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
import time
from typing import Any

import duckdb
from etude.eval.match import Tables, evaluate
from etude.eval.query import statsdb
from etude.eval.records import Code, Environment, Match
from etude.run.declaration import declare
from etude.run.delivery import sync
from etude.run.storage import DirectoryStorage
from pydantic import JsonValue
import pytest
import torch

from manabot.env.match import Match as EnvMatch
from manabot.env.observation import ObservationSpace
from manabot.eval import card_evaluation as evaluation, cards as cards_cli
from manabot.eval.card_evaluation import card_tables, evaluate_cards, trace_card_tables
from manabot.eval.card_metrics import (
    CARD_FIELDS,
    COUNT_FIELDS,
    PLAY_FIELDS,
    CardGame,
    normalize,
    reduce_cards,
)
from manabot.eval.card_page import page_data, write_page
from manabot.eval.card_query import (
    AuditResult,
    CardData,
    CardSelection,
    audit,
    card_data,
    load_game,
)
from manabot.eval.card_schedule import CardConditions, admit_card_runtime
from manabot.eval.cards import verification_spec
from manabot.eval.match import play_cell
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint
from manabot.sim.card_evidence import (
    CapturedCards,
    GameEvidence,
    GameTrace,
    capture_verified_game,
    read_evidence,
    retain_evidence,
)
from manabot.sim.execution import EvalExecution
from manabot.sim.matchup import selected_decks
from manabot.sim.registration import PlayerRegistration
from manabot.sim.replay import ReplayReceipt


def card_fixture(
    *,
    winner: int | None = 0,
    leg: int = 0,
    deal: int = 1,
    players: tuple[str, str] = ("player-one", "player-two"),
) -> GameEvidence:
    def card(
        copy: int, owner: int, zone: str, incarnation: int = 0
    ) -> dict[str, object]:
        return dict(
            copy=copy,
            owner=owner,
            definition=owner + 1,
            name="Mountain" if owner == 0 else "Forest",
            zone=zone,
            incarnation=incarnation,
            origin="deck",
        )

    initial = [
        card(0, 0, "Hand"),
        card(1, 0, "Hand"),
        card(2, 0, "Library"),
        card(3, 1, "Hand"),
        card(4, 1, "Library"),
    ]
    moves = []
    facts = [
        (2, 0, "Library", "Hand", "draw", 1),
        (2, 0, "Hand", "Battlefield", "move", 2),
        (2, 0, "Battlefield", "Hand", "return", 3),
        (2, 0, "Hand", "Library", "move", 4),
        (2, 0, "Library", "Hand", "selection", 5),
        (2, 0, "Hand", "Library", "move", 6),
        (2, 0, "Library", "Hand", "draw", 7),
        (4, 1, "Library", "Hand", "draw", 1),
    ]
    final = {c["copy"]: c for c in initial}
    for sequence, (copy, owner, from_, to, cause, incarnation) in enumerate(facts):
        c = card(copy, owner, to, incarnation)
        moves.append(
            dict(
                sequence=sequence,
                card=c,
                controller=owner,
                cause=cause,
                turn=sequence + 1,
                **{"from": from_},
            )
        )
        final[copy] = c
    capture = CapturedCards.model_validate(
        dict(
            capture=dict(
                version=1,
                content_digest="a" * 64,
                initial=initial,
                opening=[[0, 1], [3]],
                moves=moves,
            ),
            final_cards=list(final.values()),
            turn=8,
        )
    )
    setup = MatchConfig(
        hero=players[0],
        villain=players[1],
        hero_deck={"Mountain": 3},
        villain_deck={"Forest": 2},
    )
    data = dict(
        match_id=f"fixture-{deal}-{leg}",
        deal_seed=deal,
        leg=leg,
        seat_players=list(players),
        seat_decks=["red", "green"],
        match_hypers=setup.model_dump(),
        winner=winner,
        terminated=True,
        truncated=False,
        termination_reason="life_total",
        decisions=[],
        runtime_identities={"content_digest": capture.capture.content_digest},
    )
    data["game_trace_sha256"] = canonical_sha256(data)
    replay = ReplayReceipt(games=1).to_dict()
    return GameEvidence.model_validate(
        dict(
            game=data["match_id"],
            deal=deal,
            leg=leg,
            seat_players=data["seat_players"],
            seat_decks=["red", "green"],
            setup=setup,
            winner=winner,
            outcome="draw" if winner is None else "win_loss",
            trace=GameTrace(data=data),
            trace_sha256=data["game_trace_sha256"],
            replay=replay,
            replay_sha256=canonical_sha256(replay),
            cards=capture,
        )
    )


def _scheduled_fixture(
    match: Match, leg: int, runtime: dict[str, JsonValue]
) -> GameEvidence:
    """Synthetic metric facts with explicit schedule provenance; no replay claim."""
    conditions = CardConditions.model_validate(match.conditions)
    game = card_fixture(deal=conditions.deal, leg=leg)
    names = tuple(player.name for player in match.players)
    seats = names if leg % 2 == 0 else names[::-1]
    matrix = conditions.authored_matchup
    decks = matrix.decks(leg) if matrix else selected_decks(leg)
    setup = (
        matrix.match(leg)
        if matrix
        else MatchConfig.authored("ur-lessons-vs-gw-allies", *decks)
    ).model_copy(update={"hero": seats[0], "villain": seats[1]})
    raw = game.model_dump(mode="json", by_alias=True)
    raw.update(
        seat_players=list(seats), seat_decks=list(decks), setup=setup.model_dump()
    )
    raw["cards"]["capture"]["content_digest"] = runtime["content_digest"]
    raw["trace"]["data"]["runtime_identities"] = {
        **runtime,
        "matchup": setup.model_dump(),
        "matchup_sha256": canonical_sha256(setup.model_dump()),
    }
    return _rebind_fixture(raw)


def _rebind_fixture(raw: dict[str, Any]) -> GameEvidence:
    """Rehash an internally consistent artifact, so only schedule binding can reject it."""
    # Pydantic's model_dump boundary permits deliberate nested fixture edits;
    # validation below restores the typed evidence contract.
    trace = raw["trace"]["data"]
    trace.update(
        match_id=raw["game"],
        deal_seed=raw["deal"],
        leg=raw["leg"],
        seat_players=raw["seat_players"],
        seat_decks=raw["seat_decks"],
        match_hypers=raw["setup"],
    )
    trace.pop("game_trace_sha256")
    trace["game_trace_sha256"] = canonical_sha256(trace)
    raw["trace_sha256"] = trace["game_trace_sha256"]
    return GameEvidence.model_validate(raw)


def _fixture_play(
    runtime: dict[str, JsonValue],
    change: dict[int, dict[str, Any]] | None = None,
) -> Callable[[Match], Tables]:
    """A `play` that retains synthetic games for every leg, edited per leg."""

    def play(match: Match) -> Tables:
        root = Path(evaluation.os.environ["MANABOT_CARD_OUTPUT"])
        games = []
        for leg in range(8):
            game = _scheduled_fixture(match, leg, runtime)
            if change and leg in change:
                raw = game.model_dump(mode="json", by_alias=True)
                raw.update(change[leg])
                game = _rebind_fixture(raw)
            games.append((retain_evidence(root, game), game))
        return card_tables(match, games)

    return play


@pytest.mark.parametrize("native_column", ["absent", "mixed", "corrupt"])
def test_audit_preserves_card_position_before_game_seat_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_column: str
) -> None:
    spec = verification_spec(8)
    assert spec.authored_matchup is not None
    fixture = _fixture_play(
        runtime_fingerprints(match_config=spec.authored_matchup.match())
    )

    def play(match: Match) -> Tables:
        tables = fixture(match)
        for row in tables["card_plays"]:
            if native_column == "absent":
                row.pop("game_seat")
            elif native_column == "mixed":
                # Represent old rows unioned with newly written native seats.
                if row["on_play"]:
                    row["game_seat"] = None
            else:
                row["game_seat"] = 1 if row["on_play"] else 0
        return tables

    monkeypatch.setattr(evaluation, "play_card_match", play)
    evaluate_cards(spec, tmp_path)
    with statsdb(spec.experiment, tmp_path) as db:
        result = audit(db, tmp_path)
    assert result.games == 8
    assert len(result.problems) == (8 if native_column == "corrupt" else 0)


def _db(
    matches: Sequence[tuple[str, tuple[str, str], list[CardGame]]],
) -> duckdb.DuckDBPyConnection:
    """The statsdb tables card queries read, built straight from reducer rows."""
    db = duckdb.connect()
    db.execute(
        "create table card_plays (run varchar, match varchar, seat bigint, game_seat bigint, game varchar,"
        " deal bigint, deck varchar, opponent_deck varchar, on_play boolean,"
        " eligible boolean, outcome varchar, win bigint, content varchar,"
        " metric_version varchar, source_commit varchar)"
    )
    db.execute(
        "create table card_counts (run varchar, match varchar, seat bigint, game varchar,"
        " card varchar, deck_copies bigint, opening_copies bigint, drawn_copies bigint,"
        " gih_copies bigint)"
    )
    db.execute(
        "create table cards (run varchar, match varchar, seat bigint, card varchar,"
        " content varchar, definition bigint, name varchar)"
    )
    db.execute(
        "create table seats (run varchar, match varchar, seat bigint, player varchar)"
    )
    db.execute(
        "create table players (player varchar, name varchar, kind varchar,"
        " checkpoint varchar, artifact varchar,"
        " uri varchar, config json)"
    )
    db.execute("create table matches (run varchar, match varchar, state varchar)")
    names = {name for _, pair, _ in matches for name in pair}
    for name in names:
        db.execute(
            "insert into players values (?, ?, 'code', null, ?, null, null)",
            [f"id-{name}", name, "a" * 64],
        )
    for match, pair, rows in matches:
        db.execute("insert into matches values ('run', ?, 'done')", [match])
        for seat, name in enumerate(pair):
            db.execute(
                "insert into seats values ('run', ?, ?, ?)",
                [match, seat, f"id-{name}"],
            )
        normal = normalize(rows)
        for play in normal.plays:
            db.execute(
                "insert into card_plays values ('run', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'abc')",
                [match, *(play[field] for field in PLAY_FIELDS)],
            )
        for count in normal.counts:
            db.execute(
                "insert into card_counts values ('run', ?, ?, ?, ?, ?, ?, ?, ?)",
                [match, *(count[field] for field in COUNT_FIELDS)],
            )
        for card in normal.cards:
            db.execute(
                "insert into cards values ('run', ?, ?, ?, ?, ?, ?)",
                [match, *(card[field] for field in CARD_FIELDS)],
            )
    return db


def test_copy_weighted_gih_distinguishes_draw_tutor_bounce_and_redraw() -> None:
    game = card_fixture()
    rows = reduce_cards(game, ("player-two", "player-one"))
    red = next(r for r in rows if r.name == "Mountain")
    assert (
        red.seat,
        red.deck_copies,
        red.opening_copies,
        red.drawn_copies,
        red.gih_copies,
    ) == (1, 3, 2, 2, 4)
    assert red.win == 1
    losing = reduce_cards(card_fixture(winner=1, deal=2), ("player-two", "player-one"))
    # One observed copy in the losing game: 4/5 of copies, not 1/2 of games.
    losing = [
        r.model_copy(update={"opening_copies": 1, "drawn_copies": 0, "gih_copies": 1})
        if r.name == "Mountain"
        else r
        for r in losing
    ]
    db = _db([("one", ("two", "one"), rows), ("two", ("two", "one"), losing)])
    data = card_data(db, CardSelection(experiment="fixture", subjects=("one",)))
    mountain = next(c for c in data.cards if c.name == "Mountain")
    assert mountain.gih.rate == 0.8 and mountain.gih.count == 5
    assert (mountain.oh.count, mountain.oh.rate) == (3, 2 / 3)
    assert (mountain.gd.count, mountain.gd.rate) == (2, 1.0)
    assert (mountain.gp.count, mountain.gp.rate) == (2, 0.5)
    assert mountain.gns.count == 0 and mountain.iwd is None
    assert mountain.gih_interval is not None
    assert data.opponents == {"two": 2}
    assert data.subjects == {"id-one": "one"}


def test_draws_zero_denominator_and_missing_facts_are_explicit(tmp_path: Path) -> None:
    game = card_fixture(winner=None)
    rows = reduce_cards(game, game.seat_players)
    assert all(not r.eligible and r.win is None for r in rows)
    db = _db([("one", ("player-one", "player-two"), rows)])
    data = card_data(db, CardSelection(experiment="fixture", subjects=("player-one",)))
    assert data.cards == () and data.games == 0
    assert data.excluded == {"draw": 1}
    ref = retain_evidence(tmp_path, game)
    assert read_evidence((tmp_path / ref.path).read_bytes(), ref) == game
    assert retain_evidence(tmp_path, game) == ref
    with pytest.raises(ValueError, match="bytes differ"):
        read_evidence(b"broken", ref)
    raw = game.cards.model_dump(mode="json", by_alias=True)
    raw["capture"]["moves"].pop(0)
    with pytest.raises(ValueError, match="sequence"):
        CapturedCards.model_validate(raw)
    raw = game.cards.model_dump(mode="json", by_alias=True)
    raw["capture"]["moves"][0]["cause"] = "unknown"
    with pytest.raises(ValueError):
        CapturedCards.model_validate(raw)


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("games", "eight legs per deal"),
        ("participants", "eight legs per deal"),
        ("name", "participant differs from registration"),
    ],
)
def test_execution_rejects_invalid_participant_schedule(
    corruption: str, error: str
) -> None:
    spec = verification_spec(8)
    entrants = tuple(
        Code(
            name=player.player_id,
            config={"registration": player.model_dump(mode="json")},
        )
        for player in spec.players
    )
    if corruption == "participants":
        entrants = entrants[:1]
    elif corruption == "name":
        entrants = (
            entrants[0].model_copy(update={"name": "other-player"}),
            entrants[1],
        )
    match = Match(
        players=entrants,
        games=4 if corruption == "games" else 8,
        conditions=CardConditions(
            deal=spec.deals[0],
            key=spec.key,
            authored_matchup=spec.authored_matchup,
            game_seconds=spec.game_seconds,
            max_commands=spec.max_commands,
        ).model_dump(mode="json"),
    )
    with pytest.raises(ValueError, match=error):
        evaluation.play_card_match(match)


def test_native_evaluation_resumes_delivers_and_audits(tmp_path: Path) -> None:
    spec = verification_spec(8)
    source = tmp_path / "source"
    result = evaluate_cards(spec, source)
    assert result.played == 1
    selection = CardSelection(
        experiment=spec.experiment, subjects=(spec.players[0].player_id,)
    )
    with statsdb(spec.experiment, source) as db:
        data = card_data(db, selection)
        assert data.cards and sum(data.opponents.values()) == 8
        filtered = card_data(
            db, selection.model_copy(update={"on_play": True, "deck": "elemental_ug"})
        )
        assert sum(filtered.opponents.values()) == 2
        checked = audit(db, source, replay=True)
        assert (checked.games, checked.replayed, checked.problems) == (8, 8, ())
        run, game_id, path = db.execute(
            "select run, game, evidence from card_plays order by game limit 1"
        ).fetchone()
        # Capture leaves every command, frame, state and outcome unchanged.
        game = load_game(db, source, run, game_id)
        assert capture_verified_game(game.trace) == game
    resumed = evaluate_cards(spec, source)
    assert (resumed.played, resumed.skipped) == (0, 1)
    # Ordinary Etude delivery carries the evidence with the tables.
    delivered = tmp_path / "delivered"
    assert sync(source, storages=[DirectoryStorage(delivered)]) == {}
    with statsdb(spec.experiment, delivered) as db:
        assert card_data(db, selection) == data
        assert audit(db, delivered).problems == ()
        next(delivered.rglob(Path(path).name)).unlink()
        assert len(audit(db, delivered).problems) == 1


@pytest.mark.parametrize("outcome", ["crash", "timeout", "truncated", "replay_failed"])
def test_all_excluded_games_have_no_card_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    spec = verification_spec(8)
    assert spec.authored_matchup is not None
    runtime = runtime_fingerprints(match_config=spec.authored_matchup.match())
    monkeypatch.setattr(
        evaluation,
        "play_card_match",
        _fixture_play(runtime, {leg: {"outcome": outcome} for leg in range(8)}),
    )
    evaluate_cards(spec, tmp_path)
    selection = CardSelection(
        experiment=spec.experiment, subjects=(spec.players[0].player_id,)
    )
    with statsdb(spec.experiment, tmp_path) as db:
        # Etude omits a column when every retained value is null.
        assert "win" not in {
            row[0] for row in db.execute("describe card_plays").fetchall()
        }
        data = card_data(db, selection)
        assert (data.games, data.deals, data.win_rate, data.cards) == (0, 0, None, ())
        assert data.excluded == {outcome: 8}
        assert sum(data.opponents.values()) == 8
        assert data.subjects and data.runs == (spec.run,)
        assert data.failed_matches == 0
        empty = card_data(db, selection.model_copy(update={"deck": "absent"}))
        assert (empty.games, empty.win_rate, empty.cards, empty.excluded) == (
            0,
            None,
            (),
            {},
        )


def test_failed_games_are_exclusions_and_failed_matches_are_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = verification_spec(16)
    assert spec.authored_matchup is not None
    runtime = runtime_fingerprints(match_config=spec.authored_matchup.match())
    fixture = _fixture_play(
        runtime, {3: {"outcome": "crash"}, 5: {"outcome": "timeout"}}
    )
    calls = 0

    def play(match: Match) -> Tables:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted fixture")
        return fixture(match)

    monkeypatch.setattr(evaluation, "play_card_match", play)
    with pytest.raises(Exception, match="match_failed"):
        evaluate_cards(spec, tmp_path)
    selection = CardSelection(
        experiment=spec.experiment, subjects=(spec.players[0].player_id,)
    )
    with statsdb(spec.experiment, tmp_path) as db:
        data = card_data(db, selection)
    # The crash and timeout leave six eligible games of their deal, not zero.
    assert data.games == 6 and data.excluded == {"crash": 1, "timeout": 1}
    assert data.failed_matches == 1
    retried = evaluate_cards(spec, tmp_path)
    assert (retried.played, retried.skipped) == (1, 1)
    with statsdb(spec.experiment, tmp_path) as db:
        data = card_data(db, selection)
    assert data.games == 12 and data.failed_matches == 0
    with pytest.raises(ValueError, match="no player in this experiment"):
        with statsdb(spec.experiment, tmp_path) as db:
            card_data(db, selection.model_copy(update={"subjects": ("absent",)}))


def test_runs_pool_into_their_experiment_and_filters_narrow_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = verification_spec(8)
    assert spec.authored_matchup is not None
    runtime = runtime_fingerprints(match_config=spec.authored_matchup.match())
    monkeypatch.setattr(
        evaluation, "play_card_match", _fixture_play(runtime, {2: {"outcome": "crash"}})
    )
    commits = iter(["1" * 40, "2" * 40, "3" * 40])
    for run, deals, experiment in (
        ("first", (1,), spec.experiment),
        ("second", (2,), spec.experiment),
        ("elsewhere", (3,), "other-experiment"),
    ):
        commit = next(commits)
        monkeypatch.setattr(evaluation, "source_commit", lambda c=commit: c)
        evaluate_cards(
            spec.model_copy(
                update={"run": run, "deals": deals, "experiment": experiment}
            ),
            tmp_path,
        )
    subject = spec.players[0].player_id
    with statsdb(spec.experiment, tmp_path) as db:
        pooled = card_data(
            db, CardSelection(experiment=spec.experiment, subjects=(subject,))
        )
        assert pooled.runs == ("first", "second") and pooled.deals == 2
        assert pooled.games == 14 and pooled.excluded == {"crash": 2}
        one = card_data(
            db,
            CardSelection(
                experiment=spec.experiment, subjects=(subject,), commits=("2222",)
            ),
        )
        assert one.runs == ("second",) and one.games == 7
        drawn = card_data(
            db,
            CardSelection(
                experiment=spec.experiment, subjects=(subject,), on_play=False
            ),
        )
        # Etude orders players by player identity. Player 0 is on the play in
        # even legs, which hold the crashed leg 2.
        (player,) = db.execute(
            "select distinct s.seat from seats s join players a using (player)"
            " where a.name = ?",
            [subject],
        ).fetchone()
        assert drawn.games == (8 if player == 0 else 6)
        both = card_data(db, CardSelection(experiment=spec.experiment))
        assert both.games == 28 and len(both.subjects) == 2
    with statsdb("other-experiment", tmp_path) as db:
        assert card_data(db, CardSelection(experiment="other-experiment")).runs == (
            "elsewhere",
        )


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("deal", "scheduled deal"),
        ("seat", "seat/deck assignment"),
        ("deck_label", "seat/deck assignment"),
        ("deck_setup", "seat/deck assignment"),
        ("observation_abi_sha256", "runtime setup/ABI"),
        ("matchup_sha256", "runtime setup/ABI"),
    ],
)
def test_each_game_is_admitted_against_its_schedule(
    tmp_path: Path, corruption: str, error: str
) -> None:
    spec = verification_spec(8)
    assert spec.authored_matchup is not None
    runtime = runtime_fingerprints(match_config=spec.authored_matchup.match())
    match = Match(
        players=tuple(
            Code(
                name=p.player_id,
                config={"registration": p.model_dump(mode="json")},
            )
            for p in spec.players
        ),
        games=8,
        conditions=CardConditions(
            deal=spec.deals[0],
            key=spec.key,
            authored_matchup=spec.authored_matchup,
            game_seconds=spec.game_seconds,
            max_commands=spec.max_commands,
        ).model_dump(mode="json"),
    )
    raw = _scheduled_fixture(match, 1, runtime).model_dump(mode="json", by_alias=True)
    if corruption == "deal":
        raw["deal"] += 1
    elif corruption == "seat":
        raw["seat_players"].reverse()
        raw["setup"].update(hero=raw["seat_players"][0], villain=raw["seat_players"][1])
    elif corruption == "deck_label":
        raw["seat_decks"].reverse()
    elif corruption == "deck_setup":
        raw["setup"]["hero_deck"] = {"Mountain": 60}
    else:
        raw["trace"]["data"]["runtime_identities"][corruption] = "b" * 64
    game = _rebind_fixture(raw)
    with pytest.raises(ValueError, match=error):
        card_tables(match, [(retain_evidence(tmp_path, game), game)])


@pytest.mark.parametrize(
    "field",
    ["content_digest", "content_manifest_sha256", "engine_source_sha256", "world"],
)
def test_audit_binds_games_to_their_declared_runtime(field: str) -> None:
    spec = verification_spec(8)
    assert spec.authored_matchup is not None
    runtime = runtime_fingerprints(match_config=spec.authored_matchup.match())
    identity = {key: runtime[key] for key in evaluation.RUNTIME_FIELDS}
    environment = Environment(
        name="fixture",
        files=canonical_sha256(identity),
        config={**identity, "capture_version": 1, "metric_version": "gih-copies-v1"},
    )
    match = Match(
        players=tuple(
            Code(
                name=p.player_id,
                config={"registration": p.model_dump(mode="json")},
            )
            for p in spec.players
        ),
        games=8,
        conditions=CardConditions(
            deal=spec.deals[0],
            key=spec.key,
            authored_matchup=spec.authored_matchup,
            game_seconds=spec.game_seconds,
            max_commands=spec.max_commands,
        ).model_dump(mode="json"),
    )
    game = _scheduled_fixture(match, 0, runtime)
    admit_card_runtime(game, environment)
    raw = game.model_dump(mode="json", by_alias=True)
    raw["trace"]["data"]["runtime_identities"][field] = "b" * 64
    if field == "content_digest":
        raw["cards"]["capture"]["content_digest"] = "b" * 64
    with pytest.raises(ValueError, match="declared runtime/content identity"):
        admit_card_runtime(_rebind_fixture(raw), environment)


def test_loaded_evidence_rejects_capture_from_another_content(tmp_path: Path) -> None:
    game = card_fixture()
    capture = game.cards.capture.model_copy(update={"content_digest": "b" * 64})
    changed = game.model_copy(
        update={"cards": game.cards.model_copy(update={"capture": capture})}
    )
    ref = retain_evidence(tmp_path, changed)
    with pytest.raises(
        ValueError, match="captured content differs from retained runtime"
    ):
        read_evidence((tmp_path / ref.path).read_bytes(), ref)


def test_tiny_checkpoint_match_captures_verified_native_facts(tmp_path: Path) -> None:
    spec = verification_spec(8)
    assert spec.authored_matchup is not None
    torch.manual_seed(23)
    space = ObservationSpace()
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8, num_attention_heads=2, semantic_pack="elemental-world-v2"
        ),
    )
    path = tmp_path / "tiny.pt"
    save_checkpoint(
        agent,
        space,
        path,
        player_configs=EnvMatch(spec.authored_matchup.match()).to_rust(),
    )
    candidate = PlayerRegistration.model_validate(
        {
            **spec.players[0].model_dump(),
            "runner_kind": "checkpoint",
            "source_sha256": None,
            "role": "challenger",
            "checkpoint_sha256": file_sha256(path),
            "checkpoint_bytes": path.stat().st_size,
            "parameter_count": sum(p.numel() for p in agent.parameters()),
            "training_seed": 23,
            "artifact_id": "untrained-fixture-23",
            "evidence_class": "fixture",
            "player_spec": {
                "kind": "checkpoint",
                "deterministic": False,
                "device": "cpu",
                "batch_size": 1,
                "allow_deck_repetition": True,
            },
        }
    )
    spec = spec.model_copy(
        update={
            "players": (candidate, spec.players[1]),
            "checkpoints": {candidate.player_id: path},
        }
    )
    evaluate_cards(spec, tmp_path / "evaluation")
    with statsdb(spec.experiment, tmp_path / "evaluation") as db:
        # The checkpoint digest selects the subject, as an artifact would.
        data = card_data(
            db,
            CardSelection(
                experiment=spec.experiment, subjects=(file_sha256(path)[:12],)
            ),
        )
    assert sum(data.opponents.values()) + sum(data.excluded.values()) == 8
    # Some card had more qualifying copies than games in which a copy was seen.
    assert any(c.gih.count > c.gp.count - c.gns.count for c in data.cards)


def test_uncertainty_keeps_the_same_deal_across_opponents_together() -> None:
    matches = []
    for deal in (1, 2):
        for opponent in ("zero", "one"):
            winner = 0 if opponent == "zero" else 1
            game = card_fixture(
                winner=winner,
                deal=deal,
                leg=0 if opponent == "zero" else 2,
                players=("subject", opponent),
            )
            matches.append(
                (
                    f"{deal}-{opponent}",
                    ("subject", opponent),
                    reduce_cards(game, ("subject", opponent)),
                )
            )
    data = card_data(
        _db(matches), CardSelection(experiment="fixture", subjects=("subject",))
    )
    mountain = next(c for c in data.cards if c.name == "Mountain")
    assert mountain.gih.rate == 0.5
    assert mountain.gih_interval == (0.5, 0.5)
    assert data.opponents == {"zero": 2, "one": 2}


@pytest.mark.parametrize("outcome", ["crash", "timeout", "truncated", "replay_failed"])
def test_invalid_outcomes_never_receive_win_rate_credit(outcome: str) -> None:
    game = card_fixture().model_copy(update={"outcome": outcome})
    rows = reduce_cards(game, game.seat_players)
    assert all(not row.eligible and row.win is None for row in rows)


def test_outcome_only_experiment_has_no_card_data(tmp_path: Path) -> None:
    def outcome_only(match: Match) -> Tables:
        return {"plays": [{"seat": 0, "score": 1.0}]}

    evaluate(
        experiment="old",
        environment=Environment(name="old", files="a" * 64),
        matches=[Match(players=(Code(name="one"),))],
        play=outcome_only,
        run="old",
        directory=tmp_path,
    )
    with statsdb("old", tmp_path) as db:
        with pytest.raises(ValueError, match="no card evaluations"):
            card_data(db, CardSelection(experiment="old"))


def test_verifier_fails_when_the_audit_finds_a_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def evaluated(spec: evaluation.CardEvaluationSpec, directory: Path) -> None:
        pass

    def data(db: duckdb.DuckDBPyConnection, selection: CardSelection) -> CardData:
        return CardData(
            selection=selection,
            subjects={},
            opponents={},
            runs=(),
            games=8,
            deals=1,
            win_rate=0.5,
            excluded={},
            failed_matches=0,
            cards=(),
        )

    def audited(
        db: duckdb.DuckDBPyConnection, source: Path, *, replay: bool
    ) -> AuditResult:
        return AuditResult(
            games=8, replayed=8, problems=("run/game: stored card rows differ",)
        )

    monkeypatch.setattr(cards_cli, "evaluate_cards", evaluated)
    monkeypatch.setattr(cards_cli, "card_data", data)
    monkeypatch.setattr(cards_cli, "audit", audited)
    with pytest.raises(ValueError, match="stored card rows differ"):
        cards_cli.verify(8, tmp_path)


def test_card_page_embeds_every_filter_combination(tmp_path: Path) -> None:
    matches = []
    for deal, leg in ((1, 0), (2, 1), (3, 2)):
        pair = ("subject", "rival") if leg != 1 else ("rival", "subject")
        game = card_fixture(winner=deal % 2, deal=deal, leg=leg, players=pair)
        matches.append((f"m{deal}", pair, reduce_cards(game, pair)))
    db = _db(matches)
    selection = CardSelection(experiment="fixture <demo>", subjects=("subject",))
    data = page_data(db, selection)
    variants = data["variants"]
    assert isinstance(variants, dict)
    assert "||||" in variants and "subject|rival|play|red|green" in variants
    assert variants["||||"]["games"] == card_data(db, selection).games
    assert data["subjects"] == ["subject"] and data["opponents"] == ["rival"]
    path = write_page(db, selection, tmp_path / "cards.html")
    html = path.read_text()
    assert "<title>fixture &lt;demo&gt; · card data</title>" in html
    embedded = html.split('<script id="card-data" type="application/json">')[1].split(
        "</script>"
    )[0]
    assert json.loads(embedded)["variants"].keys() == variants.keys()


@pytest.mark.parametrize("shared", [False, True])
def test_any_evaluation_can_deliver_card_data_beside_its_own_tables(
    tmp_path: Path,
    shared: bool,
) -> None:
    """An eval that calls `play_cell` itself adds card tables with one call."""
    spec = verification_spec(8)
    first, second = spec.players
    root = declare(
        experiment="grid-like",
        clocks=(),
        run="grid",
        kind="evaluation",
        directory=tmp_path,
    ).root

    def play(match: Match) -> Tables:
        out = root / "games" / match.key
        out.mkdir(parents=True)
        rows, trace, _ = play_cell(
            key=spec.key,
            player_a=first,
            player_b=second,
            deal_seeds=(int(match.conditions["deal"]),),
            out_dir=out,
            game_seconds=spec.game_seconds,
            max_commands=spec.max_commands,
            authored_matchup=spec.authored_matchup,
            matchup_mode="all",
            legs=(0, 1),
            execution=EvalExecution(deadline_unix=time.time() + 120)
            if shared
            else None,
            card_evidence_dir=root,
        )
        own = {"games": [{"seat": 0, "score": float(r["score_a"] or 0)} for r in rows]}
        return {
            **own,
            **trace_card_tables(root, trace, (first.player_id, second.player_id)),
        }

    evaluate(
        experiment="grid-like",
        environment=Environment(name="grid", files="a" * 64),
        matches=[
            Match(
                players=(
                    Code(name=first.player_id),
                    Code(name=second.player_id),
                ),
                conditions={"deal": spec.deals[0]},
            )
        ],
        play=play,
        run="grid",
        directory=tmp_path,
    )
    with statsdb("grid-like", tmp_path) as db:
        data = card_data(
            db, CardSelection(experiment="grid-like", subjects=(first.player_id,))
        )
        assert data.games + sum(data.excluded.values()) == 2 and data.cards
        assert {r[0] for r in db.execute("show tables").fetchall()} >= {
            "games",
            "card_plays",
        }
        checked = audit(db, tmp_path)
        assert (checked.games, checked.problems) == (2, ())
        db.execute("update card_counts set gih_copies = gih_copies + 1")
        assert len(audit(db, tmp_path).problems) == 2
    with pytest.raises(ValueError, match="card_evidence_dir"):
        trace_card_tables(root, {}, (first.player_id, second.player_id))
