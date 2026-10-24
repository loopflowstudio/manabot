"""GIH semantics, retained evidence admission, and real Etude completion/resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from etude.eval.match import Row, Tables, evaluate
from etude.eval.records import Agent, Environment, Match
from etude.run.storage import DirectoryStorage
from pydantic import JsonValue
import pytest
import torch

from manabot.arena import card_evaluation as evaluation, cards as cards_cli
from manabot.arena.card_evaluation import evaluate_cards, publish_cards
from manabot.arena.card_evidence import (
    CAPTURE_VERSION,
    METRIC_VERSION,
    ArenaGameTrace,
    CapturedCards,
    CardMatchError,
    GameEvidence,
    capture_verified_game,
    read_evidence,
    reduce_cards,
    retain_evidence,
)
from manabot.arena.card_query import (
    CardQuery,
    CardRanking,
    Completeness,
    evidence_refs,
    rank_cards,
    summarize_cards,
)
from manabot.arena.card_schedule import CardConditions, admit_card_match
from manabot.arena.cards import verification_spec
from manabot.arena.match import selected_decks
from manabot.arena.models import PlayerRegistration, canonical_sha256, file_sha256
from manabot.arena.replay import ArenaReplayReceipt
from manabot.env import Match as EnvMatch, ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint


def card_fixture(
    *, winner: int | None = 0, leg: int = 0, deal: int = 1
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
        hero="player-one",
        villain="player-two",
        hero_deck={"Mountain": 3},
        villain_deck={"Forest": 2},
    )
    data = dict(
        match_id=f"fixture-{deal}-{leg}",
        deal_seed=deal,
        leg=leg,
        seat_players=["player-one", "player-two"],
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
    replay = ArenaReplayReceipt(games=1).to_dict()
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
            trace=ArenaGameTrace(data=data),
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
    names = tuple(agent.name for agent in match.agents)
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


def _fixture_tables(games: list[GameEvidence], match: Match, root: Path) -> Tables:
    tables: dict[str, list[Row]] = {"card_games": [], "card_evidence": []}
    names = tuple(agent.name for agent in match.agents)
    for game in games:
        ref = retain_evidence(root, game)
        tables["card_games"].extend(
            row.model_dump() for row in reduce_cards(game, names)
        )
        for player in (0, 1):
            tables["card_evidence"].append(
                dict(
                    player=player,
                    **ref.model_dump(),
                    capture_version=CAPTURE_VERSION,
                    metric_version=METRIC_VERSION,
                )
            )
    return tables


def test_copy_weighted_gih_distinguishes_draw_tutor_bounce_and_redraw() -> None:
    game = card_fixture()
    rows = reduce_cards(game, ("player-two", "player-one"))
    red = next(r for r in rows if r.name == "Mountain")
    assert (
        red.player,
        red.deck_copies,
        red.opening_copies,
        red.drawn_copies,
        red.gih_copies,
    ) == (1, 3, 2, 2, 4)
    assert red.win == 1
    loss = next(
        r
        for r in reduce_cards(
            card_fixture(winner=1, deal=2), ("player-two", "player-one")
        )
        if r.name == "Mountain"
    )
    # One observed copy in the losing game: 4/5, not 1/2 game presence.
    loss = loss.model_copy(
        update={"opening_copies": 1, "drawn_copies": 0, "gih_copies": 1}
    )
    result = summarize_cards(
        [red, loss],
        CardQuery(run="fixture", player="player-one"),
        "one",
        "env",
        Completeness(
            scheduled=2,
            completed=2,
            failed=0,
            pending=0,
            scheduled_games=2,
            failed_games=0,
        ),
        {red.game: "two", loss.game: "two"},
    )
    assert result.cards[0].rate == 0.8
    assert result.cards[0].qualifying_copies == 5
    assert result.cards[0].games == 2
    assert result.cards[0].interval is not None
    assert result.opponent_mix == {"two": 2}


def test_draws_zero_denominator_and_missing_facts_are_explicit(tmp_path: Path) -> None:
    game = card_fixture(winner=None)
    rows = reduce_cards(game, game.seat_players)
    assert all(not r.eligible and r.win is None for r in rows)
    result = summarize_cards(
        rows[:1],
        CardQuery(run="fixture", player="player-one"),
        "one",
        "env",
        Completeness(
            scheduled=1,
            completed=1,
            failed=0,
            pending=0,
            scheduled_games=1,
            failed_games=0,
        ),
        {game.game: "two"},
    )
    assert result.cards[0].rate is None
    assert result.excluded_outcomes == {"draw": 1}
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
        ("games", "complete eight-leg deal"),
        ("participants", "complete eight-leg deal"),
        ("name", "participant differs from registration"),
    ],
)
def test_execution_and_admission_reject_invalid_participant_schedule(
    corruption: str, error: str
) -> None:
    spec = verification_spec(8)
    agents = tuple(
        Agent(
            name=player.player_id,
            kind="code",
            artifact="a" * 64,
            config={"registration": player.model_dump(mode="json")},
        )
        for player in spec.players
    )
    if corruption == "participants":
        agents = agents[:1]
    elif corruption == "name":
        agents = (agents[0].model_copy(update={"name": "other-player"}), agents[1])
    match = Match(
        agents=agents,
        games=4 if corruption == "games" else 8,
        conditions=CardConditions(
            deal=spec.deals[0],
            key=spec.key,
            authored_matchup=spec.authored_matchup,
            game_seconds=spec.game_seconds,
            max_commands=spec.max_commands,
        ).model_dump(mode="json"),
    )
    # Both paths reject the schedule before executing or loading any games.
    with pytest.raises(ValueError, match=error):
        evaluation.play_card_match(match)
    with pytest.raises(ValueError, match=error):
        admit_card_match([], match, Environment(name="fixture", files="a" * 64))


def test_native_evaluation_resume_query_and_delivered_rebuild(tmp_path: Path) -> None:
    spec = verification_spec(8)
    result = evaluate_cards(spec, tmp_path / "source")
    assert result.played == 1
    query = CardQuery(run=spec.run, player=spec.players[0].player_id)
    ranking = rank_cards(tmp_path / "source", query)
    assert ranking.complete and ranking.cards
    assert ranking.completeness.completed == 1
    assert sum(ranking.opponent_mix.values()) == 8
    resumed = evaluate_cards(spec, tmp_path / "source")
    assert (resumed.played, resumed.skipped) == (0, 1)
    filtered = rank_cards(
        tmp_path / "source",
        query.model_copy(update={"on_play": True, "deck": "elemental_ug"}),
    )
    assert sum(filtered.opponent_mix.values()) == 2
    refs = evidence_refs(tmp_path / "source", spec.run)
    assert len(refs) == 8
    # Every command/frame/state/outcome still agrees when capture is enabled.
    for ref in refs[:1]:
        game = read_evidence(
            (tmp_path / "source" / spec.run / ref.path).read_bytes(), ref
        )
        assert capture_verified_game(game.trace) == game
    target = DirectoryStorage(tmp_path / "delivered")
    # Etude discovers nested runs; evidence must use the same discovered prefix.
    publish_cards(tmp_path, target)
    assert rank_cards(target, query) == ranking
    ref = refs[0]
    (tmp_path / "delivered" / spec.run / ref.path).unlink()
    with pytest.raises(FileNotFoundError):
        rank_cards(target, query)


@pytest.mark.parametrize("known_failure", [False, True])
def test_interrupted_schedule_and_partial_query_use_etude_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, known_failure: bool
) -> None:
    """No games/inference: exercise real SDK records, failed resume and scalar storage."""
    spec = verification_spec(16)
    assert spec.authored_matchup is not None
    runtime = runtime_fingerprints(match_config=spec.authored_matchup.match())
    attempts: list[int] = []
    fail = True

    def play(match: Match) -> Tables:
        nonlocal fail
        deal = int(match.conditions["deal"])
        attempts.append(deal)
        if len(attempts) == 2 and fail:
            fail = False
            if known_failure:
                raise CardMatchError({"timeout": 1, "failed_match": 7})
            raise RuntimeError("interrupted fixture")
        root = Path(evaluation.os.environ["MANABOT_CARD_OUTPUT"])
        return _fixture_tables(
            [_scheduled_fixture(match, leg, runtime) for leg in range(8)], match, root
        )

    monkeypatch.setattr(evaluation, "play_card_match", play)
    with pytest.raises(Exception, match="match_failed"):
        evaluate_cards(spec, tmp_path)
    query = CardQuery(run=spec.run, player=spec.players[0].player_id)
    with pytest.raises(ValueError, match="schedule incomplete"):
        rank_cards(tmp_path, query)
    partial = rank_cards(tmp_path, query.model_copy(update={"partial": True}))
    assert not partial.complete
    assert partial.excluded_outcomes == (
        {"timeout": 1, "failed_match": 7} if known_failure else {"unobserved": 8}
    )
    assert (
        partial.completeness.scheduled,
        partial.completeness.completed,
        partial.completeness.failed,
    ) == (2, 1, 1)
    retried = evaluate_cards(spec, tmp_path)
    assert (retried.played, retried.skipped) == (1, 1)
    complete = rank_cards(tmp_path, query)
    assert complete.complete and sum(complete.opponent_mix.values()) == 16
    assert len(attempts) == 3
    # Replicated delivery contributes each completed match exactly once.
    publish_cards(tmp_path, DirectoryStorage(tmp_path / "replica"))
    assert rank_cards(tmp_path, query) == complete
    with pytest.raises(ValueError, match="exactly one entrant"):
        rank_cards(tmp_path, query.model_copy(update={"player": "absent"}))


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("deal", "scheduled deal"),
        ("duplicate_leg", "every scheduled leg"),
        ("seat", "seat/deck assignment"),
        ("deck_label", "seat/deck assignment"),
        ("deck_setup", "seat/deck assignment"),
        ("content_digest", "runtime/content identity"),
        ("content_manifest_sha256", "runtime/content identity"),
        ("engine_source_sha256", "runtime/content identity"),
        ("world", "runtime/content identity"),
        ("observation_abi_sha256", "runtime setup/ABI"),
        ("matchup_sha256", "runtime setup/ABI"),
    ],
)
def test_schedule_admission_rejects_consistent_misassociation_after_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str, error: str
) -> None:
    spec = verification_spec(8)
    assert spec.authored_matchup is not None
    runtime = runtime_fingerprints(match_config=spec.authored_matchup.match())

    def play(match: Match) -> Tables:
        games = [_scheduled_fixture(match, leg, runtime) for leg in range(8)]
        raw = games[1].model_dump(mode="json", by_alias=True)
        if corruption == "deal":
            raw["deal"] += 1
        elif corruption == "duplicate_leg":
            # Keep eight unique game IDs but repeat an entire seat/deck leg.
            raw = games[0].model_dump(mode="json", by_alias=True)
            raw["game"] = games[1].game
        elif corruption == "seat":
            raw["seat_players"].reverse()
            raw["setup"].update(
                hero=raw["seat_players"][0], villain=raw["seat_players"][1]
            )
        elif corruption == "deck_label":
            raw["seat_decks"].reverse()
        elif corruption == "deck_setup":
            raw["setup"]["hero_deck"] = {"Mountain": 60}
        else:
            raw["trace"]["data"]["runtime_identities"][corruption] = "b" * 64
            if corruption == "content_digest":
                raw["cards"]["capture"]["content_digest"] = "b" * 64
        games[1] = _rebind_fixture(raw)
        # Reducer rows and immutable hashes agree with the misassociated artifact.
        return _fixture_tables(
            games, match, Path(evaluation.os.environ["MANABOT_CARD_OUTPUT"])
        )

    monkeypatch.setattr(evaluation, "play_card_match", play)
    source = tmp_path / "source"
    evaluate_cards(spec, source)
    delivered = DirectoryStorage(tmp_path / "delivered")
    publish_cards(source, delivered)
    for store in (DirectoryStorage(source), delivered):
        for participant in spec.players:
            with pytest.raises(ValueError, match=error):
                rank_cards(store, CardQuery(run=spec.run, player=participant.player_id))


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
    ranking = rank_cards(
        tmp_path / "evaluation", CardQuery(run=spec.run, player=candidate.player_id)
    )
    assert ranking.complete and sum(ranking.opponent_mix.values()) == 8
    assert any(card.qualifying_copies > card.games for card in ranking.cards)


def test_uncertainty_keeps_the_same_deal_across_opponents_together() -> None:
    template = reduce_cards(card_fixture(), ("player-one", "player-two"))[0]
    rows = [
        template.model_copy(
            update={"game": f"{deal}-{opponent}", "deal": deal, "win": opponent}
        )
        for deal in (1, 2)
        for opponent in (0, 1)
    ]
    ranking = summarize_cards(
        rows,
        CardQuery(run="fixture", player="one"),
        "one",
        "env",
        Completeness(
            scheduled=4,
            completed=4,
            failed=0,
            pending=0,
            scheduled_games=4,
            failed_games=0,
        ),
        {r.game: r.game.split("-")[1] for r in rows},
    )
    assert ranking.cards[0].rate == 0.5
    assert ranking.cards[0].interval == (0.5, 0.5)
    assert ranking.opponent_mix == {"0": 2, "1": 2}


@pytest.mark.parametrize("outcome", ["crash", "timeout", "truncated", "replay_failed"])
def test_invalid_outcomes_never_receive_win_rate_credit(outcome: str) -> None:
    game = card_fixture().model_copy(update={"outcome": outcome})
    rows = reduce_cards(game, game.seat_players)
    assert all(not row.eligible and row.win is None for row in rows)


def test_historical_outcome_only_evaluation_reports_unavailable(tmp_path: Path) -> None:
    def outcome_only(match: Match) -> Tables:
        return {"plays": [{"player": 0, "score": 1.0}]}

    evaluate(
        experiment="old",
        environment=Environment(name="old", files="a" * 64),
        matches=[Match(agents=(Agent(name="one", kind="code", artifact="b" * 64),))],
        play=outcome_only,
        run="old",
        directory=tmp_path,
    )
    with pytest.raises(ValueError, match="card evidence unavailable"):
        rank_cards(tmp_path, CardQuery(run="old", player="one"))


def test_scale_verifier_rejects_second_participant_table_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = verification_spec(8)
    game = card_fixture()

    def evaluated(spec: evaluation.CardEvaluationSpec, directory: Path) -> None:
        pass

    def ranking(source: Path, query: CardQuery) -> CardRanking:
        if query.player == spec.players[1].player_id:
            raise ValueError("card tables differ from retained evidence")
        return summarize_cards(
            reduce_cards(game, game.seat_players)[:1],
            query,
            query.player,
            "fixture",
            Completeness(
                scheduled=1,
                completed=1,
                failed=0,
                pending=0,
                scheduled_games=8,
                failed_games=0,
            ),
            {game.game: spec.players[1].player_id},
        )

    monkeypatch.setattr(cards_cli, "evaluate_cards", evaluated)
    monkeypatch.setattr(cards_cli, "rank_cards", ranking)
    # The second subject's corrupt table must fail before trace verification.
    with pytest.raises(ValueError, match="card tables differ from retained evidence"):
        cards_cli.verify(8, tmp_path)
