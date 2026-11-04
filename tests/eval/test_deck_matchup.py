"""Imported corpus pairs use the shared supervised four-leg evaluator."""

from pathlib import Path

from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.infra.identity import canonical_sha256
from manabot.sim.matchup import DeckMatchup
from manabot.sim.replay import read_trace
import managym
from tests.eval.arena_fixture import arena_fixture


def test_imported_pair_swaps_decks_sideboards_policies_and_replays(
    tmp_path: Path,
) -> None:
    matchup = DeckMatchup(
        population_sha256="a" * 64,
        entries=("event-a", "event-b"),
        lists=(
            managym.Deck({"Plains": 10}, {"Island": 1}),
            managym.Deck({"Mountain": 10}, {"Forest": 2}),
        ),
    )
    contract = arena_fixture()
    key = contract.key.model_copy(
        update={"world": managym.WORLD_VERSION, "content_suite": matchup.content_suite}
    )
    players = [
        p.model_copy(
            update={
                "world": managym.WORLD_VERSION,
                "content_suite": matchup.content_suite,
                "matchup_sha256": canonical_sha256(matchup.match().model_dump()),
            }
        )
        for p in contract.anchors[:2]
    ]
    request = EvaluationRequest(
        key=key,
        candidate=players[0],
        opponent=players[1],
        deal_seeds=(21390001,),
        authored_matchup=matchup,
        game_seconds=30,
    )
    assert isinstance(
        EvaluationRequest.model_validate_json(
            request.model_dump_json()
        ).authored_matchup,
        DeckMatchup,
    )
    result = evaluate(request, out_dir=tmp_path / "evaluation")
    assert result.status == "completed", result.error
    assert len(result.rows) == 4 and all(row.valid for row in result.rows)
    cells = {
        (
            row.model_extra["seat_decks"][row.model_extra["player_a_seat"]],
            row.model_extra["player_a_seat"],
        )
        for row in result.rows
    }
    assert cells == {(event, seat) for event in matchup.entries for seat in (0, 1)}
    games = read_trace(tmp_path / "evaluation" / result.rows[0].trace_path)
    assert len(games) == 4
    for game in games:
        setup = matchup.match(game["leg"])
        assert game["match_hypers"]["hero_sideboard"] == setup.hero_sideboard
        assert game["match_hypers"]["villain_sideboard"] == setup.villain_sideboard
        assert game["decisions"]

    capped = evaluate(
        request.model_copy(update={"max_commands": 1}),
        out_dir=tmp_path / "capped",
    )
    assert capped.status == "incomplete" and capped.score is None
    assert len(capped.rows) == 4
    assert all(
        row.failure == "game exceeded its Command cap" and not row.valid
        for row in capped.rows
    )
    assert all(
        row.model_extra["termination_reason"] == "command_cap" for row in capped.rows
    )
