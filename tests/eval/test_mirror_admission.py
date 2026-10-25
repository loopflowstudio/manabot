"""Exact authored-roster admission for out-of-training mirror evaluation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from manabot.env import Match, ObservationSpace
from manabot.eval.match import MatchupMode, play_cell
from manabot.eval.monitor import EvaluationRow
from manabot.infra.config import MatchConfig
from manabot.model.world import checkpoint_world, validate_agent_setup
from manabot.sim.matchup import selected_decks, selected_match
from managym import Env
from tests.eval.test_evaluation_match import selected_players


def test_selected_leg_layout_preserves_cross_setups() -> None:
    match = Match(selected_match())
    for leg in range(8):
        decks = selected_decks(leg)
        actual = MatchConfig.authored(
            "ur-lessons-vs-gw-allies", *decks, hero=match.hero, villain=match.villain
        )
        if leg < 4:
            expected = match if leg < 2 else match.swapped()
            actual.hero, actual.villain = expected.hero, expected.villain
            assert actual.model_dump() == expected.config.model_dump()
        else:
            deck = "ur_lessons" if leg < 6 else "gw_allies"
            assert decks == (deck, deck)
            assert actual.content_pack == "ur-lessons-vs-gw-allies"
    for leg in (-1, 8):
        with pytest.raises(ValueError, match="leg"):
            selected_decks(leg)


def test_authored_mirrors_keep_compiled_manifest_and_strict_roster() -> None:
    base = Match(selected_match())
    binding = checkpoint_world(base.to_rust(), ObservationSpace())
    agent = SimpleNamespace(world_binding=binding)
    engine = Env(seed=131)
    engine.reset(base.to_rust())
    expected = engine.content_pack_manifest()
    for deck in ("ur_lessons", "gw_allies"):
        mirror = Match(MatchConfig.authored("ur-lessons-vs-gw-allies", deck, deck))
        with pytest.raises(ValueError, match="setup differs"):
            validate_agent_setup(agent, mirror.to_rust())
        validate_agent_setup(agent, mirror.to_rust(), allow_deck_repetition=True)
        engine.reset(mirror.to_rust())
        assert engine.content_pack_manifest() == expected
        engine.reset(mirror.swapped().to_rust())
        assert engine.content_pack_manifest() == expected

        configs = mirror.to_rust()
        configs[0].content_pack = None
        with pytest.raises(RuntimeError, match="explicit pack"):
            engine.reset(configs)
        with pytest.raises(ValueError, match="explicit content pack"):
            validate_agent_setup(agent, configs, allow_deck_repetition=True)

        changed = mirror.config.model_copy(deep=True)
        changed.hero_deck["Island"] = changed.hero_deck.get("Island", 0) + 1
        with pytest.raises(ValueError, match="outside checkpoint"):
            validate_agent_setup(
                agent, Match(changed).to_rust(), allow_deck_repetition=True
            )
        if deck == "ur_lessons":
            changed = mirror.config.model_copy(deep=True)
            changed.hero_sideboard = {}
            with pytest.raises(ValueError, match="outside checkpoint"):
                validate_agent_setup(
                    agent, Match(changed).to_rust(), allow_deck_repetition=True
                )


@pytest.mark.parametrize("mode", ["mirrors", "all"])
def test_mirror_arena_plays_all_seats_and_replays(
    tmp_path: Path, mode: MatchupMode
) -> None:
    key, (first, second) = selected_players()
    rows, _, replay = play_cell(
        key=key,
        player_a=first,
        player_b=second,
        deal_seeds=(1913131102,),
        out_dir=tmp_path,
        matchup_mode=mode,
    )
    legs = range(8) if mode == "all" else range(4, 8)
    assert {row["leg"] for row in rows} == set(legs)
    assert {(tuple(row["seat_decks"]), row["player_a_seat"]) for row in rows} == {
        (selected_decks(leg), leg % 2) for leg in legs
    }
    assert all(EvaluationRow.model_validate(row).valid for row in rows)
    assert replay["passed"]


def test_mirror_checkpoint_requires_explicit_admission(tmp_path: Path) -> None:
    key, (first, second) = selected_players()
    # Copy only to test the call's rejection before a worker/checkpoint is opened.
    checkpoint = first.model_copy(update={"runner_kind": "checkpoint"})
    with pytest.raises(ValueError, match="explicit checkpoint"):
        play_cell(
            key=key,
            player_a=checkpoint,
            player_b=second,
            deal_seeds=(131,),
            out_dir=tmp_path,
            matchup_mode="mirrors",
        )
    assert not list(tmp_path.iterdir())
