"""Exact authored-roster admission for out-of-training mirror evaluation."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from test_match import selected_players

from manabot.arena.match import play_cell, selected_decks, selected_match
from manabot.env import Match, ObservationSpace
from manabot.infra.hypers import MatchHypers
from manabot.model.world import checkpoint_world, validate_agent_setup
from manabot.training.monitor_evaluation import ArenaRow
from managym import Env


def test_selected_leg_layout_preserves_cross_setups() -> None:
    match = Match(selected_match())
    for leg in range(8):
        decks = selected_decks(leg)
        actual = MatchHypers.authored(
            "ur-lessons-vs-gw-allies", *decks, hero=match.hero, villain=match.villain
        )
        if leg < 4:
            expected = match if leg < 2 else match.swapped()
            actual.hero, actual.villain = expected.hero, expected.villain
            assert actual.model_dump() == expected.hypers.model_dump()
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
        mirror = Match(MatchHypers.authored("ur-lessons-vs-gw-allies", deck, deck))
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

        changed = mirror.hypers.model_copy(deep=True)
        changed.hero_deck["Island"] = changed.hero_deck.get("Island", 0) + 1
        with pytest.raises(ValueError, match="outside checkpoint"):
            validate_agent_setup(
                agent, Match(changed).to_rust(), allow_deck_repetition=True
            )
        if deck == "ur_lessons":
            changed = mirror.hypers.model_copy(deep=True)
            changed.hero_sideboard = {}
            with pytest.raises(ValueError, match="outside checkpoint"):
                validate_agent_setup(
                    agent, Match(changed).to_rust(), allow_deck_repetition=True
                )


def test_mirror_only_arena_plays_all_seats_and_replays(tmp_path: Path) -> None:
    key, (first, second) = selected_players()
    rows, _, replay = play_cell(
        key=key,
        player_a=first,
        player_b=second,
        deal_seeds=(1913131102,),
        out_dir=tmp_path,
        matchup_mode="mirrors",
    )
    assert {row["leg"] for row in rows} == {4, 5, 6, 7}
    assert {(tuple(row["seat_decks"]), row["player_a_seat"]) for row in rows} == {
        (tuple([deck] * 2), seat)
        for deck in ("ur_lessons", "gw_allies")
        for seat in (0, 1)
    }
    assert all(ArenaRow.model_validate(row).valid for row in rows)
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
