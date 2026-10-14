"""Matchup strata change real setups and preserve exact collector recovery."""

from collections import Counter

import numpy as np
import pytest
import torch

from experiments.runners.experiment_mirrors import recipe
from manabot.arena.match import selected_match
from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentSpec, MatchHypers, RewardHypers
from manabot.model.agent import Agent
from manabot.model.world import checkpoint_world, validate_agent_setup
from manabot.sim.net_opponent import SeatRoutedCollector
from manabot.training.matchups import stream_matches
from manabot.training.models import TrainingRegime
import managym


def test_strata_balance_decks_and_play_order() -> None:
    base = Match(selected_match())
    for curriculum in ("cross-balanced", "mirrors-balanced"):
        matches = stream_matches(base, 12, curriculum)
        exposures: Counter[tuple[str, int]] = Counter()
        cells: Counter[tuple[str, str]] = Counter()
        for i, match in enumerate(matches):
            decks = [
                "Lessons" if p.decklist == base.to_rust()[0].decklist else "Allies"
                for p in match.to_rust()
            ]
            exposures[(decks[i % 2], i % 2)] += 1
            cells[tuple(decks)] += 1
        assert set(exposures.values()) == {3}
        if curriculum == "mirrors-balanced":
            assert cells[("Lessons", "Lessons")] == 4
            assert cells[("Allies", "Allies")] == 4
        else:
            assert not any(a == b for a, b in cells)


def test_mirror_admission_requires_exact_roster_and_explicit_option() -> None:
    base = Match(selected_match())
    space = ObservationSpace()
    agent = Agent(space, AgentSpec(hidden_dim=8, num_attention_heads=2))
    agent.world_binding = checkpoint_world(base.to_rust(), space)
    mirror = stream_matches(base, 12, "mirrors-balanced")[0]
    with pytest.raises(ValueError, match="setup differs"):
        validate_agent_setup(agent, mirror.to_rust())
    validate_agent_setup(agent, mirror.to_rust(), allow_deck_repetition=True)
    changed = mirror.hypers.model_copy(deep=True)
    changed.hero_sideboard = {}
    with pytest.raises(ValueError, match="outside checkpoint"):
        validate_agent_setup(
            agent, Match(changed).to_rust(), allow_deck_repetition=True
        )


def test_native_mixed_collector_replays_terminal_resets() -> None:
    torch.set_num_threads(1)
    match = Match(selected_match())
    space = ObservationSpace()
    agent = Agent(
        space,
        AgentSpec(
            hidden_dim=8, num_attention_heads=2, semantic_pack="ur-lessons-vs-gw-allies"
        ),
    )

    def collector() -> SeatRoutedCollector:
        return SeatRoutedCollector(
            space,
            match,
            Reward(RewardHypers()),
            num_envs=12,
            seed=1250,
            opponent_mode="self",
            recovery_max_microsteps=100000,
            matchup_curriculum="mirrors-balanced",
        )

    first = collector()
    first.collect(agent, 128)
    assert first.stats.games > 0
    restored = collector()
    restored.restore(first.snapshot(), lambda: None)
    left, right = first.collect(agent, 8), restored.collect(agent, 8)
    np.testing.assert_array_equal(left.actions, right.actions)
    assert first.stats.games == restored.stats.games


def test_authored_mirrors_keep_complete_content_manifest() -> None:
    engine = managym.Env(seed=125)
    engine.reset(Match(selected_match()).to_rust())
    expected = engine.content_pack_manifest()
    for deck in ("ur_lessons", "gw_allies"):
        mirror = Match(MatchHypers.authored("ur-lessons-vs-gw-allies", deck, deck))
        engine.reset(mirror.to_rust())
        assert engine.content_pack_manifest() == expected
        engine.reset(mirror.swapped().to_rust())
        assert engine.content_pack_manifest() == expected
        wrong = mirror.to_rust()
        wrong[0].content_pack = None
        with pytest.raises(RuntimeError, match="explicit pack"):
            engine.reset(wrong)


def test_linked_stages_cannot_change_curriculum() -> None:
    value = recipe("cross-balanced", 100, 3600).model_dump()
    value["stages"][1]["matchup_curriculum"] = "mirrors-balanced"
    with pytest.raises(ValueError, match="preserve streams, curriculum"):
        TrainingRegime.model_validate(value)
