"""Matchup strata change real setups and preserve exact collector recovery."""

from collections import Counter

import pytest

from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.model.world import checkpoint_world, validate_agent_setup
from manabot.sim.matchup import selected_match
from manabot.training.rollout.matchups import stream_matches
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
    agent = ActorCritic(space, ModelConfig(hidden_dim=8, num_attention_heads=2))
    agent.world_binding = checkpoint_world(base.to_rust(), space)
    mirror = stream_matches(base, 12, "mirrors-balanced")[0]
    with pytest.raises(ValueError, match="setup differs"):
        validate_agent_setup(agent, mirror.to_rust())
    validate_agent_setup(agent, mirror.to_rust(), allow_deck_repetition=True)
    changed = mirror.config.model_copy(deep=True)
    changed.hero_sideboard = {}
    with pytest.raises(ValueError, match="outside checkpoint"):
        validate_agent_setup(
            agent, Match(changed).to_rust(), allow_deck_repetition=True
        )


def test_authored_mirrors_keep_complete_content_manifest() -> None:
    engine = managym.Env(seed=125)
    engine.reset(Match(selected_match()).to_rust())
    expected = engine.content_pack_manifest()
    for deck in ("ur_lessons", "gw_allies"):
        mirror = Match(MatchConfig.authored("ur-lessons-vs-gw-allies", deck, deck))
        engine.reset(mirror.to_rust())
        assert engine.content_pack_manifest() == expected
        engine.reset(mirror.swapped().to_rust())
        assert engine.content_pack_manifest() == expected
