"""Temur setup: learning catalog binding and training observations."""

import numpy as np

from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import (
    CARD_DIM,
    PLAYER_DIM,
    ObservationSpace,
    column,
)
from manabot.infra.config import MatchConfig, ObservationConfig, RewardConfig
from manabot.semantic.learning import BoundSemanticPack, LearningSchema
from manabot.verify.text_observation import priority_root
import managym
from tests.fixtures.observation_rows import observation_arrays


def test_temur_learning_catalog_closes_the_registered_cards() -> None:
    env = managym.Env(seed=163, skip_trivial=False)
    env.reset(
        [
            managym.authored_deck_setup("elemental-temur-v1", key)
            for key in ("temur_left", "temur_right")
        ]
    )
    binding = BoundSemanticPack.bind(env.content_pack_manifest())
    assert binding.schema.raw == LearningSchema.load().raw
    assert binding.pad_catalog()
    assert len(binding.definition_row_by_card_def_id) == managym.CARD_DEFINITION_COUNT


def test_temur_finishes_through_the_production_training_observation_path() -> None:
    match = MatchConfig.authored("elemental-temur-v1", "temur_left", "temur_right")
    space = ObservationSpace()
    env = Env(Match(match), space, Reward(RewardConfig()), seed=163)
    observation, _ = env.reset(seed=163)
    encoded = observation_arrays(space, observation)
    assert encoded["agent_player"].shape[-1] == PLAYER_DIM
    assert encoded["agent_cards"].shape[-1] == CARD_DIM
    rng = np.random.default_rng(163)
    for _ in range(5000):
        action_count = len(observation.action_space.actions)
        observation, _, terminal, truncated, info = env.step(
            int(rng.integers(action_count))
        )
        assert not truncated
        assert not info["action_space_truncated"]
        assert not info["card_space_truncated"]
        assert not info["permanent_space_truncated"]
        if terminal:
            return
    raise AssertionError("Temur game did not terminate")


def test_temur_restricted_mana_and_legendary_flags_encode() -> None:
    env = priority_root(
        (), ("Hermitic Herbalist", "Iroh, Grand Lotus"), mana_floats="all"
    )
    observation = env.observation_for_player(env.current_agent_index())
    activation = next(
        i
        for i, action in enumerate(observation.action_space.actions)
        if action.ability_index == 1
    )
    observation = env.step(activation)[0]
    red_pair = next(
        i
        for i, action in enumerate(observation.action_space.actions)
        if action.mode_index == 12
    )
    observation = env.step(red_pair)[0]
    assert observation.agent.lesson_mana[3] == 2
    iroh = next(
        i
        for i, permanent in enumerate(observation.agent_permanents)
        if permanent.characteristics.name == "Iroh, Grand Lotus"
    )
    assert observation.agent_permanents[iroh].characteristics.is_legendary
    space = ObservationSpace(ObservationConfig())
    encoded = observation_arrays(space, observation)
    assert encoded["agent_player"][
        0, column("agent_player", "lesson_mana_red")
    ] == np.float32(0.2)
    assert (
        encoded["agent_permanents"][iroh, column("agent_permanents", "is_legendary")]
        == 1
    )
