"""Temur compiled admission and current-world learning observations."""

from pathlib import Path

import numpy as np

from manabot.env import Env, Reward
from manabot.env.match import Match
from manabot.env.observation import ObservationEncoder, ObservationSpace
<<<<<<< HEAD
from manabot.infra.hypers import MatchHypers, ObservationSpaceHypers, RewardHypers
=======
from manabot.infra.config import MatchConfig, ObservationConfig, RewardConfig
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
from manabot.semantic.compiler import compile_paths, pretty_json
from manabot.semantic.learning import BoundSemanticPack, LearningSchema
import managym

BASE = Path("content/semantic/v1")


def test_temur_artifacts_and_learning_catalog_close_the_registered_cards() -> None:
    ir, coverage = compile_paths(
        BASE / "temur.source.json", BASE / "temur.fixtures.json"
    )
    assert (BASE / "generated/temur.ir.json").read_text() == pretty_json(ir)
    assert (BASE / "generated/temur.coverage.json").read_text() == pretty_json(coverage)
    assert coverage["definition_count"] == 9
    assert coverage["admission_closure_complete"]
    env = managym.Env(seed=163, skip_trivial=False)
    env.reset(
        [
            managym.authored_deck_setup("elemental-temur-v1", key)
            for key in ("temur_left", "temur_right")
        ]
    )
    binding = BoundSemanticPack.bind(env.content_pack_manifest())
    assert binding.schema.raw["schema_version"] == 5
    assert binding.pad_catalog()
    assert len(binding.definition_row_by_card_def_id) == 9
    assert (
        len(LearningSchema.load("content/semantic/w8/learning_schema.json").opcodes)
        == 55
    )


def test_temur_finishes_through_the_production_training_observation_path() -> None:
<<<<<<< HEAD
    match = MatchHypers.authored("elemental-temur-v1", "temur_left", "temur_right")
    env = Env(Match(match), ObservationSpace(), Reward(RewardHypers()), seed=163)
=======
    match = MatchConfig.authored("elemental-temur-v1", "temur_left", "temur_right")
    env = Env(Match(match), ObservationSpace(), Reward(RewardConfig()), seed=163)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    observation, _ = env.reset(seed=163)
    assert observation["agent_player"].shape[-1] == 40
    assert observation["agent_cards"].shape[-1] == 41
    rng = np.random.default_rng(163)
    for _ in range(5000):
        actions = np.flatnonzero(observation["actions_valid"])
        observation, _, terminal, truncated, info = env.step(int(rng.choice(actions)))
        assert not truncated
        assert not info["action_space_truncated"]
        assert not info["card_space_truncated"]
        assert not info["permanent_space_truncated"]
        if terminal:
            return
    raise AssertionError("compiled Temur game did not terminate")


def test_temur_restricted_mana_and_legendary_flags_encode_identically() -> None:
    env = managym.Env(seed=163, skip_trivial=False)
    env.reset(
        [
            managym.authored_deck_setup("elemental-temur-v1", key)
            for key in ("temur_left", "temur_right")
        ]
    )
    env.scenario_force_battlefield(0, "Hermitic Herbalist")
    env.scenario_force_battlefield(0, "Iroh, Grand Lotus")
    observation = env.scenario_refresh()
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
    assert any(card.is_legendary for card in observation.agent_cards)
<<<<<<< HEAD
    expected = ObservationEncoder(ObservationSpaceHypers()).encode(observation)
=======
    expected = ObservationEncoder(ObservationConfig()).encode(observation)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    actual = env.encode_observation(observation)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value, err_msg=key)
    assert actual["agent_player"][0, 37] == np.float32(0.2)
