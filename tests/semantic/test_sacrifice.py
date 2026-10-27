"""Compiled sacrifice admission and typed learning data for the owned card pool."""

from pathlib import Path

import numpy as np

from manabot.env import Env, Reward
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
<<<<<<< HEAD
from manabot.infra.hypers import MatchHypers, RewardHypers
=======
from manabot.infra.config import MatchConfig, RewardConfig
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
from manabot.semantic.compiler import compile_paths, pretty_json
from manabot.semantic.learning import BoundSemanticPack, LearningSchema
import managym

BASE = Path("content/semantic/v1")


def test_sacrifice_artifacts_close_the_owned_cards_and_tokens() -> None:
    ir, coverage = compile_paths(
        BASE / "sacrifice.source.json", BASE / "sacrifice.fixtures.json"
    )
    assert (BASE / "generated/sacrifice.ir.json").read_text() == pretty_json(ir)
    assert (BASE / "generated/sacrifice.coverage.json").read_text() == pretty_json(
        coverage
    )
    assert coverage["definition_count"] == 27
    assert coverage["admission_closure_complete"]
    assert coverage["no_card_name_dispatch"]


def test_sacrifice_live_manifest_binds_every_learning_program() -> None:
    env = managym.Env(seed=162, skip_trivial=False)
    env.reset(
        [
            managym.authored_deck_setup("elemental-sacrifice-v1", key)
            for key in ("sacrifice_left", "sacrifice_right")
        ]
    )
    manifest = env.content_pack_manifest()
    binding = BoundSemanticPack.bind(manifest)
    assert binding.schema.raw["schema_version"] == 4
    assert len(binding.definition_row_by_card_def_id) == 27
    padded = binding.pad_catalog()
    assert padded
    # Existing artifacts retain exactly their original opcode vocabulary.
    assert (
        len(LearningSchema.load("content/semantic/w7/learning_schema.json").opcodes)
        == 50
    )


def test_sacrifice_game_uses_the_training_observation_path() -> None:
<<<<<<< HEAD
    match = MatchHypers.authored(
        "elemental-sacrifice-v1", "sacrifice_left", "sacrifice_right"
    )
    env = Env(Match(match), ObservationSpace(), Reward(RewardHypers()), seed=162)
=======
    match = MatchConfig.authored(
        "elemental-sacrifice-v1", "sacrifice_left", "sacrifice_right"
    )
    env = Env(Match(match), ObservationSpace(), Reward(RewardConfig()), seed=162)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    observation, _ = env.reset(seed=162)
    rng = np.random.default_rng(162)
    for _ in range(5000):
        actions = np.flatnonzero(observation["actions_valid"])
        observation, _, terminal, truncated, info = env.step(int(rng.choice(actions)))
        assert not truncated
        assert not info["action_space_truncated"]
        assert not info["card_space_truncated"]
        assert not info["permanent_space_truncated"]
        if terminal:
            return
    raise AssertionError("compiled sacrifice game did not terminate")
