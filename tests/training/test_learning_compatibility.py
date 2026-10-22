"""Frozen recipe bytes keep their identity while runtime dispatch is descriptive."""

import json
from pathlib import Path

import pytest

from manabot.arena.models import canonical_sha256
from manabot.training.models import DampedPolicyLearning, TrainingRegime, TrainSelfPlay


def test_frozen_recipe_round_trip_preserves_digest_and_one_seat() -> None:
    raw = json.loads(Path("manabot/training/presets/ataraxos-mtg-v1.json").read_text())
    recipe = TrainingRegime.model_validate(raw)
    for stage in recipe.stages:
        assert isinstance(stage, TrainSelfPlay)
        assert isinstance(stage.learning, DampedPolicyLearning)
        assert stage.learning.gradient == "damped_policy"
        assert stage.learning_seats == "one"
    assert recipe.model_dump(mode="json") == raw
    assert canonical_sha256(recipe.model_dump(mode="json")) == (
        "43e7ece151480504a795f3d355c792080d8088215151bf9fe24d64e6c22166cb"
    )
    assert (
        TrainingRegime.model_validate_json(recipe.model_dump_json()).model_dump() == raw
    )


def test_input_alias_preserves_math_without_becoming_runtime_dispatch() -> None:
    old = DampedPolicyLearning.model_validate({"gradient": "ataraxos_move"})
    new = DampedPolicyLearning(gradient="damped_policy")
    for iteration in (1, 25, 10000):
        assert old.rates(iteration) == new.rates(iteration)
    assert old.gradient == new.gradient == "damped_policy"
    assert old.model_copy(deep=True).model_dump()["gradient"] == "ataraxos_move"
    assert new.model_dump()["gradient"] == "damped_policy"
    assert old.model_dump(exclude={"gradient"}) == new.model_dump(exclude={"gradient"})
    with pytest.raises(ValueError):
        DampedPolicyLearning.model_validate({"gradient": "unknown"})
