"""Training contracts, credit boundaries, and experiment controls."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.direct_self_play import direct_self_play
from manabot.backends.torch.rl import selected_rows
from manabot.policy.prepare import reference_distribution
from manabot.training.learning import DampedPolicyLearning
from manabot.training.records import StageRecord, TrainingRun
from manabot.training.regime import TrainingRegime
from manabot.training.validation import validate_regime
from manabot.verify.store import VerifyStore
from managym import WORLD_VERSION

ROOT = Path(__file__).resolve().parents[2]


def recipe() -> TrainingRegime:
    value = direct_self_play()
    value.world = WORLD_VERSION
    value.agent.world = "ur-lessons-vs-gw-allies"
    return value


def test_reference_has_full_support_and_is_permutation_invariant():
    obs = {
        "actions_valid": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        "actions": torch.tensor(
            [
                [
                    [1.0, *([0.0] * 15), 1.0, 1.0],
                    [1.0, *([0.0] * 15), 0.0, 1.0],
                    [0.0, 1.0, *([0.0] * 14), 0.0, 1.0],
                    [0.0] * 18,
                ]
            ]
        ),
    }
    obs = {key: value.numpy() for key, value in obs.items()}
    probs = reference_distribution(obs, "action_type_uniform")
    np.testing.assert_allclose(probs, [[0.25, 0.25, 0.5, 0.0]], rtol=1e-6)
    permutation = np.array([2, 0, 3, 1])
    moved = reference_distribution(
        {k: v[:, permutation] for k, v in obs.items()}, "action_type_uniform"
    )
    np.testing.assert_allclose(moved, probs[:, permutation], rtol=1e-6)
    assert selected_rows(torch.zeros(4), 0.25, 0.01).numel() == 0


def test_store_exports_committed_run_and_preserves_old_schema(tmp_path):
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = TrainingRun(
            id="fixture",
            regime=recipe(),
            regime_digest="test",
            seed=1,
            seed_streams={},
            identities={},
            stages=[StageRecord(id="policy-0")],
        )
        store.save_training_run(run)
        run.status = "failed"
        assert store.training_run(run.id).status == "pending"
        store.save_training_run(run)
        assert store.training_run(run.id).status == "failed"
        assert store.con.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_modified_models_are_revalidated_before_execution():
    value = recipe()
    value.stages[0].transitions = 0
    with pytest.raises(ValueError):
        validate_regime(value)


def test_legacy_learning_discriminator_is_rejected() -> None:
    with pytest.raises(ValueError):
        DampedPolicyLearning.model_validate({"gradient": "ataraxos_move"})
    rule = DampedPolicyLearning(gradient="damped_policy")
    assert rule.model_dump()["gradient"] == "damped_policy"


def test_frozen_regimes_read_their_pack_selectors_as_world() -> None:
    value = json.loads(
        (ROOT / "experiments/regimes/elemental-long-183-v1.json").read_text()
    )
    assert value["match"]["content_pack"] == "elemental-world-v2"
    regime = TrainingRegime.model_validate(value)
    assert regime.match is not None
    assert regime.match.world == regime.agent.world == "elemental-world-v2"
    assert regime.match.model_dump(mode="json")["world"] == "elemental-world-v2"
