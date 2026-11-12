"""The damped update, written once, agrees across every installed backend.

One rollout batch is collected with the torch policy; each backend
(`manabot.backends.parity.BACKENDS`) then loads the same weights through the
shared container and runs `update_damped_iteration` on that batch. The torch
backend is the reference. With the whole batch in one minibatch every backend
measures its loss and gradient at the same weights, and the bars are the
parity check's: loss and gradient norm within 0.1%, policy terms within 1e-4.
With several minibatches the later ones start from weights that already
differ by rounding (Adam turns a rounding-sized gradient into a full step), so
that run checks the row and step counts exactly and the loss within 1%. A
backend whose framework is absent is skipped.
"""

from collections.abc import Mapping
from types import ModuleType

import numpy as np
import pytest
import torch

from manabot.backends.parity import (
    BACKENDS,
    LOGIT_TOLERANCE,
    LOSS_TOLERANCE,
    backend_module,
    installed_backends,
)
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.policy.checkpoint import (
    PolicyCheckpoint,
    canonical_parameters,
    from_torch_state,
)
from manabot.training.regime import DampedPolicyLearning
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.selection import UpdateDiagnostics
import managym

LEARNING = DampedPolicyLearning(
    gradient="damped_policy", advantage_quantile=0.5, min_advantage=0
)
# Every count the update reports; equal in every backend.
_COUNTS = (
    "rows",
    "retained",
    "optimizer_exposures",
    "actor_exposures",
    "critic_exposures",
)
_POLICY_TERMS = ("entropy", "collection_kl", "reference_kl", "advantage_abs_mean")


@pytest.fixture(scope="module")
def captured() -> tuple[PolicyCheckpoint, RolloutBatch]:
    """Initial weights in the shared container and one batch collected with them."""
    torch.manual_seed(7)
    space = ObservationSpace()
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=16,
            num_attention_heads=2,
            value_aggregation="masked_mean",
            world=managym.WORLD_SETUP,
        ),
    )
    state = {k: v.detach().numpy().copy() for k, v in agent.state_dict().items()}
    collector = SeatRoutedCollector(
        space,
        Match(MatchConfig()),
        Reward(RewardConfig()),
        num_envs=2,
        opponent_mode="self",
        learning_seats="one",
        batch_closure="exact-quota",
        device="cpu",
    )
    batch = collector.collect(agent, 6)
    return from_torch_state(agent.config, space.encoder.config, state), batch


def _update(
    module: ModuleType,
    weights: PolicyCheckpoint,
    batch: RolloutBatch,
    minibatch_rows: int,
) -> tuple[UpdateDiagnostics, Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    """Diagnostics of one update from `weights`, with parameters before and after."""
    learner = module.learner.Learner(
        module.checkpoint.assemble(weights), "cpu", minibatch_rows=minibatch_rows
    )
    before = learner.parameters()
    diagnostics = update_damped_iteration(module, learner, batch, LEARNING, 1)
    return diagnostics, before, learner.parameters()


def _assert_close(actual: object, expected: object) -> None:
    """Floats within the parity bar, everything else (counts, labels, absent
    values, structure) equal."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_close(actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for item, value in zip(actual, expected, strict=True):
            _assert_close(item, value)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=LOGIT_TOLERANCE)
    else:
        assert actual == expected


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_damped_update_agrees_with_the_torch_backend(
    name: str, captured: tuple[PolicyCheckpoint, RolloutBatch]
) -> None:
    if name not in installed_backends():
        pytest.skip(f"the {BACKENDS[name]} framework is not installed")
    weights, batch = captured
    reference, module = backend_module("torch"), backend_module(name)
    rows = batch.actions.size
    expected, _, reference_after = _update(reference, weights, batch, rows)
    actual, before, after = _update(module, weights, batch, rows)

    assert expected["optimizer_exposures"] > 0
    assert expected["numerical"]["optimizer_steps"] == 1
    for key in _COUNTS:
        assert actual[key] == expected[key], key
    assert actual["numerical"]["optimizer_steps"] == 1
    assert actual["loss"] == pytest.approx(expected["loss"], rel=LOSS_TOLERANCE)
    assert actual["gradient_norm"] == pytest.approx(
        expected["gradient_norm"], rel=LOSS_TOLERANCE
    )
    for key in ("policy_loss", "value_loss", *_POLICY_TERMS):
        assert actual[key] == pytest.approx(expected[key], abs=LOGIT_TOLERANCE), key
    _assert_close(actual["selection_groups"], expected["selection_groups"])
    _assert_close(actual["value_scale"], expected["value_scale"])

    # The step moved the same parameters to within Adam's step size: a
    # backend's gradient differs from torch's by rounding, which Adam's
    # normalization turns into at most one learning-rate step per element in
    # either direction.
    layout = module.checkpoint.BACKEND.parameter_layout(
        canonical_parameters(reference_after)
    )
    bound = 2.01 * expected["learning_rate"]
    assert after.keys() == layout.keys()
    assert any(not np.array_equal(after[key], before[key]) for key in after)
    for key, value in after.items():
        assert np.abs(value - layout[key]).max() <= bound, key

    expected, _, _ = _update(reference, weights, batch, 2)
    actual, _, _ = _update(module, weights, batch, 2)
    assert expected["numerical"]["optimizer_steps"] > 1
    for key in _COUNTS:
        assert actual[key] == expected[key], key
    # Step and row counts; the sampled gradient detail is torch's alone.
    for key in (
        "optimizer_steps",
        "skipped_steps",
        "rejected_steps",
        "actor_rows",
        "critic_rows",
        "zero_advantage_rows",
    ):
        assert actual["numerical"][key] == expected["numerical"][key], key
    assert actual["loss"] == pytest.approx(expected["loss"], rel=1e-2)
