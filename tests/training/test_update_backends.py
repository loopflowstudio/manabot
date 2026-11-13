"""The damped update, written once, agrees across every installed backend.

One rollout batch is collected with the torch policy; each backend
(`manabot.backends.parity.BACKENDS`) then loads the same weights through the
shared container and runs `update_damped_iteration` on that batch. The torch
backend is the reference. With the whole batch in one minibatch every backend
measures its loss and gradient at the same weights, and the bars are the
parity check's: loss and gradient norm within 0.1%, policy terms within 1e-4,
and every post-Adam weight within 1e-6, with clipping active.
With several minibatches the later ones start from weights that already
differ by rounding, so
that run checks the row and step counts exactly and the loss within 1%. A
backend whose framework is absent is skipped.
"""

from collections.abc import Mapping
from dataclasses import replace
from types import ModuleType

import numpy as np
import pytest
import torch

from manabot.backends.parity import (
    BACKENDS,
    LOGIT_TOLERANCE,
    LOSS_TOLERANCE,
    STEP_PARAMETER_TOLERANCE,
    Report,
    backend_module,
    installed_backends,
)
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import PERMANENT_ATTACHED_INDEX, ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.policy.checkpoint import (
    PolicyCheckpoint,
    canonical_parameters,
    from_torch_state,
)
from manabot.training.regime import DampedPolicyLearning
from manabot.training.rollout.actions import ActionRows
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.selection import UpdateDiagnostics
import managym

LEARNING = DampedPolicyLearning(
    gradient="damped_policy",
    advantage_quantile=0.5,
    min_advantage=0,
    max_grad_norm=0.01,
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
    assert expected["gradient_norm"] > LEARNING.max_grad_norm
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

    # Every weight after one clipped Adam step agrees, including both GRU
    # biases and tensors whose gradients are only rounding noise.
    layout = module.checkpoint.BACKEND.parameter_layout(
        canonical_parameters(reference_after)
    )
    assert after.keys() == layout.keys()
    assert any(not np.array_equal(after[key], before[key]) for key in after)
    for key, value in after.items():
        assert np.abs(value - layout[key]).max() <= STEP_PARAMETER_TOLERANCE, key

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


def test_parity_requires_a_matching_optimizer_step() -> None:
    report = Report()
    assert "clipped Adam weights" in report.failures()
    report.step_checked = True
    assert report.failures() == []
    report.step_parameter_max_abs = 2 * STEP_PARAMETER_TOLERANCE
    assert "clipped Adam weights" in report.failures()


@pytest.mark.parametrize("name", sorted(BACKENDS))
@pytest.mark.parametrize("bootstrap", [False, True])
def test_backends_refuse_malformed_rows_before_preparing_them(
    name: str,
    bootstrap: bool,
    captured: tuple[PolicyCheckpoint, RolloutBatch],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if name not in installed_backends():
        pytest.skip(f"the {BACKENDS[name]} framework is not installed")
    weights, batch = captured
    module = backend_module(name)
    agent = module.checkpoint.assemble(weights)
    original = RolloutBatch.observations
    # Each corruption is otherwise a real batch. Check the raw action mask
    # before boolean conversion and semantic joins before any indexing.
    for key, column, value, message in (
        ("actions_valid", None, 2, "invalid_action_mask"),
        ("semantic_cards", None, -1, "unadmitted semantic definition"),
        ("agent_permanents", PERMANENT_ATTACHED_INDEX, -1, "invalid permanent link"),
        ("facts", 0, -1, "invalid fact object"),
        ("effects", 13, -1, "effect CSR"),
        ("name_candidates", 0, -1, "unadmitted card definition"),
        ("actions", -2, -1, "action continuation"),
    ):

        def corrupt(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            obs = {k: v.copy() for k, v in obs.items()}
            if key in ("facts", "effects"):
                obs[key][0, 0, -1] = 1
            if key == "effects":
                obs[key][0, 0, 10] = 1
            slot = (
                int(np.flatnonzero(obs["actions_valid"][0])[0])
                if key in ("name_candidates", "actions")
                else 0
            )
            index = (0, slot) if column is None else (0, slot, column)
            obs[key][index] = value
            return obs

        def malformed(
            self: RolloutBatch, indexes: np.ndarray | None = None
        ) -> ActionRows:
            rows = original(self, indexes)
            return replace(rows, observation=corrupt(rows.observation))

        if bootstrap:
            with pytest.raises(ValueError, match=message):
                module.rollout.AdmittedRollout(
                    agent, replace(batch, next_obs=corrupt(batch.next_obs))
                )
        else:
            admitted = module.rollout.AdmittedRollout(agent, batch)
            with monkeypatch.context() as patch:
                patch.setattr(RolloutBatch, "observations", malformed)
                with pytest.raises(ValueError, match=message):
                    admitted.admit_rows(np.array([0]), 1)
            with pytest.raises(ValueError, match="were not admitted"):
                admitted.rows(np.array([0]), "cpu")
