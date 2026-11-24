"""The profiling controls must leave the ordinary Adam update intact."""

import numpy as np
import pytest

from manabot.backends import resolve
from manabot.backends.parity import STEP_PARAMETER_TOLERANCE, minibatch_loss
from manabot.policy.prepare import reference_distribution
from manabot.policy.protocol import Settings, Targets
from tests.training.backend_fixtures import legacy_capture

pytest.importorskip("mlx.core")

from scripts.profile_mlx_learner import profile_step  # noqa: E402


def test_profile_updates_every_weight_with_the_full_gradient() -> None:
    weights, rollout = legacy_capture(seed=11, num_envs=2, decisions=4)
    actions = rollout.actions.reshape(-1)
    rows = rollout.observations(np.arange(len(actions)))
    obs = rows.observation
    targets = Targets(
        actions,
        np.linspace(-1, 1, len(actions), dtype=np.float32),
        np.ones(len(actions), np.float32),
        np.ones(len(actions), bool),
        rows.log_probabilities,
        reference_distribution(obs, "action_type_uniform"),
    )
    settings = Settings(1e-4, 0.2, 0.1, 0.05, 0.01)
    backend = resolve("mlx", precision="float32")
    policy, expected = backend.load(weights)
    _, profiled = backend.load(weights)
    policy.bucketed_shapes = profiled.agent.bucketed_shapes = True
    catalog_key = "semantic_cards.encoder.Wh"
    before = expected.parameters()[catalog_key]
    expected.set_learning_rate(settings.learning_rate)
    loss = minibatch_loss(
        backend, policy, expected, policy.prepare(obs), targets, settings
    )
    norm = expected.step(loss, settings.max_grad_norm)
    sample = profile_step(backend, profiled, obs, targets, settings, 0)
    assert len(sample.attention_shapes) == 2
    assert sample.rows == len(actions)
    assert sample.loss == pytest.approx(float(loss), abs=1e-7)
    assert sample.norm == pytest.approx(float(norm), rel=1e-6)
    actual = profiled.parameters()
    after = expected.parameters()
    assert np.max(np.abs(after[catalog_key] - before)) > STEP_PARAMETER_TOLERANCE
    for name, weight in after.items():
        np.testing.assert_allclose(
            actual[name], weight, atol=STEP_PARAMETER_TOLERANCE, rtol=0, err_msg=name
        )
