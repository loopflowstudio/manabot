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

import numpy as np
import pytest

from manabot.backends import resolve
from manabot.backends.parity import (
    BACKENDS,
    LOGIT_TOLERANCE,
    LOSS_TOLERANCE,
    STEP_PARAMETER_TOLERANCE,
    TOLERANCES,
    Report,
)
from manabot.env.observation import (
    ACTION_CONTINUATION_INDEX,
    PERMANENT_ATTACHED_INDEX,
)
from manabot.infra.profiler import Profiler
from manabot.policy.checkpoint import (
    PolicyCheckpoint,
    canonical_parameters,
)
from manabot.policy.protocol import Backend, Precision
from manabot.training.learning import DampedPolicyLearning
from manabot.training.rollout.actions import ActionRows
from manabot.training.rollout.collection import (
    RolloutBatch,
)
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.selection import UpdateDiagnostics
from tests.training.backend_fixtures import installed_backend, legacy_capture

LEARNING = DampedPolicyLearning(
    gradient="damped_policy",
    advantage_quantile=0.5,
    reference_forward_kl=0.002,
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
_POLICY_TERMS = (
    "entropy",
    "collection_kl",
    "reference_kl",
    "reference_forward_kl",
    "advantage_abs_mean",
)


@pytest.fixture(scope="module")
def captured() -> tuple[PolicyCheckpoint, RolloutBatch]:
    """Initial weights in the shared container and one batch collected with them."""
    return legacy_capture(seed=7, num_envs=2, decisions=6)


def _update(
    module: Backend,
    weights: PolicyCheckpoint,
    batch: RolloutBatch,
    minibatch_rows: int,
    *,
    execution_rows: int | None = None,
) -> tuple[UpdateDiagnostics, Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    """Diagnostics of one update from `weights`, with parameters before and after."""
    learner = module.learner.Learner(
        module.load(weights)[0],
        "cpu",
        minibatch_rows=minibatch_rows,
        execution_rows=execution_rows,
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
    weights, batch = captured
    reference, module = resolve("torch"), installed_backend(name)
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
    layout = module.parameter_layout(canonical_parameters(reference_after))
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


def test_mlx_chunking_preserves_the_collection_draw_and_logical_adam_step(
    captured: tuple[PolicyCheckpoint, RolloutBatch],
) -> None:
    weights, batch = captured
    backend = installed_backend("mlx")
    agent = backend.load(weights)[0].to("mps")
    rows = batch.observations(np.arange(5)).observation
    whole = backend.collection.CollectionSession(device="mps", seed=111)
    split = backend.collection.CollectionSession(device="mps", seed=111)
    with whole.scope(agent) as scope:
        expected = scope.sample(rows, Profiler())
    with split.scope(agent) as scope:
        actual = scope.sample(rows, Profiler(), execution_rows=2)
    np.testing.assert_array_equal(split.snapshot_rng(), whole.snapshot_rng())
    np.testing.assert_array_equal(actual.actions, expected.actions)
    np.testing.assert_allclose(
        actual.log_probabilities, expected.log_probabilities, atol=1e-6
    )
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-6)

    size = batch.actions.size
    reference, _, unchunked = _update(backend, weights, batch, size)
    chunked, _, after = _update(backend, weights, batch, size, execution_rows=2)
    assert (
        reference["numerical"]["optimizer_steps"]
        == chunked["numerical"]["optimizer_steps"]
        == 1
    )
    for key in _COUNTS:
        assert reference[key] == chunked[key]
    assert chunked["loss"] == pytest.approx(reference["loss"], rel=LOSS_TOLERANCE)
    for name, value in unchunked.items():
        np.testing.assert_allclose(
            after[name], value, atol=STEP_PARAMETER_TOLERANCE, rtol=0, err_msg=name
        )


def test_parity_requires_a_matching_optimizer_step() -> None:
    report = Report()
    assert "clipped Adam weights" in report.failures()
    report.step_checked = True
    assert report.failures() == []
    report.step_parameter_max_abs = 2 * STEP_PARAMETER_TOLERANCE
    assert "clipped Adam weights" in report.failures()


def test_float32_parity_requires_every_tensor_gradient_to_agree() -> None:
    report = Report(
        step_checked=True,
        gradient_cosine=1.0,
        gradient_min_cosine=0.98,
        gradient_worst="policy_head.0.bias",
    )
    assert report.failures() == ["gradient policy_head.0.bias"]
    report.gradient_min_cosine = 1.0
    assert report.failures() == []


def test_half_damped_update_after_cached_inference(
    captured: tuple[PolicyCheckpoint, RolloutBatch],
) -> None:
    precision: Precision = "bfloat16"
    module = installed_backend("mlx", precision=precision)
    weights, batch = captured
    agents = [module.load(weights)[0] for _ in range(2)]
    learners = [module.learner.Learner(agent, minibatch_rows=2) for agent in agents]
    before = learners[0].parameters()
    # A fixed-weight collection cache must not sever catalog gradients in training.
    agents[0].cache_programs()
    result = update_damped_iteration(module, learners[0], batch, LEARNING, 1)
    assert result["numerical"]["optimizer_steps"] > 1
    assert result["numerical"]["rejected_steps"] == 0
    assert np.isfinite(result["loss"])
    after = learners[0].parameters()
    assert all(value.dtype == np.float32 for value in after.values())
    assert any(
        np.any(after[key] != value)
        for key, value in before.items()
        if key.startswith("semantic_cards.")
    )
    # A precision change must invalidate the old inference catalog.
    agents[0].cache_programs()
    agents[0].set_precision("float32")
    agents[1].update(agents[0].parameters())
    agents[1].set_precision("float32")
    obs = batch.observations(np.array([0, 1])).observation
    for actual, expected in zip(
        agents[0].evaluate(agents[0].prepare(obs)),
        agents[1].evaluate(agents[1].prepare(obs)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_half_parity_uses_measured_dtype_tolerances() -> None:
    precision: Precision = "bfloat16"
    tolerance = TOLERANCES[precision]
    report = Report(
        precision=precision,
        step_checked=True,
        step_parameter_max_abs=0.00017,
        gradient_min_cosine=-1,
        gradient_cosine=tolerance.cosine + 0.0001,
        forward_logits=tolerance.logits / 2,
        forward_values=tolerance.values / 2,
        loss_relative=tolerance.loss / 2,
    )
    assert report.failures() == []
    report.forward_logits = tolerance.logits * 2
    report.loss_relative = tolerance.loss * 2
    report.gradient_cosine = tolerance.cosine - 0.01
    assert set(report.failures()) == {"forward logits", "learner loss", "full gradient"}
    report.step_parameter_max_abs = float("nan")
    assert "finite clipped Adam step" in report.failures()


@pytest.mark.parametrize("name", sorted(BACKENDS))
@pytest.mark.parametrize("bootstrap", [False, True])
def test_backends_refuse_malformed_rows_before_preparing_them(
    name: str,
    bootstrap: bool,
    captured: tuple[PolicyCheckpoint, RolloutBatch],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights, batch = captured
    module = installed_backend(name)
    agent = module.load(weights)[0]
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
        ("actions", ACTION_CONTINUATION_INDEX, -1, "action continuation"),
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
                ).next_observation("cpu")
        else:
            admitted = module.rollout.AdmittedRollout(agent, batch)
            with monkeypatch.context() as patch:
                patch.setattr(RolloutBatch, "observations", malformed)
                with pytest.raises(ValueError, match=message):
                    admitted.admit_rows(np.array([0]), 1)
            with pytest.raises(ValueError, match="were not admitted"):
                admitted.rows(np.array([0]), "cpu")


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_actor_filtered_floor_is_the_reported_penalty(
    name: str,
    captured: tuple[PolicyCheckpoint, RolloutBatch],
) -> None:
    weights, batch = captured
    backend = installed_backend(name)
    results: list[UpdateDiagnostics] = []
    for coefficient in (0.0, 0.002):
        learning = LEARNING.model_copy(
            update={"filter_scope": "actor", "reference_forward_kl": coefficient}
        )
        learner = backend.learner.Learner(
            backend.load(weights)[0],
            "cpu",
            minibatch_rows=batch.actions.size,
            execution_rows=2,
            bootstrap_rows=1,
        )
        results.append(update_damped_iteration(backend, learner, batch, learning, 1))
    control, floor = results
    assert 0 < floor["actor_exposures"] < floor["critic_exposures"]
    assert floor["reference_forward_kl"] == control["reference_forward_kl"] > 0
    assert floor["policy_loss"] - control["policy_loss"] == pytest.approx(
        0.002 * floor["reference_forward_kl"], rel=1e-5
    )
