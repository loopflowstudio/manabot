"""The torch backend through the NumPy backend protocol parity uses."""

import numpy as np
import pytest
import torch

from manabot.backends.parity import minibatch_loss
import manabot.backends.torch as torch_backend
from manabot.backends.torch.arrays import array_fields, on_device
from manabot.backends.torch.checkpoint import BACKEND
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy.checkpoint import canonical_parameters, from_torch_state
from manabot.policy.prepare import reference_distribution
from manabot.policy.protocol import Settings, Targets
import managym
from tests.model.prepared_routes import device_input


def _batch(space: ObservationSpace, count: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    env = managym.Env(seed=11)
    env.reset(Match().to_rust())
    rows = []
    while len(rows) < count:
        obs = env.observation_for_player(env.current_agent_index())
        rows.append(space.encode(obs))
        _, _, terminated, truncated, _ = env.step(
            int(rng.integers(len(obs.action_space.actions)))
        )
        if terminated or truncated:
            env.reset(Match().to_rust())
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def _targets(obs: dict[str, np.ndarray], logits: np.ndarray) -> Targets:
    legal = obs["actions_valid"] > 0
    masked = np.where(legal, logits, -np.inf)
    logs = masked - np.logaddexp.reduce(masked, axis=-1, keepdims=True)
    rows = len(legal)
    return Targets(
        actions=legal.argmax(-1),
        advantages=np.linspace(-1, 1, rows, dtype=np.float32),
        value_targets=np.zeros(rows, np.float32),
        actor_mask=np.ones(rows, bool),
        behavior_logs=logs.astype(np.float32),
        reference=reference_distribution(obs, "action_type_uniform"),
    )


def test_torch_backend_runs_the_policy_and_damped_step_over_numpy() -> None:
    space = ObservationSpace()
    torch.manual_seed(0)
    model = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8, value_aggregation="masked_mean", world=managym.WORLD_SETUP
        ),
    ).eval()
    state = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
    policy, learner = BACKEND.load(
        from_torch_state(
            model.config,
            space.encoder.config,
            state,
            parameter_names=dict(model.named_parameters()),
        )
    )
    obs = _batch(space, 6)

    logits, values = policy.evaluate(policy.prepare(obs, inference=True))
    with torch.no_grad():
        expected_logits, expected_values = model(
            device_input(model, {k: torch.from_numpy(v) for k, v in obs.items()})
        )
    np.testing.assert_array_equal(logits, expected_logits.numpy())
    np.testing.assert_array_equal(values, expected_values.numpy())

    batch = policy.prepare(obs)
    targets = _targets(obs, logits)
    settings = Settings(
        learning_rate=1e-3, clip=0.2, collection_kl=0.1, tau=0.01, max_grad_norm=0.5
    )
    loss = minibatch_loss(torch_backend, policy, learner, batch, targets, settings)
    gradients = learner.gradients(loss)
    names = {name for name, _ in model.named_parameters()}
    assert np.isfinite(float(loss.detach()))
    assert gradients.keys() == names
    assert BACKEND.gradient_layout(canonical_parameters(gradients)).keys() == names
    assert any(np.abs(g).sum() > 0 for g in gradients.values())

    before = learner.parameters()
    learner.set_learning_rate(settings.learning_rate)
    step_loss = minibatch_loss(torch_backend, policy, learner, batch, targets, settings)
    norm = learner.step(step_loss, settings.max_grad_norm)
    after = learner.parameters()
    assert float(step_loss.detach()) == float(loss.detach())
    assert float(norm) > 0
    moved = [name for name in names if not np.array_equal(before[name], after[name])]
    assert moved


def test_prepared_policy_boundary_and_cached_program_lifetime() -> None:
    """The direct backend API refuses host arrays in forward and stale cached weights."""
    torch.manual_seed(17)
    model = ActorCritic(
        ObservationSpace(), ModelConfig(hidden_dim=8, world=managym.WORLD_SETUP)
    ).eval()
    obs = _batch(model.observation_space, 3)
    batch = model.prepare(obs, inference=True)
    with pytest.raises(TypeError, match="device Prepared"):
        model(batch)
    expected = model.evaluate(batch)
    model.cache_programs()
    for wanted, actual in zip(expected, model.evaluate(batch), strict=True):
        np.testing.assert_array_equal(wanted, actual)
    assert model.semantic_cards is not None
    with torch.no_grad():
        model.semantic_cards.token_embedding.weight.add_(0.01)
    with pytest.raises(RuntimeError, match="cached inference policy changed"):
        model.evaluate(batch)
    model.clear_cache()
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(expected, model.evaluate(batch), strict=True)
    )
    empty = {**obs, "actions_valid": np.zeros_like(obs["actions_valid"])}
    with pytest.raises(ValueError, match="No valid actions"):
        model.get_action_and_value(on_device(model.prepare(empty), "cpu"))


@pytest.mark.parametrize("bad", [float("nan"), -1.0, 0.5])
def test_host_admission_retains_invalid_mask_evidence(bad: float) -> None:
    model = ActorCritic(ObservationSpace(), ModelConfig(hidden_dim=8))
    obs = _batch(model.observation_space, 1)
    obs["actions_valid"][0, 0] = bad
    with pytest.raises(
        torch_backend.rl.NumericalError, match="invalid_action_mask"
    ) as error:
        model.prepare(obs)
    assert error.value.tensors["failed"][0, 0]
    np.testing.assert_array_equal(
        error.value.tensors["mask"].numpy(), obs["actions_valid"]
    )


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("bucketed", [False, True])
def test_preparation_purpose_is_independent_of_grad_mode(
    inference: bool, bucketed: bool
) -> None:
    model = ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=8, world=managym.WORLD_SETUP),
        bucketed_shapes=bucketed,
    ).eval()
    obs = _batch(model.observation_space, 3)
    with torch.enable_grad():
        learner = model.prepare(obs, inference=inference)
    with torch.no_grad():
        frozen = model.prepare(obs, inference=inference)
    expected = array_fields(learner, np.ndarray)
    actual = array_fields(frozen, np.ndarray)
    assert expected.keys() == actual.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(value, actual[name])
    assert learner.programs is not None
    assert model.semantic_cards is not None
    count = len(learner.programs.rows)
    total = model.semantic_cards.catalog_shape.definitions
    if inference or bucketed:
        assert count == total
    else:
        assert 0 < count < total
    for wanted, output in zip(
        model.evaluate(learner), model.evaluate(frozen), strict=True
    ):
        np.testing.assert_array_equal(wanted, output)
