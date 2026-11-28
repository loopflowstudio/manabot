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
from manabot.env.observation import ObservationSpace, encode_policy_batch
from manabot.infra.config import ModelConfig
from manabot.policy.checkpoint import canonical_parameters, from_torch_state
from manabot.policy.prepare import reference_distribution
from manabot.policy.protocol import Settings, Targets
import managym
from tests.model.prepared_routes import device_input


def _play(count: int) -> tuple[list[managym.Observation], managym.Observation]:
    """`count` decisions of one random game, then a terminal observation."""
    rng = np.random.default_rng(3)
    env = managym.Env(seed=11)
    env.reset(Match().to_rust())
    rows: list[managym.Observation] = []
    while True:
        obs = env.observation_for_player(env.current_agent_index())
        if len(rows) < count:
            rows.append(obs)
        _, _, terminated, truncated, _ = env.step(
            int(rng.integers(len(obs.action_space.actions)))
        )
        if terminated or truncated:
            return rows, env.observation_for_player(0)


def _batch(space: ObservationSpace, count: int) -> managym.ObservationBatch:
    rows, _ = _play(count)
    assert len(rows) == count
    return encode_policy_batch(space, rows)


def _targets(obs: managym.ObservationBatch, logits: np.ndarray) -> Targets:
    counts = np.asarray(obs.action_counts)
    legal = np.arange(logits.shape[-1]) < counts[:, None]
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
            space.config,
            state,
            parameter_names=dict(model.named_parameters()),
            readings=model.readings,
        )
    )
    obs = _batch(space, 6)

    logits, values = policy.evaluate(policy.prepare(obs, inference=True))
    with torch.no_grad():
        expected_logits, expected_values = model(device_input(model, obs))
    np.testing.assert_array_equal(logits, expected_logits.numpy())
    np.testing.assert_array_equal(values, expected_values.numpy())

    policy.train()
    modes = [module.training for module in policy.modules()]
    probabilities, probability_values = policy.evaluate_probabilities(
        policy.prepare(obs, inference=True)
    )
    np.testing.assert_array_equal(
        probabilities, torch.softmax(expected_logits, dim=-1).numpy()
    )
    np.testing.assert_array_equal(probability_values, expected_values.numpy())
    assert [module.training for module in policy.modules()] == modes
    policy.eval()

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
    _, terminal = _play(0)
    empty = encode_policy_batch(model.observation_space, [terminal])
    assert list(empty.action_counts) == [0]
    with pytest.raises(ValueError, match="No valid actions"):
        model.get_action_and_value(on_device(model.prepare(empty), "cpu"))


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
    if inference:
        assert count == total
    else:
        assert 0 < count < total
    for wanted, output in zip(
        model.evaluate(learner), model.evaluate(frozen), strict=True
    ):
        np.testing.assert_array_equal(wanted, output)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_bucketed_learner_catalog_coverage(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=8, world=managym.WORLD_SETUP),
        bucketed_shapes=True,
    )
    obs = _batch(model.observation_space, 3)
    needs = model.check_observation(obs)
    # Catalog scheduling runs entirely on the host, including for a GPU learner.
    monkeypatch.setattr(ActorCritic, "device_type", device)
    learner = model.prepare_admitted(obs, needs=needs).programs
    inference = model.prepare_admitted(obs, needs=needs, inference=True).programs
    assert model.semantic_cards is not None
    assert learner is not None and inference is not None
    total = model.semantic_cards.catalog_shape.definitions
    assert len(inference.rows) == total
    if device == "cuda":
        np.testing.assert_array_equal(learner.rows, inference.rows)
    else:
        assert 0 < len(learner.rows) < total
