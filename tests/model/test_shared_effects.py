"""Shared payload routing retains addressable objects and shared learning gradients."""

from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig, StateEncoder
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.model.shared_effects import SharedEffectEncoder
from manabot.recipes.local import local_smoke
from manabot.sim.rollout import BatchedSampler
from manabot.sim.search.value import ValueScorer
from manabot.training.execution import execute_regime
from manabot.training.regime import TrainingRegime
from manabot.verify.action_cases import reorder_objects
from manabot.verify.representation import shared_effect_root
from manabot.verify.store import VerifyStore
import managym


def _batch(values: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value)[None] for key, value in values.items()}


def test_memberships_share_gradients_preserve_multiplicity_and_ignore_padding() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(7)
    env = shared_effect_root(1)
    obs = env.observation_for_player(env.current_agent_index())
    values = _batch(
        ObservationSpace(ObservationConfig(policy_input_version=7)).encode(obs)
    )
    encoder = SharedEffectEncoder(8)
    objects = torch.randn(1, 202, 8)
    headers, details, members = (
        values[name].clone() for name in ("effects", "effect_details", "effect_members")
    )
    assert headers[0, 0, 16] == 10
    headers[0, 0, 16] = 1
    one = encoder(headers, details, members, objects) - objects
    one.sum().backward()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in encoder.named_parameters()
        if parameter.grad is not None
    }
    assert gradients and any(
        gradient.abs().sum() > 0 for gradient in gradients.values()
    )
    encoder.zero_grad(set_to_none=True)
    headers[0, 0, 16] = 2
    members[0, 1, 0] = members[0, 0, 0]
    two = encoder(headers, details, members, objects) - objects
    torch.testing.assert_close(two, 2 * one)
    two.sum().backward()
    for name, parameter in encoder.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, 2 * gradients[name])
    # Unused storage is not another record or membership.
    headers[:, 1:, :17] = float("nan")
    details[:, 2:] = float("nan")
    members[:, 2:] = float("nan")
    torch.testing.assert_close(
        encoder(headers, details, members, objects) - objects, two
    )
    headers[0, 0, 16] = 2049
    with pytest.raises(ValueError, match="CSR"):
        encoder(headers, details, members, objects)


@pytest.mark.parametrize("arm", ("rows", "typed_pool", "typed_gru"))
def test_v7_policy_routes_shared_effects_and_reloads(
    tmp_path: Path, arm: StateEncoder
) -> None:
    torch.set_num_threads(1)
    env = shared_effect_root(7)
    obs = env.observation_for_player(env.current_agent_index())
    space = ObservationSpace(ObservationConfig(policy_input_version=7))
    model = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=16,
            state_encoder=arm,
            policy_film=True,
            world=managym.WORLD_SETUP,
        ),
    )
    values = _batch(space.encode(obs))
    logits, value = model(values)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    reordered_logits, reordered_value = model(reorder_objects(values))
    torch.testing.assert_close(reordered_logits, logits)
    torch.testing.assert_close(reordered_value, value)
    (
        logits[values["actions_valid"].bool()].square().mean() + value.square().mean()
    ).backward()
    assert model.shared_effects is not None
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.shared_effects.parameters()
    )
    path = tmp_path / "shared.pt"
    save_checkpoint(model, space, path, player_configs=None)
    restored, restored_space = load_checkpoint(str(path))
    assert restored_space.encoder.config.policy_input_version == 7
    restored.eval()
    model.eval()
    with torch.no_grad():
        for expected, actual in zip(model(values), restored(values), strict=True):
            torch.testing.assert_close(expected, actual)


def test_shared_effect_policy_uses_ordinary_learner_and_export(tmp_path: Path) -> None:
    regime = local_smoke(steps=1, seconds=120)
    regime.id = "shared-effects-v7-engineering-smoke"
    regime.observation = ObservationConfig(policy_input_version=7)
    regime = TrainingRegime.model_validate(regime.model_dump())
    with VerifyStore(tmp_path / "verify.sqlite") as store:
        run = execute_regime(regime, 73200, tmp_path / "training", store)
    assert run.status == "completed" and run.updates_through() == 1
    assert run.stages[0].learner_transitions > 0
    model, space = load_checkpoint(run.stages[0].artifacts["raw"]["path"])
    assert (
        model.shared_effects is not None
        and space.encoder.config.policy_input_version == 7
    )


def test_shared_memberships_are_object_joins_under_permutation() -> None:
    torch.manual_seed(11)
    env = shared_effect_root(1)
    observation = env.observation_for_player(env.current_agent_index())
    values = _batch(
        ObservationSpace(ObservationConfig(policy_input_version=7)).encode(observation)
    )
    encoder = SharedEffectEncoder(8)
    objects = torch.randn(1, 202, 8, requires_grad=True)
    expected = encoder(
        values["effects"], values["effect_details"], values["effect_members"], objects
    )
    headers, members = values["effects"].clone(), values["effect_members"].clone()
    for column in (10, 12):
        live = headers[..., column] > 0
        headers[..., column][live] = 203 - headers[..., column][live]
    live = members[..., 0] > 0
    members[..., 0][live] = 203 - members[..., 0][live]
    actual = encoder(headers, values["effect_details"], members, objects.flip(1))
    torch.testing.assert_close(actual, expected.flip(1))
    expected.square().sum().backward()
    assert objects.grad is not None and torch.isfinite(objects.grad).all()
    # The public source/controller addresses remain joins during serving too.
    model = ActorCritic(
        ObservationSpace(ObservationConfig(policy_input_version=7)),
        ModelConfig(hidden_dim=16),
    )
    buffers = {key: value.numpy() for key, value in values.items()}
    selected = BatchedSampler(model, deterministic=True).select(buffers, np.array([0]))
    assert buffers["actions_valid"][0, selected[0]] == 1
    assert np.isfinite(ValueScorer(model).score(buffers, np.array([0]))).all()
