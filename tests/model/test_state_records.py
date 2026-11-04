"""Typed state treatments preserve native decisions and train/export normally."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from experiments.state_representation import recipe
from manabot.env.observation import FACT_ROWS, ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.architecture import architecture_identity
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.model.state_records import SharedFactEncoder, TypedContinuationEncoder
from manabot.training.execution import execute_regime
from manabot.verify.representation import shared_effect_root
from manabot.verify.store import VerifyStore
from manabot.verify.text_observation import collision_branches, native_fixtures
import managym

Arm = Literal["typed_pool", "typed_gru"]


def _batch(encoded: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value)[None] for key, value in encoded.items()}


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_native_choices_remain_distinct_and_reload(tmp_path: Path, arm: Arm) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(73200)
    space = ObservationSpace()
    agent = ActorCritic(
        space, ModelConfig(hidden_dim=16, state_encoder=arm, policy_film=True)
    )
    branches = collision_branches()
    encoded = [
        space.encode(env.observation_for_player(env.current_agent_index()))
        for env in branches
    ]
    outputs = [agent(_batch(obs)) for obs in encoded]
    assert not torch.equal(outputs[0][0], outputs[1][0])
    assert not torch.equal(outputs[0][1], outputs[1][1])
    (
        outputs[0][1].sum()
        + outputs[0][0][_batch(encoded[0])["actions_valid"].bool()].sum()
    ).backward()
    assert agent.continuation is not None
    assert (
        sum(
            p.grad.abs().sum()
            for p in agent.continuation.parameters()
            if p.grad is not None
        )
        > 0
    )
    path = tmp_path / "policy.pt"
    players = [managym.PlayerConfig(name, {"Mountain": 40}) for name in ("a", "b")]
    save_checkpoint(agent, space, path, player_configs=players)
    restored, _ = load_checkpoint(str(path))
    assert restored.config.state_encoder == arm
    for expected, actual in zip(outputs[0], restored(_batch(encoded[0])), strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    payload = torch.load(path, weights_only=False)
    payload["hypers"]["agent_hypers"]["state_encoder"] = "rows"
    torch.save(payload, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint(str(path))


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_shared_facts_preserve_membership_multiplicity_and_padding(arm: Arm) -> None:
    torch.manual_seed(73200)
    encoder = SharedFactEncoder(8, arm)
    objects = torch.randn(1, 3, 8, requires_grad=True)
    facts = torch.zeros(1, FACT_ROWS, 12)
    # Same payload, distinct objects; a duplicate on object one is significant.
    facts[0, :3] = torch.tensor(
        [[obj, 6, 1, 2, 3, 4, 0.1, 0.2, 3, 7, 11, 1] for obj in (1, 2, 1)]
    )
    expected = encoder(facts, objects)
    delta = expected - objects
    torch.testing.assert_close(delta[:, 0], 2 * delta[:, 1])
    assert delta[:, 2].abs().sum() == 0
    permuted = facts[:, torch.randperm(FACT_ROWS)]
    torch.testing.assert_close(expected, encoder(permuted, objects))
    poisoned = facts.clone()
    poisoned[0, 3:, :11] = float("nan")
    torch.testing.assert_close(expected, encoder(poisoned, objects))
    changed = facts.clone()
    changed[0, 0, 8] = 1
    assert not torch.equal(expected, encoder(changed, objects))
    expected.sum().backward()
    assert encoder.source.weight.grad is not None
    assert encoder.source.weight.grad.abs().sum() > 0
    changed[0, 0, 0] = 4
    with pytest.raises(ValueError, match="object"):
        encoder(changed, objects)


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_native_viewer_projection_and_object_joins(arm: Arm) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(73200)
    space = ObservationSpace()
    agent = ActorCritic(space, ModelConfig(hidden_dim=16, state_encoder=arm))
    for env in native_fixtures().values():
        actor = env.current_agent_index()
        for viewer in (actor, 1 - actor):
            obs = env.observation_for_player(viewer)
            native = env.encode_observation(obs)
            reference = space.encode(obs)
            for key in native:
                np.testing.assert_array_equal(native[key], reference[key])
            if viewer == actor:
                logits, value = agent(_batch(native))
                assert torch.isfinite(
                    logits[_batch(native)["actions_valid"].bool()]
                ).all()
                assert torch.isfinite(value).all()


def test_historical_defaults_and_admission() -> None:
    assert "state_encoder" not in ModelConfig().model_dump()
    space = ObservationSpace()
    identities = {
        architecture_identity(ModelConfig(state_encoder=arm), space)
        for arm in ("rows", "typed_pool", "typed_gru")
    }
    assert len(identities) == 3
    with pytest.raises(ValueError, match="v5"):
        ActorCritic(
            ObservationSpace(ObservationConfig(policy_input_version=4)),
            ModelConfig(state_encoder="typed_gru"),
        )


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_treatment_trains_with_existing_executor(tmp_path: Path, arm: Arm) -> None:
    plan = recipe(arm, smoke=True)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        run = execute_regime(plan, 73200, tmp_path / "run", store)
    assert run.status == "completed"
    stage = run.stages[0]
    assert stage.updates == 1 and stage.learner_transitions > 0
    assert stage.optimizer_exposures > 0
    restored, _ = load_checkpoint(stage.artifacts["raw"]["path"])
    assert restored.config.state_encoder == arm


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_frame_and_object_relabeling_preserves_joins(arm: Arm) -> None:
    torch.manual_seed(73200)
    encoder = TypedContinuationEncoder(8, arm)
    objects = torch.randn(1, 3, 8, requires_grad=True)
    definitions = torch.randn(1, 64, 2, 8, requires_grad=True)
    rows = torch.zeros(1, 64, 49)
    for i, (handle, parent, kind, source) in enumerate(((1, 0, 7, 1), (2, 1, 8, 2))):
        rows[0, i, :7] = torch.tensor([1, handle, parent, kind, 1, 1, source])
        rows[0, i, 31] = 1
    rows[0, 2] = rows[0, 1]
    rows[0, 2, 0] = 2
    rows[0, 2, 22] = rows[0, 2, 29] = 1
    rows[0, 2, 24] = 3
    expected_context, expected_frames = encoder(rows, objects, definitions)
    renamed = rows.clone()
    renamed[0, :3, 1] = torch.tensor([9, 4, 4])
    renamed[0, :3, 2] = torch.tensor([0, 9, 9])
    # Reverse objects, repairing source and target references.
    for col in (6, 24):
        named = renamed[..., col] > 0
        renamed[..., col][named] = 4 - renamed[..., col][named]
    actual_context, actual_frames = encoder(
        renamed.flip(1), objects.flip(1), definitions.flip(1)
    )
    torch.testing.assert_close(actual_context, expected_context)
    torch.testing.assert_close(actual_frames[:, [8, 3]], expected_frames[:, :2])
    expected_context.sum().backward()
    assert objects.grad is not None and (objects.grad.abs().sum(-1) > 0).all()
    assert definitions.grad is not None and definitions.grad.abs().sum() > 0
    renamed[0, 0, 2] = 4
    with pytest.raises(ValueError, match="cyclic"):
        encoder(renamed, objects, definitions)


@pytest.mark.parametrize("arm", ("typed_pool", "typed_gru"))
def test_stack_order_is_semantic_but_row_storage_order_is_not(arm: Arm) -> None:
    torch.manual_seed(73200)
    encoder = TypedContinuationEncoder(8, arm)
    objects = torch.randn(1, 2, 8)
    definitions = torch.randn(1, 64, 2, 8)
    rows = torch.zeros(1, 64, 49)
    for i in range(2):
        rows[0, i, :7] = torch.tensor([1, i + 1, 0, 1, 1, 1, i + 1])
        rows[0, i, 31] = 1
    summary, frames = encoder(rows, objects, definitions)
    permuted = encoder(rows.flip(1), objects, definitions.flip(1))
    torch.testing.assert_close(permuted[0], summary)
    torch.testing.assert_close(permuted[1], frames)
    rows[0, :2, 1] = torch.tensor([2, 1])
    changed, _ = encoder(rows, objects, definitions)
    assert not torch.allclose(changed, summary, atol=1e-7, rtol=1e-7)


def test_native_shared_effects_keep_timestamps_and_reject_v5_overflow() -> None:
    env = shared_effect_root(6)
    space = ObservationSpace()
    obs = env.observation_for_player(env.current_agent_index())
    native = env.encode_observation(obs)
    expected = space.encode(obs)
    for key in native:
        np.testing.assert_array_equal(native[key], expected[key])
    facts = native["facts"][native["facts"][:, 11] == 1]
    assert len(facts) == 120
    assert len(np.unique(facts[:, 1:11], axis=0)) == 12
    assert len(np.unique(facts[:, 9])) == 6
    overflowing = shared_effect_root(7)
    obs = overflowing.observation_for_player(overflowing.current_agent_index())
    for encode in (space.encode, overflowing.encode_observation):
        with pytest.raises((ValueError, RuntimeError), match="facts"):
            encode(obs)
