"""Mana joins reach the actual policy; arbitrary class labels do not in v9."""

from pathlib import Path

import pytest
import torch

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig, StateEncoder
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.verify.mana_risk import (
    empty_fact_objects,
    mana_pairs,
    open_fact_adapter,
    tensor_batch,
)
import managym


@pytest.mark.parametrize("arm", ("rows", "typed_pool", "typed_gru"))
def test_joined_mana_separates_meaning_and_preserves_class_permutations(
    arm: StateEncoder,
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(242)
    space = ObservationSpace(ObservationConfig(policy_input_version=9))
    model = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=16,
            state_encoder=arm,
            policy_film=True,
            world=managym.WORLD_SETUP,
        ),
    ).eval()
    assert model.facts is not None
    pairs = mana_pairs(space)
    sample = tensor_batch(pairs[0].left, pairs[0].right)
    # At initialization the additive fact adapter is intentionally closed.
    objects = empty_fact_objects(model, 2)
    torch.testing.assert_close(model.facts(sample["facts"], objects), objects)
    _, values = model(sample)
    values.sum().backward()
    assert model.facts.output.weight.grad is not None
    assert model.facts.output.weight.grad.abs().sum() > 0
    assert model.facts.numeric.weight.grad is not None
    assert model.facts.numeric.weight.grad.abs().sum() == 0
    model.zero_grad(set_to_none=True)
    open_fact_adapter(model)
    for pair in pairs:
        batch = tensor_batch(pair.left, pair.right)
        logits, values = model(batch)
        valid = batch["actions_valid"][0].bool()
        left, right = logits[0, valid], logits[1, valid]
        assert len(left) >= 2, pair.name
        if pair.equivalent:
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(values[0], values[1], atol=1e-6, rtol=1e-5)
        else:
            assert (values[0] - values[1]).abs().max() > 1e-8, pair.name
            # A common logit shift cannot change the policy. Check actual
            # normalized action preferences, not only differing raw logits.
            assert (left.log_softmax(0) - right.log_softmax(0)).abs().max() > 1e-7, (
                pair.name
            )
    # Opposite preferences through the full action path send gradients back
    # through the class join, amount projection and predicate embeddings.
    logits, _ = model(sample)
    valid = sample["actions_valid"][0].bool()
    scores = logits[:, valid]
    loss = -(scores[0].log_softmax(0)[0] + scores[1].log_softmax(0)[1])
    loss.backward()
    assert model.facts.mana_join is not None
    for parameter in (
        model.facts.mana_join.weight,
        model.facts.numeric.weight,
        model.facts.detail.weight,
        model.facts.kind.weight,
    ):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    path = tmp_path / "joined-mana.pt"
    save_checkpoint(model, space, path, player_configs=None)
    restored, restored_space = load_checkpoint(str(path))
    assert restored_space.encoder.config.policy_input_version == 9
    restored.eval()
    for expected, actual in zip(model(sample), restored(sample), strict=True):
        torch.testing.assert_close(expected, actual)


def test_v8_class_label_sensitivity_is_retained_only_for_frozen_input() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(242)
    space = ObservationSpace(ObservationConfig(policy_input_version=8))
    model = ActorCritic(space, ModelConfig(hidden_dim=16, world=managym.WORLD_SETUP))
    open_fact_adapter(model)
    assert model.facts is not None and model.facts.mana_join is None
    pair = next(pair for pair in mana_pairs(space) if pair.name == "class_labels")
    batch = tensor_batch(pair.left, pair.right)
    representations = model.facts(batch["facts"], empty_fact_objects(model, 2))
    assert (representations[0] - representations[1]).abs().max() > 0.1
    assert ObservationConfig().policy_input_version == 6
    assert ObservationConfig.from_checkpoint({}).policy_input_version == 1
