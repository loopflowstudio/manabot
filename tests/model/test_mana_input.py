"""Mana restrictions and selected tap costs reach v8 policies and checkpoints."""

import json
from pathlib import Path

import pytest
import torch

from manabot.env.observation import (
    FACT_KIND_INDEX,
    FACT_MANA_LOCATION,
    RELATION_MANA_TAPS_INDEX,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig, ObservationConfig, StateEncoder
from manabot.managym.enums import ActionEnum
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.verify.mana_risk import tensor_batch
from manabot.verify.text_observation import priority_root
import managym


@pytest.mark.parametrize("arm", ("rows", "typed_pool", "typed_gru"))
def test_v8_mana_policy_trains_and_reloads(tmp_path: Path, arm: StateEncoder) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(242)
    env = priority_root((), ("Supportive Parents", "Gray Ogre", "Gray Ogre"))
    obs = env.scenario_refresh()
    index = next(
        i
        for i, action in enumerate(obs.action_space.actions)
        if int(action.action_type) == ActionEnum.PRIORITY_ACTIVATE_ABILITY
    )
    pending, *_ = env.step(index)
    viewer = env.current_agent_index()
    payload = json.loads(env.semantic_observation_json(viewer))["viewer_state"]
    payload["agent"]["lesson_mana"] = [0, 1, 0, 0, 0, 0]
    payload["agent"]["restricted_mana"] = [
        {"spells": ["Citizen"], "mana": [0, 1, 0, 0, 0, 0]}
    ]
    obs = managym.Observation.from_json(json.dumps(payload))
    space = ObservationSpace(ObservationConfig(policy_input_version=8))
    model = ActorCritic(
        space,
        ModelConfig(hidden_dim=16, state_encoder=arm, world=managym.WORLD_SETUP),
    )
    # Historical viewer JSON has no continuation frames. Exercise the live tap
    # decision and the synthetic open-subtype pool as separate batch entries.
    values = tensor_batch(space.encode(pending), space.encode(obs))
    assert (values["relations"][..., RELATION_MANA_TAPS_INDEX] == 2).any()
    assert (values["facts"][..., FACT_KIND_INDEX] == FACT_MANA_LOCATION).any()
    logits, value = model(values)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    (
        logits[values["actions_valid"].bool()].square().mean() + value.square().mean()
    ).backward()
    for encoder in (model.facts, model.continuation):
        assert encoder is not None
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in encoder.parameters()
        )
    path = tmp_path / "mana.pt"
    save_checkpoint(model, space, path, player_configs=None)
    restored, restored_space = load_checkpoint(str(path))
    assert restored_space.encoder.config.policy_input_version == 8
    model.eval()
    restored.eval()
    with torch.no_grad():
        for expected, actual in zip(model(values), restored(values), strict=True):
            torch.testing.assert_close(expected, actual)
