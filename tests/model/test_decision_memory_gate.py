"""Common-trunk identity and staged learning through the only memory join."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_decision_memory import forward, memory_input, policy
import torch

from manabot.backends.torch.checkpoint import (
    branch_from_trunk,
    load_checkpoint,
    load_metadata,
    save_checkpoint,
)
from manabot.backends.torch.policy import ActorCritic
from manabot.env.decision_memory import blank_memory
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
@pytest.mark.parametrize("summaries", [1, 4])
@pytest.mark.parametrize(
    "live,bucketed", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("aggregation", ["masked_mean", "value_token"])
def test_trunk_branch_is_exact_and_checkpoint_retains_gates(
    tmp_path: Path,
    encoder: Literal["transformer", "gru"],
    summaries: int,
    live: bool,
    bucketed: bool,
    aggregation: str,
) -> None:
    space, obs = memory_input()
    space = ObservationSpace(
        space.encoder.config.model_copy(update={"max_decision_records": 128})
    )
    obs["decision_records"] = np.pad(
        obs["decision_records"], ((0, 0), (0, 124), (0, 0))
    )
    template = policy(
        space, encoder, summaries, live=live, bucketed=bucketed, aggregation=aggregation
    )
    trunk = ActorCritic(
        space,
        template.config.model_copy(update={"decision_memory": "none"}),
        live_tokens=live,
        bucketed_shapes=bucketed,
    ).eval()
    path = tmp_path / "trunk.safetensors"
    save_checkpoint(trunk, space, path, player_configs=None)
    arm, _ = branch_from_trunk(
        str(path), template.config, live_tokens=live, bucketed_shapes=bucketed
    )
    longer = {key: value.copy() for key, value in obs.items()}
    for index in range(128):
        longer["decision_records"][0, index] = obs["decision_records"][0, 0]
        for column in (1, 7, 9):
            longer["decision_records"][0, index, column] = index
    mixed = {
        key: np.concatenate((value, blank_memory(obs)[key]), axis=0)
        for key, value in longer.items()
    }
    for rows in (obs, blank_memory(obs), mixed):
        for context in (torch.enable_grad(), torch.inference_mode()):
            with context:
                for actual, expected in zip(
                    forward(arm, rows), forward(trunk, rows), strict=True
                ):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert arm.training_metrics() == {"gate_0": 0, "gate_1": 0}
    with torch.no_grad():
        assert arm.attention.memory_gate is not None
        arm.attention.memory_gate.fill_(0.25)
    saved = tmp_path / "arm.safetensors"
    save_checkpoint(arm, space, saved, player_configs=None)
    restored, _ = load_checkpoint(str(saved))
    assert restored.training_metrics() == arm.training_metrics()
    assert load_metadata(saved)["trunk_source"] == load_metadata(path)["source"]


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
@pytest.mark.parametrize("live", [False, True])
def test_zero_gate_receives_gradient_then_encoder_learns(
    encoder: Literal["transformer", "gru"],
    live: bool,
) -> None:
    space, obs = memory_input()
    model = policy(space, encoder, 1, live=live)
    with torch.no_grad():
        for layer in (model.attention, *model.extra_attention):
            assert layer.memory_gate is not None
            layer.memory_gate.zero_()
    logits, value = forward(model, obs)
    (logits.softmax(-1)[:, 0].sum() + value.sum()).backward()
    for layer in (model.attention, *model.extra_attention):
        assert layer.memory_gate is not None
        assert layer.memory_gate.grad is not None
        assert layer.memory_gate.grad.abs() > 0
    assert model.decision_memory is not None
    assert all(
        p.grad is None or torch.count_nonzero(p.grad) == 0
        for p in model.decision_memory.parameters()
    )
    # Real damped update is covered by the real-prefix collection test; this isolates the staged
    # gradient contract without depending on sampled return signs.
    torch.optim.Adam(model.parameters(), lr=0.01).step()
    model.zero_grad(set_to_none=True)
    logits, value = forward(model, obs)
    (logits.softmax(-1)[:, 0].sum() + value.sum()).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.decision_memory.parameters()
    )


def test_branch_refuses_other_architecture_and_nonmemory_space(tmp_path: Path) -> None:
    space, _ = memory_input()
    template = policy(space, "gru", 1)
    trunk = ActorCritic(
        space, template.config.model_copy(update={"decision_memory": "none"})
    )
    path = tmp_path / "trunk.safetensors"
    save_checkpoint(trunk, space, path, player_configs=None)
    with pytest.raises(ValueError, match="settings"):
        branch_from_trunk(
            str(path), template.config.model_copy(update={"hidden_dim": 32})
        )
    with pytest.raises(ValueError, match="memory arm"):
        branch_from_trunk(str(path), trunk.config)
    ordinary = ObservationSpace()
    save_checkpoint(
        ActorCritic(ordinary, trunk.config), ordinary, path, player_configs=None
    )
    with pytest.raises(ValueError, match="capacities"):
        branch_from_trunk(str(path), template.config)


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan")])
def test_gate_scale_requires_finite_positive_value(scale: float) -> None:
    with pytest.raises(ValueError):
        ModelConfig(decision_memory_gate_scale=scale)
