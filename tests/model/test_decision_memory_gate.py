"""Common-trunk identity and staged learning through the only memory join."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_decision_memory import forward, memory_input, policy
import torch

from manabot.backends.torch.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from manabot.backends.torch.policy import ActorCritic, GameObjectAttention
from manabot.env.decision_memory import blank_memory
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.training.loops.self_play import load_pretrained


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
    admitted, _ = load_checkpoint(str(path))
    arm = ActorCritic(
        space, template.config, live_tokens=live, bucketed_shapes=bucketed
    ).eval()
    load_pretrained(arm, admitted.state_dict(), admitted.config)
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
    assert arm.memory_gates() == {"gate_0": 0, "gate_1": 0}
    with torch.no_grad():
        assert arm.attention.memory_gate is not None
        arm.attention.memory_gate.fill_(0.25)
    for actual, expected in zip(
        forward(arm, blank_memory(obs)), forward(trunk, blank_memory(obs)), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    saved = tmp_path / "arm.safetensors"
    save_checkpoint(arm, space, saved, player_configs=None)
    restored, _ = load_checkpoint(
        str(saved), live_tokens=live, bucketed_shapes=bucketed
    )
    assert restored.memory_gates() == arm.memory_gates()
    for actual, expected in zip(
        forward(restored, mixed), forward(arm, mixed), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


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
        assert torch.isfinite(layer.memory_gate.grad)
        assert layer.memory_gate.grad.abs() > 0
        assert layer.memory_projection is not None
        assert torch.count_nonzero(layer.memory_projection.weight) > 0
        assert layer.memory_projection.weight.grad is not None
        assert torch.count_nonzero(layer.memory_projection.weight.grad) == 0
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
    for layer in (model.attention, *model.extra_attention):
        assert layer.memory_projection is not None
        assert layer.memory_projection.weight.grad is not None
        assert layer.memory_projection.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.decision_memory.parameters()
    )


@pytest.mark.parametrize("fault", ["unexpected", "missing", "shape", "strict"])
def test_pretrained_restore_keeps_shared_state_strict(fault: str) -> None:
    space, _ = memory_input()
    arm = policy(space, "gru", 1)
    trunk = ActorCritic(
        space, arm.config.model_copy(update={"decision_memory": "none"})
    )
    state = dict(trunk.state_dict())
    # Choose a shared learned weight without depending on its local module name.
    key = next(name for name, value in state.items() if value.ndim == 2)
    if fault == "unexpected":
        state["unowned"] = torch.zeros(())
    elif fault == "missing":
        del state[key]
    elif fault == "shape":
        state[key] = state[key][:0]
    with pytest.raises((ValueError, RuntimeError)):
        load_pretrained(arm, state, arm.config if fault == "strict" else trunk.config)


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan")])
def test_gate_scale_requires_finite_positive_value(scale: float) -> None:
    with pytest.raises(ValueError):
        ModelConfig(decision_memory_gate_scale=scale)


def test_additive_join_changes_only_real_board_inputs() -> None:
    """One owner-neutral summary reaches both seats, never padding or the value token."""
    layer = GameObjectAttention(4, 1, ownership=False)
    layer.memory_gate = torch.nn.Parameter(torch.tensor(0.5))
    layer.memory_projection = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        layer.memory_projection.weight.copy_(torch.eye(4))
    objects = torch.randn(2, 4, 4)
    owners = torch.tensor([[True, False, False], [False, True, False]])
    padding = torch.tensor([[False, False, True, False]]).expand(2, -1)
    summaries = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    inputs: list[torch.Tensor] = []

    def retain_input(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        inputs.append(args[0].detach().clone())

    handle = layer.mha.register_forward_pre_hook(retain_input)
    try:
        output = layer(objects, owners, padding, summaries)
    finally:
        handle.remove()
    assert len(inputs) == 1
    expected = objects.clone()
    expected[:, :2] += (
        torch.tanh(layer.memory_gate.detach()) * summaries.mean(1)[:, None]
    )
    torch.testing.assert_close(inputs[0], expected, atol=0, rtol=0)
    assert torch.count_nonzero(output[:, 2]) == 0
