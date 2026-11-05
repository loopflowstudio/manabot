"""Operator diagnosis includes real backward/Adam without changing updates."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from manabot.env.observation import ObservationConfig, ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.model.actor_critic import ActorCritic
from manabot.training.learning.health import NumericalHealth, OptimizerHealth
from manabot.verify.representation import shared_effect_root
from scripts.profile_representation import OperatorTraces


def test_trace_covers_one_complete_step_and_preserves_updates(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(74200)
    space = ObservationSpace(ObservationConfig(policy_input_version=7))
    env = shared_effect_root(2)
    obs = {
        key: torch.from_numpy(value)[None]
        for key, value in space.encode(
            env.observation_for_player(env.current_agent_index())
        ).items()
    }
    model = ActorCritic(space, ModelConfig(hidden_dim=16, state_encoder="typed_gru"))
    reference = deepcopy(model)

    def update(agent: ActorCritic) -> None:
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
        health = NumericalHealth(optimizer_steps=0, sampled_steps=0, rejected_steps=0)
        observer = OptimizerHealth(agent, optimizer, health, iteration=1)
        with torch.no_grad():
            agent.forward_distribution(obs)
        for _ in range(2):
            _, values = agent.forward_distribution(obs)
            observer.step(values.square().mean(), 0.5)
        assert health["optimizer_steps"] == 2

    update(reference)
    traces = OperatorTraces(tmp_path, "cpu")
    with traces.capture():
        update(model)
    assert traces.completed == {"inference", "learning"}
    for actual, expected in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    events = json.loads((tmp_path / "learning.trace.json").read_text())["traceEvents"]
    names = [event.get("name") for event in events]
    assert names.count("learning.forward") == 1
    assert names.count("learning.backward_and_clipping") == 1
    assert names.count("learning.optimizer_and_health") == 1
    assert "Optimizer.step#Adam.step" in names
    assert any("Backward" in str(name) for name in names)
    assert (
        "aten::_foreach_norm" in names
    )  # Actual gradient clipping, not a label alone.


def test_failed_capture_is_retained_as_partial(tmp_path: Path) -> None:
    traces = OperatorTraces(tmp_path, "cpu")
    with pytest.raises(RuntimeError, match="injected"):
        with traces.capture():
            traces._start("learning")
            raise RuntimeError("injected learner failure")
    assert not traces.completed
    assert traces.profiler is None
    assert (tmp_path / "learning.partial.trace.json").exists()
    assert not (tmp_path / "learning.trace.json").exists()
