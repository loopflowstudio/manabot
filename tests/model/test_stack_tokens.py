"""Stack rows carry spells and abilities into each policy forward."""

from dataclasses import replace

import numpy as np
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import STACK_DEFINITION_INDEX
from manabot.policy.prepare import Prepared
from manabot.policy.spans import ObservationSelection
from manabot.training.rollout.collection import collect
from tests.model.prepared_routes import device_input
from tests.model.test_inference import PolicyFixture, _policy


def _decisions(case: PolicyFixture, steps: int) -> ObservationSelection:
    """Every collected decision of both environments, in rollout order."""
    batch = collect(case.collector, case.agent, steps)
    return batch.observations(np.arange(batch.actions.size)).observation


def _without_stack(prepared: Prepared[np.ndarray]) -> Prepared[np.ndarray]:
    """The same rows with every stack token absent; the stack is the last segment."""
    assert prepared.stack is not None and prepared.stack_rows is not None
    assert prepared.stack_mask is not None and prepared.belief is None
    capacity = prepared.stack_rows.shape[1]
    validity, owners = prepared.validity.copy(), prepared.is_agent.copy()
    validity[:, -capacity:] = 0
    owners[:, -capacity:] = False
    return replace(
        prepared,
        validity=validity,
        is_agent=owners,
        stack=np.zeros_like(prepared.stack),
        stack_rows=np.zeros_like(prepared.stack_rows),
        stack_mask=np.zeros_like(prepared.stack_mask),
    )


def _variants(case: PolicyFixture) -> list[ActorCritic]:
    """Plain, bucketed and live-token copies with identical perturbed weights."""
    space, config = case.agent.observation_space, case.agent.config
    plain = ActorCritic(space, config).eval()
    with torch.no_grad():
        for parameter in plain.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    models = [plain]
    for options in ({"bucketed_shapes": True}, {"live_tokens": True}):
        model = ActorCritic(space, config, **options).eval()
        model.load_state_dict(plain.state_dict())
        models.append(model)
    return models


def test_bucketed_and_live_forwards_match_plain() -> None:
    torch.manual_seed(344)
    case = _policy()
    obs = _decisions(case, 48)
    stack, valid = obs.project("stack"), obs.project("stack_valid") > 0
    assert valid.any(), "fixture should put a spell on the stack"
    assert (stack[..., STACK_DEFINITION_INDEX][valid] > 0).all()
    outputs = []
    with torch.no_grad():
        for model in _variants(case):
            outputs.append(model.forward_distribution(device_input(model, obs)))
    (logits, values), *others = outputs
    for other_logits, other_values in others:
        torch.testing.assert_close(other_logits, logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(other_values, values, atol=1e-5, rtol=1e-5)


def test_stack_rows_change_the_policy() -> None:
    """A stack row is an attended token: removing it changes some output."""
    torch.manual_seed(345)
    case = _policy()
    obs = _decisions(case, 48)
    rows = torch.from_numpy(obs.project("stack_valid").any(1))
    assert rows.any()
    model = _variants(case)[0]
    prepared = model.prepare(obs, inference=True)
    with torch.no_grad():
        _, values = model.forward_distribution(device_input(model, prepared))
        _, without = model.forward_distribution(
            device_input(model, _without_stack(prepared))
        )
    assert not torch.allclose(values[rows], without[rows])
