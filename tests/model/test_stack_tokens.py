"""Stack rows carry spells and abilities into each policy forward."""

import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import STACK_DEFINITION_INDEX
from manabot.training.rollout.collection import collect
from tests.fixtures.rollout_arrays import trajectory_rows
from tests.model.prepared_routes import device_input
from tests.model.test_inference import PolicyFixture, _policy


def _decisions(case: PolicyFixture, steps: int) -> dict[str, torch.Tensor]:
    """Every collected decision of both environments, flattened to rows."""
    batch = collect(case.collector, case.agent, steps)
    return {
        key: torch.from_numpy(value).flatten(0, 1).float()
        for key, value in trajectory_rows(batch).observation.items()
    }


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
    assert obs["stack_valid"].any(), "fixture should put a spell on the stack"
    assert (obs["stack"][..., STACK_DEFINITION_INDEX][obs["stack_valid"] > 0] > 0).all()
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
    rows = obs["stack_valid"].any(1)
    assert rows.any()
    model = _variants(case)[0]
    erased = dict(obs)
    erased["stack"] = torch.zeros_like(obs["stack"])
    erased["stack_valid"] = torch.zeros_like(obs["stack_valid"])
    with torch.no_grad():
        _, values = model.forward_distribution(device_input(model, obs))
        _, without = model.forward_distribution(device_input(model, erased))
    assert not torch.allclose(values[rows], without[rows])
