"""Input v11 stack rows reach the policy, and v9 checkpoints keep their rows.

Input v11 appends one stack row per spell or ability after the permanents.
Bucketed shapes and live tokens must agree with the plain forward on batches
that hold stack rows, and the deprecated v9 layout (no stack) must still run.
"""

import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import STACK_DEFINITION_INDEX, encode_observation
from manabot.training.rollout.collection import collect
from manabot.verify.text_observation import priority_root
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


@pytest.mark.parametrize("input_version", [11, 9])
def test_bucketed_and_live_forwards_match_plain(input_version: int) -> None:
    torch.manual_seed(344)
    case = _policy(input_version=input_version)
    if input_version == 9:
        # Legacy compatibility uses a representable board. Full games may
        # acquire positional library facts that this frozen layout cannot hold.
        env = priority_root(("Lightning Bolt",), ("Mountain", "Gray Ogre"))
        observation = env.observation_for_player(env.current_agent_index())
        obs = {
            key: torch.from_numpy(value[None]).float()
            for key, value in encode_observation(
                case.agent.observation_space, observation
            ).items()
        }
    else:
        obs = _decisions(case, 48)
    if input_version == 11:
        assert case.agent.stack_tokens
        assert obs["stack_valid"].any(), "fixture should put a spell on the stack"
        assert (
            obs["stack"][..., STACK_DEFINITION_INDEX][obs["stack_valid"] > 0] > 0
        ).all()
    else:
        assert not case.agent.stack_tokens and "stack" not in obs
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


def test_v11_model_refuses_v9_weights_by_shape() -> None:
    v9 = _policy(input_version=9).agent
    v11 = _policy().agent
    assert v9.perm_embedding.projection[0].in_features < (
        v11.perm_embedding.projection[0].in_features
    )
    with pytest.raises(RuntimeError, match="size mismatch|Missing key"):
        v11.load_state_dict(v9.state_dict())
