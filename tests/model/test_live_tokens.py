"""Live-token attention agrees with the padded path on admitted inputs."""

import pytest
import torch

from manabot.backends.torch.live_tokens import LiveTokens
from manabot.backends.torch.policy import ActorCritic, PolicyInference
from manabot.policy.prepare import OBJECT_VALIDITY_KEYS
from manabot.training.rollout.collection import collect
from tests.model.prepared_routes import device_input, host_inputs
from tests.model.test_inference import PolicyFixture, _policy


def _pair(
    case: PolicyFixture, value_aggregation: str, bucketed: bool
) -> tuple[ActorCritic, ActorCritic]:
    """Padded and live-token policies with identical, perturbed weights."""
    config = case.agent.config.model_copy(
        update={"value_aggregation": value_aggregation, "attention_layers": 2}
    )
    space = case.agent.observation_space
    padded = ActorCritic(space, config, bucketed_shapes=bucketed).eval()
    live = ActorCritic(space, config, bucketed_shapes=bucketed, live_tokens=True)
    live.eval().load_state_dict(padded.state_dict())
    # Fresh fact and link outputs are zero; give those joins real signal.
    with torch.no_grad():
        for reference, copy in zip(padded.parameters(), live.parameters(), strict=True):
            noise = torch.randn_like(reference) * 0.05
            reference.add_(noise)
            copy.add_(noise)
    return padded, live


def _later_decisions(case: PolicyFixture) -> dict[str, torch.Tensor]:
    """Decisions from played turns, so boards hold different object counts."""
    batch = collect(case.collector, case.agent, 24)
    return {
        key: torch.from_numpy(value[-8:, 0]).float()
        for key, value in batch.observations().observation.items()
    }


@pytest.mark.parametrize(
    ("value_aggregation", "bucketed"),
    [("masked_mean", False), ("value_token", False), ("masked_mean", True)],
)
def test_outputs_and_gradients_match_padded(
    value_aggregation: str, bucketed: bool
) -> None:
    torch.manual_seed(319)
    case = _policy()
    obs = _later_decisions(case)
    valid = sum(
        obs[key].sum(1) for key in ("agent_cards_valid", "agent_permanents_valid")
    )
    assert len(set(valid.tolist())) > 1, "fixture boards should differ in size"
    padded, live = _pair(case, value_aggregation, bucketed)
    outputs = []
    for model in (padded, live):
        logits, values = model.forward_distribution(device_input(model, obs))
        outputs.append((logits, values))
        legal = obs["actions_valid"].bool()
        (logits[legal].square().mean() + values.square().mean()).backward()
    (expected_logits, expected_values), (logits, values) = outputs
    torch.testing.assert_close(logits, expected_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(values, expected_values, atol=1e-5, rtol=1e-5)
    for (name, a), b in zip(padded.named_parameters(), live.parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None or not b.grad.any(), name
        else:
            assert b.grad is not None, name
            torch.testing.assert_close(b.grad, a.grad, atol=1e-5, rtol=1e-4)
    with torch.no_grad(), PolicyInference(live) as scope:
        prepared = scope.prepare(host_inputs(obs))
        present = sum(
            int(obs[key].count_nonzero())
            for key in (*OBJECT_VALIDITY_KEYS, "stack_valid")
            if key in obs
        )
        tokens = len(obs["actions_valid"]) if value_aggregation == "value_token" else 0
        assert prepared is not None
        coordinates = prepared
        assert coordinates.live_index is not None
        assert coordinates.live_offsets is not None
        assert coordinates.live_index.numel() == present + tokens
        assert coordinates.live_offsets[-1].item() == present + tokens
        served, served_values = live.forward_distribution(prepared)
    torch.testing.assert_close(served, expected_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(served_values, expected_values, atol=1e-5, rtol=1e-5)


def test_padded_policy_measures_no_live_count() -> None:
    case = _policy()
    with torch.no_grad(), PolicyInference(case.agent) as scope:
        batch = scope.prepare(host_inputs(case.obs))
        assert batch is not None and batch.live_index is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA attention backward")
def test_short_packed_minibatch_cuda_backward() -> None:
    attention = torch.nn.MultiheadAttention(64, 4, batch_first=True, device="cuda")
    packed = torch.randn(5, 64, requires_grad=True, device="cuda")
    live = LiveTokens(
        torch.tensor([0, 1, 58, 59, 60], device="cuda"),
        torch.tensor([0, 2, 5], device="cuda"),
        batch=2,
        slots=58,
    )
    (actual,) = torch.autograd.grad(
        live.self_attention(attention, packed).sum(), packed
    )
    reference = sum(
        attention(row, row, row, need_weights=False)[0].sum()
        for row in packed.split((2, 3))
    )
    (expected,) = torch.autograd.grad(reference, packed)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
