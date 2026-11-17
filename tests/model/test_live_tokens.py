"""Live-token attention agrees with the padded path on admitted inputs."""

from copy import deepcopy

import pytest
import torch

from manabot.backends.torch.live_tokens import LiveTokens
from manabot.backends.torch.policy import ActorCritic, PolicyInference
from manabot.policy.prepare import OBJECT_VALIDITY_KEYS
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
    batch = case.collector.collect(case.agent, 24)
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


@pytest.mark.parametrize("lengths", [(3,), (3, 5)])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_sparse_short_chunk_matches_padded_backward(
    lengths: tuple[int, ...], device: str
) -> None:
    """A final chunk can have fewer live tokens than its 234 padded slots."""
    torch.manual_seed(39100)
    slots, width = 234, 16
    reference = torch.nn.MultiheadAttention(width, 4, batch_first=True).to(device)
    attention = deepcopy(reference)
    padded = torch.randn(len(lengths), slots, width, device=device, requires_grad=True)
    actual = padded.detach().clone().requires_grad_()
    index = torch.tensor(
        [row * slots + col for row, size in enumerate(lengths) for col in range(size)],
        device=device,
    )
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device=device
    )
    coordinates = LiveTokens(index, offsets, len(lengths), slots)
    absent = (
        torch.arange(slots, device=device)[None]
        >= torch.tensor(lengths, device=device)[:, None]
    )
    expected, _ = reference(
        padded, padded, padded, key_padding_mask=absent, need_weights=False
    )
    expected = coordinates.gather(expected)
    # Float32 jagged attention on CUDA exercises the memory-efficient kernel's
    # backward bound check; CPU retains the same output/gradient parity proof.
    result = coordinates.self_attention(attention, coordinates.gather(actual))
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
    result.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual.grad, padded.grad, atol=1e-5, rtol=1e-5)
    for parameter, target in zip(
        attention.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter.grad, target.grad, atol=1e-5, rtol=1e-5)
