"""Semantic program joins: float32 parity and mandatory rented-CUDA admission.

MANABOT_REQUIRE_CUDA=1 makes missing CUDA a failure, never a successful skip.
The admission command prints actual recurrent output dtypes for the run log.
"""

from copy import deepcopy
import io
import json
import os
from typing import Literal

import numpy as np
import pytest
import torch
from torch import nn

from experiments.elemental_variants import elemental_recipe
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.inference import PolicyInference
from manabot.model.semantic_cards import SemanticCardEncoder
from manabot.training.precision import forward_precision
from manabot.training.rollout.collection import SeatRoutedCollector


def test_definition_join_preserves_float32_outputs_and_gradients() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(26410)
    encoder = SemanticCardEncoder("ur-lessons-vs-gw-allies", 8)
    reference = deepcopy(encoder)
    rows = torch.tensor([1, 2, 3])
    actual = encoder.definitions(rows)
    expected = reference.token_embedding.weight.new_zeros(actual.shape).index_copy(
        0, rows, reference._encode_programs(rows)
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for a, b in zip(encoder.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_definition_join_accepts_recurrent_output_dtype(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject the observed CUDA boundary on CPU; this is not CUDA admission."""
    encoder = SemanticCardEncoder("ur-lessons-vs-gw-allies", 8)
    original = encoder._encode_programs

    def reduced(rows: torch.Tensor) -> torch.Tensor:
        return original(rows).to(dtype)

    monkeypatch.setattr(encoder, "_encode_programs", reduced)
    table = encoder.definitions(torch.tensor([1, 2]))
    assert table.dtype == torch.float32
    table.sum().backward()
    assert encoder.token_embedding.weight.grad is not None
    empty = encoder.definitions(torch.empty(0, dtype=torch.long))
    assert empty.dtype == table.dtype and not empty.any()


@pytest.mark.parametrize("declared", ["float32", "bf16"])
def test_cuda_semantic_policy_admission(declared: Literal["float32", "bf16"]) -> None:
    if not torch.cuda.is_available():
        if os.environ.get("MANABOT_REQUIRE_CUDA") == "1":
            pytest.fail("CUDA admission requires a real CUDA device")
        pytest.skip("CUDA admission runs on the bounded rental")
    if declared == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.fail("CUDA device does not support BF16")
    torch.set_num_threads(1)
    torch.manual_seed(26400)
    regime = elemental_recipe("precision-admission-fixture", 2)
    assert regime.observation.policy_input_version == 6
    assert not regime.agent.catalog_records(regime.observation.policy_input_version)
    regime.agent.hidden_dim = 16
    regime.agent.num_attention_heads = 2
    space = ObservationSpace(regime.observation)
    agent = ActorCritic(space, regime.agent).cuda().train()
    cards = agent.semantic_cards
    assert cards is not None
    dtypes: set[str] = set()

    def record_dtype(
        module: nn.Module, args: tuple[object, ...], result: object
    ) -> None:
        assert isinstance(result, tuple) and isinstance(result[1], torch.Tensor)
        dtypes.add(str(result[1].dtype))

    hook = cards.encoder.register_forward_hook(record_dtype)
    collector = SeatRoutedCollector(
        space,
        Match(regime.match),
        Reward(RewardConfig()),
        num_envs=2,
        seed=26400,
        opponent_mode="self",
        recovery_max_microsteps=10000,
    )
    try:
        host = collector._slice_obs_tensors(np.arange(2))
        obs = {key: value.cuda() for key, value in host.items()}
        with forward_precision(declared, torch.device("cuda")):
            full = cards.definitions()
            selected = cards.definitions(torch.tensor([1, 2], device="cuda"))
            empty = cards.definitions(torch.empty(0, dtype=torch.long, device="cuda"))
        assert full.dtype == selected.dtype == empty.dtype == torch.float32
        assert not empty.any() and torch.isfinite(full).all()
        # Both full and selected paths retain GRU/embedding gradients.
        (full.square().mean() + selected.square().mean()).backward()
        assert cards.token_embedding.weight.grad is not None
        assert torch.isfinite(cards.token_embedding.weight.grad).all()
        agent.zero_grad(set_to_none=True)
        with (
            PolicyInference(agent) as scope,
            torch.no_grad(),
            forward_precision(declared, torch.device("cuda")),
        ):
            prepared = scope.prepare(host)
            first = agent.forward_distribution(prepared)
            second = agent.forward_distribution(prepared)
            for a, b in zip(first, second, strict=True):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        optimizer = torch.optim.Adam(agent.parameters())
        with forward_precision(declared, torch.device("cuda")):
            logits, values = agent.forward_distribution(obs)
        loss = (
            logits[obs["actions_valid"].bool()].float().square().mean()
            + values.float().square().mean()
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in agent.parameters()
            if p.grad is not None
        )
        optimizer.step()
        checkpoint = io.BytesIO()
        torch.save(agent.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = ActorCritic(space, regime.agent).cuda().train()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        with torch.no_grad(), forward_precision(declared, torch.device("cuda")):
            for a, b in zip(
                agent.forward_distribution(obs),
                restored.forward_distribution(obs),
                strict=True,
            ):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        print(
            json.dumps(
                {
                    "precision": declared,
                    "gru_output_dtypes": sorted(dtypes),
                    "table_dtype": str(full.dtype),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(),
                    "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                }
            )
        )
    finally:
        hook.remove()
        collector.close()
