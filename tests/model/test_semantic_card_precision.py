"""Semantic program and GRU decision-memory joins: float32 parity and CUDA admission.

MANABOT_REQUIRE_CUDA=1 makes missing CUDA a failure, never a successful skip.
The admission command prints actual recurrent output dtypes for the run log.
"""

from copy import deepcopy
import io
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from manabot.backends.torch.arrays import device_of, on_device
from manabot.backends.torch.collection import (
    CollectionSession as TorchCollectionSession,
)
from manabot.backends.torch.learner import forward_precision
from manabot.backends.torch.policy import ActorCritic, PolicyInference
from manabot.backends.torch.semantic_cards import SemanticCardEncoder
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.policy.spans import ObservationSelection
from manabot.training.rollout import collection
from manabot.training.rollout.collection import (
    close_collector,
    open_collector,
)
from tests.model.prepared_routes import device_input, program_input
from tests.model.test_decision_memory import policy
from tests.model.test_packed_gru import cast_gru_output
from tests.training.test_memory_learner_fold import memory_fold_observation

ROOT = Path(__file__).resolve().parents[2]


def exercise_gru_memory(
    model: ActorCritic,
    retained: ObservationSelection,
    precision: Literal["float32", "bf16"] = "float32",
) -> None:
    """Exercise cold/warm serving folds, learner fold, and stateless GRU memory."""
    assert model.decision_memory is not None and model.decision_memory.gru is not None
    device = device_of(model)
    with (
        PolicyInference(model) as scope,
        torch.no_grad(),
        forward_precision(precision, device),
    ):
        cold = scope.prepare(retained)
        assert cold.memory is not None and cold.memory.fold is not None
        assert any(level.record_nodes is not None for level in cold.memory.fold.levels)
        assert cold.memory.fold.retain is not None
        assert cold.memory.fold.root_states is None
        cold_logits, cold_values = model.forward_distribution(cold)
        assert torch.isfinite(cold_logits).all() and torch.isfinite(cold_values).all()

        warm = scope.prepare(retained)
        assert warm.memory is not None and warm.memory.fold is not None
        assert warm.memory.fold.root_states is not None
        assert warm.memory.fold.root_references is not None
        warm_logits, warm_values = model.forward_distribution(warm)
        assert torch.isfinite(warm_logits).all() and torch.isfinite(warm_values).all()

    for batch, expect_fold in (
        (on_device(model.prepare(retained), device), True),
        (
            on_device(
                model.prepare(
                    ObservationSelection(retained.space, retained.coordinates)
                ),
                device,
            ),
            False,
        ),
    ):
        assert batch.memory is not None
        assert (batch.memory.fold is not None) is expect_fold
        model.zero_grad(set_to_none=True)
        with forward_precision(precision, device):
            logits, values = model.forward_distribution(batch)
        assert torch.isfinite(logits).all() and torch.isfinite(values).all()
        loss = logits.float().square().mean() + values.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        gru_grads = [
            p.grad for p in model.decision_memory.gru.parameters() if p.grad is not None
        ]
        assert gru_grads and all(torch.isfinite(g).all() for g in gru_grads)
        assert sum(float(g.abs().sum()) for g in gru_grads) > 0


def test_definition_join_preserves_float32_outputs_and_gradients() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(26410)
    encoder = SemanticCardEncoder("ur-lessons-vs-gw-allies", 8)
    reference = deepcopy(encoder)
    rows = torch.tensor([1, 2, 3])
    actual = encoder.definitions(program_input(encoder, rows))
    expected = reference.token_embedding.weight.new_zeros(actual.shape).index_copy(
        0, rows, reference._encode_programs(program_input(reference, rows))
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for a, b in zip(encoder.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_definition_join_accepts_recurrent_output_dtype(dtype: torch.dtype) -> None:
    """Inject the observed CUDA boundary at `nn.GRU` on CPU; not CUDA admission."""
    encoder = SemanticCardEncoder("ur-lessons-vs-gw-allies", 8)
    encoder.encoder.register_forward_hook(cast_gru_output(dtype))
    table = encoder.definitions(program_input(encoder, torch.tensor([1, 2])))
    assert table.dtype == torch.float32
    assert torch.isfinite(table).all()
    table.sum().backward()
    assert encoder.token_embedding.weight.grad is not None
    assert torch.isfinite(encoder.token_embedding.weight.grad).all()
    assert encoder.token_embedding.weight.grad.abs().sum() > 0
    empty = encoder.definitions(
        program_input(encoder, torch.empty(0, dtype=torch.long))
    )
    assert empty.dtype == table.dtype and not empty.any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decision_memory_joins_accept_recurrent_output_dtype(
    dtype: torch.dtype,
) -> None:
    """Inject reduced `nn.GRU` outputs across serving, learner and stateless memory."""
    space, retained = memory_fold_observation()
    model = policy(space, "gru", 4, live=True, bucketed=True)
    assert model.decision_memory is not None and model.decision_memory.gru is not None
    model.decision_memory.gru.register_forward_hook(cast_gru_output(dtype))
    exercise_gru_memory(model, retained)


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
    # The MTG-183 run's model, inputs and match, read from its saved record.
    record = json.loads(
        (ROOT / "experiments/regimes/elemental-long-183-v1.json").read_bytes()
    )
    model = ModelConfig.model_validate(
        {**record["agent"], "hidden_dim": 16, "num_attention_heads": 2}
    )
    space = ObservationSpace(ObservationConfig.model_validate(record["observation"]))
    agent = ActorCritic(space, model).cuda().train()
    cards = agent.semantic_cards
    assert cards is not None
    mem_space, retained = memory_fold_observation()
    mem_agent = policy(mem_space, "gru", 4, live=True, bucketed=True).cuda().train()
    assert (
        mem_agent.decision_memory is not None
        and mem_agent.decision_memory.gru is not None
    )
    dtypes: set[str] = set()

    def record_dtype(
        module: nn.Module, args: tuple[object, ...], result: object
    ) -> None:
        assert isinstance(result, tuple) and isinstance(result[1], torch.Tensor)
        dtypes.add(str(result[1].dtype))
        if isinstance(result[0], PackedSequence):
            dtypes.add(str(result[0].data.dtype))

    hooks = [
        cards.encoder.register_forward_hook(record_dtype),
        mem_agent.semantic_cards.encoder.register_forward_hook(record_dtype),
        mem_agent.decision_memory.gru.register_forward_hook(record_dtype),
    ]
    collector = open_collector(
        space,
        Match(MatchConfig.model_validate(record["match"])),
        Reward(RewardConfig()),
        num_envs=2,
        seed=26400,
        opponent_mode="self",
        numerics=TorchCollectionSession(device="cpu", seed=26400, precision="float32"),
    )
    try:
        host = collection._retain_obs_rows(collector, np.arange(2))
        with forward_precision(declared, torch.device("cuda")):
            full = cards.definitions(program_input(cards))
            selected = cards.definitions(
                program_input(cards, torch.tensor([1, 2], device="cuda"))
            )
            empty = cards.definitions(
                program_input(cards, torch.empty(0, dtype=torch.long, device="cuda"))
            )
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
            logits, values = agent.forward_distribution(device_input(agent, host))
        loss = (
            logits[torch.from_numpy(host["actions_valid"]).cuda().bool()]
            .float()
            .square()
            .mean()
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
        restored = ActorCritic(space, model).cuda().train()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        with torch.no_grad(), forward_precision(declared, torch.device("cuda")):
            for a, b in zip(
                agent.forward_distribution(device_input(agent, host)),
                restored.forward_distribution(device_input(restored, host)),
                strict=True,
            ):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        exercise_gru_memory(mem_agent, retained, declared)
        assert dtypes == {"torch.float32"}
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
        for hook in hooks:
            hook.remove()
        close_collector(collector)
