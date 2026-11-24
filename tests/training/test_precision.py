"""Declared training precisions: admission, TF32 switches and bf16 forwards."""

from contextlib import AbstractContextManager
from typing import Literal, cast

import numpy as np
import pytest
import torch

import manabot.backends.torch as torch_backend
from manabot.backends.torch import learner as torch_learner
from manabot.backends.torch.collection import (
    CollectionSession as TorchCollectionSession,
)
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, RewardConfig
from manabot.training.learning import DampedPolicyLearning, Precision
from manabot.training.rollout.collection import close_collector, collect, open_collector
from manabot.training.updates.damped_policy import update_damped_iteration


def test_tf32_sets_both_switches_and_float32_restores_the_process_defaults() -> None:
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch_learner.apply_matmul_precision("tf32")
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
        torch_learner.apply_matmul_precision("bf16")
        assert (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        ) == before
    finally:
        torch_learner.apply_matmul_precision("float32")
    assert (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    ) == before


@pytest.mark.parametrize("declared", ["float32", "tf32"])
def test_only_bf16_autocasts(declared: Precision) -> None:
    with torch_learner.forward_precision(declared, torch.device("cuda")):
        assert not torch.is_autocast_enabled("cuda")


@pytest.mark.filterwarnings("ignore:CUDA is not available")
@pytest.mark.filterwarnings("ignore:.*MPS.*:UserWarning")
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_bf16_autocasts_gpu_forwards(device: str) -> None:
    context = torch_learner.forward_precision("bf16", torch.device(device))
    assert isinstance(context, torch.autocast)
    assert (context.device, context.fast_dtype) == (device, torch.bfloat16)


@pytest.mark.parametrize("precision", ["bf16", "bfloat16", "fp32"])
def test_cpu_refuses_unsupported_torch_precision(precision: str) -> None:
    declared = cast(Precision, precision)  # unchecked input from a caller
    with pytest.raises(ValueError, match="precision"):
        torch_learner.forward_precision(declared, torch.device("cpu"))
    agent = ActorCritic(
        ObservationSpace(), ModelConfig(hidden_dim=8, num_attention_heads=2)
    )
    with pytest.raises(ValueError, match="precision"):
        Learner(agent, "cpu", minibatch_rows=1, precision=declared)


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_bf16_wraps_exactly_the_policy_forwards_and_keeps_float32_outputs(
    kind: Literal["scalar", "categorical_wdl"], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both collection and learning run their real GPU forwards under autocast."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_macos_or_newer(
        14, 0
    ):
        device = "mps"
    else:
        pytest.skip("bfloat16 collection and learning require CUDA or MPS")
    torch.manual_seed(5)
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=16, num_attention_heads=2, value_kind=kind),
    ).to(device)
    forwards: list[bool] = []
    entries: list[bool] = []
    for name in ("forward", "forward_distribution"):
        method = getattr(agent, name)

        def tracked(*args: object, _method: object = method) -> object:
            forwards.append(torch.is_autocast_enabled(device))
            return _method(*args)  # type: ignore[operator]

        monkeypatch.setattr(agent, name, tracked)
    real = torch_learner.forward_precision

    def recorded(
        declared: Precision, device: torch.device
    ) -> AbstractContextManager[object]:
        entries.append(declared == "bf16")
        return real(declared, device)

    for module in (
        "manabot.backends.torch.collection",
        "manabot.backends.torch.learner",
    ):
        monkeypatch.setattr(f"{module}.forward_precision", recorded)
    collector = open_collector(
        agent.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=2,
        opponent_mode="self",
        learning_seats="one",
        batch_closure="exact-quota",
        numerics=TorchCollectionSession(device=device, seed=0, precision="bf16"),
    )

    batch = collect(collector, agent, 1)
    learner = Learner(agent, device, minibatch_rows=2, precision="bf16")
    result = update_damped_iteration(
        torch_backend,
        learner,
        batch,
        DampedPolicyLearning(
            gradient="damped_policy", advantage_quantile=0, min_advantage=0
        ),
        1,
    )
    close_collector(collector)

    assert forwards and all(forwards)
    assert entries and all(entries)
    assert batch.values.dtype == np.float32 and np.isfinite(batch.values).all()
    assert np.isfinite(batch.logprobs).all()
    assert result["rows"] > 0
    assert all(p.dtype == torch.float32 for p in agent.parameters())
    assert not torch.is_autocast_enabled(device)
