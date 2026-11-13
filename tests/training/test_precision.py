"""Declared training precisions: admission, TF32 switches and bf16 forwards."""

from typing import Literal

import numpy as np
from pydantic import ValidationError
import pytest
import torch

import manabot.backends.torch as torch_backend
from manabot.backends.torch import learner as torch_learner
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, RewardConfig
from manabot.training.learning import DampedPolicyLearning, Precision
from manabot.training.regime import Execution
from manabot.training.rollout.collection import close_collector, collect, open_collector
from manabot.training.updates.damped_policy import update_damped_iteration


@pytest.mark.parametrize("declared", ["tf32", "bf16"])
def test_reduced_precision_requires_cuda(declared: Precision) -> None:
    with pytest.raises(ValidationError, match="requires CUDA"):
        Execution(precision=declared)
    assert Execution(device="cuda", precision=declared).precision == declared


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
def test_bf16_autocasts_cuda_forwards() -> None:
    context = torch_learner.forward_precision("bf16", torch.device("cuda"))
    assert isinstance(context, torch.autocast)
    assert (context.device, context.fast_dtype) == ("cuda", torch.bfloat16)


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_bf16_wraps_exactly_the_policy_forwards_and_keeps_float32_outputs(
    kind: Literal["scalar", "categorical_wdl"], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real attention agent collects and learns under bf16 autocast (CPU).

    This proves CPU context wiring only. CUDA's recurrent kernels choose
    different dtypes; the CUDA admission test covers the semantic-card path.
    """
    torch.manual_seed(5)
    torch.set_num_threads(1)
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=16, num_attention_heads=2, value_kind=kind),
    )
    forwards: list[bool] = []
    entries: list[bool] = []
    for name in ("forward", "forward_distribution"):
        method = getattr(agent, name)

        def tracked(*args: object, _method: object = method) -> object:
            forwards.append(torch.is_autocast_enabled("cpu"))
            return _method(*args)  # type: ignore[operator]

        monkeypatch.setattr(agent, name, tracked)
    real = torch_learner.forward_precision

    def recorded(declared: Precision, device: torch.device) -> object:
        entries.append(declared == "bf16")
        return real(declared, device)

    for module in (
        "manabot.training.rollout.collection",
        "manabot.backends.torch.learner",
    ):
        monkeypatch.setattr(f"{module}.forward_precision", recorded)
    collector = open_collector(
        agent.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=2,
        opponent_mode="self",
        learning_seats="both",
        precision="bf16",
    )

    batch = collect(collector, agent, 1)
    learner = Learner(agent, "cpu", minibatch_rows=2, precision="bf16")
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
    assert not torch.is_autocast_enabled("cpu")
