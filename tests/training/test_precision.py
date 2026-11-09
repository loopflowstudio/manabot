"""Declared training precisions: admission, TF32 switches and bf16 forwards."""

from typing import Literal

import numpy as np
from pydantic import ValidationError
import pytest
import torch

from experiments import training_throughput_packing as packing
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.experiment.job import TrainingJob
from manabot.infra.config import ModelConfig, RewardConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.training import precision
from manabot.training.learning.learner import Learner
from manabot.training.learning.objectives import update_iteration
from manabot.training.regime import DampedPolicyLearning, Execution, Precision
from manabot.training.rollout.collection import SeatRoutedCollector


def test_float32_regimes_keep_their_bytes() -> None:
    # Step 2's frozen D1 regime (protocol freeze receipt, `0366ff8d…1c3e`).
    d1 = TrainingJob.model_validate_json(packing.runs("HEAD", "d1")[0].job.config or "")
    assert (
        canonical_sha256(d1.regime.model_dump(mode="json"))
        == "0366ff8d8548a0540976b7e3bb26b96f9326b82b250bfa11539cf197bc731c3e"
    )
    assert Execution().model_dump()["precision"] == "float32"


@pytest.mark.parametrize("declared", ["tf32", "bf16"])
def test_reduced_precision_requires_cuda(declared: Precision) -> None:
    with pytest.raises(ValidationError, match="requires CUDA"):
        Execution(precision=declared)
    assert Execution(device="cuda", precision=declared).precision == declared


def test_tf32_sets_both_switches_and_float32_restores_the_process_defaults() -> None:
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        precision.apply_matmul_precision("tf32")
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
        precision.apply_matmul_precision("bf16")
        assert (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        ) == before
    finally:
        precision.apply_matmul_precision("float32")
    assert (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    ) == before


@pytest.mark.parametrize("declared", ["float32", "tf32"])
def test_only_bf16_autocasts(declared: Precision) -> None:
    with precision.forward_precision(declared, torch.device("cuda")):
        assert not torch.is_autocast_enabled("cuda")


@pytest.mark.filterwarnings("ignore:CUDA is not available")
def test_bf16_autocasts_cuda_forwards() -> None:
    context = precision.forward_precision("bf16", torch.device("cuda"))
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
    real = precision.forward_precision

    def recorded(declared: Precision, device: torch.device) -> object:
        entries.append(declared == "bf16")
        return real(declared, device)

    for module in (
        "manabot.training.rollout.collection",
        "manabot.training.learning.learner",
    ):
        monkeypatch.setattr(f"{module}.forward_precision", recorded)
    collector = SeatRoutedCollector(
        agent.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=2,
        opponent_mode="self",
        learning_seats="both",
        recovery_max_microsteps=10_000,
        precision="bf16",
    )

    batch = collector.collect(agent, 1)
    learner = Learner(agent, "cpu", minibatch_rows=2, precision="bf16")
    result = update_iteration(
        learner,
        batch,
        DampedPolicyLearning(
            gradient="damped_policy", advantage_quantile=0, min_advantage=0
        ),
        0,
        np.random.default_rng(1),
        iteration=1,
    )
    collector.close()

    assert forwards and all(forwards)
    assert entries and all(entries)
    assert batch.values.dtype == np.float32 and np.isfinite(batch.values).all()
    assert np.isfinite(batch.logprobs).all()
    assert result["rows"] > 0
    assert all(p.dtype == torch.float32 for p in agent.parameters())
    assert not torch.is_autocast_enabled("cpu")
