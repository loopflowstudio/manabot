"""Declared training precisions: admission, TF32 switches and bf16 forwards."""

from typing import Literal

import numpy as np
from pydantic import ValidationError
import pytest
import torch

from experiments import training_throughput_packing as packing
from experiments.batched_baseline import batched_baseline
import manabot.backends.torch as torch_backend
from manabot.backends.torch import learner as torch_learner
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.experiment.job import TrainingJob
from manabot.infra.config import BF16_MIN_HIDDEN_DIM, ModelConfig, RewardConfig
from manabot.infra.identity import canonical_sha256
from manabot.recipes.baseline import baseline
from manabot.recipes.builders import with_capacity
from manabot.training.regime import (
    DampedPolicyLearning,
    Device,
    Execution,
    Precision,
    TrainingRegime,
    ladder_precision,
)
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.updates.damped_policy import update_damped_iteration


def test_float32_regimes_keep_their_bytes() -> None:
    # The D1 packing regime omits the input version, so it resolves to v9.
    d1 = TrainingJob.model_validate_json(packing.runs("HEAD", "d1")[0].job.config or "")
    assert (
        canonical_sha256(d1.regime.model_dump(mode="json"))
        == "5684d15da200eef2c9169b75cce8b54dd643e22a0201a148be80bf2958505b84"
    )
    assert batched_baseline().stages[0].execution.precision == "float32"


def _at_width(regime: TrainingRegime, width: int) -> TrainingRegime:
    return with_capacity(regime, id=f"w{width}", width=width, depth=8, heads=8)


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (BF16_MIN_HIDDEN_DIM - 8, "float32"),
        (BF16_MIN_HIDDEN_DIM, "bf16"),
        (384, "bf16"),
    ],
)
def test_auto_is_bf16_on_cuda_from_the_threshold_width(
    width: int, expected: Precision
) -> None:
    base = batched_baseline()
    assert base.stages[0].execution.device == "cuda"
    wide = _at_width(base, width)
    assert wide.stages[0].execution.resolved_precision == expected
    # A narrower rebuild of a wide regime resolves the ladder again.
    assert _at_width(wide, 128).stages[0].execution.resolved_precision == "float32"


def _on_device(regime: TrainingRegime, device: Device) -> TrainingRegime:
    data = regime.authored_dump()
    stages = data["stages"]
    assert isinstance(stages, list)
    stages[0]["execution"]["device"] = device
    return TrainingRegime.model_validate(data)


@pytest.mark.parametrize(
    ("device", "width", "expected"),
    [
        ("cpu", BF16_MIN_HIDDEN_DIM - 8, "float32"),
        ("cpu", BF16_MIN_HIDDEN_DIM, "float32"),
        ("cpu", 384, "float32"),
        ("mps", BF16_MIN_HIDDEN_DIM - 8, "float32"),
        ("mps", BF16_MIN_HIDDEN_DIM, "bf16"),
        ("mps", 384, "bf16"),
        ("cuda", BF16_MIN_HIDDEN_DIM - 8, "float32"),
        ("cuda", BF16_MIN_HIDDEN_DIM, "bf16"),
        ("cuda", 384, "bf16"),
    ],
)
def test_one_width_rule_resolves_auto_on_every_device(
    device: Device, width: int, expected: Precision
) -> None:
    assert ladder_precision(device, width) == expected
    regime = _on_device(_at_width(batched_baseline(), width), device)
    execution = regime.stages[0].execution
    assert execution.device == device
    assert execution.resolved_precision == expected
    # The record holds the resolved precision and reads back unchanged.
    saved = regime.model_dump(mode="json")
    assert saved["stages"][0]["execution"]["precision"] == expected
    assert TrainingRegime.model_validate(saved) == regime


def test_explicit_float32_wins_on_wide_mps() -> None:
    wide = _on_device(_at_width(batched_baseline(), 384), "mps")
    data = wide.authored_dump()
    stages = data["stages"]
    assert isinstance(stages, list)
    stages[0]["execution"]["precision"] = "float32"
    pinned = TrainingRegime.model_validate(data)
    assert pinned.stages[0].execution.resolved_precision == "float32"
    assert _at_width(pinned, 512).stages[0].execution.resolved_precision == "float32"


def test_regimes_serialize_the_resolved_precision() -> None:
    wide = _at_width(batched_baseline(), BF16_MIN_HIDDEN_DIM)
    saved = wide.model_dump(mode="json")
    assert saved["stages"][0]["execution"]["precision"] == "bf16"
    assert '"auto"' not in wide.model_dump_json()
    assert TrainingRegime.model_validate(saved) == wide
    # A record that omits precision predates `auto` and ran float32.
    del saved["stages"][0]["execution"]["precision"]
    restored = TrainingRegime.model_validate(saved)
    assert restored.stages[0].execution.resolved_precision == "float32"
    with pytest.raises(ValueError, match="resolved by the regime"):
        Execution().resolved_precision


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("declared", ["float32", "tf32", "bf16"])
def test_explicit_and_saved_precisions_are_kept(
    declared: Precision, saved: bool
) -> None:
    base = batched_baseline()
    base.stages[0].execution = Execution(device="cuda", precision=declared)
    if saved:
        base = TrainingRegime.model_validate(base.model_dump(mode="json"))
    wide = _at_width(base, BF16_MIN_HIDDEN_DIM)
    assert wide.stages[0].execution.resolved_precision == declared
    assert _at_width(wide, 128).stages[0].execution.resolved_precision == declared


def test_precision_and_layout_share_authored_and_saved_resolution() -> None:
    regime = baseline(2, device="cuda").regime
    authored = regime.authored_dump()
    stages = authored["stages"]
    assert isinstance(stages, list)
    execution = stages[0]["execution"]
    for field in ("precision", "bucketed_shapes", "live_tokens"):
        assert execution[field] == "auto"
    # Loading a record freezes every resolved choice, including omitted off values.
    saved = regime.model_dump(mode="json")
    saved_execution = saved["stages"][0]["execution"]
    for field in ("precision", "bucketed_shapes", "live_tokens"):
        saved_execution.pop(field, None)
    restored = TrainingRegime.model_validate(saved)
    execution = restored.stages[0].execution
    assert (execution.precision, execution.bucketed_shapes, execution.live_tokens) == (
        "float32",
        False,
        False,
    )
    assert "auto" not in str(restored.authored_dump())


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_tf32_requires_cuda(device: Device) -> None:
    with pytest.raises(ValidationError, match="precision tf32 requires CUDA"):
        Execution(device=device, precision="tf32")
    assert Execution(device="cuda", precision="tf32").precision == "tf32"


def test_bf16_requires_a_gpu_device() -> None:
    with pytest.raises(ValidationError, match="precision bf16 requires CUDA or MPS"):
        Execution(precision="bf16")
    for device in ("cuda", "mps"):
        assert Execution(device=device, precision="bf16").precision == "bf16"


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
    result = update_damped_iteration(
        torch_backend,
        learner,
        batch,
        DampedPolicyLearning(
            gradient="damped_policy", advantage_quantile=0, min_advantage=0
        ),
        1,
    )
    collector.close()

    assert forwards and all(forwards)
    assert entries and all(entries)
    assert batch.values.dtype == np.float32 and np.isfinite(batch.values).all()
    assert np.isfinite(batch.logprobs).all()
    assert result["rows"] > 0
    assert all(p.dtype == torch.float32 for p in agent.parameters())
    assert not torch.is_autocast_enabled("cpu")
