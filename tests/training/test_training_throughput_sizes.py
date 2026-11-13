"""Step 3's size variants and launch legs (Task b015d5)."""

from pathlib import Path

from etude.launch.job_file import read_job_file
import pytest
from torch import nn

from experiments import training_throughput_sizes as sizes
from experiments.elemental_variants import elemental_recipe
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.training.regime import TrainingRegime

ROOT = Path(__file__).parents[2]
SIZE_FIELDS = {
    "hidden_dim",
    "attention_layers",
    "num_attention_heads",
    "attention_feedforward_dim",
}


def _compute_parameters(regime: TrainingRegime) -> int:
    """Parameters outside the embedding lookup tables, as the protocol counts them."""
    agent = ActorCritic(ObservationSpace(regime.observation), regime.agent)
    lookups = sum(
        m.weight.numel() for m in agent.modules() if isinstance(m, nn.Embedding)
    )
    return sum(p.numel() for p in agent.parameters()) - lookups


def _without_size(regime: TrainingRegime) -> dict[str, object]:
    return regime.model_dump(mode="json", exclude={"id": True, "agent": SIZE_FIELDS})


@pytest.mark.parametrize(
    ("size", "parameters"),
    [
        ("w64", 307_010),
        ("w128l4", 1_584_002),
        ("w160l8", 5_341_922),
        ("a384l8", 21_090_434),
    ],
)
def test_sizes_change_only_the_network_and_count_as_the_protocol(
    size: sizes.Size, parameters: int
) -> None:
    variant = sizes.recipe(size, "float32", sizes.SPEED_UPDATES)
    control = elemental_recipe("control", sizes.SPEED_UPDATES)

    assert _without_size(variant) == _without_size(control)
    assert _compute_parameters(variant) == parameters
    shape = sizes.SIZES[size]
    agent = variant.agent
    assert (agent.hidden_dim, agent.attention_layers, agent.num_attention_heads) == (
        shape.width,
        shape.layers,
        shape.heads,
    )


def test_w64_float32_keeps_the_control_bytes_and_precisions_differ_only_there() -> None:
    control = elemental_recipe("control", sizes.SPEED_UPDATES)
    assert sizes.recipe("w64", "float32", 4).model_dump(
        exclude={"id"}
    ) == control.model_dump(exclude={"id"})

    bf16 = sizes.recipe("w160l8", "bf16", 4).model_dump(mode="json")
    fp32 = sizes.recipe("w160l8", "float32", 4).model_dump(mode="json")
    assert bf16["stages"][0]["execution"]["precision"] == "bf16"
    for declared in (bf16, fp32):
        declared["id"] = None
        declared["stages"][0]["execution"]["precision"] = None
    assert bf16 == fp32


def _launched(*arguments: str) -> list[sizes.Run]:
    return read_job_file(
        ROOT / "experiments/training_throughput_sizes.py", list(arguments)
    )


def test_unknown_legs_and_seeds_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown leg"):
        sizes.runs("HEAD", "a4000-w64-k1-float32")
    with pytest.raises(ValueError, match="declares no seeds"):
        sizes.runs("HEAD", "a4000-w128l4-k2-float32", seeds="26303")
    with pytest.raises(ValueError, match="pace"):
        sizes.limits(sizes.LEGS["l4-w128l4-k1-float32"], 0)
