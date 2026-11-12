"""The MLX port's layout conversions agree with the PyTorch modules they replace.

Apple Silicon only (the optional `mlx` extra); elsewhere the module is skipped.
Whole-policy parity on real checkpoints is `manabot.model.mlx.parity`.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from manabot.env.match import Match  # noqa: E402
from manabot.env.observation import ObservationSpace  # noqa: E402
from manabot.infra.config import MatchConfig, ModelConfig  # noqa: E402
from manabot.model.actor_critic import ActorCritic  # noqa: E402
from manabot.model.checkpoint import save_checkpoint  # noqa: E402
from manabot.model.mlx.checkpoint import (  # noqa: E402
    convert,
    convert_state,
    load_mlx_policy,
    mlx_parameters,
    torch_to_mlx,
)
from manabot.model.mlx.model import (  # noqa: E402
    ObjectAttention,
    PackedGRU,
    attend,
    packed_steps,
)
from manabot.model.mlx.prepare import pack  # noqa: E402


def test_packed_gru_matches_torch_final_and_every_step() -> None:
    torch.manual_seed(3)
    width = 8
    reference = torch.nn.GRU(width, width, batch_first=True)
    state = {k: v.detach().numpy() for k, v in reference.state_dict().items()}
    gru = PackedGRU(width, width)
    prefix = "facts.program_sequence"
    for key, value in mlx_parameters(
        {f"{prefix}.{name}": tensor for name, tensor in state.items()}, width
    ).items():
        setattr(gru, key.rsplit(".", 1)[1], mx.array(value))

    lengths = [3, 5, 1, 5, 2]
    tokens = (
        np.random.default_rng(0).normal(size=(sum(lengths), width)).astype(np.float32)
    )
    schedule = pack(lengths)
    outputs, finals = gru.run(mx.array(tokens), packed_steps(schedule.gather, schedule))
    every = np.array(outputs)[schedule.unpack]
    last = np.array(finals)[schedule.restore]

    offsets = np.concatenate(([0], np.cumsum(lengths)))
    for index, length in enumerate(lengths):
        sequence = torch.from_numpy(tokens[offsets[index] : offsets[index + 1]])[None]
        expected, hidden = reference(sequence)
        np.testing.assert_allclose(
            every[offsets[index] : offsets[index + 1]],
            expected[0].detach().numpy(),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            last[index], hidden[0, 0].detach().numpy(), atol=1e-6
        )


def test_split_attention_projections_match_packed_multihead_attention() -> None:
    torch.manual_seed(4)
    width, heads, count = 16, 4, 6
    reference = torch.nn.MultiheadAttention(width, heads, batch_first=True)
    layer = ObjectAttention(width, heads, 2 * width, ownership=False)
    for name, tensor in reference.state_dict().items():
        for key, value in torch_to_mlx(
            f"attention.mha.{name}", tensor.numpy(), width
        ).items():
            module, leaf = key.split(".")[-2:]
            setattr(getattr(layer.mha, module), leaf, mx.array(value))
    objects = torch.randn(2, count, width)
    padding = torch.tensor([[False] * count, [False] * 4 + [True] * 2])
    expected, _ = reference(
        objects, objects, objects, key_padding_mask=padding, need_weights=False
    )

    mask = mx.where(mx.array(~padding.numpy()), 0.0, -mx.inf)[:, None, None, :]
    actual = attend(layer.mha, mx.array(objects.numpy()), mask)
    np.testing.assert_allclose(np.array(actual), expected.detach().numpy(), atol=1e-5)


def test_sequential_children_move_under_layers() -> None:
    weight = np.zeros((2, 2), np.float32)
    assert list(torch_to_mlx("player_embedding.projection.2.weight", weight, 2)) == [
        "player_embedding.projection.layers.2.weight"
    ]
    assert list(torch_to_mlx("extra_attention.3.mlp.0.bias", weight, 2)) == [
        "extra_attention.3.mlp.layers.0.bias"
    ]
    assert list(torch_to_mlx("value_head.5.weight", weight, 2)) == [
        "value_head.layers.5.weight"
    ]


def _converted(tmp_path: Path) -> tuple[Path, Path, ActorCritic]:
    space = ObservationSpace()
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8,
            num_attention_heads=2,
            world="ur-lessons-vs-gw-allies",
            value_aggregation="masked_mean",
        ),
    )
    original = tmp_path / "policy.pt"
    match = Match(
        MatchConfig.authored("ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies")
    )
    save_checkpoint(agent, space, original, player_configs=match.to_rust())
    target, tensors = convert(original)
    assert tensors == len(agent.state_dict())
    return original, target, agent


def test_converted_checkpoint_is_one_file_carrying_the_original_source(
    tmp_path: Path,
) -> None:
    original, target, agent = _converted(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "policy.mlx.safetensors",
        "policy.pt",
    ]
    _, metadata = mx.load(str(target), return_metadata=True)
    source = torch.load(original, weights_only=False)["source"]
    assert json.loads(metadata["source"]) == source
    assert metadata["converted_from"] == str(original)

    policy, _ = load_mlx_policy(target)
    state = {k: v.detach().numpy() for k, v in agent.state_dict().items()}
    expected, _ = convert_state(state, agent.config.hidden_dim)
    loaded = dict(tree_flatten(policy.parameters()))
    assert set(loaded) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(np.array(loaded[key]), value)


def test_a_refused_converted_checkpoint_names_the_original_revision(
    tmp_path: Path,
) -> None:
    _, target, _ = _converted(tmp_path)
    arrays, metadata = mx.load(str(target), return_metadata=True)
    metadata["source"] = json.dumps(
        {"commit": "7c1e9a2", "dirty": False, "etude_run": "3a1380d9"}
    )
    binding = json.loads(metadata["world_binding"])
    binding["input_schema"]["agent_permanents"]["shape"] = [40, 37]
    metadata["world_binding"] = json.dumps(binding)
    mx.save_safetensors(str(target), arrays, metadata=metadata)
    with pytest.raises(ValueError) as refused:
        load_mlx_policy(target)
    assert "written at revision 7c1e9a2, Etude run 3a1380d9; run it from there" in str(
        refused.value
    )
