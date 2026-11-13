"""The MLX backend's layout conversions agree with the PyTorch modules they replace.

Apple Silicon only (the optional `mlx` extra); elsewhere the module is skipped.
Whole-policy parity on real checkpoints is `manabot.backends.parity`.
"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.model.checkpoint_fixture import read_checkpoint

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from manabot.backends.convert import convert  # noqa: E402
from manabot.backends.mlx.checkpoint import (  # noqa: E402
    load_mlx_policy,
    mlx_parameters,
)
from manabot.backends.mlx.policy import (  # noqa: E402
    ObjectAttention,
    PackedGRU,
    attend,
    packed_steps,
)
from manabot.backends.torch.checkpoint import (  # noqa: E402
    load_metadata,
    save_checkpoint,
)
from manabot.backends.torch.policy import ActorCritic  # noqa: E402
from manabot.env.match import Match  # noqa: E402
from manabot.env.observation import ObservationSpace  # noqa: E402
from manabot.infra.config import MatchConfig, ModelConfig  # noqa: E402
from manabot.policy.checkpoint import (  # noqa: E402
    canonical_parameters,
    read_safetensors,
    write_safetensors,
)
from manabot.policy.prepare import Tables, pack  # noqa: E402
import managym  # noqa: E402


def _mlx_names(torch_named: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return mlx_parameters(canonical_parameters(torch_named))


def test_packed_gru_matches_torch_final_and_every_step() -> None:
    torch.manual_seed(3)
    width = 8
    reference = torch.nn.GRU(width, width, batch_first=True)
    state = {k: v.detach().numpy() for k, v in reference.state_dict().items()}
    gru = PackedGRU(width, width)
    prefix = "facts.program_sequence"
    for key, value in _mlx_names(
        {f"{prefix}.{name}": tensor for name, tensor in state.items()}
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
    torch_named = {
        f"attention.mha.{name}": tensor.numpy()
        for name, tensor in reference.state_dict().items()
    }
    for key, value in _mlx_names(torch_named).items():
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
    names = {
        "player_embedding.projection.2.weight": (
            "player_embedding.projection.layers.2.weight"
        ),
        "extra_attention.3.mlp.0.bias": "extra_attention.3.mlp.layers.0.bias",
        "value_head.5.weight": "value_head.layers.5.weight",
    }
    for torch_name, mlx_name in names.items():
        assert list(_mlx_names({torch_name: weight})) == [mlx_name]


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
    torch.save(read_checkpoint(original), original)
    target, tensors = convert(original)
    assert tensors == len(agent.state_dict())
    return original, target, agent


def test_converted_checkpoint_is_one_file_carrying_the_original_source(
    tmp_path: Path,
) -> None:
    original, target, agent = _converted(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "policy.pt",
        "policy.safetensors",
    ]
    _, metadata = read_safetensors(target)
    source = load_metadata(original)["source"]
    assert json.loads(metadata["source"]) == source
    assert metadata["converted_from"] == str(original)

    policy, _ = load_mlx_policy(target)
    state = {k: v.detach().numpy() for k, v in agent.state_dict().items()}
    buffers = set(Tables.BUFFERS.values())
    expected = _mlx_names({k: v for k, v in state.items() if k not in buffers})
    loaded = dict(tree_flatten(policy.parameters()))
    assert set(loaded) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(np.array(loaded[key]), value)


def test_a_refused_converted_checkpoint_names_the_original_revision(
    tmp_path: Path,
) -> None:
    _, target, _ = _converted(tmp_path)
    arrays, metadata = read_safetensors(target)
    metadata["source"] = json.dumps(
        {"commit": "7c1e9a2", "dirty": False, "etude_run": "3a1380d9"}
    )
    binding = json.loads(metadata["world_binding"])
    binding["input_schema"]["agent_permanents"]["shape"] = [40, 37]
    metadata["world_binding"] = json.dumps(binding)
    write_safetensors(target, arrays, metadata)
    with pytest.raises(ValueError) as refused:
        load_mlx_policy(target)
    assert "written at revision 7c1e9a2, Etude run 3a1380d9; run it from there" in str(
        refused.value
    )


def test_prepared_batch_refuses_another_configuration(tmp_path: Path) -> None:
    _, target, _ = _converted(tmp_path)
    policy, space = load_mlx_policy(target)
    env = managym.Env(seed=7)
    env.reset(Match().to_rust())
    obs = space.encode(env.observation_for_player(env.current_agent_index()))
    batch = policy.prepare({key: value[None] for key, value in obs.items()})
    with pytest.raises(ValueError, match="another policy configuration"):
        policy(replace(batch, binding="another model"))


@pytest.mark.parametrize("cache_catalog", [False, True])
def test_mlx_policy_serves_and_samples_through_the_shared_interface(
    tmp_path: Path, cache_catalog: bool
) -> None:
    _, target, _ = _converted(tmp_path)
    policy, space = load_mlx_policy(target)
    policy.to("cpu").eval()
    env = managym.Env(seed=7)
    env.reset(Match().to_rust())
    obs = space.encode(env.observation_for_player(env.current_agent_index()))
    inputs = {key: value[None] for key, value in obs.items()}
    batch = policy.prepare(inputs, inference=True)
    logits, values = policy.evaluate(batch)
    assert policy.device_type == "cpu"
    assert policy.parameter_count > 0
    assert policy.resident_bytes > 4 * policy.parameter_count
    assert policy.world_binding
    np.testing.assert_allclose(
        policy.win_probabilities(batch), 1 / (1 + np.exp(-values)), atol=1e-7
    )
    with policy.inference_scope(cache_catalog=cache_catalog) as scope:
        result = scope.infer(inputs)
        np.testing.assert_array_equal(result.logits, logits)
        assert result.transfers == 0
        assert result.transfer_bytes == 0
        action = policy.sample_rows(result.logits, inputs["actions_valid"], [13], False)
        assert action == policy.sample_rows(
            logits, inputs["actions_valid"], [13], False
        )
        assert inputs["actions_valid"][0, action[0]]
        assert (
            policy.sample(batch, deterministic=True).tolist()
            == logits.argmax(-1).tolist()
        )
    with pytest.raises(RuntimeError, match="closed"):
        scope.infer(inputs)
    with policy.inference_scope() as changed:
        policy.player_embedding.projection.layers[0].weight = (
            policy.player_embedding.projection.layers[0].weight + 1
        )
        with pytest.raises(RuntimeError, match="changed"):
            changed.infer(inputs)
