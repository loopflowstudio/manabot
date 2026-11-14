"""The MLX backend's layout conversions agree with the PyTorch modules they replace.

Apple Silicon only (the optional `mlx` extra); elsewhere the module is skipped.
Whole-policy parity on real checkpoints is `manabot.backends.parity`.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

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
from manabot.backends.torch.checkpoint import save_checkpoint  # noqa: E402
from manabot.backends.torch.policy import ActorCritic  # noqa: E402
from manabot.env.match import Match  # noqa: E402
from manabot.env.observation import ObservationSpace  # noqa: E402
from manabot.infra.config import (  # noqa: E402
    BF16_MIN_HIDDEN_DIM,
    MatchConfig,
    ModelConfig,
)
from manabot.policy.checkpoint import (  # noqa: E402
    canonical_parameters,
    from_torch_state,
    read_safetensors,
    write_safetensors,
)
from manabot.policy.prepare import Tables, pack  # noqa: E402
from manabot.policy.protocol import Precision  # noqa: E402


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
    source = torch.load(original, weights_only=False)["source"]
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


def test_bucket_trim_preserves_mlx_logits_values_and_gradients(tmp_path: Path) -> None:
    """Remapped opponent joins and ownership survive changes in object capacity."""
    import manabot.backends.mlx as backend
    from manabot.backends.mlx.checkpoint import build
    from manabot.backends.mlx.learner import Learner
    from manabot.backends.parity import minibatch_loss
    from manabot.model.inference import trim_observation
    from manabot.policy.buckets import bucket_sizes
    from manabot.policy.protocol import Settings, Targets

    torch.set_num_threads(1)
    _, _, agent = _converted(tmp_path)
    encoder = agent.observation_space.encoder
    obs = {k: torch.zeros((2, *shape)) for k, shape in encoder.shapes.items()}
    obs["agent_player_valid"][:] = 1
    obs["opponent_player_valid"][:] = 1
    obs["agent_cards_valid"][:, 1] = 1
    obs["opponent_permanents_valid"][:, 0] = 1
    obs["actions_valid"][:, :2] = 1
    obs["action_focus"].fill_(-1)
    obs["action_focus"][:, 0, 0] = 0
    obs["action_focus"][:, 1, 0] = (
        2 + 2 * encoder.cards_per_player + encoder.perms_per_player
    )
    sizes = bucket_sizes({k: v.numpy() for k, v in obs.items()}, encoder, objects=True)
    assert sizes is not None and sizes.trims_objects
    trimmed = trim_observation(obs, sizes)
    policy, _ = build(
        agent.config,
        encoder.config,
        {k: v.numpy() for k, v in agent.state_dict().items()},
    )
    learner = Learner(policy, minibatch_rows=2)
    settings = Settings(1e-4, 0.2, 0.1, 0.05, 0.3)
    policy.check_observation({k: v.numpy() for k, v in obs.items()})
    results = []
    for tensors in (obs, trimmed):
        prepared = policy.prepare_admitted({k: v.numpy() for k, v in tensors.items()})
        assert prepared.ownership.shape == (prepared.validity.shape[1],)
        logits, values = policy.evaluate(prepared)
        legal = prepared.legal
        logs = np.where(legal, np.log(0.5), -np.inf).astype(np.float32)
        reference = np.where(legal, 0.5, 0).astype(np.float32)
        targets = Targets(
            np.zeros(2, np.int64),
            np.array([1, -1], np.float32),
            np.ones(2, np.float32),
            np.ones(2, bool),
            logs,
            reference,
        )
        loss = minibatch_loss(backend, policy, learner, prepared, targets, settings)
        gradients = learner.gradients(loss)
        results.append((logits[:, :2], values, float(loss), gradients))
    expected, actual = results
    for a, b in zip(actual[:3], expected[:3], strict=True):
        np.testing.assert_allclose(a, b, atol=2e-6)
    assert actual[3].keys() == expected[3].keys()
    for key in expected[3]:
        np.testing.assert_allclose(
            actual[3][key], expected[3][key], atol=2e-6, err_msg=key
        )


def test_mixed_precision_preserves_master_updates(tmp_path: Path) -> None:
    """Half forwards preserve small updates in float32 masters and Adam state."""
    precision: Precision = "bfloat16"
    import manabot.backends.mlx as backend
    from manabot.backends.mlx.arrays import precision_dtype
    from manabot.backends.mlx.checkpoint import build
    from manabot.backends.mlx.learner import Learner
    from manabot.backends.parity import minibatch_loss
    from manabot.policy.protocol import Settings, Targets

    _, _, agent = _converted(tmp_path)
    encoder = agent.observation_space.encoder
    obs = {k: np.zeros((2, *shape), np.float32) for k, shape in encoder.shapes.items()}
    obs["agent_player_valid"][:] = obs["opponent_player_valid"][:] = 1
    obs["actions_valid"][:, :2] = 1
    obs["action_focus"][:] = -1
    policy, _ = build(
        agent.config,
        encoder.config,
        {k: v.numpy() for k, v in agent.state_dict().items()},
    )
    assert policy.precision == "float32"
    learner = Learner(policy, precision=precision, minibatch_rows=2)
    dtype = precision_dtype(precision)
    assert learner.precision == policy.precision == precision
    batch = policy.prepare(obs)
    settings = Settings(1e-5, 0.2, 0.1, 0.05, 0.01)
    targets = Targets(
        np.zeros(2, np.int64),
        np.array([1, -1], np.float32),
        np.ones(2, np.float32),
        np.ones(2, bool),
        np.where(batch.legal, np.log(0.5), -np.inf).astype(np.float32),
        np.where(batch.legal, 0.5, 0).astype(np.float32),
    )
    with learner.forward_precision():
        assert policy.programs().dtype == dtype
        assert all(p.dtype == dtype for _, p in tree_flatten(policy.parameters()))
    policy.cache_programs()
    inference_logits, inference_values = policy.evaluate(batch)
    logits, values = policy.forward_distribution(batch)
    assert logits.dtype == values.dtype == mx.float32
    np.testing.assert_array_equal(inference_logits, np.array(logits))
    np.testing.assert_array_equal(inference_values, np.array(values).reshape(-1))
    before = learner.parameters()
    loss = minibatch_loss(backend, policy, learner, batch, targets, settings)
    assert loss.dtype == mx.float32 and bool(mx.isfinite(loss))
    gradients = learner.gradients(loss)
    norm = np.sqrt(
        sum(np.square(g.astype(np.float64)).sum() for g in gradients.values())
    )
    assert norm > settings.max_grad_norm
    learner.set_learning_rate(settings.learning_rate)
    with policy.forward_precision(), pytest.raises(ValueError, match="before stepping"):
        learner.step(loss, settings.max_grad_norm)
    learner.step(loss, settings.max_grad_norm)
    after = learner.parameters()
    changed = False
    for name, value in after.items():
        assert value.dtype == np.float32
        grad = gradients[name] * min(1, settings.max_grad_norm / norm)
        expected = before[name] - settings.learning_rate * grad / (np.abs(grad) + 1e-5)
        np.testing.assert_allclose(value, expected, atol=1e-7, rtol=1e-6)
        changed |= bool(np.any(value != before[name]))
    assert changed
    assert all(
        p.dtype == mx.float32
        for _, p in tree_flatten(learner.optimizer.state)
        if isinstance(p, mx.array) and mx.issubdtype(p.dtype, mx.floating)
    )


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (BF16_MIN_HIDDEN_DIM - 8, "float32"),
        (BF16_MIN_HIDDEN_DIM, "bfloat16"),
        (384, "bfloat16"),
    ],
)
def test_omitted_precision_follows_the_shared_width_rule(
    width: int, expected: Precision
) -> None:
    """A policy, backend and learner built without a precision run bfloat16
    from the width torch's GPU devices use, with float32 masters and outputs."""
    from manabot.backends.mlx.arrays import default_precision, precision_dtype
    from manabot.backends.mlx.checkpoint import MLXBackend, assemble
    from manabot.training.regime import ladder_precision

    assert default_precision(width) == expected
    assert (ladder_precision("mps", width) == "bf16") == (expected == "bfloat16")
    space = ObservationSpace()
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=width,
            num_attention_heads=8,
            world="ur-lessons-vs-gw-allies",
            value_aggregation="masked_mean",
        ),
    )
    weights = from_torch_state(
        agent.config,
        space.encoder.config,
        {k: v.numpy() for k, v in agent.state_dict().items()},
    )
    policy, learner = MLXBackend().load(weights)
    assert policy.precision == learner.precision == expected
    assert all(p.dtype == mx.float32 for _, p in tree_flatten(policy.parameters()))
    assert policy.programs().dtype == precision_dtype(expected)
    # An explicit precision always wins over the width.
    for explicit in ("float32", "bfloat16"):
        assert assemble(weights, precision=explicit).precision == explicit
        assert MLXBackend(explicit).load(weights)[0].precision == explicit


def test_half_policy_loading_and_float32_action_logs(tmp_path: Path) -> None:
    precision: Precision = "bfloat16"
    from manabot.backends.mlx.arrays import precision_dtype
    from manabot.backends.mlx.rl import action_logs, policy_logs

    _, path, _ = _converted(tmp_path)
    policy, _ = load_mlx_policy(str(path), precision=precision)
    assert policy.precision == precision
    assert policy.programs().dtype == precision_dtype(precision)
    policy.cache_programs()
    policy.set_precision("float32")
    assert policy.programs().dtype == mx.float32
    assert all(p.dtype == mx.float32 for _, p in tree_flatten(policy.parameters()))
    # Torch regimes spell "bf16"; the refusal names MLX's dtype spellings.
    with pytest.raises(ValueError, match="expected one of float32, bfloat16"):
        policy.set_precision("bf16")  # type: ignore[arg-type]

    logits = mx.array(
        [[0.123, 0.124, 20], [-3, 2, 0]], dtype=precision_dtype(precision)
    )
    valid = mx.array([[True, True, False], [True, True, True]])
    logs = policy_logs(logits, valid)
    expected = policy_logs(logits.astype(mx.float32), valid)
    assert logs.dtype == mx.float32
    np.testing.assert_array_equal(np.array(logs), np.array(expected))
    actions = mx.array([1, 2])
    selected = action_logs(logs, actions)
    assert selected.dtype == mx.float32
    np.testing.assert_array_equal(np.array(selected), np.array(logs)[[0, 1], [1, 2]])
    np.testing.assert_allclose(np.array(mx.exp(logs).sum(-1)), 1, atol=1e-7)


def test_half_overflow_refuses_update(tmp_path: Path) -> None:
    from manabot.backends.mlx.learner import Learner
    from manabot.backends.mlx.rl import NumericalError

    _, path, _ = _converted(tmp_path)
    policy, _ = load_mlx_policy(str(path), precision="bfloat16")
    learner = Learner(policy, minibatch_rows=1)
    before = learner.parameters()

    def overflow() -> tuple[mx.array, ...]:
        with learner.forward_precision():
            return (
                policy.semantic_cards.token_embedding.weight.sum() * mx.array(mx.inf),
            )

    loss = learner.loss(overflow)[0]
    with pytest.raises(NumericalError, match="nonfinite_objective"):
        learner.step(loss, 0.1)
    for key, value in learner.parameters().items():
        np.testing.assert_array_equal(value, before[key])
    assert int(learner.optimizer.step) == 0
