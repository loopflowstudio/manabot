"""The MLX backend's layout conversions agree with the PyTorch modules they replace.

Apple Silicon only (the optional `mlx` extra); elsewhere the module is skipped.
Whole-policy parity on real checkpoints is `manabot.backends.parity`.
"""

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from tests.model.checkpoint_fixture import read_checkpoint
from tests.training.backend_fixtures import policy_weights, self_play_batch

mx = pytest.importorskip("mlx.core")

# ruff: noqa: E402 (optional MLX imports follow importorskip)

from mlx.utils import tree_flatten

from manabot.backends import resolve
from manabot.backends.convert import convert
import manabot.backends.mlx as mlx_backend
from manabot.backends.mlx.arrays import (
    default_precision,
    flat_numpy,
    precision_dtype,
    to_numpy,
)
from manabot.backends.mlx.checkpoint import (
    MLXBackend,
    assemble,
    canonical_from_mlx,
    mlx_parameters,
    require_supported,
)
from manabot.backends.mlx.learner import Learner as MLXLearner, update_ema
from manabot.backends.mlx.policy import (
    ObjectAttention,
    PackedGRU,
    attend,
    packed_steps,
)
from manabot.backends.mlx.rl import NumericalError, action_logs, policy_logs
from manabot.backends.parity import (
    COSINE_TOLERANCE,
    LOGIT_TOLERANCE,
    LOSS_TOLERANCE,
    STEP_PARAMETER_TOLERANCE,
    gradient_agreement,
    minibatch_loss,
)
import manabot.backends.torch as torch_module
from manabot.backends.torch.checkpoint import (
    load_metadata,
    save_checkpoint,
)
from manabot.backends.torch.collection import (
    CollectionSession as TorchCollectionSession,
)
from manabot.backends.torch.learner import Learner as TorchLearner
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import (
    ObservationSpace,
    encode_observation,
)
from manabot.infra.config import (
    BF16_MIN_HIDDEN_DIM,
    MatchConfig,
    ModelConfig,
    ObservationConfig,
)
from manabot.policy.buckets import bucket_sizes, trim_observation
from manabot.policy.checkpoint import (
    UnsupportedPolicy,
    canonical_parameters,
    read_safetensors,
    write_safetensors,
)
from manabot.policy.prepare import Tables, pack, reference_distribution
from manabot.policy.protocol import Precision, Settings, Targets
from manabot.training.learning import DampedPolicyLearning
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.verify.choice_corpus import collect_choices
from manabot.verify.learning_regression import inspect_policy
import managym


def _mlx_names(torch_named: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
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


def lesson_match() -> Match:
    return Match(
        MatchConfig.authored("ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies")
    )


@pytest.fixture
def baseline_model() -> ModelConfig:
    return ModelConfig(
        hidden_dim=16,
        attention_layers=2,
        num_attention_heads=4,
        world="ur-lessons-vs-gw-allies",
        value_kind="scalar",
        value_aggregation="masked_mean",
        policy_film=True,
        focus_film=True,
        stream_norm=True,
    )


def _converted(tmp_path: Path) -> tuple[Path, Path, ActorCritic]:
    space = ObservationSpace(ObservationConfig(policy_input_version=9))
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
    save_checkpoint(agent, space, original, player_configs=lesson_match().to_rust())
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

    policy, _ = resolve("mlx").load_policy(str(target))
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
        resolve("mlx").load_policy(str(target))
    assert "written at revision 7c1e9a2, Etude run 3a1380d9; run it from there" in str(
        refused.value
    )


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"value_kind": "categorical_wdl"}, "value_kind"),
        ({"value_aggregation": "value_token"}, "value_aggregation"),
        ({"attention_on": False}, "attention_on=False"),
        ({"recent_events": True}, "recent_events"),
    ],
)
def test_unimplemented_layouts_are_refused_by_name(
    kwargs: dict[str, object], pattern: str
) -> None:
    model = ModelConfig.model_validate(
        {"world": "elemental-world-v2", "value_aggregation": "masked_mean", **kwargs}
    )
    with pytest.raises(UnsupportedPolicy, match=pattern):
        require_supported(model, ObservationConfig())


@pytest.mark.parametrize("bucketed", [False, True])
def test_v11_baseline_architecture_matches_torch_forward_gradients_and_step(
    bucketed: bool,
    baseline_model: ModelConfig,
) -> None:
    """Latest baseline config (v11 + stream_norm + focus_film + policy_film)
    agrees with PyTorch on forward, loss, every gradient tensor and clipped Adam."""

    torch.manual_seed(11)
    space = ObservationSpace(ObservationConfig())
    assert space.one_token
    agent = ActorCritic(space, baseline_model, bucketed_shapes=bucketed)
    # Give zero-initialized residual/FiLM layers non-zero weights so every
    # branch (stack_source, policy_film, links, focus_film) actually fires.
    with torch.no_grad():
        for name, param in agent.named_parameters():
            if param.abs().max() == 0:
                param.copy_(torch.randn_like(param) * 0.1)
    checkpoint = policy_weights(agent)
    rollout = self_play_batch(
        agent, lesson_match(), TorchCollectionSession(device="cpu", seed=3), 8
    )
    rows = rollout.observations(np.arange(rollout.actions.size))
    obs = rows.observation

    policy = assemble(checkpoint, bucketed_shapes=bucketed)
    round_tripped = canonical_from_mlx(flat_numpy(policy.parameters()))
    assert round_tripped.keys() == checkpoint.parameters.keys()
    for name, expected in checkpoint.parameters.items():
        np.testing.assert_array_equal(round_tripped[name], expected)

    torch_batch = agent.prepare(obs)
    mlx_batch = policy.prepare(obs)
    torch_logits, torch_values = agent.evaluate(torch_batch)
    mlx_logits, mlx_values = policy.evaluate(mlx_batch)
    legal = torch_batch.legal
    assert float(np.abs(mlx_logits - torch_logits)[legal].max()) <= LOGIT_TOLERANCE
    assert float(np.abs(mlx_values - torch_values).max()) <= LOGIT_TOLERANCE

    targets = Targets(
        actions=rollout.actions.reshape(-1),
        advantages=np.linspace(-1.0, 1.0, rollout.actions.size, dtype=np.float32),
        value_targets=np.linspace(-0.5, 0.5, rollout.actions.size, dtype=np.float32),
        actor_mask=np.ones(rollout.actions.size, dtype=bool),
        behavior_logs=rows.log_probabilities,
        reference=reference_distribution(obs, "action_type_uniform"),
    )
    settings = Settings(
        learning_rate=2.5e-4,
        clip=0.2,
        collection_kl=0.1,
        tau=0.1,
        max_grad_norm=0.05,
        reference_forward_kl=0.002,
    )
    torch_learner = TorchLearner(agent, "cpu", minibatch_rows=rollout.actions.size)
    mlx_learner = MLXLearner(policy, minibatch_rows=rollout.actions.size)

    torch_loss = minibatch_loss(
        torch_module, agent, torch_learner, torch_batch, targets, settings
    )
    mlx_loss = minibatch_loss(
        mlx_backend, policy, mlx_learner, mlx_batch, targets, settings
    )
    ref_loss = float(torch_loss.detach())
    got_loss = float(mlx_loss)
    assert abs(got_loss - ref_loss) / max(abs(ref_loss), 1e-12) <= LOSS_TOLERANCE

    expected_grads = _mlx_names(torch_learner.gradients(torch_loss))
    agreement = gradient_agreement(mlx_learner.gradients(mlx_loss), expected_grads)
    assert agreement.min_cosine >= COSINE_TOLERANCE, (
        f"worst gradient {agreement.worst}: {agreement.min_cosine}"
    )

    torch_learner.set_learning_rate(settings.learning_rate)
    mlx_learner.set_learning_rate(settings.learning_rate)
    torch_loss2 = minibatch_loss(
        torch_module, agent, torch_learner, torch_batch, targets, settings
    )
    torch_learner.step(torch_loss2, settings.max_grad_norm)
    mlx_learner.step(mlx_loss, settings.max_grad_norm)
    expected_after = _mlx_names(torch_learner.parameters())
    actual_after = mlx_learner.parameters()
    for name, value in actual_after.items():
        assert (
            float(np.abs(value - expected_after[name]).max())
            <= STEP_PARAMETER_TOLERANCE
        ), name


def test_bfloat16_host_arrays_widen_to_float32() -> None:
    widened = to_numpy(mx.array([1.25, -0.5], dtype=mx.bfloat16))
    assert widened.dtype == np.float32
    np.testing.assert_array_equal(widened, [1.25, -0.5])


def test_mlx_full_self_play_cycle_create_collect_update_save_and_torch_reload(
    tmp_path: Path,
    baseline_model: ModelConfig,
) -> None:
    """MLX creates a fresh v11 baseline policy, collects games, runs damped
    self-play update + EMA, saves a portable checkpoint, and Torch loads it
    with matching inference outputs."""

    mlx = resolve("mlx")
    torch_backend = resolve("torch")
    space = ObservationSpace(ObservationConfig())
    mlx.seed(19)
    policy = mlx.runtime.create_policy(space, baseline_model, bucketed_shapes=True)
    ema = mlx.runtime.create_policy(space, baseline_model, bucketed_shapes=True)
    learner = MLXLearner(policy, minibatch_rows=4)
    match = lesson_match()
    batch = self_play_batch(
        policy, match, mlx.collection.CollectionSession(device="mps", seed=19), 6
    )
    diagnostics = update_damped_iteration(
        mlx,
        learner,
        batch,
        DampedPolicyLearning(gradient="damped_policy", tau_scale=0.1),
        iteration=1,
    )
    assert diagnostics["numerical"]["optimizer_steps"] >= 1
    assert np.isfinite(diagnostics["loss"])
    update_ema(ema, policy, 0.99)

    saved = tmp_path / "mlx-baseline.safetensors"
    mlx.save_policy(
        policy,
        space,
        saved,
        player_configs=match.to_rust(),
        extra={"value_semantic": "signed_outcome"},
    )
    reloaded_torch, _ = torch_backend.load_policy(str(saved))
    reloaded_mlx, _ = mlx.load_policy(str(saved), bucketed_shapes=True)
    obs = batch.observations(np.array([0, 1])).observation
    torch_logits, torch_values = reloaded_torch.evaluate(
        reloaded_torch.prepare(obs, inference=True)
    )
    mlx_logits, mlx_values = reloaded_mlx.evaluate(
        reloaded_mlx.prepare(obs, inference=True)
    )
    legal = obs["actions_valid"] > 0
    assert float(np.abs(mlx_logits - torch_logits)[legal].max()) <= 1e-4
    assert float(np.abs(mlx_values - torch_values).max()) <= 1e-4

    cases, _, findings = collect_choices(
        tmp_path / "choices", 4000, time.perf_counter() + 30, closest=True
    )
    assert not findings and cases
    snapshot_mlx = inspect_policy(saved, cases[:2], updates=1, backend=mlx)
    snapshot_torch = inspect_policy(saved, cases[:2], updates=1, backend=torch_backend)
    assert snapshot_mlx.checkpoint_sha256 == snapshot_torch.checkpoint_sha256
    assert snapshot_mlx.maximum_invariance_error <= 1e-4
    for mlx_choice, torch_choice in zip(
        snapshot_mlx.choices, snapshot_torch.choices, strict=True
    ):
        assert mlx_choice.case == torch_choice.case
        assert mlx_choice.log_odds == pytest.approx(torch_choice.log_odds, abs=1e-4)
        assert mlx_choice.value == pytest.approx(torch_choice.value, abs=1e-4)


def test_prepared_batch_refuses_another_configuration(tmp_path: Path) -> None:
    _, target, _ = _converted(tmp_path)
    policy, space = resolve("mlx").load_policy(str(target))
    env = managym.Env(seed=7)
    env.reset(Match().to_rust())
    obs = encode_observation(
        space, env.observation_for_player(env.current_agent_index())
    )
    batch = policy.prepare({key: value[None] for key, value in obs.items()})
    with pytest.raises(ValueError, match="another policy configuration"):
        policy(replace(batch, binding="another model"))


@pytest.mark.parametrize("cache_catalog", [False, True])
def test_mlx_policy_serves_and_samples_through_the_shared_interface(
    tmp_path: Path, cache_catalog: bool
) -> None:
    _, target, _ = _converted(tmp_path)
    policy, space = resolve("mlx").load_policy(str(target))
    policy.to("cpu").eval()
    env = managym.Env(seed=7)
    env.reset(Match().to_rust())
    obs = encode_observation(
        space, env.observation_for_player(env.current_agent_index())
    )
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


def test_bucket_trim_preserves_mlx_logits_values_and_gradients(tmp_path: Path) -> None:
    """Remapped opponent joins and ownership survive changes in object capacity."""

    _, _, agent = _converted(tmp_path)
    space = agent.observation_space
    obs = {
        k: np.zeros((2, *shape), dtype=np.float32) for k, shape in space.shapes.items()
    }
    obs["agent_player_valid"][:] = 1
    obs["opponent_player_valid"][:] = 1
    obs["agent_cards_valid"][:, 1] = 1
    obs["opponent_permanents_valid"][:, 0] = 1
    obs["actions_valid"][:, :2] = 1
    obs["action_focus"].fill(-1)
    obs["action_focus"][:, 0, 0] = 0
    obs["action_focus"][:, 1, 0] = (
        2 + 2 * space.shapes["agent_cards"][0] + space.shapes["agent_permanents"][0]
    )
    sizes = bucket_sizes(obs, space, objects=True)
    assert sizes is not None and sizes.trims_objects
    trimmed = trim_observation(obs, sizes)
    policy = assemble(policy_weights(agent))
    learner = MLXLearner(policy, minibatch_rows=2)
    settings = Settings(1e-4, 0.2, 0.1, 0.05, 0.3)
    policy.check_observation(obs)
    results = []
    for tensors in (obs, trimmed):
        prepared = policy.prepare_admitted(tensors)
        assert prepared.is_agent.shape == prepared.validity.shape
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
        loss = minibatch_loss(mlx_backend, policy, learner, prepared, targets, settings)
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

    _, _, agent = _converted(tmp_path)
    space = agent.observation_space
    obs = {k: np.zeros((2, *shape), np.float32) for k, shape in space.shapes.items()}
    obs["agent_player_valid"][:] = obs["opponent_player_valid"][:] = 1
    obs["actions_valid"][:, :2] = 1
    obs["action_focus"][:] = -1
    policy = assemble(policy_weights(agent))
    assert policy.precision == "float32"
    learner = MLXLearner(policy, precision=precision, minibatch_rows=2)
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
    loss = minibatch_loss(mlx_backend, policy, learner, batch, targets, settings)
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
def test_omitted_precision_follows_the_width_rule(
    width: int, expected: Precision
) -> None:
    """A policy, backend and learner built without a precision run bfloat16
    from the threshold width, with float32 masters and outputs."""

    assert default_precision(width) == expected
    space = ObservationSpace(ObservationConfig(policy_input_version=9))
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=width,
            num_attention_heads=8,
            world="ur-lessons-vs-gw-allies",
            value_aggregation="masked_mean",
        ),
    )
    weights = policy_weights(agent)
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

    _, path, _ = _converted(tmp_path)
    policy, _ = resolve("mlx", precision=precision).load_policy(str(path))
    assert policy.precision == precision
    served, _ = MLXBackend(precision=precision).load_policy(str(path))
    assert served.precision == precision
    assert served.programs().dtype == precision_dtype(precision)
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

    _, path, _ = _converted(tmp_path)
    policy, _ = resolve("mlx", precision="bfloat16").load_policy(str(path))
    learner = MLXLearner(policy, minibatch_rows=1)
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
