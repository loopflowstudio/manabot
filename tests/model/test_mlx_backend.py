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
from manabot.backends.mlx.learner import Learner as MLXLearner
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
    STACK_SOURCE_INDEX,
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
    read_saved_container,
    write_safetensors,
)
from manabot.policy.prepare import Tables, pack, reference_distribution
from manabot.policy.protocol import Precision, Settings, Targets
from manabot.recipes import baseline, elemental
from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.loops.self_play import (
    Counts,
    create,
    device_settings,
    fresh_iteration,
    learn,
)
from manabot.training.rollout.matchups import elemental_roster
from manabot.training.updates.health import NumericalHealth, OptimizerHealth
from manabot.training.updates.ppo import update_iteration
from manabot.verify.choice_corpus import collect_choices
from manabot.verify.learning_regression import inspect_policy
from manabot.verify.representation import kellan_upgrade_root
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


def test_fresh_backends_export_the_same_input_contract(
    baseline_model: ModelConfig, tmp_path: Path
) -> None:
    space = ObservationSpace()
    for backend in (resolve("torch"), resolve("mlx")):
        policy = backend.runtime.create_policy(space, baseline_model)
        backend.save_policy(
            policy,
            space,
            tmp_path / backend.name,
            player_configs=lesson_match().to_rust(),
        )
    torch_file, mlx_file = (
        read_saved_container(tmp_path / name) for name in ("torch", "mlx")
    )
    assert mlx_file.readings == torch_file.readings
    assert mlx_file.dimensions() == torch_file.dimensions()
    assert {name: p.shape for name, p in mlx_file.parameters.items()} == {
        name: p.shape for name, p in torch_file.parameters.items()
    }
    assert mlx_file.buffers.keys() == torch_file.buffers.keys()
    for name, expected in torch_file.buffers.items():
        np.testing.assert_array_equal(
            mlx_file.buffers[name], expected, strict=True, err_msg=name
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
    # Put an ability on the stack with a source on the acting battlefield.
    source = 2 + obs["agent_cards"].shape[1] + obs["opponent_cards"].shape[1]
    obs["agent_permanents_valid"][0, 0] = 1
    obs["stack_valid"][0, 0] = 1
    obs["stack"][0, 0, STACK_SOURCE_INDEX] = source + 1
    obs["player_turn_state"][0, 0, 0] = 1
    obs["card_turn_state"][0, 0, 0] = 1

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


@pytest.mark.parametrize("world", [None, managym.WORLD_SETUP])
def test_hashed_records_match_torch_with_a_live_shared_effect(
    world: str | None,
) -> None:
    torch.manual_seed(23)
    env = kellan_upgrade_root()
    space = ObservationSpace()
    obs = env.observation_for_player(env.current_agent_index())
    rows = {key: value[None] for key, value in encode_observation(space, obs).items()}
    assert (rows["effect_details"][..., 0] >= 11).any()
    config = ModelConfig(hidden_dim=16, world=world, record_vocabulary="hashed")
    reference = ActorCritic(space, config)
    with torch.no_grad():
        # A fresh policy's zero residuals would hide a wrong record embedding.
        for parameter in (
            reference.facts.output.weight,
            reference.shared_effects.reducer.output.weight,
        ):
            parameter.normal_(std=0.1)
    policy = assemble(policy_weights(reference))
    for actual, expected in zip(
        policy.evaluate(policy.prepare(rows)),
        reference.evaluate(reference.prepare(rows)),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=LOGIT_TOLERANCE)


def test_gray_ogre_ppo_matches_torch_on_one_clipped_step() -> None:
    torch.manual_seed(17)
    settings = gray_ogre_self_play(horizon=1)
    space = ObservationSpace(settings.observation)
    reference = ActorCritic(space, settings.model)
    batch = self_play_batch(
        reference,
        Match(settings.match),
        TorchCollectionSession(device="cpu", seed=17),
        6,
    )
    # Exercise both loss terms and distinct actor/critic denominators on a
    # short nonterminal rollout, independently of when the first game ends.
    batch = replace(
        batch,
        rewards=np.linspace(-1, 1, batch.actions.size, dtype=np.float32).reshape(
            batch.actions.shape
        ),
    )
    learning = settings.learning.model_copy(
        update={"filter_scope": "actor", "max_grad_norm": 0.01}
    )
    size = batch.actions.size
    policy = assemble(policy_weights(reference))
    rows = batch.observations(np.arange(size)).observation
    for actual, expected in zip(
        policy.evaluate(policy.prepare(rows)),
        reference.evaluate(reference.prepare(rows)),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=LOGIT_TOLERANCE)
    torch_learner = TorchLearner(reference, "cpu", minibatch_rows=size)
    mlx_learner = MLXLearner(policy, minibatch_rows=size, execution_rows=2)
    expected = update_iteration(
        torch_module, torch_learner, batch, learning, 0, np.random.default_rng(13)
    )
    actual = update_iteration(
        mlx_backend, mlx_learner, batch, learning, 0, np.random.default_rng(13)
    )

    assert actual.keys() == expected.keys()
    assert actual["numerical"].keys() == expected["numerical"].keys()
    assert (
        actual["numerical"]["optimizer_steps"]
        == expected["numerical"]["optimizer_steps"]
        == 1
    )
    assert (
        0
        < actual["actor_exposures"]
        == expected["actor_exposures"]
        < actual["critic_exposures"]
        == size
    )
    assert expected["gradient_norm"] > learning.max_grad_norm
    for key in ("gradient_norm", "loss", "value_loss", "policy_loss"):
        assert actual[key] == pytest.approx(
            expected[key], rel=LOSS_TOLERANCE, abs=1e-6
        ), key

    # Adam's first moment after one step is proportional to the complete
    # clipped gradient, including every execution chunk.
    moments = {
        name: state["exp_avg"].numpy()
        if (state := torch_learner.optimizer.state.get(param))
        else np.zeros(tuple(param.shape), np.float32)
        for name, param in reference.named_parameters()
    }
    mlx_moments = {
        key.removesuffix(".m"): value
        for key, value in flat_numpy(mlx_learner.optimizer.state).items()
        if key.endswith(".m")
    }
    agreement = gradient_agreement(mlx_moments, _mlx_names(moments))
    assert agreement.min_cosine >= COSINE_TOLERANCE, agreement.worst
    after = mlx_learner.parameters()
    for name, value in _mlx_names(torch_learner.parameters()).items():
        np.testing.assert_allclose(
            after[name], value, rtol=0, atol=STEP_PARAMETER_TOLERANCE, err_msg=name
        )


def test_bfloat16_host_arrays_widen_to_float32() -> None:
    widened = to_numpy(mx.array([1.25, -0.5], dtype=mx.bfloat16))
    assert widened.dtype == np.float32
    np.testing.assert_array_equal(widened, [1.25, -0.5])


def test_mlx_selection_draw_uses_reported_logits_and_private_seed(
    baseline_model: ModelConfig,
) -> None:
    backend = resolve("mlx", precision="bfloat16")
    space = ObservationSpace()
    policy = backend.runtime.create_policy(space, baseline_model)
    env = managym.Env(seed=7)
    env.reset(lesson_match().to_rust())
    obs = encode_observation(
        space, env.observation_for_player(env.current_agent_index())
    )
    logits, values = policy.evaluate(
        policy.prepare({key: row[None] for key, row in obs.items()}, inference=True)
    )
    scores = mx.array(logits[0])
    predictor = backend.collection.SelectionPredictor(policy, seed=17)
    key = mx.random.key(17)
    for _ in range(2):
        key, draw = mx.random.split(key)
        result = predictor.predict(obs)
        assert result.action == int(mx.random.categorical(scores, key=draw))
        assert result.value == float(values[0])
        np.testing.assert_array_equal(
            result.probabilities, to_numpy(mx.softmax(scores))
        )


@pytest.mark.parametrize("recipe", ["baseline", "elemental"])
def test_mlx_fresh_recipe_exports_both_live_policies(
    tmp_path: Path, recipe: str
) -> None:
    mlx = resolve("mlx", precision="float32")
    torch_backend = resolve("torch")
    settings = (
        baseline.settings(baseline.BaselineConfig(device="mps"))
        if recipe == "baseline"
        else elemental.settings(elemental.ElementalConfig(device="mps", streams=2))
    ).model_copy(
        update={
            "streams": 2,
            "games_per_stream": 1,
            "transitions": 1,
            "bucketed_shapes": True,
        }
    )
    seed = 19
    initial, initial_ema = create(settings, seed=seed, backend=mlx)
    assert initial_ema is not None
    starting = initial.parameters()
    for name, value in flat_numpy(initial_ema.parameters()).items():
        np.testing.assert_array_equal(value, starting[name])
    for name, value in initial.agent.buffers().items():
        np.testing.assert_array_equal(initial_ema.buffers()[name], value)

    result = fresh_iteration(settings, backend=mlx, seed=seed, out=tmp_path / recipe)
    assert result.counts.updates == 1 and result.counts.games == 2
    assert result.counts.optimizer_updates > 0
    assert result.counts.environment_decisions == result.counts.learner_transitions > 0
    assert result.checkpoint.name == "00000001-ema.safetensors"
    assert result.ema is not None
    decay = settings.learning.ema
    raw, ema = flat_numpy(result.raw.parameters()), flat_numpy(result.ema.parameters())
    assert any(not np.array_equal(raw[name], value) for name, value in starting.items())
    for name, value in starting.items():
        np.testing.assert_allclose(
            ema[name],
            value * decay + raw[name] * (1 - decay),
            atol=1e-7,
            rtol=1e-7,
            err_msg=name,
        )

    env = managym.Env(seed=7)
    env.reset(Match(settings.match).to_rust())
    encoded = encode_observation(
        result.raw.observation_space,
        env.observation_for_player(env.current_agent_index()),
    )
    obs = {key: value[None] for key, value in encoded.items()}
    legal = obs["actions_valid"] > 0
    for label, source in (("raw", result.raw), ("ema", result.ema)):
        saved = result.checkpoint.with_name(f"00000001-{label}.safetensors")
        record = mlx.metadata(saved)["bc"]
        assert (
            record["counts"] == result.counts.coordinates()
            and record["weights"] == label
        )
        expected_logits, expected_values = source.evaluate(
            source.prepare(obs, inference=True)
        )
        canonical = canonical_from_mlx(flat_numpy(source.parameters()))
        stored = mlx.read_policy(str(saved))
        assert canonical.keys() == stored.parameters.keys()
        for name, value in stored.parameters.items():
            np.testing.assert_array_equal(value, canonical[name])
        for reader in (mlx, torch_backend):
            loaded, _ = reader.load_policy(saved)
            logits, values = loaded.evaluate(loaded.prepare(obs, inference=True))
            np.testing.assert_allclose(
                logits[legal], expected_logits[legal], atol=1e-4, rtol=0
            )
            np.testing.assert_allclose(values, expected_values, atol=1e-4, rtol=0)

    if recipe == "elemental":
        (ledger,) = (tmp_path / recipe / "game-attempts").glob("*.jsonl")
        matchups = [
            json.loads(line)["attempt"]["matchup"]
            for line in ledger.read_text().splitlines()
        ]
        assert len(matchups) == 2
        roster = {player.name for player in elemental_roster(Match(settings.match))}
        pairs = {(m["hero"], m["villain"]) for m in matchups}
        assert len(pairs) == 2 and all(set(pair) <= roster for pair in pairs)
        return
    saved = result.checkpoint
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


def test_mlx_filtered_iteration_advances_ema_and_invalidates_cached_programs(
    baseline_model: ModelConfig,
) -> None:
    backend = resolve("mlx")
    settings = baseline.settings(baseline.BaselineConfig(device="mps")).model_copy(
        update={"model": baseline_model}
    )
    settings = settings.model_copy(
        update={
            "learning": settings.learning.model_copy(
                update={"ema": 0.75, "min_advantage": 100.0}
            )
        }
    )
    learner, ema = create(settings, seed=21, backend=backend)
    assert ema is not None
    batch = self_play_batch(
        learner.agent,
        lesson_match(),
        backend.collection.CollectionSession(device="mps", seed=21),
        2,
    )
    # Make the two catalogs differ, as they would after an earlier optimizer step.
    ema.cache_programs()
    initial = ema.semantic_cards.token_embedding.weight
    learner.agent.semantic_cards.token_embedding.weight = initial + 0.25
    diagnostics = learn(
        settings,
        learner,
        ema,
        Counts(),
        batch,
        np.random.default_rng(3),
        backend=backend,
    )
    assert diagnostics["numerical"]["optimizer_steps"] == 0
    np.testing.assert_allclose(
        np.asarray(ema.semantic_cards.token_embedding.weight),
        np.asarray(initial + 0.0625),
        atol=1e-6,
    )
    obs = batch.observations(np.array([0, 1])).observation
    prepared = ema.prepare(obs)
    actual = ema.evaluate(prepared)
    ema.clear_cache()
    for observed, expected in zip(actual, ema.evaluate(prepared), strict=True):
        np.testing.assert_array_equal(observed, expected)


def test_configured_mlx_precision_and_device_scope(baseline_model: ModelConfig) -> None:
    backend = resolve("mlx", precision="bfloat16")
    settings = baseline.settings(baseline.BaselineConfig(device="cpu")).model_copy(
        update={"model": baseline_model}
    )
    before = mx.default_device()
    with pytest.raises(ValueError, match="differs from MLX policy precision"):
        with device_settings(settings, backend=backend):
            assert mx.default_device() == mx.cpu
            create(settings, seed=7, backend=backend)
    assert mx.default_device() == before
    agent = backend.runtime.create_policy(
        ObservationSpace(settings.observation), settings.model
    )
    assert agent.precision == "bfloat16"


@pytest.mark.parametrize("exceptional", [False, True])
def test_mlx_seeded_diagnostics_restore_outer_random_stream(exceptional: bool) -> None:
    backend = resolve("mlx")
    backend.seed(71)
    mx.random.uniform().item()
    expected = mx.random.uniform().item()
    backend.seed(71)
    mx.random.uniform().item()
    try:
        with backend.seeded_inference(21, cpu_only=True):
            mx.random.uniform().item()
            if exceptional:
                raise LookupError
    except LookupError:
        assert exceptional
    assert mx.random.uniform().item() == expected


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


def test_mlx_inference_checks_master_and_cast_weights(tmp_path: Path) -> None:
    _, target, _ = _converted(tmp_path)
    policy, _ = resolve("mlx", precision="bfloat16").load_policy(target)
    with policy.inference_scope() as scope:
        with policy.forward_precision():
            with pytest.raises(RuntimeError, match="master weights"):
                policy.inference_scope()
            with policy.forward_precision():
                scope.cache_programs()
            layer = policy.player_embedding.projection.layers[0]
            layer.weight = layer.weight + 1
            with pytest.raises(RuntimeError, match="changed"):
                scope.cache_programs()
        scope.cache_programs()
        policy.set_precision("float32")
        with pytest.raises(RuntimeError, match="changed"):
            scope.cache_programs()

    policy.set_precision("bfloat16")
    with policy.inference_scope() as scope:
        layer.weight = layer.weight + 1
        with policy.forward_precision(), pytest.raises(RuntimeError, match="changed"):
            scope.cache_programs()


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
    health = NumericalHealth(optimizer_steps=0, rejected_steps=0, sampled_steps=0)
    with pytest.raises(NumericalError, match="nonfinite_objective") as caught:
        learner.step(
            loss, 0.1, recorder=OptimizerHealth(health, 1, backend=mlx_backend)
        )
    assert (
        caught.value.health
        == health
        == dict(optimizer_steps=0, rejected_steps=1, sampled_steps=0)
    )
    for key, value in learner.parameters().items():
        np.testing.assert_array_equal(value, before[key])
    assert int(learner.optimizer.step) == 0
