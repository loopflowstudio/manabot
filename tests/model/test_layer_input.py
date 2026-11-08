"""Explicit layer input preserves retained architectures and admits complete grants."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from manabot.env.match import Match, Reward
from manabot.env.observation import (
    FACT_CONTINUATION_PROGRAM,
    FACT_DETAIL_INDEX,
    FACT_DIM,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_KINDS_V8,
    FACT_OBJECT_INDEX,
    FACT_VALID_INDEX,
    FACT_VALUE_INDEX,
    ObservationSpace,
)
from manabot.eval.admission import register_checkpoint
from manabot.eval.evaluate import validate_registered_player
from manabot.infra.config import ModelConfig, ObservationConfig, RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.model.facts import FactEncoder
from manabot.model.shared_effects import SharedEffectEncoder
from manabot.model.state_records import SharedFactEncoder
from manabot.recipes.local import local_smoke
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.training.regime import TrainingRegime
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.verify.representation import kellan_trigger_root, kellan_upgrade_root
from manabot.verify.text_observation import priority_root
import managym


def test_v8_recipe_collector_and_evaluation_select_saved_input(tmp_path: Path) -> None:
    """Exercise production selection/admission without launching an experiment."""
    torch.set_num_threads(1)
    recipe = local_smoke(steps=1)
    recipe.observation = ObservationConfig(policy_input_version=8)
    recipe = TrainingRegime.model_validate_json(recipe.model_dump_json())
    space = ObservationSpace(recipe.observation)
    collector = SeatRoutedCollector(
        space,
        Match(recipe.match),
        Reward(RewardConfig()),
        num_envs=1,
        opponent_mode="self",
        learning_seats="one",
        batch_closure="exact-quota",
    )
    try:
        model = ActorCritic(space, recipe.agent)
        batch = collector.collect(model, 1)
        assert batch.ordered_obs()["effect_details"].shape == (1, 1, 4096, 3)
        assert batch.actions.size == 1
        path = tmp_path / "explicit-v8.pt"
        save_checkpoint(
            model, space, path, player_configs=Match(recipe.match).to_rust()
        )
        admitted = register_checkpoint(
            path,
            match=recipe.match,
            content_suite="layer-input-fixture",
            player_id="v8-fixture",
            display_name="v8 fixture",
        )
        paths, selected = validate_registered_player(admitted.registration, path)
        assert paths["v8-fixture"] == str(path)
        assert selected is not None and selected.encoder.config == recipe.observation
        assert ObservationConfig().policy_input_version == 6
    finally:
        collector.close()


def _assert_encoder_parity(
    env: managym.Env, obs: managym.Observation, space: ObservationSpace
) -> dict[str, NDArray[np.float32]]:
    config = space.encoder.config
    reference = space.encode(obs)
    native = env.encode_observation(obs, config.policy_input_version, config.max_facts)
    assert reference.keys() == native.keys()
    for key, value in reference.items():
        np.testing.assert_array_equal(value, native[key], err_msg=key)
    return reference


@pytest.mark.parametrize("state_encoder", ["rows", "typed_pool", "typed_gru"])
def test_layer_input_native_python_text_forward_backward_and_reload(
    tmp_path: Path,
    state_encoder: Literal["rows", "typed_pool", "typed_gru"],
) -> None:
    torch.set_num_threads(1)
    env = kellan_upgrade_root()
    obs = env.observation_for_player(env.current_agent_index())
    assert any(
        r.effect.modification == "GrantTriggered" and r.effect.program
        for r in obs.shared_effects or []
    )
    config = ObservationConfig(policy_input_version=8)
    space = ObservationSpace(config)
    reference = _assert_encoder_parity(env, obs, space)
    assert reference["effect_details"].shape == (4096, 3)
    assert (reference["effect_details"][:, 0] >= 11).any()
    assert "exile" in capture_text_decision(env).text.lower()
    for version in range(1, 8):
        with pytest.raises(ValueError, match=rf"v{version}.*select v8"):
            ObservationSpace(ObservationConfig(policy_input_version=version)).encode(
                obs
            )
        with pytest.raises(ValueError, match=rf"v{version}.*select v8"):
            env.encode_observation(obs, version)
    values = {key: torch.from_numpy(value)[None] for key, value in reference.items()}
    model = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=16, world=managym.WORLD_SETUP, state_encoder=state_encoder
        ),
    )
    logits, value = model(values)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    (
        logits[values["actions_valid"].bool()].square().mean() + value.square().mean()
    ).backward()
    assert model.shared_effects is not None
    assert any(
        p.grad is not None and p.grad.abs().sum()
        for p in model.shared_effects.parameters()
    )
    path = tmp_path / "layer.pt"
    save_checkpoint(model, space, path, player_configs=None)
    restored, restored_space = load_checkpoint(str(path))
    assert restored_space.encoder.config == config
    for expected, actual in zip(model(values), restored(values), strict=True):
        torch.testing.assert_close(expected, actual)


@pytest.mark.parametrize("overflow", ["error", "first-legal"])
def test_layer_admission_precedes_action_prefix_selection(
    overflow: Literal["error", "first-legal"],
) -> None:
    env = kellan_upgrade_root()
    obs = env.observation_for_player(env.current_agent_index())
    count = len(obs.action_space.actions)
    assert count > 1
    space = ObservationSpace(
        ObservationConfig(
            policy_input_version=7, max_actions=1, action_overflow=overflow
        )
    )
    with pytest.raises(ValueError, match="v7.*select v8"):
        space.encode(obs)
    buffers = space.encoder.allocate()
    with pytest.raises(ValueError, match="v7.*select v8"):
        env.encode_observation_into(obs, buffers, action_overflow=overflow)
    assert all(not array.any() for array in buffers.values())
    assert len(obs.action_space.actions) == count


def test_layer_colors_counter_order_and_signed_program_payload_parity() -> None:
    env = kellan_upgrade_root()
    obs = env.observation_for_player(env.current_agent_index())
    payload = json.loads(obs.toJSON())
    permanent = payload["agent_permanents"][0]
    permanent["colors"] = ["Blue", "Red"]
    permanent["keyword_counter_timestamps"] = {"trample": 12345}
    # Retained observation transport can carry any signed i32 program integer.
    payload["shared_effects"][0]["effect"]["program"] = [[4, -17], [4, 2**31 - 1]]
    observed = managym.Observation.from_json(json.dumps(payload))
    config = ObservationConfig(policy_input_version=8)
    encoded = _assert_encoder_parity(env, observed, ObservationSpace(config))
    assert set(encoded["facts"][:, 1]) >= {12, 13}
    assert encoded["facts"][encoded["facts"][:, 1] == 13, 9].min() > 0


def test_retained_input_architectures_keep_shapes_and_admit_representable_state() -> (
    None
):
    torch.set_num_threads(1)
    env = priority_root((), ("Mountain",))
    obs = env.observation_for_player(env.current_agent_index())
    assert ObservationConfig().policy_input_version == 6
    assert (
        ObservationConfig.from_checkpoint({}).model_dump().get("policy_input_version")
        is None
    )
    assert (
        ObservationConfig(policy_input_version=5).model_dump().get("max_facts") is None
    )
    assert FactEncoder(8, 128).kind.weight.shape == (11, 8)
    assert SharedFactEncoder(8, "typed_pool").kind.weight.shape == (11, 8)
    assert SharedEffectEncoder(8).detail_kind.weight.shape == (4, 8)
    for version, arm in (
        (5, "rows"),
        (5, "typed_pool"),
        (6, "rows"),
        (7, "typed_pool"),
    ):
        space = ObservationSpace(ObservationConfig(policy_input_version=version))
        model = ActorCritic(space, ModelConfig(hidden_dim=16, state_encoder=arm))
        reference = _assert_encoder_parity(env, obs, space)
        output = model(
            {key: torch.from_numpy(value)[None] for key, value in reference.items()}
        )
        assert all(torch.isfinite(value).all() for value in output)


def test_v8_scalar_and_vector_buffers_select_the_explicit_schema() -> None:
    config = ObservationConfig(policy_input_version=8)
    space = ObservationSpace(config)
    env = kellan_upgrade_root()
    obs = env.observation_for_player(env.current_agent_index())
    buffers = space.encoder.allocate()
    env.encode_observation_into(obs, buffers)
    expected = space.encode(obs)
    for key in expected:
        np.testing.assert_array_equal(buffers[key], expected[key])
    with pytest.raises(ValueError, match="v7.*select v8"):
        env.encode_observation_into(
            obs,
            ObservationSpace(
                ObservationConfig(policy_input_version=7)
            ).encoder.allocate(),
        )
    vector_buffers = space.encoder.allocate(1)
    vector_buffers.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    players = [
        managym.PlayerConfig(name, {"Mountain": 20, "Gray Ogre": 20})
        for name in ("a", "b")
    ]
    vector = managym.VectorEnv(num_envs=1, seed=23)
    vector.set_buffers(vector_buffers)
    vector.reset_all(players)
    vector.reset_seeds_into_buffers([23])
    scalar = managym.Env(seed=23)
    scalar.reset(players)
    observed = scalar.observation_for_player(scalar.current_agent_index())
    expected = space.encode(observed)
    for key in expected:
        np.testing.assert_array_equal(
            vector_buffers[key][0], expected[key], err_msg=key
        )


@pytest.mark.parametrize("state_encoder", ["rows", "typed_pool", "typed_gru"])
def test_captured_continuation_program_reaches_native_python_and_policy(
    state_encoder: Literal["rows", "typed_pool", "typed_gru"],
) -> None:
    torch.set_num_threads(1)
    env = kellan_trigger_root(("Lightning Bolt",))
    for prefix, name in (("Cast ", "Lightning Bolt"), ("Target ", "Kellan")):
        decision = capture_text_decision(env)
        action = next(
            a
            for a in decision.actions
            if a.description.startswith(prefix) and name in a.description
        )
        submit_response(env, decision, action.reference)
    for _ in range(8):
        obs = env.observation_for_player(env.current_agent_index())
        if not obs.shared_effects:
            break
        decision = capture_text_decision(env)
        action = next(
            a
            for a in decision.actions
            if a.description.startswith(("Pay", "Pass priority"))
        )
        submit_response(env, decision, action.reference)
    else:
        raise AssertionError("Lightning Bolt did not remove the grant's source")
    assert any(frame.kind == 3 and frame.program for frame in obs.continuation)
    assert "captured program" in capture_text_decision(env).text.lower()
    config = ObservationConfig(policy_input_version=8)
    space = ObservationSpace(config)
    reference = _assert_encoder_parity(env, obs, space)
    assert (reference["facts"][:, 1] == 14).sum() > 0
    with pytest.raises(ValueError, match="v7.*select v8"):
        env.encode_observation(obs, 7)
    with pytest.raises(ValueError, match="v6.*select v8"):
        ObservationSpace(ObservationConfig()).encode(obs)
    values = {key: torch.from_numpy(value)[None] for key, value in reference.items()}
    model = ActorCritic(space, ModelConfig(hidden_dim=16, state_encoder=state_encoder))
    logits, value = model(values)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    value.sum().backward()
    assert model.facts is not None and model.facts.output.weight.grad is not None
    assert model.facts.output.weight.grad.abs().sum() > 0


def _program_rows(holder: int, frame: int, values: tuple[int, ...]) -> torch.Tensor:
    """A captured program of integer tokens, with exact signed-i32 transport."""
    rows = torch.zeros(len(values), FACT_DIM)
    rows[:, FACT_OBJECT_INDEX] = holder
    rows[:, FACT_KIND_INDEX] = FACT_CONTINUATION_PROGRAM
    rows[:, FACT_ID_INDEX + 1] = frame
    rows[:, FACT_VALUE_INDEX] = 4  # TokenKind::Integer
    rows[:, FACT_VALUE_INDEX + 1] = torch.arange(len(values))
    rows[:, FACT_VALID_INDEX] = 1
    for row, value in zip(rows, values, strict=True):
        unsigned = value & 0xFFFFFFFF
        row[FACT_DETAIL_INDEX] = unsigned >> 16
        row[FACT_DETAIL_INDEX + 1] = unsigned & 0xFFFF
    return rows


def test_local_programs_preserve_ownership_order_multiplicity_and_gradients() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(244)
    encoder = FactEncoder(8, 32, kinds=FACT_KINDS_V8)
    # A trained output must expose the representation; fresh fact output is zero.
    with torch.no_grad():
        encoder.output.weight.copy_(torch.eye(8))
    objects = torch.zeros(1, 2, 8)
    first = _program_rows(1, 1, (1, -2, 3))
    second = _program_rows(2, 2, (7, 8))

    def encode(rows: torch.Tensor) -> torch.Tensor:
        facts = torch.zeros(1, 32, FACT_DIM)
        facts[0, : len(rows)] = rows
        return encoder(facts, objects)

    combined = encode(torch.cat((first, second)))
    torch.testing.assert_close(combined, encode(first) + encode(second))
    torch.testing.assert_close(combined, encode(torch.cat((second, first))))
    # Independent records are also independent when they name the same holder.
    same_holder = second.clone()
    same_holder[:, FACT_OBJECT_INDEX] = 1
    torch.testing.assert_close(
        encode(torch.cat((first, same_holder))), encode(first) + encode(same_holder)
    )
    torch.testing.assert_close(encode(torch.cat((first, first))), 2 * encode(first))
    remapped = torch.cat((first, second))
    remapped[:, FACT_OBJECT_INDEX] = 3 - remapped[:, FACT_OBJECT_INDEX]
    torch.testing.assert_close(encode(remapped), combined.flip(1))
    padding = torch.full((3, FACT_DIM), float("nan"))
    padding[:, FACT_VALID_INDEX] = 0
    torch.testing.assert_close(
        combined,
        encode(torch.cat((padding, first[:1], padding, first[1:], second, padding))),
    )
    # Reordering a program's tokens changes meaning, even with equal token sums.
    reordered = _program_rows(1, 1, (3, -2, 1))
    assert not torch.allclose(encode(first), encode(reordered))
    # Detail hashes collide modulo 4096; signed numeric meaning must still reach
    # the model rather than disappearing into the vocabulary bucket.
    assert not torch.allclose(
        encode(_program_rows(1, 1, (0,))),
        encode(_program_rows(1, 1, (-4096,))),
    )
    assert not torch.allclose(
        encode(_program_rows(1, 1, (4096,))),
        encode(_program_rows(1, 1, (8192,))),
    )
    combined.square().sum().backward()
    assert encoder.program_sequence is not None
    assert encoder.program_sequence.weight_ih_l0.grad is not None
    assert encoder.program_sequence.weight_ih_l0.grad.abs().sum() > 0
    assert encoder.program_numeric is not None
    assert encoder.program_numeric.weight.grad is not None
    assert encoder.program_numeric.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("malformed", ["ordinal", "holder", "frame"])
def test_local_program_rejects_broken_boundaries(malformed: str) -> None:
    rows = _program_rows(1, 1, (1, 2, 3))
    if malformed == "ordinal":
        rows[1, FACT_VALUE_INDEX + 1] = 3
    elif malformed == "holder":
        rows[1, FACT_OBJECT_INDEX] = 2
    else:
        rows[1, FACT_ID_INDEX + 1] = 2
    encoder = FactEncoder(8, len(rows), kinds=FACT_KINDS_V8)
    with pytest.raises(ValueError, match="fact program"):
        encoder(rows[None], torch.zeros(1, 2, 8))


def test_shared_program_order_joins_padding_multiplicity_and_numeric_gradients() -> (
    None
):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    encoder = SharedEffectEncoder(16, version=8)
    objects = torch.randn(2, 4, 16, requires_grad=True)
    headers = torch.zeros(2, 128, 18)
    details = torch.zeros(2, 4096, 3)
    members = torch.zeros(2, 2048, 2)
    # Two independent records with explicit source/controller and target joins.
    for i in range(2):
        headers[:, i, 0] = i + 1
        headers[:, i, 9] = i + 1
        headers[:, i, 10] = 1
        headers[:, i, 12] = i + 1
        headers[:, i, 13:17] = torch.tensor([i * 2, 2, i, 1])
        headers[:, i, 17] = 1
        details[:, i * 2 : i * 2 + 2] = torch.tensor([[14, 0, 4096], [14, 0, 3 + i]])
        members[:, i] = torch.tensor([i + 3, i + 1])
    expected = encoder(headers, details, members, objects)
    # Packed programs from different observations have no shared recurrent state.
    for b in range(2):
        torch.testing.assert_close(
            expected[b : b + 1],
            encoder(
                headers[b : b + 1],
                details[b : b + 1],
                members[b : b + 1],
                objects[b : b + 1],
            ),
        )
    changed = details.clone()
    changed[:, 0, 2] = 8192  # same embedding bucket, distinct magnitude
    actual = encoder(headers, changed, members, objects)
    assert not torch.allclose(actual[:, 2], expected[:, 2])
    torch.testing.assert_close(actual[:, 3], expected[:, 3])
    changed[:, :2] = details[:, :2].flip(1)
    assert not torch.allclose(encoder(headers, changed, members, objects), expected)
    for column in (9, 12):  # timestamp and source are strategically meaningful
        changed_headers = headers.clone()
        changed_headers[:, 0, column] += 1
        assert not torch.allclose(
            encoder(changed_headers, details, members, objects), expected
        )
    # Physical record order is immaterial after CSR slices move with the records.
    swapped = headers.clone()
    swapped[:, :2] = headers[:, :2].flip(1)
    swapped[:, :2, 13:17] = headers[:, :2, 13:17]
    swapped_details = details.clone()
    swapped_details[:, :4] = details[:, :4].reshape(2, 2, 2, 3).flip(1).reshape(2, 4, 3)
    swapped_members = members.clone()
    swapped_members[:, :2] = members[:, :2].flip(1)
    torch.testing.assert_close(
        encoder(swapped, swapped_details, swapped_members, objects), expected
    )
    # Remap all public object addresses together.
    remapped = headers.clone()
    for column in (10, 12):
        remapped[:, :2, column] = 5 - remapped[:, :2, column]
    remapped_members = members.clone()
    remapped_members[:, :2, 0] = 5 - remapped_members[:, :2, 0]
    torch.testing.assert_close(
        encoder(remapped, details, remapped_members, objects.flip(1)), expected.flip(1)
    )
    # Duplicate targets preserve multiplicity, not a mean over memberships.
    duplicate_headers = headers.clone()
    duplicate_headers[:, 1, 17] = 0
    one = encoder(duplicate_headers, details, members, objects) - objects
    duplicate_headers[:, 0, 16] = 2
    duplicate_members = members.clone()
    duplicate_members[:, 1, 0] = members[:, 0, 0]
    torch.testing.assert_close(
        encoder(duplicate_headers, details, duplicate_members, objects) - objects,
        2 * one,
    )
    headers[:, 2:, :17] = float("nan")
    details[:, 4:] = float("nan")
    members[:, 2:] = float("nan")
    torch.testing.assert_close(encoder(headers, details, members, objects), expected)
    expected.square().sum().backward()
    assert objects.grad is not None and objects.grad.abs().sum() > 0
    assert encoder.detail_number.weight.grad is not None
    assert encoder.detail_number.weight.grad.abs().sum() > 0
    assert encoder.reducer.gru is not None
    assert encoder.reducer.gru.weight_ih_l0.grad is not None
    assert encoder.reducer.gru.weight_ih_l0.grad.abs().sum() > 0
