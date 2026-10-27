"""The actual policy input separates native pending choices and survives storage."""

from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.semantic.decision_contract import SemanticDecisionContract
from manabot.verify.text_observation import collision_branches, native_fixtures
from manabot.verify.trigger_order import ordering_root
import managym


def test_native_continuations_separate_policy_inputs_and_model_outputs() -> None:
    torch.manual_seed(37)
    branches = collision_branches()
    observations = [
        env.observation_for_player(env.current_agent_index()) for env in branches
    ]
    assert observations[0].toJSON() == observations[1].toJSON()
    for version in (1, 2):
        space = ObservationSpace(ObservationConfig(policy_input_version=version))
        encoded = [space.encode(obs) for obs in observations]
        for env, obs, expected in zip(branches, observations, encoded, strict=True):
            actual = env.encode_observation(obs, policy_input_version=version)
            assert actual.keys() == expected.keys()
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        differences = [
            key
            for key in encoded[0]
            if not np.array_equal(encoded[0][key], encoded[1][key])
        ]
        assert differences == ([] if version == 1 else ["relations"])
        if version == 2:
            agent = ActorCritic(space, ModelConfig(hidden_dim=16, policy_film=True))
            batch = {
                key: torch.from_numpy(np.stack([obs[key] for obs in encoded]))
                for key in encoded[0]
            }
            logits, value = agent(batch)
            assert not torch.equal(value[0], value[1])
            assert not torch.equal(logits[0], logits[1])
            (value.sum() + logits[batch["actions_valid"].bool()].sum()).backward()
            assert agent.continuation is not None
            assert any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in agent.continuation.parameters()
            )
    # Enabling history does not conceal the current-state distinction.
    space = ObservationSpace(ObservationConfig(policy_history_version=1))
    assert not np.array_equal(
        space.encode(observations[0])["relations"],
        space.encode(observations[1])["relations"],
    )


def test_native_fixture_relation_parity_and_other_viewer_safety() -> None:
    space = ObservationSpace()
    for name, env in native_fixtures().items():
        actor = env.current_agent_index()
        assert actor is not None
        for viewer in (actor, 1 - actor):
            observation = env.observation_for_player(viewer)
            expected = space.encode(observation)
            actual = env.encode_observation(observation)
            for key in actual:
                np.testing.assert_array_equal(
                    actual[key], expected[key], err_msg=f"{name}: {key}"
                )
            if name == "mana_color" and viewer == actor:
                outputs = [
                    output
                    for frame in observation.continuation
                    for output in frame.mana_outputs
                ]
                assert len(outputs) == 5
                assert {tuple(output.mana) for output in outputs} == {
                    tuple(int(index == color) for index in range(6))
                    for color in range(5)
                }
                assert all(not any(output.lesson) for output in outputs)
                assert np.count_nonzero(actual["relations"][:, 0] == 6) == 10
            if viewer != actor:
                assert all(frame.kind <= 3 for frame in observation.continuation)
    left, right = collision_branches()
    viewer = 1 - left.current_agent_index()
    assert (
        left.observation_for_player(viewer).continuation_json()
        == right.observation_for_player(viewer).continuation_json()
    )


@pytest.mark.parametrize("version", (1, 2))
def test_checkpoint_retains_its_policy_input_version(
    tmp_path: Path, version: int
) -> None:
    space = ObservationSpace(ObservationConfig(policy_input_version=version))
    agent = ActorCritic(space, ModelConfig(hidden_dim=16))
    players = [managym.PlayerConfig(name, {"Mountain": 40}) for name in ("a", "b")]
    path = tmp_path / "policy.pt"
    save_checkpoint(agent, space, path, player_configs=players)
    payload = torch.load(path, weights_only=False)
    assert payload["world_binding"]["policy_input_version"] == version
    assert ("policy_input_version" in payload["hypers"]["observation_hypers"]) == (
        version == 2
    )
    restored, restored_space = load_checkpoint(str(path))
    assert restored_space.shapes == space.shapes
    assert restored_space.encoder.config.policy_input_version == version
    assert restored.state_dict().keys() == agent.state_dict().keys()
    for key, value in agent.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    # Changing a saved configuration cannot silently upgrade trained weights.
    payload["hypers"]["observation_hypers"]["policy_input_version"] = 3 - version
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(str(path))


def test_stack_targets_remain_distinct_after_the_same_committed_cost() -> None:
    branches = collision_branches()
    for env in branches:
        contract = SemanticDecisionContract.from_env(env)
        contract.apply(env, contract.frame.find_verb("select_card")["id"])
    observations = [
        env.observation_for_player(env.current_agent_index()) for env in branches
    ]
    assert all(
        any(frame.kind == 1 for frame in obs.continuation) for obs in observations
    )
    legacy = ObservationSpace(ObservationConfig(policy_input_version=1))
    left, right = [legacy.encode(obs) for obs in observations]
    assert all(np.array_equal(left[key], right[key]) for key in left)
    current = ObservationSpace()
    left, right = [current.encode(obs) for obs in observations]
    assert not np.array_equal(left["relations"], right["relations"])


@pytest.mark.parametrize("difference", ["ability", "context"])
def test_same_source_trigger_choices_follow_facts_not_frame_positions(
    difference: str,
) -> None:
    torch.manual_seed(195)
    _, raw = ordering_root()
    space = ObservationSpace()
    encoded = space.encode(raw)
    batch = {
        key: torch.from_numpy(value).unsqueeze(0) for key, value in encoded.items()
    }
    # A bounded policy fixture isolates two occurrences of the same source.
    # Native projection of these facts is covered by trigger_order_same_source_joins.
    rows = batch["relations"]
    header = rows[0, (rows[0, :, 0] == 1) & (rows[0, :, 3] == 7)][0].clone()
    rows.zero_()
    for index in range(2):
        rows[0, index] = header
        rows[0, index, 1] = index + 1
        rows[0, index, 2] = 0
        rows[0, index, 8] = index + 1 if difference == "ability" else 1
        if difference == "context":
            rows[0, index + 2] = rows[0, index]
            rows[0, index + 2, 0] = 3
            rows[0, index + 2, 24:26] = index + 1
    batch["actions"][0, 1, :-2] = batch["actions"][0, 0, :-2]
    batch["actions"][0, :2, -2] = torch.tensor([1, 2])
    batch["action_focus"].fill_(-1)
    agent = ActorCritic(space, ModelConfig(hidden_dim=16))
    logits, _ = agent(batch)
    assert not torch.isclose(logits[0, 0], logits[0, 1])
    reordered = {key: value.clone() for key, value in batch.items()}
    reordered["relations"] = rows.flip(1)
    torch.testing.assert_close(agent(reordered)[0], logits)
    swapped = {key: value.clone() for key, value in batch.items()}
    swapped["actions"][0, :2, -2] = torch.tensor([2, 1])
    torch.testing.assert_close(agent(swapped)[0][0, :2], logits[0, :2].flip(0))
    batch["actions"][0, :2, -2] = 0
    ablated, _ = agent(batch)
    torch.testing.assert_close(ablated[0, 0], ablated[0, 1])
    batch["actions"][0, 0, -2] = 64
    with pytest.raises(ValueError, match="reference has no frame"):
        agent(batch)
