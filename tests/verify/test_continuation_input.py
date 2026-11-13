"""The actual policy input separates native pending choices and survives storage."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, JsonValue, TypeAdapter
import pytest
import torch

from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import RELATION_ELIGIBILITY_INDEX, ObservationSpace
from manabot.infra.config import (
    ModelConfig,
    ObservationConfig,
    UnsupportedPolicyInput,
)
from manabot.model.admission import admit_observation
from manabot.semantic.decision_contract import SemanticDecisionContract
from manabot.semantic.text_observation import render_decision
from manabot.verify.text_observation import collision_branches, native_fixtures
from manabot.verify.trigger_order import ordering_root
import managym
from managym.decision_capture import DecisionInput


def test_native_continuations_separate_policy_inputs_and_model_outputs() -> None:
    torch.manual_seed(37)
    branches = collision_branches()
    observations = [
        env.observation_for_player(env.current_agent_index()) for env in branches
    ]
    assert observations[0].toJSON() == observations[1].toJSON()
    space = ObservationSpace()
    encoded = [space.encode(obs) for obs in observations]
    for env, obs, expected in zip(branches, observations, encoded, strict=True):
        actual = env.encode_observation(obs)
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    differences = [
        key
        for key in encoded[0]
        if not np.array_equal(encoded[0][key], encoded[1][key])
    ]
    assert differences == ["relations"]
    agent = ActorCritic(space, ModelConfig(hidden_dim=16, policy_film=True))
    batch = {
        key: torch.from_numpy(np.stack([obs[key] for obs in encoded]))
        for key in encoded[0]
    }
    logits, value = agent(admit_observation(agent, batch))
    assert not torch.equal(value[0], value[1])
    assert not torch.equal(logits[0], logits[1])
    (value.sum() + logits[batch["actions_valid"].bool()].sum()).backward()
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


def test_checkpoint_retains_its_policy_input_version(tmp_path: Path) -> None:
    space = ObservationSpace()
    version = space.encoder.config.policy_input_version
    agent = ActorCritic(space, ModelConfig(hidden_dim=16))
    players = [managym.PlayerConfig(name, {"Mountain": 40}) for name in ("a", "b")]
    path = tmp_path / "policy.pt"
    save_checkpoint(agent, space, path, player_configs=players)
    payload = torch.load(path, weights_only=False)
    assert payload["world_binding"]["policy_input_version"] == version
    assert payload["hypers"]["observation_hypers"]["policy_input_version"] == version
    restored, restored_space = load_checkpoint(str(path))
    assert restored_space.shapes == space.shapes
    assert restored_space.encoder.config.policy_input_version == version
    assert restored.state_dict().keys() == agent.state_dict().keys()
    for key, value in agent.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    # Weights saved under any other input, or none, are never read as this one.
    for other in (8, None):
        saved = payload["hypers"]["observation_hypers"]
        if other is None:
            del saved["policy_input_version"]
        else:
            saved["policy_input_version"] = other
        torch.save(payload, path)
        with pytest.raises(UnsupportedPolicyInput):
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
    space = ObservationSpace()
    left, right = [space.encode(obs) for obs in observations]
    assert not np.array_equal(left["relations"], right["relations"])


class EligibilityEvidence(BaseModel):
    legal: bool
    capture: dict[str, JsonValue]
    relations: list[float]


def _eligibility_evidence() -> list[EligibilityEvidence]:
    """Native directed captures; no invented continuation or reconstructed legality."""
    path = (
        Path(__file__).parents[1]
        / "fixtures/resolution-eligibility-captures-v4-w15.json"
    )
    return TypeAdapter(list[EligibilityEvidence]).validate_json(path.read_text())


def _without_active_effects(viewer_state: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """The retained viewer state with the empty shared projection it implies.

    These captures predate shared-effect provenance. Each records no effect on
    any player or permanent, so the projection is empty rather than unknown; a
    capture that did record one could not be joined this way.
    """
    holders: list[JsonValue] = [viewer_state["agent"], viewer_state["opponent"]]
    for key in ("agent_permanents", "opponent_permanents"):
        permanents = viewer_state[key]
        assert isinstance(permanents, list)
        holders.extend(permanents)
    for holder in holders:
        assert isinstance(holder, dict) and not holder.get("effects")
    assert "shared_effects" not in viewer_state
    return viewer_state | {"shared_effects": []}


def test_retained_eligibility_separates_portable_text_and_policy_input() -> None:
    torch.manual_seed(608)
    space = ObservationSpace()
    agent = ActorCritic(space, ModelConfig(hidden_dim=16))
    encoded: list[dict[str, np.ndarray]] = []
    texts: list[str] = []
    for record in _eligibility_evidence():
        capture_json = json.dumps(record.capture)
        capture = DecisionInput.model_validate_json(capture_json)
        observation = managym.Observation.from_json(
            json.dumps(_without_active_effects(capture.observation.viewer_state))
        )
        # Schema 7 intentionally excludes continuation. Join the validated capture
        # frames only for the stateless reference encoder, never an execution Env.
        tensors = space.encode(observation)
        rows = space.encoder._encode_relations(
            capture.frames, observation.agent.player_index
        )
        tensors["relations"] = rows
        # The receipt retains the columns through eligibility; X and granted
        # keywords follow them and are empty in these positions.
        retained = RELATION_ELIGIBILITY_INDEX + 1
        np.testing.assert_array_equal(
            rows[:, :retained], np.array(record.relations).reshape(64, retained)
        )
        assert not rows[:, retained:].any()
        selected = rows[(rows[:, 0] == 2) & (rows[:, 3] == 5)]
        np.testing.assert_array_equal(selected[:, 22], [1, 2])
        np.testing.assert_array_equal(selected[:, 29], [1, 2])
        np.testing.assert_array_equal(
            selected[:, RELATION_ELIGIBILITY_INDEX], [1 if record.legal else 2, 1]
        )
        assert not rows[rows[:, 3] != 5, RELATION_ELIGIBILITY_INDEX].any()
        text = render_decision(capture_json).text
        assert f"Legal when resolution began: {'yes' if record.legal else 'no'}" in text
        texts.append(text)
        encoded.append(tensors)
    for start in (0, 2):
        left, right = encoded[start : start + 2]
        assert texts[start] != texts[start + 1]
        for key in left:
            if key != "relations":
                np.testing.assert_array_equal(left[key], right[key], err_msg=key)
        np.testing.assert_array_equal(
            left["relations"][:, :RELATION_ELIGIBILITY_INDEX],
            right["relations"][:, :RELATION_ELIGIBILITY_INDEX],
        )
        batch = {
            key: torch.from_numpy(np.stack([left[key], right[key]])) for key in left
        }
        _, value = agent(admit_observation(agent, batch))
        assert not torch.equal(value[0], value[1])


def test_historical_capture_eligibility_is_unknown_and_invalid_facts_fail_closed() -> (
    None
):
    record = _eligibility_evidence()[1]
    payload = record.capture
    payload["capture_version"] = 3
    frames = payload["frames"]
    assert isinstance(frames, list)
    for frame in frames:
        assert isinstance(frame, dict) and isinstance(frame["targets"], list)
        for target in frame["targets"]:
            assert isinstance(target, dict)
            del target["legal_at_resolution_start"]
    del payload["digest"]
    payload["digest"] = sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    capture = DecisionInput.model_validate_json(json.dumps(payload))
    assert all(
        t.legal_at_resolution_start is None for f in capture.frames for t in f.targets
    )
    assert (
        "Legal when resolution began: unknown"
        in render_decision(json.dumps(payload)).text
    )
    for bad in (0, "true", [], {}):
        fresh = _eligibility_evidence()[0].capture
        fresh["frames"][1]["targets"][0]["legal_at_resolution_start"] = bad
        with pytest.raises(ValueError):
            DecisionInput.model_validate_json(json.dumps(fresh))
    fresh = _eligibility_evidence()[0].capture
    fresh["capture_version"] = 3
    with pytest.raises(ValueError, match="capture v3"):
        DecisionInput.model_validate_json(json.dumps(fresh))
    fresh = _eligibility_evidence()[0].capture
    fresh["frames"][0]["targets"][0]["legal_at_resolution_start"] = True
    with pytest.raises(ValueError, match="frame relationship"):
        DecisionInput.model_validate_json(json.dumps(fresh))


def test_scalar_and_vector_buffers_match_the_reference_encoder() -> None:
    space = ObservationSpace()
    players = [managym.PlayerConfig(name, {"Mountain": 40}) for name in ("a", "b")]
    scalar = managym.Env(seed=608, skip_trivial=False)
    observation, _ = scalar.reset(players)
    expected = space.encode(observation)
    out = {key: np.zeros_like(value) for key, value in expected.items()}
    scalar.encode_observation_into(observation, out)
    for key in out:
        np.testing.assert_array_equal(out[key], expected[key], err_msg=key)
    vector = managym.VectorEnv(1, seed=608, skip_trivial=False)
    buffers = {
        key: np.zeros((1, *value.shape), dtype=value.dtype)
        for key, value in expected.items()
    }
    buffers.update(
        {
            "rewards": np.zeros(1, dtype=np.float64),
            "terminated": np.zeros(1, dtype=np.uint8),
            "truncated": np.zeros(1, dtype=np.uint8),
        }
    )
    vector.set_buffers(buffers)
    vector.reset_all_into_buffers(players)
    for key in expected:
        np.testing.assert_array_equal(buffers[key][0], expected[key], err_msg=key)


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
    logits, _ = agent(admit_observation(agent, batch))
    assert not torch.isclose(logits[0, 0], logits[0, 1])
    reordered = {key: value.clone() for key, value in batch.items()}
    reordered["relations"] = rows.flip(1)
    torch.testing.assert_close(agent(admit_observation(agent, reordered))[0], logits)
    swapped = {key: value.clone() for key, value in batch.items()}
    swapped["actions"][0, :2, -2] = torch.tensor([2, 1])
    torch.testing.assert_close(
        agent(admit_observation(agent, swapped))[0][0, :2], logits[0, :2].flip(0)
    )
    batch["actions"][0, :2, -2] = 0
    ablated, _ = agent(admit_observation(agent, batch))
    torch.testing.assert_close(ablated[0, 0], ablated[0, 1])
    batch["actions"][0, 0, -2] = 64
    with pytest.raises(ValueError, match="reference has no frame"):
        admit_observation(agent, batch)


def test_retained_capture_schema_cannot_admit_current_name_references() -> None:
    payload = _eligibility_evidence()[0].capture
    payload["observation"]["viewer_state"]["action_space"]["actions"][0].update(
        name_index=0, name_definition=0
    )
    with pytest.raises(ValueError):
        DecisionInput.model_validate_json(json.dumps(payload))
    fresh = _eligibility_evidence()[0].capture
    fresh["capture_version"] = 5
    with pytest.raises(ValueError, match="noncanonical semantic observation"):
        DecisionInput.model_validate_json(json.dumps(fresh))


def test_exported_observation_keeps_complete_policy_input_with_continuation() -> None:
    space = ObservationSpace(ObservationConfig(policy_input_version=10))
    for env in collision_branches():
        actor = env.current_agent_index()
        assert actor is not None
        observation = env.observation_for_player(actor)
        assert observation.continuation
        exported = managym.Observation.from_json(
            observation.toJSON(), continuation_json=observation.continuation_json()
        )
        assert exported.toJSON() == observation.toJSON()
        assert exported.continuation_json() == observation.continuation_json()
        expected = space.encode(observation)
        actual = space.encode(exported)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        frames = json.loads(observation.continuation_json())
        frames[0]["kind"] = 99
        with pytest.raises(ValueError, match="continuation"):
            managym.Observation.from_json(
                observation.toJSON(), continuation_json=json.dumps(frames)
            )
