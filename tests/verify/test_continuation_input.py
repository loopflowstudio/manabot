"""The actual policy input separates native pending choices and survives storage."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, JsonValue, TypeAdapter
import pytest
import torch

from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import (
    ACTION_CONTINUATION_INDEX,
    RELATION_VALID_INDEX,
    ObservationSpace,
    encode_policy_batch,
    step_buffers,
)
from manabot.infra.config import (
    ModelConfig,
    ObservationConfig,
    UnsupportedPolicyInput,
)
from manabot.policy.spans import ObservationSelection
from manabot.semantic.decision_contract import SemanticDecisionContract
from manabot.semantic.text_observation import render_decision
from manabot.verify.text_observation import collision_branches, native_fixtures
from manabot.verify.trigger_order import ordering_root
import managym
from managym.decision_capture import DecisionInput
from tests.fixtures.observation_rows import observation_arrays, vector_arrays
from tests.model.checkpoint_fixture import read_checkpoint, write_checkpoint
from tests.model.prepared_routes import device_input, native_rows


def test_native_continuations_separate_policy_inputs_and_model_outputs() -> None:
    torch.manual_seed(37)
    branches = collision_branches()
    observations = [
        env.observation_for_player(env.current_agent_index()) for env in branches
    ]
    assert observations[0].toJSON() == observations[1].toJSON()
    space = ObservationSpace()
    encoded = [observation_arrays(space, obs) for obs in observations]
    differences = [
        key
        for key in encoded[0]
        if not np.array_equal(encoded[0][key], encoded[1][key])
    ]
    assert differences == ["relations"]
    agent = ActorCritic(space, ModelConfig(hidden_dim=16, policy_film=True))
    prepared = device_input(agent, encode_policy_batch(space, observations))
    logits, value = agent(prepared)
    assert not torch.equal(value[0], value[1])
    assert not torch.equal(logits[0], logits[1])
    (value.sum() + logits[prepared.legal].sum()).backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in agent.continuation.parameters()
    )
    # Enabling history does not conceal the current-state distinction.
    space = ObservationSpace(ObservationConfig(policy_history_version=1))
    assert not np.array_equal(
        observation_arrays(space, observations[0])["relations"],
        observation_arrays(space, observations[1])["relations"],
    )


def test_native_fixture_mana_outputs_and_other_viewer_safety() -> None:
    space = ObservationSpace()
    for name, env in native_fixtures().items():
        actor = env.current_agent_index()
        assert actor is not None
        for viewer in (actor, 1 - actor):
            observation = env.observation_for_player(viewer)
            encoded = observation_arrays(space, observation)
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
                assert np.count_nonzero(encoded["relations"][:, 0] == 6) == 10
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
    version = space.config.policy_input_version
    agent = ActorCritic(space, ModelConfig(hidden_dim=16))
    players = [managym.PlayerConfig(name, {"Mountain": 40}) for name in ("a", "b")]
    path = tmp_path / "policy.pt"
    save_checkpoint(agent, space, path, player_configs=players)
    payload = read_checkpoint(path, weights_only=False)
    assert payload["world_binding"]["policy_input_version"] == version
    assert payload["hypers"]["observation_hypers"]["policy_input_version"] == version
    restored, restored_space = load_checkpoint(str(path))
    assert restored_space.shapes == space.shapes
    assert restored_space.config.policy_input_version == version
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
        write_checkpoint(payload, path)
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
    left, right = [observation_arrays(space, obs) for obs in observations]
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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_retained_eligibility_renders_as_text_and_refuses_incomplete_policy_input() -> (
    None
):
    layout = managym.ObservationLayout()
    texts: list[str] = []
    for record in _eligibility_evidence():
        capture_json = json.dumps(record.capture)
        capture = DecisionInput.model_validate_json(capture_json)
        # Schema 7 intentionally excludes continuation. Join the validated capture
        # frames only into a stateless observation, never an execution Env.
        observation = managym.Observation.from_json(
            json.dumps(_without_active_effects(capture.observation.viewer_state)),
            continuation_json=json.dumps(record.capture["frames"]),
        )
        # These historical captures lack the identities v11 requires.
        with pytest.raises(ValueError, match="requires permanent and spell identities"):
            managym.encode_observation_batch([observation], layout)
        text = render_decision(capture_json).text
        assert f"Legal when resolution began: {'yes' if record.legal else 'no'}" in text
        texts.append(text)
    assert texts[0] != texts[1] and texts[2] != texts[3]


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


def test_vector_rows_match_scalar_encoding() -> None:
    space = ObservationSpace()
    players = [managym.PlayerConfig(name, {"Mountain": 40}) for name in ("a", "b")]
    scalar = managym.Env(seed=608, skip_trivial=False)
    observation, _ = scalar.reset(players)
    expected = observation_arrays(space, observation)
    vector = managym.VectorEnv(1, seed=608, skip_trivial=False)
    vector.set_observation_layout(step_buffers(1), layout=space.layout)
    vector.reset_all_into_buffers(players)
    published = vector_arrays(space, vector.observation_rows())
    for key in expected:
        np.testing.assert_array_equal(published[key][0], expected[key], err_msg=key)


@pytest.mark.parametrize("difference", ["ability", "context"])
def test_same_source_trigger_choices_follow_facts_not_frame_positions(
    difference: str,
) -> None:
    torch.manual_seed(195)
    _, raw = ordering_root()
    space = ObservationSpace()
    spans = ObservationSelection(space, ((encode_policy_batch(space, [raw]), 0),)).spans
    relations = spans.dense("relations")
    actions = spans.dense("actions")
    # A bounded policy fixture isolates two occurrences of the same source.
    # Native projection of these facts is covered by trigger_order_same_source_joins.
    header = relations[0, (relations[0, :, 0] == 1) & (relations[0, :, 3] == 7)][0]
    rows = np.zeros_like(relations)
    for index in range(2):
        rows[0, index] = header
        rows[0, index, 1] = index + 1
        rows[0, index, 2] = 0
        rows[0, index, 8] = index + 1 if difference == "ability" else 1
        if difference == "context":
            rows[0, index + 2] = rows[0, index]
            rows[0, index + 2, 0] = 3
            rows[0, index + 2, 24:26] = index + 1
    actions[0, 1, :ACTION_CONTINUATION_INDEX] = actions[
        0, 0, :ACTION_CONTINUATION_INDEX
    ]
    agent = ActorCritic(space, ModelConfig(hidden_dim=16))

    def forward(relations: np.ndarray, continuations: tuple[int, int]) -> torch.Tensor:
        edited = actions.copy()
        edited[0, :2, ACTION_CONTINUATION_INDEX] = continuations
        tables = {
            **spans.tables,
            "relations": native_rows(
                "relations", relations, relations[..., RELATION_VALID_INDEX] == 1
            ),
            "actions": native_rows(
                "actions", edited, spans.dense("actions_valid") == 1
            ),
            "action_focus": replace(
                spans.tables["action_focus"],
                records=np.full_like(spans.tables["action_focus"].records, -1),
            ),
        }
        # Synthetic rows bypass native certification, so prepare them directly.
        prepared = agent._prepare_rows(
            replace(spans, tables=tables),
            agent._admission_spec().tables,
            False,
            None,
            None,
        )
        return agent(device_input(agent, prepared))[0]

    logits = forward(rows, (1, 2))
    assert not torch.isclose(logits[0, 0], logits[0, 1])
    torch.testing.assert_close(forward(rows[:, ::-1].copy(), (1, 2)), logits)
    torch.testing.assert_close(forward(rows, (2, 1))[0, :2], logits[0, :2].flip(0))
    ablated = forward(rows, (0, 0))
    torch.testing.assert_close(ablated[0, 0], ablated[0, 1])


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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_exported_observation_keeps_complete_policy_input_with_continuation() -> None:
    space = ObservationSpace()
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
        expected = observation_arrays(space, observation)
        actual = observation_arrays(space, exported)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        frames = json.loads(observation.continuation_json())
        frames[0]["kind"] = 99
        with pytest.raises(ValueError, match="continuation"):
            managym.Observation.from_json(
                observation.toJSON(), continuation_json=json.dumps(frames)
            )
