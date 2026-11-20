"""Native retained-state pairs reach the policy input, text and real model.

The fixture is produced by effect_execution_inputs_separate_results_iteration_and_completion
with EFFECT_EXECUTION_EVIDENCE set. It records constructed positions, not a
training receipt or an admitted replay world.
"""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, JsonValue, TypeAdapter
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace, encode_observation
from manabot.infra.config import ModelConfig
from manabot.managym.decision_capture import DecisionInput
from manabot.semantic.text_observation import render_decision
from manabot.verify.text_observation import priority_root
import managym
from tests.model.prepared_routes import device_input

# Relation decision code of a pending native `Decision::SelectCards`.
SELECTION_OBLIGATION = 28
RELATIONS = managym.observation_layout()["relations"]["columns"]


def _relation(name: str) -> int:
    return RELATIONS[name]["index"]


class ExecutionEvidence(BaseModel):
    case: str
    choice: int
    capture: dict[str, JsonValue]
    facts: list[float]
    relations: list[float]


def _evidence() -> list[ExecutionEvidence]:
    return TypeAdapter(list[ExecutionEvidence]).validate_json(
        (Path(__file__).parents[1] / "fixtures/effect-execution-v8.json").read_text()
    )


def test_real_fetch_search_encodes_its_pending_selection_frame() -> None:
    env = priority_root((), ("Fabled Passage", "Mountain"))
    observation = env.scenario_refresh()
    activate = next(
        index
        for index, action in enumerate(observation.action_space.actions)
        if action.action_type == managym.ActionEnum.PRIORITY_ACTIVATE_ABILITY
    )
    observation, *_ = env.step(activate)
    for _ in range(8):
        if any(
            frame.decision == SELECTION_OBLIGATION for frame in observation.continuation
        ):
            break
        passing = next(
            index
            for index, action in enumerate(observation.action_space.actions)
            if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
        )
        observation, *_ = env.step(passing)
    else:
        raise AssertionError("fetch search not reached")
    encoded = encode_observation(ObservationSpace(), observation)
    rows = encoded["relations"][encoded["relations"][:, _relation("valid")] == 1]
    # Frame header rows (kind 1), one per continuation frame, each under its parent.
    headers = rows[rows[:, _relation("kind")] == 1]
    frames = observation.continuation
    np.testing.assert_array_equal(
        headers[:, _relation("frame")], np.arange(1, len(frames) + 1)
    )
    np.testing.assert_array_equal(
        headers[:, _relation("parent")], np.arange(len(frames))
    )
    np.testing.assert_array_equal(
        headers[:, _relation("frame_kind")], [frame.kind for frame in frames]
    )
    # The resolving fetch ability names Fabled Passage (definition id + 1).
    passage = list(managym.catalog_tokens()["keys"]).index("eld.fabled_passage") + 1
    np.testing.assert_array_equal(headers[:-1, _relation("source_definition")], passage)
    # The innermost frame is the pending, declinable library selection.
    selection = headers[-1]
    assert selection[_relation("decision")] == SELECTION_OBLIGATION
    assert selection[_relation("can_decline")] == 1
    assert (rows[:, _relation("decision")] == SELECTION_OBLIGATION).sum() == 2
    assert encoded["actions_valid"].sum() == len(observation.action_space.actions)


def test_effect_execution_pairs_reach_actual_model_and_text() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(241)
    space = ObservationSpace()
    model = ActorCritic(space, ModelConfig(hidden_dim=16))
    # Facts deliberately start with zero output weights. A fixed nonzero
    # readout tests representability without fitting weights or running training.
    with torch.no_grad():
        model.facts.output.weight.copy_(
            torch.eye(model.facts.output.weight.shape[0]) * 0.01
        )
    records = _evidence()
    for start in range(0, len(records), 2):
        encoded = []
        texts: list[str] = []
        for record in records[start : start + 2]:
            capture_json = json.dumps(record.capture)
            capture = DecisionInput.model_validate_json(capture_json)
            # The receipt's arrays came from the full live observation; storage
            # keeps its frames apart from the viewer JSON, so rejoin them.
            obs = managym.Observation.from_json(
                json.dumps(capture.observation.viewer_state),
                continuation_json=json.dumps(record.capture["frames"]),
            )
            tensors = encode_observation(space, obs)
            for name, width, expected in (
                ("facts", 12, record.facts),
                ("relations", 49, record.relations),
            ):
                native = np.zeros_like(tensors[name])
                native[: len(expected) // width] = np.asarray(expected).reshape(
                    -1, width
                )
                np.testing.assert_array_equal(
                    tensors[name], native, err_msg=f"{record.case}: {name}"
                )
            encoded.append(tensors)
            text = render_decision(capture_json).text
            assert "effect execution:" in text
            texts.append(text)
        assert texts[0] != texts[1]
        batch = {
            key: torch.from_numpy(np.stack([value[key] for value in encoded]))
            for key in encoded[0]
        }
        logits, values = model(device_input(model, batch))
        assert torch.isfinite(values).all()
        assert not torch.equal(values[0], values[1]), records[start].case
        assert not torch.equal(logits[0], logits[1]), records[start].case
        model.zero_grad()
        (values.sum() + logits[batch["actions_valid"].bool()].sum()).backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.facts.parameters()
        )
