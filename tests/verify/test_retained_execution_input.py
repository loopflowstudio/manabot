"""Native retained-state pairs reach the reference encoder, text and real model.

The fixture is produced by retained_execution_inputs_separate_results_iteration_and_completion
with RETAINED_EXECUTION_EVIDENCE set. It records constructed positions, not a
training receipt or an admitted replay world.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, JsonValue, TypeAdapter
import pytest
import torch

from manabot.env.observation import ObservationSpace, _timestamp_ranks
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.managym.decision_capture import DecisionInput
from manabot.model.actor_critic import ActorCritic
from manabot.semantic.text_observation import render_decision
from manabot.verify.text_observation import priority_root
import managym


class ExecutionEvidence(BaseModel):
    case: str
    choice: int
    capture: dict[str, JsonValue]
    facts: list[float]
    relations: list[float]


@dataclass
class CapturedObservation:
    """Stateless test projection: viewer JSON plus native, validated capture frames."""

    observation: managym.Observation
    continuation: list[managym.ContinuationFrame]

    # PyO3 observations are opaque; forward their already validated public fields.
    def __getattr__(self, name: str) -> Any:
        return getattr(self.observation, name)


def _evidence() -> list[ExecutionEvidence]:
    return TypeAdapter(list[ExecutionEvidence]).validate_json(
        (Path(__file__).parents[1] / "fixtures/retained-execution-v8.json").read_text()
    )


def test_real_fetch_search_keeps_frozen_input_admission_and_encoder_parity() -> None:
    env = priority_root((), ("Fabled Passage", "Mountain"))
    observation = env.scenario_refresh()
    activate = next(
        index
        for index, action in enumerate(observation.action_space.actions)
        if action.action_type == managym.ActionEnum.PRIORITY_ACTIVATE_ABILITY
    )
    observation, *_ = env.step(activate)
    for _ in range(8):
        if any(frame.legacy_decision == 6 for frame in observation.continuation):
            break
        passing = next(
            index
            for index, action in enumerate(observation.action_space.actions)
            if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
        )
        observation, *_ = env.step(passing)
    else:
        raise AssertionError("fetch search not reached")
    for version in range(1, 9):
        expected = ObservationSpace(
            ObservationConfig(policy_input_version=version)
        ).encode(observation)
        actual = env.encode_observation(observation, policy_input_version=version)
        for key, value in expected.items():
            np.testing.assert_array_equal(
                actual[key], value, err_msg=f"v{version}: {key}"
            )


@pytest.mark.parametrize("state_encoder", ["rows", "typed_pool", "typed_gru"])
def test_retained_execution_pairs_reach_actual_model_and_text(
    state_encoder: Literal["rows", "typed_pool", "typed_gru"],
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(241)
    space = ObservationSpace(ObservationConfig(policy_input_version=8))
    model = ActorCritic(space, ModelConfig(hidden_dim=16, state_encoder=state_encoder))
    assert model.facts is not None
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
            obs = managym.Observation.from_json(
                json.dumps(capture.observation.viewer_state)
            )
            tensors = space.encode(obs)
            joined = CapturedObservation(obs, capture.frames)
            # Native arrays in the receipt came directly from the full live
            # observation. Only schema-7 storage separates frames from viewer JSON.
            tensors["facts"] = space.encoder._encode_facts(
                cast(managym.Observation, joined), _timestamp_ranks(obs, shared=True)
            )
            tensors["relations"] = space.encoder._encode_relations(
                capture.frames, obs.agent.player_index
            )
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
            assert "retained execution:" in text
            texts.append(text)
        assert texts[0] != texts[1]
        batch = {
            key: torch.from_numpy(np.stack([value[key] for value in encoded]))
            for key in encoded[0]
        }
        logits, values = model(batch)
        assert torch.isfinite(values).all()
        assert not torch.equal(values[0], values[1]), records[start].case
        assert not torch.equal(logits[0], logits[1]), records[start].case
        model.zero_grad()
        (values.sum() + logits[batch["actions_valid"].bool()].sum()).backward()
        assert model.facts is not None
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.facts.parameters()
        )
