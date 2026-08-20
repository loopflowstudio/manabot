"""Behavioral checks for autonomous and supplied-belief manabot paths."""

import numpy as np
import pytest

from manabot.belief import AgentMemory, BeliefError, CompatibleDealBeliefModel, Manabot
from manabot.belief.demo import run_demo
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model import Agent
from managym.decision import Observation
from managym.possible_worlds import PossibleWorldSpace
from tests.belief.support import (
    fixture_decision,
    fixture_history,
    fixture_manabot,
    fixture_space,
)


def test_autonomous_and_exact_override_share_policy_value_bytes() -> None:
    decision, schema = fixture_decision()
    manabot = fixture_manabot(schema)
    memory = AgentMemory()

    autonomous = manabot.decide(decision, memory)
    supplied = manabot.evaluate_under_belief(decision, autonomous.belief_update.belief)

    assert autonomous.result.output_bytes == supplied.output_bytes
    assert memory.belief is None
    assert autonomous.next_memory.belief is autonomous.belief_update.belief


def test_belief_enabled_agent_has_no_silent_fallback() -> None:
    decision, schema = fixture_decision()
    agent = Agent(
        ObservationSpace(),
        AgentHypers(
            hidden_dim=8,
            num_attention_heads=2,
            belief_count_buckets=schema.count_buckets,
            belief_card_vocab_size=3,
        ),
    )

    with pytest.raises(ValueError, match="missing inputs"):
        agent(dict(decision.observation))


def test_manabot_rejects_the_legacy_positional_condition_channel() -> None:
    _, schema = fixture_decision()
    agent = Agent(
        ObservationSpace(),
        AgentHypers(
            hidden_dim=8,
            num_attention_heads=2,
            max_conditions=5,
            belief_count_buckets=schema.count_buckets,
            belief_card_vocab_size=3,
        ),
    )

    with pytest.raises(BeliefError, match="legacy positional"):
        Manabot(
            policy_value=agent,
            belief_model=CompatibleDealBeliefModel(),
            belief_schema=schema,
        )


def test_world_space_mismatch_fails_before_inference() -> None:
    decision, schema = fixture_decision()
    manabot = fixture_manabot(schema)
    history = fixture_history()
    original = fixture_space(history)
    other_history = type(history).from_observation(
        Observation(
            schema_version=history.schema_version,
            revision=original.source_revision + 1,
            viewer=history.viewer,
            viewer_state_hash="another-viewer-state",
            viewer_state={},
            events=history.events,
            decision=None,
        )
    )
    other_space = PossibleWorldSpace.from_fixture(
        viewer=history.viewer,
        source_revision=original.source_revision + 1,
        source_viewer_state_hash="another-viewer-state",
        pool=dict(original.pool),
        hands=((dict(world.hand), world.weight) for world in original.worlds),
    )
    other_belief = (
        CompatibleDealBeliefModel()
        .update(previous=None, world_space=other_space, viewer_history=other_history)
        .belief
    )

    with pytest.raises(BeliefError, match="world-space identity"):
        manabot.evaluate_under_belief(decision, other_belief)


def test_keystone_demo_changes_belief_tokens_and_hides_actual_truth() -> None:
    evidence = run_demo()

    assert evidence["generated_override_byte_identical"] is True
    assert evidence["viewer_hidden_swap_identical"] is True
    assert evidence["viewer_hidden_swap_materialized"] is True
    assert (
        evidence["bolt_count_tokens"]["hand"]["has"]
        != evidence["bolt_count_tokens"]["hand"]["lacks"]
    )
    assert (
        evidence["bolt_count_tokens"]["library"]["has"]
        != evidence["bolt_count_tokens"]["library"]["lacks"]
    )
    deltas = np.asarray(
        list(evidence["policy_delta_has_minus_lacks"].values()), dtype=np.float64
    )
    assert np.max(np.abs(deltas)) > 0.0
    assert set(evidence["receipts"]) == {
        "belief_update",
        "belief_encoding",
        "policy_inference",
    }
