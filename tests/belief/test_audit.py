from __future__ import annotations

import math

import pytest

from manabot.belief.audit import (
    aggregate_known_truth,
    score_known_truth,
    score_paired_known_truth,
    viewer_equivalence_audit,
)
from manabot.belief.tracker import BeliefTracker
from manabot.sim.teacher1_evidence import _fresh_env


def test_authority_truth_is_scored_outside_the_tracker_boundary() -> None:
    env = _fresh_env(149)
    viewer = int(env._engine.current_agent_index())
    tracker = BeliefTracker.from_engine(
        env._engine, viewer=viewer, likelihood=None, epsilon=0.0
    )

    point = score_known_truth(env._engine, tracker, game_index=3, step=0)

    assert sum(dict(point.true_hand).values()) == tracker.posterior.space.hand_size
    assert point.true_hand_probability > 0.0
    assert math.isfinite(point.true_hand_log_loss)
    assert point.true_hand_rank is not None
    assert point.posterior_digest == tracker.posterior.digest
    assert len(point.cards) == len(tracker.posterior.space.pool)
    assert point.space_id == tracker.posterior.space.identity


def test_known_truth_aggregate_reports_per_definition_calibration() -> None:
    env = _fresh_env(151)
    viewer = int(env._engine.current_agent_index())
    tracker = BeliefTracker.from_engine(
        env._engine, viewer=viewer, likelihood=None, epsilon=0.0
    )
    points = [
        score_known_truth(env._engine, tracker, game_index=0, step=step)
        for step in range(2)
    ]

    metrics = aggregate_known_truth(points, bins=5)

    assert metrics["points"] == 2
    assert metrics["mean_true_hand_log_loss"] == pytest.approx(
        points[0].true_hand_log_loss
    )
    assert 0.0 <= metrics["mean_per_card_brier"] <= 1.0
    assert 0.0 <= metrics["per_card_ece"] <= 1.0
    assert set(metrics["per_card"]) == {
        name for name, _ in tracker.posterior.space.pool
    }


def test_paired_truth_scores_the_same_hand_and_world_space() -> None:
    env = _fresh_env(157)
    viewer = int(env._engine.current_agent_index())
    tracker = BeliefTracker.from_engine(
        env._engine, viewer=viewer, likelihood=None, epsilon=0.0
    )

    point = score_paired_known_truth(env._engine, tracker, game_index=4, step=-1)

    assert point.transition_sequence == 0
    assert point.posterior.space_id == point.prior.space_id == tracker.space.identity
    assert point.posterior.belief_digest == tracker.posterior.digest
    assert point.prior.belief_digest == tracker.prior.digest
    assert point.posterior.true_world_index == point.prior.true_world_index
    assert point.posterior.true_hand_probability == pytest.approx(
        point.prior.true_hand_probability
    )
    assert point.log_loss_improvement_nats == pytest.approx(0.0)
    assert point.truth_mass_ratio == pytest.approx(1.0)
    assert point.to_dict()["true_hand"] == dict(point.true_hand)


def test_viewer_equivalent_exact_hands_hide_authority_differences() -> None:
    env = _fresh_env(163)
    viewer = int(env._engine.current_agent_index())
    tracker = BeliefTracker.from_engine(
        env._engine, viewer=viewer, likelihood=None, epsilon=0.0
    )

    result = viewer_equivalence_audit(
        env._engine, tracker, first_seed=167, second_seed=173
    )

    assert result == {
        "worlds_distinct": True,
        "authority_states_distinct": True,
        "viewer_projection_mismatches": 0,
        "opponent_private_cards_exposed": 0,
    }
