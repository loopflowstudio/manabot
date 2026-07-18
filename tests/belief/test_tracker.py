from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import numpy as np
import pytest

from manabot.belief.likelihood import LikelihoodResult, RulesProviderGap
from manabot.belief.tracker import BeliefTracker
from managym.decision import Observation, SemanticTransition, TransitionReceipt
from managym.possible_worlds import PossibleWorld, PossibleWorldSpace


@dataclass
class FakeEngine:
    space: PossibleWorldSpace | None = None

    def possible_world_space_json(self, viewer: int) -> str:
        assert self.space is not None and viewer == self.space.viewer
        return json.dumps(
            {
                "schema_version": 1,
                "identity": self.space.identity,
                "viewer": self.space.viewer,
                "opponent": self.space.opponent,
                "source_observation": {
                    "schema_version": 1,
                    "revision": self.space.source_revision,
                    "viewer": self.space.viewer,
                    "viewer_state_hash": self.space.source_viewer_state_hash,
                },
                "hand_size": self.space.hand_size,
                "pool": dict(self.space.pool),
                "total_weight": str(self.space.total_weight),
                "worlds": [
                    {
                        "index": world.index,
                        "hand": dict(world.hand),
                        "weight": str(world.weight),
                    }
                    for world in self.space.worlds
                ],
            }
        )

    def semantic_observation_json(self, viewer: int) -> str:
        assert self.space is not None and viewer == self.space.viewer
        return json.dumps(
            {
                "identity": {
                    "schema_version": 1,
                    "revision": self.space.source_revision,
                    "viewer": viewer,
                    "viewer_state_hash": self.space.source_viewer_state_hash,
                },
                "viewer_state": {"opponent_cards": []},
                "events": [],
                "decision": None,
            }
        )


def _state(
    identity: str,
    revision: int,
    pool: dict[str, int],
    hand_size: int,
    rows: list[tuple[dict[str, int], int]],
) -> tuple[FakeEngine, PossibleWorldSpace, Observation]:
    engine = FakeEngine()
    space = PossibleWorldSpace(
        identity=identity,
        viewer=0,
        opponent=1,
        source_revision=revision,
        source_viewer_state_hash=f"hash-{identity}",
        hand_size=hand_size,
        pool=tuple(sorted(pool.items())),
        total_weight=sum(weight for _, weight in rows),
        worlds=tuple(
            PossibleWorld(index, tuple(sorted(hand.items())), weight)
            for index, (hand, weight) in enumerate(rows)
        ),
        _engine=engine,
    )
    engine.space = space
    observation = Observation(
        schema_version=1,
        revision=revision,
        viewer=0,
        viewer_state_hash=f"hash-{identity}",
        viewer_state={"opponent_cards": []},
        events=(),
        decision=None,
    )
    return engine, space, observation


def _transition(
    before: int,
    after: int,
    commitment: Mapping[str, Any] | None,
    observation: Observation,
) -> SemanticTransition:
    return SemanticTransition(
        receipt=TransitionReceipt(
            schema_version=1,
            before_revision=before,
            after_revision=after,
            command_id=f"command-{before}",
            public_commitment=commitment,
            events=(f"event-{before}",),
            next_decision=None,
        ),
        observation=observation,
    )


class FixedLikelihood:
    def __init__(self, likelihoods: tuple[float, ...]) -> None:
        self.likelihoods = likelihoods
        self.calls: list[tuple[Any, int, Mapping[str, Any]]] = []

    def evaluate(
        self,
        root_engine: Any,
        *,
        viewer: int,
        commitment: Mapping[str, Any],
        belief,
    ) -> LikelihoodResult:
        self.calls.append((root_engine, viewer, commitment))
        return LikelihoodResult(
            likelihoods=np.asarray(self.likelihoods, dtype=np.float64),
            legal_action_counts=np.full(belief.support_size, 2, dtype=np.int64),
            matching_action_counts=np.ones(belief.support_size, dtype=np.int64),
            seconds=0.125,
        )


def test_action_update_changes_posterior_but_not_compatible_prior() -> None:
    _, before, observation = _state(
        "before",
        1,
        {"A": 1, "B": 1},
        1,
        [({"A": 1}, 1), ({"B": 1}, 1)],
    )
    after_engine, after, after_observation = _state(
        "after",
        2,
        {"A": 1, "B": 1},
        1,
        [({"A": 1}, 1), ({"B": 1}, 1)],
    )
    model = FixedLikelihood((0.9, 0.1))
    tracker = BeliefTracker(
        before,
        observation,
        likelihood=model,
        epsilon=0.0,
        model_id="test-model",
    )

    tracker.observe(
        after_engine,
        acting=1,
        transition=_transition(1, 2, {"kind": "pass_priority"}, after_observation),
        likelihood_root="public-root",
    )

    assert tracker.posterior.probability_of_hand({"A": 1}) == pytest.approx(0.9)
    assert tracker.prior.probability_of_hand({"A": 1}) == pytest.approx(0.5)
    assert tracker.space.identity == after.identity
    assert tracker.stats.action_updates == 1
    assert tracker.stats.likelihood_seconds == pytest.approx(0.125)
    assert model.calls == [("public-root", 0, {"kind": "pass_priority"})]


def test_hidden_draw_convolves_posterior_and_rebuilds_prior() -> None:
    _, before, observation = _state(
        "before-draw",
        1,
        {"A": 2, "B": 1},
        1,
        [({"A": 1}, 2), ({"B": 1}, 1)],
    )
    after_engine, _, after_observation = _state(
        "after-draw",
        2,
        {"A": 2, "B": 1},
        2,
        [({"A": 2}, 1), ({"A": 1, "B": 1}, 2)],
    )
    tracker = BeliefTracker(
        before,
        observation,
        likelihood=None,
        epsilon=0.0,
        model_id="test-model",
    )

    tracker.observe(
        after_engine,
        acting=0,
        transition=_transition(1, 2, None, after_observation),
    )

    assert tracker.stats.hidden_draws == 1
    assert tracker.posterior.space.hand_size == 2
    assert tracker.prior.space.hand_size == 2


def test_known_exit_and_return_follow_canonical_pool_changes() -> None:
    _, first, first_observation = _state(
        "first",
        1,
        {"A": 2, "B": 2},
        2,
        [({"A": 2}, 1), ({"A": 1, "B": 1}, 4), ({"B": 2}, 1)],
    )
    exit_engine, exited, exit_observation = _state(
        "exited",
        2,
        {"A": 1, "B": 2},
        1,
        [({"A": 1}, 1), ({"B": 1}, 2)],
    )
    return_engine, _, return_observation = _state(
        "returned",
        3,
        {"A": 2, "B": 2},
        2,
        [({"A": 2}, 1), ({"A": 1, "B": 1}, 4), ({"B": 2}, 1)],
    )
    model = FixedLikelihood((1.0, 1.0, 1.0))
    tracker = BeliefTracker(
        first,
        first_observation,
        likelihood=model,
        epsilon=0.0,
        model_id="test-model",
    )
    tracker.observe(
        exit_engine,
        acting=1,
        transition=_transition(1, 2, {"kind": "cast", "card": "A"}, exit_observation),
        likelihood_root="cast-root",
    )
    assert tracker.space.identity == exited.identity
    assert tracker.stats.known_exits == 1

    tracker.observe(
        return_engine,
        acting=0,
        transition=_transition(2, 3, None, return_observation),
    )
    assert tracker.stats.known_returns == 1
    assert tracker.posterior.space.hand_size == 2


def test_unidentified_hidden_pool_exit_is_typed_provider_gap() -> None:
    _, first, observation = _state("gap-first", 1, {"A": 2}, 1, [({"A": 1}, 2)])
    after_engine, _, after_observation = _state("gap-after", 2, {"A": 1}, 0, [({}, 1)])
    tracker = BeliefTracker(
        first,
        observation,
        likelihood=None,
        epsilon=0.0,
        model_id="test-model",
    )

    with pytest.raises(RulesProviderGap, match="no canonical public commitment"):
        tracker.observe(
            after_engine,
            acting=1,
            transition=_transition(1, 2, None, after_observation),
        )


def test_one_public_commitment_cannot_explain_multiple_hidden_pool_exits() -> None:
    _, first, observation = _state(
        "multi-exit-first",
        1,
        {"A": 1, "B": 1},
        2,
        [({"A": 1, "B": 1}, 1)],
    )
    after_engine, _, after_observation = _state("multi-exit-after", 2, {}, 0, [({}, 1)])
    tracker = BeliefTracker(
        first,
        observation,
        likelihood=FixedLikelihood((1.0,)),
        epsilon=0.0,
        model_id="test-model",
    )

    with pytest.raises(RulesProviderGap, match="does not match canonical pool"):
        tracker.observe(
            after_engine,
            acting=1,
            transition=_transition(
                1, 2, {"kind": "cast", "card": "A"}, after_observation
            ),
            likelihood_root="cast-root",
        )


def test_tracker_receipt_binds_observation_space_and_belief_identities() -> None:
    _, before, observation = _state(
        "receipt-before",
        1,
        {"A": 1, "B": 1},
        1,
        [({"A": 1}, 1), ({"B": 1}, 1)],
    )
    after_engine, _, after_observation = _state(
        "receipt-after",
        2,
        {"A": 1, "B": 1},
        1,
        [({"A": 1}, 1), ({"B": 1}, 1)],
    )
    tracker = BeliefTracker(
        before,
        observation,
        likelihood=None,
        epsilon=0.0,
        model_id="test-model",
    )
    tracker.observe(
        after_engine,
        acting=0,
        transition=_transition(1, 2, None, after_observation),
    )

    receipt = tracker.replay_receipt()

    assert receipt["viewer"] == 0
    assert receipt["initial_space_id"] == "receipt-before"
    assert receipt["transitions"][0]["after_space_id"] == "receipt-after"
    assert receipt["transitions"][0]["posterior_normalization_error"] < 1e-12
    assert len(receipt["history_digest"]) == 64
