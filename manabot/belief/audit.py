"""Known-truth calibration isolated from the acting information path."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np

from manabot.belief.range import BeliefError, BeliefState
from manabot.belief.tracker import BeliefTracker

_HAND_ZONE = 1


@dataclass(frozen=True, slots=True)
class CardCalibrationPoint:
    card: str
    predicted_inclusion: float
    present: bool

    def to_dict(self) -> dict[str, str | float | bool]:
        return {
            "card": self.card,
            "predicted_inclusion": self.predicted_inclusion,
            "present": self.present,
        }


@dataclass(frozen=True, slots=True)
class KnownTruthPoint:
    """Authority-only witness scored against one viewer-derived posterior."""

    game_index: int
    step: int
    viewer: int
    transition_sequence: int
    true_world_index: int | None
    true_hand: tuple[tuple[str, int], ...]
    true_hand_probability: float
    true_hand_log_loss: float
    true_hand_rank: int | None
    top_hand_mass: float
    effective_range_size: float
    posterior_digest: str
    space_id: str
    cards: tuple[CardCalibrationPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_index": self.game_index,
            "step": self.step,
            "viewer": self.viewer,
            "transition_sequence": self.transition_sequence,
            "true_world_index": self.true_world_index,
            "true_hand": dict(self.true_hand),
            "true_hand_probability": self.true_hand_probability,
            "true_hand_log_loss": self.true_hand_log_loss,
            "true_hand_rank": self.true_hand_rank,
            "top_hand_mass": self.top_hand_mass,
            "effective_range_size": self.effective_range_size,
            "posterior_digest": self.posterior_digest,
            "space_id": self.space_id,
            "cards": [card.to_dict() for card in self.cards],
        }


def authority_hand(engine: Any, tracker: BeliefTracker) -> dict[str, int]:
    """Read truth for audit only; player/tracker never receive this value."""

    opponent = (tracker.viewer + 1) % 2
    authority_view = engine.observation_for_player(opponent)
    counts: dict[str, int] = {}
    for card in authority_view.agent_cards:
        if int(card.zone) == _HAND_ZONE:
            counts[str(card.name)] = counts.get(str(card.name), 0) + 1
    if sum(counts.values()) != tracker.posterior.space.hand_size:
        raise BeliefError(
            "authority hand size does not match the canonical public hand size"
        )
    return counts


def score_known_truth(
    engine: Any,
    tracker: BeliefTracker,
    *,
    game_index: int,
    step: int,
) -> KnownTruthPoint:
    posterior = tracker.posterior
    hand = authority_hand(engine, tracker)
    world_index = posterior.index_for_hand(hand)
    probability = (
        posterior.probability_at(world_index) if world_index is not None else 0.0
    )
    inclusions = posterior.inclusion_probabilities()
    cards = tuple(
        CardCalibrationPoint(
            card=card,
            predicted_inclusion=inclusion,
            present=hand.get(card, 0) > 0,
        )
        for card, inclusion in inclusions.items()
    )
    return KnownTruthPoint(
        game_index=game_index,
        step=step,
        viewer=tracker.viewer,
        transition_sequence=len(tracker.records),
        true_world_index=world_index,
        true_hand=tuple(sorted(hand.items())),
        true_hand_probability=probability,
        true_hand_log_loss=(-math.log(probability) if probability > 0.0 else math.inf),
        true_hand_rank=posterior.rank(world_index) if world_index is not None else None,
        top_hand_mass=float(np.max(posterior.probabilities)),
        effective_range_size=posterior.effective_range_size,
        posterior_digest=posterior.digest,
        space_id=posterior.space.identity,
        cards=cards,
    )


def _ece(points: Iterable[CardCalibrationPoint], bins: int) -> float:
    rows = list(points)
    if not rows:
        return 0.0
    weighted_error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        bucket = [
            row
            for row in rows
            if low <= row.predicted_inclusion < high
            or (index == bins - 1 and row.predicted_inclusion == 1.0)
        ]
        if bucket:
            predicted = float(np.mean([row.predicted_inclusion for row in bucket]))
            observed = float(np.mean([row.present for row in bucket]))
            weighted_error += len(bucket) * abs(predicted - observed)
    return weighted_error / len(rows)


def aggregate_known_truth(
    points: Iterable[KnownTruthPoint], *, bins: int = 10
) -> dict[str, Any]:
    rows = list(points)
    if bins < 1:
        raise ValueError("calibration bins must be positive")
    if not rows:
        return {
            "points": 0,
            "mean_true_hand_log_loss": None,
            "mean_per_card_brier": None,
            "per_card": {},
        }
    card_rows = [card for row in rows for card in row.cards]
    ranks = [row.true_hand_rank for row in rows if row.true_hand_rank is not None]
    names = sorted({card.card for card in card_rows})
    per_card: dict[str, Any] = {}
    for name in names:
        selected = [card for card in card_rows if card.card == name]
        per_card[name] = {
            "points": len(selected),
            "brier": float(
                np.mean(
                    [
                        (card.predicted_inclusion - float(card.present)) ** 2
                        for card in selected
                    ]
                )
            ),
            "ece": _ece(selected, bins),
            "mean_predicted_inclusion": float(
                np.mean([card.predicted_inclusion for card in selected])
            ),
            "observed_inclusion": float(np.mean([card.present for card in selected])),
        }
    return {
        "points": len(rows),
        "mean_true_hand_log_loss": float(
            np.mean([row.true_hand_log_loss for row in rows])
        ),
        "mean_per_card_brier": float(
            np.mean(
                [
                    (card.predicted_inclusion - float(card.present)) ** 2
                    for card in card_rows
                ]
            )
        ),
        "per_card_ece": _ece(card_rows, bins),
        "mean_top_hand_mass": float(np.mean([row.top_hand_mass for row in rows])),
        "median_true_hand_rank": float(np.median(ranks)) if ranks else None,
        "mean_effective_range_size": float(
            np.mean([row.effective_range_size for row in rows])
        ),
        "true_hand_outside_support": sum(row.true_world_index is None for row in rows),
        "per_card": per_card,
    }


def score_belief_for_hand(
    belief: BeliefState, hand: Mapping[str, int]
) -> tuple[float, int | None]:
    index = belief.index_for_hand(hand)
    probability = belief.probability_at(index) if index is not None else 0.0
    return (
        -math.log(probability) if probability > 0.0 else math.inf,
        belief.rank(index) if index is not None else None,
    )


def viewer_equivalence_audit(
    engine: Any, tracker: BeliefTracker, *, first_seed: int, second_seed: int
) -> dict[str, int | bool]:
    """Materialize by canonical index and compare fixed-viewer projections."""

    space = tracker.posterior.space
    first_index = 0
    second_index = space.support_size - 1
    first = space.materialize(first_index, seed=first_seed)
    second = space.materialize(second_index, seed=second_seed)
    first_view = json.loads(first.semantic_observation_json(tracker.viewer))
    second_view = json.loads(second.semantic_observation_json(tracker.viewer))
    exposed = sum(
        int(card["zone"] == _HAND_ZONE)
        for view in (first_view, second_view)
        for card in view["viewer_state"]["opponent_cards"]
    )
    return {
        "worlds_distinct": first_index != second_index,
        "authority_states_distinct": first.state_digest() != second.state_digest(),
        "viewer_projection_mismatches": int(first_view != second_view),
        "opponent_private_cards_exposed": exposed,
    }


__all__ = [
    "CardCalibrationPoint",
    "KnownTruthPoint",
    "aggregate_known_truth",
    "authority_hand",
    "score_belief_for_hand",
    "score_known_truth",
    "viewer_equivalence_audit",
]
