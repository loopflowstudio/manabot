"""Experimental ragged decoder for the structured-offer prototype.

This module is intentionally isolated from the production policy network and
the fixed-width observation encoder. Rust remains authoritative for IDs and
legality; the decoder only scores public rows and emits an ID-only submission.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import sys
from typing import Any, Mapping, Sequence


class StructuredPolicyError(ValueError):
    """The projection or policy scores cannot produce a safe submission."""


@dataclass(frozen=True)
class ChoiceRow:
    offer_index: int
    role: int
    minimum: int
    maximum: int
    candidate_start: int
    candidate_stop: int


@dataclass(frozen=True)
class RaggedOfferBatch:
    """One flattened offer projection with explicit candidate offsets."""

    projection: Mapping[str, Any]
    offers: tuple[Mapping[str, Any], ...]
    choices: tuple[ChoiceRow, ...]
    candidates: tuple[Mapping[str, Any], ...]
    choice_offsets: tuple[int, ...]

    @property
    def max_candidate_count(self) -> int:
        return max(
            (row.candidate_stop - row.candidate_start for row in self.choices),
            default=0,
        )

    @property
    def max_legal_branches(self) -> int:
        branches = 1
        for row in self.choices:
            count = row.candidate_stop - row.candidate_start
            branches *= sum(
                math.comb(count, selected)
                for selected in range(row.minimum, row.maximum + 1)
            )
        return branches


@dataclass(frozen=True)
class PolicyScores:
    offer_scores: tuple[float, ...]
    candidate_scores: tuple[float, ...]


@dataclass(frozen=True)
class DecodedSubmission:
    offer_id: int
    answers: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {"offer_id": self.offer_id, "answers": list(self.answers)}

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), separators=(",", ":"), sort_keys=True)


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StructuredPolicyError(f"{field} must be an integer")
    if value < 0:
        raise StructuredPolicyError(f"{field} must be non-negative")
    return value


def flatten_projection(projection: Mapping[str, Any]) -> RaggedOfferBatch:
    """Validate and flatten a wire projection without fixed-width padding."""

    raw_offers = projection.get("offers")
    if not isinstance(raw_offers, list) or not raw_offers:
        raise StructuredPolicyError("projection must contain at least one offer")
    if len(raw_offers) > sys.maxsize:
        raise StructuredPolicyError("offer count exceeds platform index range")

    offers: list[Mapping[str, Any]] = []
    choices: list[ChoiceRow] = []
    candidates: list[Mapping[str, Any]] = []
    choice_offsets = [0]
    seen_offers: set[int] = set()

    for offer_index, raw_offer in enumerate(raw_offers):
        if not isinstance(raw_offer, Mapping):
            raise StructuredPolicyError("offer row must be an object")
        offer_id = _integer(raw_offer.get("id"), "offer.id")
        if offer_id in seen_offers:
            raise StructuredPolicyError(f"duplicate offer id {offer_id}")
        seen_offers.add(offer_id)
        offers.append(raw_offer)

        raw_choices = raw_offer.get("choices")
        if not isinstance(raw_choices, list):
            raise StructuredPolicyError("offer.choices must be a list")
        seen_roles: set[int] = set()
        for raw_choice in raw_choices:
            if (
                not isinstance(raw_choice, Mapping)
                or raw_choice.get("kind") != "select"
            ):
                raise StructuredPolicyError("only select choices are supported")
            role = _integer(raw_choice.get("role"), "choice.role")
            if role in seen_roles:
                raise StructuredPolicyError(f"duplicate role id {role}")
            seen_roles.add(role)
            minimum = _integer(raw_choice.get("min"), "choice.min")
            maximum = _integer(raw_choice.get("max"), "choice.max")
            if maximum < minimum:
                raise StructuredPolicyError("choice max is below min")
            if raw_choice.get("distinct") is not True:
                raise StructuredPolicyError("decoder requires distinct candidates")

            source = raw_choice.get("candidates")
            if not isinstance(source, Mapping):
                raise StructuredPolicyError("choice candidates must be an object")
            if source.get("depends_on") not in ([], ()):
                raise StructuredPolicyError(
                    "dynamic candidate dependencies are unsupported"
                )
            initial = source.get("initial")
            if not isinstance(initial, list):
                raise StructuredPolicyError(
                    "candidate source must contain an initial list"
                )
            if len(initial) > sys.maxsize - len(candidates):
                raise StructuredPolicyError(
                    "candidate count exceeds platform index range"
                )
            if maximum > len(initial):
                raise StructuredPolicyError("choice max exceeds candidate count")

            start = len(candidates)
            seen_candidates: set[int] = set()
            for raw_candidate in initial:
                if not isinstance(raw_candidate, Mapping):
                    raise StructuredPolicyError("candidate row must be an object")
                candidate_id = _integer(raw_candidate.get("id"), "candidate.id")
                if candidate_id in seen_candidates:
                    raise StructuredPolicyError(
                        f"duplicate candidate id {candidate_id} in role {role}"
                    )
                seen_candidates.add(candidate_id)
                candidates.append(raw_candidate)
            choices.append(
                ChoiceRow(
                    offer_index=offer_index,
                    role=role,
                    minimum=minimum,
                    maximum=maximum,
                    candidate_start=start,
                    candidate_stop=len(candidates),
                )
            )
        choice_offsets.append(len(choices))

    return RaggedOfferBatch(
        projection=projection,
        offers=tuple(offers),
        choices=tuple(choices),
        candidates=tuple(candidates),
        choice_offsets=tuple(choice_offsets),
    )


class SeededSemanticScorer:
    """Stable synthetic score tape shared by both benchmark adapters."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def _score(self, category: str, ordinal: int, row: Mapping[str, Any]) -> float:
        payload = json.dumps(
            [self.seed, category, ordinal, row],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
        return (value / ((1 << 64) - 1)) * 2.0 - 1.0

    def score(self, batch: RaggedOfferBatch, decision_ordinal: int) -> PolicyScores:
        return PolicyScores(
            offer_scores=tuple(
                self._score("offer", decision_ordinal, offer) for offer in batch.offers
            ),
            candidate_scores=tuple(
                self._score("candidate", decision_ordinal, candidate)
                for candidate in batch.candidates
            ),
        )


class RaggedPolicyDecoder:
    """Decode scored ragged rows into one Rust-validated offer submission."""

    def __init__(self, selection_threshold: float = 0.0) -> None:
        if not math.isfinite(selection_threshold):
            raise StructuredPolicyError("selection threshold must be finite")
        self.selection_threshold = selection_threshold

    def decode(
        self, batch: RaggedOfferBatch, scores: PolicyScores
    ) -> DecodedSubmission:
        self._validate_scores(scores.offer_scores, len(batch.offers), "offer")
        self._validate_scores(
            scores.candidate_scores, len(batch.candidates), "candidate"
        )

        offer_index = max(
            range(len(batch.offers)),
            key=lambda index: (scores.offer_scores[index], -index),
        )
        offer = batch.offers[offer_index]
        answers: list[Mapping[str, Any]] = []
        start = batch.choice_offsets[offer_index]
        stop = batch.choice_offsets[offer_index + 1]
        for row in batch.choices[start:stop]:
            indexes = list(range(row.candidate_start, row.candidate_stop))
            ranked = sorted(
                indexes, key=lambda index: (-scores.candidate_scores[index], index)
            )
            selected = [
                index
                for index in indexes
                if scores.candidate_scores[index] >= self.selection_threshold
            ]
            if len(selected) < row.minimum:
                selected = ranked[: row.minimum]
            elif len(selected) > row.maximum:
                selected = ranked[: row.maximum]
            selected_set = set(selected)
            selected_ids = [
                _integer(batch.candidates[index].get("id"), "candidate.id")
                for index in indexes
                if index in selected_set
            ]
            answers.append(
                {"kind": "candidates", "role": row.role, "candidates": selected_ids}
            )

        return DecodedSubmission(
            offer_id=_integer(offer.get("id"), "offer.id"), answers=tuple(answers)
        )

    @staticmethod
    def _validate_scores(scores: Sequence[float], expected: int, label: str) -> None:
        if len(scores) != expected:
            raise StructuredPolicyError(
                f"{label} score count {len(scores)} does not match {expected} rows"
            )
        if any(isinstance(score, bool) or not math.isfinite(score) for score in scores):
            raise StructuredPolicyError(f"{label} scores must all be finite numbers")


def decode_projection(
    projection: Mapping[str, Any], *, seed: int, decision_ordinal: int
) -> tuple[RaggedOfferBatch, DecodedSubmission]:
    """Convenience entry point used by the benchmark harness."""

    batch = flatten_projection(projection)
    scores = SeededSemanticScorer(seed).score(batch, decision_ordinal)
    return batch, RaggedPolicyDecoder().decode(batch, scores)
