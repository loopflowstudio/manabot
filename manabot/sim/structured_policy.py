"""Validated ragged offers shared by prototype and learned compound decoders.

Rust remains authoritative for IDs and legality. `flatten_projection` validates
public choice rows without fixed-width padding; decoders return ID-only
submissions. The deterministic prototype scorer remains a parity instrument.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import managym
from managym.choice import OfferProjection, Subject


@dataclass(frozen=True)
class ChoiceFeatures:
    """Visible object rows and numeric public parameters for one choice row."""

    objects: tuple[int, ...]
    mana: tuple[int, ...] = ()
    ordinal: int | None = None
    attack: bool | None = None


def _subject_row(subject: Subject, actor: int, rows: Mapping[int, int]) -> int:
    if subject.kind == "player":
        if subject.entity not in (0, 1):
            raise ValueError("choice references an invalid player")
        return 0 if subject.entity == actor else 1
    try:
        return rows[subject.entity]
    except KeyError as error:
        raise ValueError(
            "choice references an object absent from the visible encoding"
        ) from error


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
    support: managym.ChoiceSupport
    fingerprint: str
    offer_inputs: tuple[ChoiceFeatures, ...] = ()
    role_inputs: tuple[ChoiceFeatures, ...] = ()
    candidate_inputs: tuple[ChoiceFeatures, ...] = ()

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


def flatten_projection(
    projection: Mapping[str, Any],
    *,
    object_rows: Mapping[int, int] | None = None,
    viewer_json: str | None = None,
) -> RaggedOfferBatch:
    """Validate and flatten a wire projection without fixed-width padding."""

    try:
        typed = OfferProjection.from_json(json.dumps(projection))
        support = typed.support
    except ValueError as error:
        raise StructuredPolicyError(str(error)) from error
    offers = tuple(projection["offers"])
    choices: list[ChoiceRow] = []
    candidates: list[Mapping[str, Any]] = []
    offsets = [0]
    for offer_index, offer in enumerate(offers):
        for choice in offer["choices"]:
            start = len(candidates)
            candidates.extend(choice["candidates"]["initial"])
            choices.append(
                ChoiceRow(
                    offer_index,
                    choice["role"],
                    choice["min"],
                    choice["max"],
                    start,
                    len(candidates),
                )
            )
        offsets.append(len(choices))
    offer_inputs: list[ChoiceFeatures] = []
    role_inputs: list[ChoiceFeatures] = []
    candidate_inputs: list[ChoiceFeatures] = []
    if object_rows is not None:
        actor = typed.actor
        for offer in typed.offers:
            details = offer.details
            objects = tuple(
                _subject_row(subject, actor, object_rows)
                for subject in (offer.source, details.subject, details.target)
                if subject is not None
            )
            if details.outside_candidate is not None:
                try:
                    objects += (object_rows[details.outside_candidate],)
                except KeyError as error:
                    raise StructuredPolicyError(
                        "outside choice has no visible row"
                    ) from error
            offer_inputs.append(
                ChoiceFeatures(
                    objects,
                    details.mana,
                    None if details.program is None else details.program.ordinal,
                    details.attack,
                )
            )
            for choice in offer.choices:
                context = choice.context
                role_inputs.append(
                    ChoiceFeatures(
                        ()
                        if context.subject is None
                        else (_subject_row(context.subject, actor, object_rows),),
                        context.mana,
                        context.requirement,
                    )
                )
                for candidate in choice.candidates:
                    candidate_inputs.append(
                        ChoiceFeatures(
                            (_subject_row(candidate.subject, actor, object_rows),)
                        )
                    )
    return RaggedOfferBatch(
        projection,
        offers,
        tuple(choices),
        tuple(candidates),
        tuple(offsets),
        support,
        hashlib.sha256(
            json.dumps(
                [projection, viewer_json], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        tuple(offer_inputs),
        tuple(role_inputs),
        tuple(candidate_inputs),
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
