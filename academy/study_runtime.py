"""Game-owned joins for ephemeral Study attempts and historical evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .experience_protocol import Command, InteractionOffer
from .replay_index import CanonicalReplayProjectionV1, RestoredReplayDecision
from .study_protocol import StudyArtifact, StudyLandmark


class StudyRuntimeError(ValueError):
    """Base error for the player-facing Study runtime."""


class StudyEvidenceUnavailableError(StudyRuntimeError):
    """No honest historical evidence exists for this replay decision."""


class StudyEvidenceMismatchError(StudyRuntimeError):
    """Historical evidence does not bind the selected canonical decision."""


class StudyPlanUnavailableError(StudyRuntimeError):
    """A selected comparison plan cannot be executed through authority."""


@dataclass(frozen=True)
class HistoricalStudyEvidenceRequest:
    projection: CanonicalReplayProjectionV1
    source_replay_sha256: str
    address: str
    restored: RestoredReplayDecision


class HistoricalStudyEvidenceProvider(Protocol):
    def artifact_for(
        self,
        request: HistoricalStudyEvidenceRequest,
    ) -> StudyArtifact | dict[str, Any]: ...


class UnavailableHistoricalStudyEvidenceProvider:
    def artifact_for(
        self,
        request: HistoricalStudyEvidenceRequest,
    ) -> StudyArtifact:
        del request
        raise StudyEvidenceUnavailableError(
            "Study evidence is unavailable for this recording."
        )


@dataclass(frozen=True)
class JoinedStudyEvidence:
    artifact: StudyArtifact
    landmark: StudyLandmark


StudyPlanKind = Literal["played", "policy", "search"]


@dataclass(frozen=True)
class StudyPlanSelection:
    kind: StudyPlanKind
    command: Command
    offer: InteractionOffer
    alternative_id: str | None


def join_historical_study_evidence(
    request: HistoricalStudyEvidenceRequest,
    raw_artifact: StudyArtifact | dict[str, Any],
    *,
    allow_fixture_evidence: bool,
) -> JoinedStudyEvidence:
    """Validate one provider result against Game's selected replay authority."""
    try:
        artifact = StudyArtifact.model_validate(raw_artifact)
    except ValueError as exc:
        raise StudyEvidenceMismatchError("study_artifact_invalid") from exc

    identity = artifact.identity
    if (
        identity.source_replay_id != request.projection.replay_id
        or identity.source_replay_sha256 != request.source_replay_sha256
        or identity.match_id != request.projection.match_id
        or identity.content_pack.content_hash != request.projection.content_hash
        or identity.content_pack.asset_manifest_sha256
        != request.projection.asset_manifest_hash
    ):
        raise StudyEvidenceMismatchError("study_evidence_identity_mismatch")

    fixture_only = identity.analysis_budget.id == "fixture-only" or any(
        landmark.evidence.provenance.producer == "canonical-replay-fixture"
        for landmark in artifact.landmarks
    )
    if fixture_only and not allow_fixture_evidence:
        raise StudyEvidenceMismatchError("fixture_evidence_forbidden")

    landmarks = [
        landmark
        for landmark in artifact.landmarks
        if landmark.decision_id == request.address
    ]
    if len(landmarks) != 1:
        raise StudyEvidenceMismatchError("study_landmark_not_found")
    landmark = landmarks[0]
    restored = request.restored
    if (
        landmark.viewer != restored.viewer
        or landmark.prompt_id
        != (restored.frame.prompt.id if restored.frame.prompt is not None else None)
        or landmark.offer_id != restored.offer.id
        or landmark.frame != restored.frame
        or landmark.offer != restored.offer
        or landmark.played != restored.command
    ):
        raise StudyEvidenceMismatchError("study_landmark_identity_mismatch")

    return JoinedStudyEvidence(
        artifact=artifact.model_copy(
            update={"landmarks": [landmark.model_copy(deep=True)]},
            deep=True,
        ),
        landmark=landmark.model_copy(deep=True),
    )


def select_study_plan(
    landmark: StudyLandmark,
    kind: StudyPlanKind,
) -> StudyPlanSelection:
    """Resolve a labelled plan without collapsing its distinct evidence."""
    if kind == "played":
        return StudyPlanSelection(
            kind=kind,
            command=landmark.played.model_copy(deep=True),
            offer=landmark.offer.model_copy(deep=True),
            alternative_id=None,
        )

    alternatives = {
        alternative.id: alternative for alternative in landmark.alternatives
    }
    offer_order = {offer.id: index for index, offer in enumerate(landmark.frame.offers)}
    if kind == "policy":
        selected_id = max(
            landmark.evidence.policy_mass,
            key=lambda row: (
                row.probability,
                -offer_order[alternatives[row.alternative].command.offer_id],
            ),
        ).alternative
    elif kind == "search":
        visits = {row.alternative: row.visits for row in landmark.evidence.visits}
        uncertainty = {
            row.alternative: row.standard_error for row in landmark.evidence.uncertainty
        }
        selected_id = max(
            landmark.evidence.search_value,
            key=lambda row: (
                row.expected_match_points,
                visits[row.alternative],
                -uncertainty[row.alternative],
                -offer_order[alternatives[row.alternative].command.offer_id],
            ),
        ).alternative
    else:  # pragma: no cover - Literal plus HTTP validation closes this branch.
        raise StudyPlanUnavailableError("study_plan_invalid")

    alternative = alternatives[selected_id]
    offer = next(
        (
            candidate
            for candidate in landmark.frame.offers
            if candidate.id == alternative.command.offer_id
        ),
        None,
    )
    if offer is None:
        raise StudyPlanUnavailableError("study_plan_offer_missing")
    return StudyPlanSelection(
        kind=kind,
        command=alternative.command.model_copy(deep=True),
        offer=offer.model_copy(deep=True),
        alternative_id=selected_id,
    )
