"""Closed Python consumer for the Rust-owned study artifact v1 contract."""

from enum import Enum
import math
from typing import Literal

from pydantic import model_validator

from etude.experience_protocol import (
    Command,
    ExperienceFrame,
    InteractionOffer,
    ProtocolModel,
    UInt8,
    UInt32,
    UInt64,
)

STUDY_VERSION: Literal[1] = 1


class KnowledgeScope(str, Enum):
    HISTORICAL_VIEWER = "historical_viewer"


class ContentPackIdentity(ProtocolModel):
    id: str
    version: str
    content_hash: str
    asset_manifest_sha256: str


class EngineIdentity(ProtocolModel):
    version: str
    build_sha256: str


class ModelIdentity(ProtocolModel):
    id: str
    checkpoint_sha256: str


class AnalysisBudgetIdentity(ProtocolModel):
    id: str
    max_nodes: UInt64
    sampled_worlds: UInt32
    rollouts_per_world: UInt32


class StudyIdentity(ProtocolModel):
    artifact_id: str
    source_replay_id: str
    source_replay_sha256: str
    match_id: str
    content_pack: ContentPackIdentity
    engine: EngineIdentity
    model: ModelIdentity
    analysis_budget: AnalysisBudgetIdentity
    knowledge_scope: KnowledgeScope


class DecisionAlternative(ProtocolModel):
    id: str
    command: Command


class PolicyMass(ProtocolModel):
    alternative: str
    probability: float


class SearchValue(ProtocolModel):
    alternative: str
    perspective: UInt8
    expected_match_points: float


class VisitCount(ProtocolModel):
    alternative: str
    visits: UInt64


class SampledWorldRobustness(ProtocolModel):
    alternative: str
    favorable_worlds: UInt32
    sampled_worlds: UInt32


class UncertaintyEvidence(ProtocolModel):
    alternative: str
    standard_error: float
    method: str


class EvidenceProvenance(ProtocolModel):
    producer: str
    producer_version: str
    generated_at: str
    evidence_sha256: str


class DecisionEvidence(ProtocolModel):
    policy_mass: list[PolicyMass]
    search_value: list[SearchValue]
    visits: list[VisitCount]
    sampled_world_robustness: list[SampledWorldRobustness]
    uncertainty: list[UncertaintyEvidence]
    provenance: EvidenceProvenance


class StudyLandmark(ProtocolModel):
    id: str
    decision_id: str
    match_state_hash: str
    viewer: UInt8
    prompt_id: UInt64
    offer_id: UInt32
    frame: ExperienceFrame
    offer: InteractionOffer
    played: Command
    alternatives: list[DecisionAlternative]
    evidence: DecisionEvidence


class StudyArtifact(ProtocolModel):
    version: Literal[1]
    identity: StudyIdentity
    landmarks: list[StudyLandmark]

    @model_validator(mode="after")
    def validate_bindings_and_privacy(self) -> "StudyArtifact":
        if self.identity.knowledge_scope is not KnowledgeScope.HISTORICAL_VIEWER:
            raise ValueError("study v1 requires historical_viewer knowledge")
        if not self.landmarks:
            raise ValueError("study artifact must contain at least one landmark")
        for landmark in self.landmarks:
            self._validate_landmark(landmark)
        return self

    def _validate_landmark(self, landmark: StudyLandmark) -> None:
        context = f"landmark {landmark.id}"
        frame = landmark.frame
        pack = self.identity.content_pack
        if frame.match_id != self.identity.match_id:
            raise ValueError(f"{context}: frame match does not match study identity")
        if (
            frame.content_hash != pack.content_hash
            or frame.asset_manifest_hash != pack.asset_manifest_sha256
        ):
            raise ValueError(f"{context}: frame content pack hashes drifted")
        if frame.asset_pack is not None and (
            frame.asset_pack.id != pack.id
            or frame.asset_pack.version != pack.version
            or frame.asset_pack.manifest_sha256 != pack.asset_manifest_sha256
        ):
            raise ValueError(f"{context}: frame asset pack identity drifted")
        if frame.projection.opponent.hand:
            raise ValueError(
                f"{context}: opponent-private hand identities are forbidden"
            )
        if (
            frame.prompt is None
            or frame.prompt.id != landmark.prompt_id
            or frame.prompt.actor != landmark.viewer
            or landmark.offer.id != landmark.offer_id
            or landmark.offer.actor != landmark.viewer
        ):
            raise ValueError(f"{context}: viewer, prompt, or offer binding drifted")
        frame_offer = next(
            (offer for offer in frame.offers if offer.id == landmark.offer_id), None
        )
        if frame_offer is None or frame_offer != landmark.offer:
            raise ValueError(f"{context}: selected offer differs from frame offer")

        self._validate_command(landmark.played, landmark, context, selected=True)
        if not landmark.alternatives:
            raise ValueError(f"{context}: no decision alternatives recorded")
        alternative_ids = {alternative.id for alternative in landmark.alternatives}
        if len(alternative_ids) != len(landmark.alternatives):
            raise ValueError(f"{context}: duplicate alternative")
        for alternative in landmark.alternatives:
            self._validate_command(alternative.command, landmark, context, selected=False)

        evidence = landmark.evidence
        for label, ids in (
            ("policy mass", [row.alternative for row in evidence.policy_mass]),
            ("search value", [row.alternative for row in evidence.search_value]),
            ("visits", [row.alternative for row in evidence.visits]),
            (
                "sampled-world robustness",
                [row.alternative for row in evidence.sampled_world_robustness],
            ),
            ("uncertainty", [row.alternative for row in evidence.uncertainty]),
        ):
            if len(set(ids)) != len(ids) or set(ids) != alternative_ids:
                raise ValueError(f"{context}: {label} does not cover alternatives")

        probabilities = [row.probability for row in evidence.policy_mass]
        if any(
            not math.isfinite(probability) or not 0 <= probability <= 1
            for probability in probabilities
        ) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
            raise ValueError(f"{context}: invalid policy mass")
        if any(
            row.perspective != landmark.viewer
            or not math.isfinite(row.expected_match_points)
            for row in evidence.search_value
        ):
            raise ValueError(f"{context}: invalid search value")
        if any(
            row.sampled_worlds == 0 or row.favorable_worlds > row.sampled_worlds
            for row in evidence.sampled_world_robustness
        ):
            raise ValueError(f"{context}: invalid sampled-world robustness")
        if any(
            not math.isfinite(row.standard_error) or row.standard_error < 0
            for row in evidence.uncertainty
        ):
            raise ValueError(f"{context}: invalid uncertainty")

    @staticmethod
    def _validate_command(
        command: Command,
        landmark: StudyLandmark,
        context: str,
        *,
        selected: bool,
    ) -> None:
        if (
            command.match_id != landmark.frame.match_id
            or command.expected_revision != landmark.frame.revision
            or command.prompt_id != landmark.prompt_id
            or (selected and command.offer_id != landmark.offer_id)
            or not any(
                offer.id == command.offer_id for offer in landmark.frame.offers
            )
        ):
            raise ValueError(f"{context}: command identity drifted")
