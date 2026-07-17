"""Closed Python consumer for the Rust-owned study artifact v1 contract."""

from enum import Enum
import math
from typing import Literal

from pydantic import model_validator

from etude.experience_protocol import (
    Command,
    ExperienceFrame,
    InteractionOffer,
    PresentationEvent,
    ProtocolModel,
    UInt8,
    UInt32,
    UInt64,
)
from etude.replay_index import ReplayDecisionAddress, decision_payload_sha256

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


class RecordedDecision(ProtocolModel):
    ordinal: UInt32
    event_cursor: UInt64
    automatic: bool
    frame: ExperienceFrame
    offer: InteractionOffer
    played: Command
    presentation: list[PresentationEvent]


class RecordedDecisionInput(ProtocolModel):
    version: Literal[1]
    source_replay_id: str
    decision_count: UInt32
    decisions: list[RecordedDecision]

    @model_validator(mode="after")
    def validate_chronology_and_bindings(self) -> "RecordedDecisionInput":
        if self.decision_count != len(self.decisions):
            raise ValueError("declared decision count does not match input length")
        previous_cursor: int | None = None
        for expected_ordinal, decision in enumerate(self.decisions):
            context = f"decision {decision.ordinal}"
            if decision.ordinal != expected_ordinal:
                raise ValueError(
                    f"{context}: ordinals must be contiguous and preserve source order"
                )
            if previous_cursor is not None and decision.event_cursor <= previous_cursor:
                raise ValueError(f"{context}: event cursors must strictly increase")
            previous_cursor = decision.event_cursor
            self._validate_decision(decision, context)
        return self

    @staticmethod
    def _validate_decision(decision: RecordedDecision, context: str) -> None:
        frame = decision.frame
        prompt = frame.prompt
        if prompt is None:
            raise ValueError(f"{context}: decision frame has no prompt")
        if prompt.actor != decision.offer.actor:
            raise ValueError(f"{context}: prompt and offer actors differ")
        frame_offer = next(
            (offer for offer in frame.offers if offer.id == decision.offer.id), None
        )
        if frame_offer is None or frame_offer != decision.offer:
            raise ValueError(f"{context}: selected offer differs from frame offer")
        if (
            decision.played.match_id != frame.match_id
            or decision.played.expected_revision != frame.revision
            or decision.played.prompt_id != prompt.id
            or decision.played.offer_id != decision.offer.id
        ):
            raise ValueError(f"{context}: played command identity drifted")
        if frame.projection.opponent.hand:
            raise ValueError(
                f"{context}: opponent-private hand identities are forbidden"
            )


class StudyDecisionKind(str, Enum):
    PRIORITY = "priority"
    TARGETING = "targeting"
    ATTACK = "attack"
    BLOCK = "block"
    OTHER = "other"


class LandmarkReason(str, Enum):
    PRIORITY_COMMITMENT = "priority_commitment"
    PRIORITY_RESPONSE = "priority_response"
    TARGET_SELECTION = "target_selection"
    ATTACK_DECLARATION = "attack_declaration"
    BLOCK_DECLARATION = "block_declaration"
    BRANCHING_CHOICE = "branching_choice"
    PUBLIC_SEMANTIC_IMPACT = "public_semantic_impact"


class StudyDecision(ProtocolModel):
    id: str
    ordinal: UInt32
    viewer: UInt8
    event_cursor: UInt64
    automatic: bool
    kind: StudyDecisionKind
    frame: ExperienceFrame
    offer: InteractionOffer
    played: Command


class RankedStudyLandmark(ProtocolModel):
    decision_id: str
    rank: UInt8
    reasons: list[LandmarkReason]


class StudyDecisionIndex(ProtocolModel):
    version: Literal[1]
    identity: StudyIdentity
    decisions: list[StudyDecision]
    landmarks: list[RankedStudyLandmark]

    @model_validator(mode="after")
    def validate_bindings_and_recommendations(self) -> "StudyDecisionIndex":
        if self.identity.knowledge_scope is not KnowledgeScope.HISTORICAL_VIEWER:
            raise ValueError("study v1 requires historical_viewer knowledge")

        ids: set[str] = set()
        previous_cursor: int | None = None
        for expected_ordinal, decision in enumerate(self.decisions):
            context = f"decision {decision.ordinal}"
            if decision.ordinal != expected_ordinal:
                raise ValueError(
                    f"{context}: ordinals must be contiguous and preserve source order"
                )
            if previous_cursor is not None and decision.event_cursor <= previous_cursor:
                raise ValueError(f"{context}: event cursors must strictly increase")
            previous_cursor = decision.event_cursor
            if decision.id in ids:
                raise ValueError(f"{context}: duplicate study decision id")
            ids.add(decision.id)
            self._validate_decision(decision, context)

        if len(self.landmarks) > 7:
            raise ValueError(
                "study decision index cannot recommend more than seven landmarks"
            )
        landmark_ids: set[str] = set()
        reason_order = list(LandmarkReason)
        decisions_by_id = {decision.id: decision for decision in self.decisions}
        for expected_rank, landmark in enumerate(self.landmarks, start=1):
            if landmark.rank != expected_rank:
                raise ValueError("landmark ranks must be contiguous and one-based")
            decision = decisions_by_id.get(landmark.decision_id)
            if decision is None:
                raise ValueError("landmark references a missing decision")
            if landmark.decision_id in landmark_ids:
                raise ValueError("a decision can be recommended only once")
            landmark_ids.add(landmark.decision_id)
            if decision.automatic or len(decision.frame.offers) <= 1:
                raise ValueError("automatic or forced decisions cannot be landmarks")
            if not landmark.reasons:
                raise ValueError("landmark reasons cannot be empty")
            expected_reasons = sorted(set(landmark.reasons), key=reason_order.index)
            if landmark.reasons != expected_reasons:
                raise ValueError("landmark reasons must be unique and enum-ordered")
        return self

    def _validate_decision(self, decision: StudyDecision, context: str) -> None:
        frame = decision.frame
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
        prompt = frame.prompt
        if (
            prompt is None
            or prompt.actor != decision.viewer
            or decision.offer.actor != decision.viewer
        ):
            raise ValueError(f"{context}: viewer, prompt, or offer binding drifted")
        frame_offer = next(
            (offer for offer in frame.offers if offer.id == decision.offer.id), None
        )
        if frame_offer is None or frame_offer != decision.offer:
            raise ValueError(f"{context}: selected offer differs from frame offer")
        if (
            decision.played.match_id != frame.match_id
            or decision.played.expected_revision != frame.revision
            or decision.played.prompt_id != prompt.id
            or decision.played.offer_id != decision.offer.id
        ):
            raise ValueError(f"{context}: played command identity drifted")


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
        try:
            address = ReplayDecisionAddress.parse(landmark.decision_id)
        except ValueError as exc:
            raise ValueError(f"{context}: decision_id is not an erd1 address") from exc
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
        if (
            address.replay_id != self.identity.source_replay_id
            or address.match_id != self.identity.match_id
            or address.viewer != landmark.viewer
            or address.revision != frame.revision
            or address.prompt_id != landmark.prompt_id
            or address.offer_id != landmark.offer_id
            or address.command_id != landmark.played.command_id
            or address.decision_sha256
            != decision_payload_sha256(
                frame,
                landmark.offer,
                landmark.played,
                address.presentation_cursor,
            )
        ):
            raise ValueError(f"{context}: replay decision address drifted")
        if not landmark.alternatives:
            raise ValueError(f"{context}: no decision alternatives recorded")
        alternative_ids = {alternative.id for alternative in landmark.alternatives}
        if len(alternative_ids) != len(landmark.alternatives):
            raise ValueError(f"{context}: duplicate alternative")
        for alternative in landmark.alternatives:
            self._validate_command(
                alternative.command, landmark, context, selected=False
            )

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
            or not any(offer.id == command.offer_id for offer in landmark.frame.offers)
        ):
            raise ValueError(f"{context}: command identity drifted")
