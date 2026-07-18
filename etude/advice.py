"""Versioned belief-conditioned strategy advice on Etude's shared endpoint.

The advice-v1 provider joins a Game-owned replay/root capability, canonical
Game viewer and belief metadata, manabot belief normalization and conditional
search, and managym semantic offers/world materialization. Its public response
is closed and viewer-safe. The legacy GAM-6 fixture adapter remains available
through the same request, response, and ``/api/advice`` imports while callers
migrate to the fully pinned identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
)

from pydantic import Field, TypeAdapter, model_validator

from etude.advice_identity import (
    AbiIdentity,
    AdviceIdentity,
    AdviceRequestIdentity,
    AdvisorIdentity,
    CheckpointArtifact,
    CodeSourceArtifact,
)
from etude.experience_protocol import ExperienceFrame, InteractionOffer, ProtocolModel
from etude.replay_index import (
    DecisionNotFoundError,
    InvalidAddressError,
    ReplayDecisionAddress,
)
from etude.study_branch import (
    StudyBranchUnavailableError,
    StudyForkProvider,
)
from etude.study_protocol import DecisionEvidence, StudyArtifact, StudyLandmark
from etude.testing_house_protocol import BeliefScenario, ViewerIdentity
from managym.decision import SEMANTIC_DECISION_VERSION, DecisionFrame
from managym.possible_worlds import (
    POSSIBLE_WORLD_SPACE_VERSION,
    PossibleWorldError,
    PossibleWorldSpace,
)

if TYPE_CHECKING:
    from manabot.belief.range import BeliefState
    from manabot.sim.conditional_search import (
        ConditionalStrategyResult,
        ConditionResult,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "protocol" / "fixtures" / "advice-curated-decision.json"
VERSIONED_FIXTURE_PATH = (
    REPO_ROOT / "protocol" / "fixtures" / "advice-belief-conditioned-v1.json"
)
CHECKPOINT_VERSIONED_FIXTURE_PATH = (
    REPO_ROOT / "protocol" / "fixtures" / "advice-checkpoint-policy-v1.json"
)
FLIP_VERSIONED_FIXTURE_PATH = (
    REPO_ROOT / "protocol" / "fixtures" / "advice-belief-conditioned-flip-v1.json"
)


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_bundle_sha256(paths: Iterable[Path]) -> str:
    """Hash repository-relative source names and bytes."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item.relative_to(REPO_ROOT))):
        relative = str(path.relative_to(REPO_ROOT)).encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


class AdviceScenarioSummary(ProtocolModel):
    """Viewer-safe presentation metadata for one belief scenario."""

    landmark_id: str
    label: str
    description: str
    inferred_range: str
    belief_kind: str


class AdviceScenarioRecord(AdviceScenarioSummary):
    """The fixture's scenario metadata, including its conditional-search binding.

    ``condition_id`` and ``seed_plan`` are generator provenance, not request
    fields; the adapter cross-references each record to a landmark by
    ``landmark_id``. Both scenarios use the same seed plan, so their visible
    delta is attributable to belief conditioning rather than random-seed drift.
    """

    condition_id: str
    seed_plan: str


class AdviceArtifact(ProtocolModel):
    """Wrapper around a ``StudyArtifact`` plus a prototype presentation layer.

    The ``artifact`` is a standard ``StudyArtifact`` (two landmarks at the same
    ``erd1`` decision, each with evidence from one paired conditional-PUCT run)
    and validates through both the Pydantic model and the Rust-owned
    ``study-v1`` schema. The ``scenarios`` array is a prototype presentation
    layer validated here, not by the Rust schema.
    """

    artifact: StudyArtifact
    scenarios: list[AdviceScenarioRecord]

    @model_validator(mode="after")
    def validate_comparison_identity(self) -> AdviceArtifact:
        landmarks = self.artifact.landmarks
        if len(landmarks) != 2 or len(self.scenarios) != 2:
            raise ValueError("advice comparison must contain exactly two scenarios")
        landmark_ids = {landmark.id for landmark in landmarks}
        scenario_ids = {scenario.landmark_id for scenario in self.scenarios}
        if scenario_ids != landmark_ids:
            raise ValueError("advice scenarios do not cover the artifact landmarks")
        if len({scenario.condition_id for scenario in self.scenarios}) != 2:
            raise ValueError("advice scenarios must bind distinct conditions")
        if len({scenario.seed_plan for scenario in self.scenarios}) != 1:
            raise ValueError("advice scenarios must share one paired seed plan")
        if len({landmark.decision_id for landmark in landmarks}) != 1:
            raise ValueError("advice landmarks must share one decision address")

        first, second = landmarks
        if (
            second.frame != first.frame
            or second.offer != first.offer
            or second.played != first.played
            or second.alternatives != first.alternatives
        ):
            raise ValueError(
                "advice landmarks must share facts, command, and action vocabulary"
            )
        if first.evidence == second.evidence:
            raise ValueError("advice conditions must produce distinct evidence")
        if (
            len({landmark.evidence.provenance.producer for landmark in landmarks}) != 1
            or len(
                {landmark.evidence.provenance.generated_at for landmark in landmarks}
            )
            != 1
        ):
            raise ValueError("advice scenarios must share one producer identity")
        return self


class AvailableQuantity(ProtocolModel):
    status: Literal["available"]
    value: float
    method: str | None = None
    visits: int | None = Field(default=None, ge=0)
    simulations: int | None = Field(default=None, ge=0)
    covered_worlds: int | None = Field(default=None, ge=0)
    co_best_worlds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_finite(self) -> "AvailableQuantity":
        if not math.isfinite(self.value):
            raise ValueError("available advice quantity must be finite")
        return self


class UnavailableQuantity(ProtocolModel):
    status: Literal["unavailable"]
    reason: Literal[
        "no_realized_visits",
        "no_realized_simulations",
        "world_coverage_unrecorded",
        "no_realized_world_coverage",
        "insufficient_world_coverage",
        "comparison_quantity_unavailable",
    ]


AdviceQuantity: TypeAlias = Annotated[
    AvailableQuantity | UnavailableQuantity,
    Field(discriminator="status"),
]

AdviceUnavailableReason: TypeAlias = Literal[
    "legacy_identity_incomplete",
    "scenario_not_found",
    "decision_not_found",
    "replay_mismatch",
    "match_mismatch",
    "world_mismatch",
    "content_mismatch",
    "observation_abi_mismatch",
    "action_abi_mismatch",
    "possible_world_abi_mismatch",
    "information_boundary_mismatch",
    "planner_mismatch",
    "evaluator_mismatch",
    "advisor_artifact_unavailable",
    "advisor_artifact_mismatch",
    "compute_mismatch",
    "seed_plan_mismatch",
    "decision_identity_mismatch",
    "decision_root_unavailable",
    "action_identity_mismatch",
    "belief_artifact_unavailable",
    "belief_distribution_unavailable",
    "belief_distribution_mismatch",
    "belief_viewer_mismatch",
    "belief_decision_mismatch",
    "belief_provenance_mismatch",
    "belief_space_mismatch",
    "belief_invalid",
    "policy_mass_invalid",
    "private_projection_failure",
]


class BeliefNormalizationReceipt(ProtocolModel):
    scenario_id: str
    space_identity: str
    belief_model_id: str
    distribution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_belief_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    positive_support: int = Field(ge=1)
    normalization_error: float = Field(ge=0.0)
    provenance_kind: Literal["player_authored", "model_inferred"]
    provenance_identity: str


class AdvisorOfferEvidence(ProtocolModel):
    offer_id: int = Field(ge=0)
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    visits: int = Field(ge=0)
    q: AdviceQuantity
    robustness: AdviceQuantity
    uncertainty: AdviceQuantity


class AdvisorScenarioEvidence(ProtocolModel):
    scenario_id: str
    belief: BeliefNormalizationReceipt
    condition_mass: float = Field(ge=0.0, le=1.0)
    support: int = Field(ge=1)
    sampled_worlds: int = Field(ge=1)
    actions: list[AdvisorOfferEvidence]
    root_value: AdviceQuantity
    root_uncertainty: AdviceQuantity
    simulations: int = Field(ge=1)
    cap_hits: int = Field(ge=0)
    tree_nodes: int = Field(ge=0)


class AdvisorDeltaQuantity(ProtocolModel):
    status: Literal["available", "unavailable"]
    value: float | None = None
    reason: Literal["comparison_quantity_unavailable"] | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "AdvisorDeltaQuantity":
        if self.status == "available":
            if self.value is None or not math.isfinite(self.value) or self.reason:
                raise ValueError("available delta requires one finite value")
        elif self.value is not None or self.reason is None:
            raise ValueError("unavailable delta requires only a reason")
        return self


class AdvisorOfferDelta(ProtocolModel):
    offer_id: int = Field(ge=0)
    policy_probability: float
    q: AdvisorDeltaQuantity
    robustness: AdvisorDeltaQuantity
    uncertainty: AdvisorDeltaQuantity


class AdvisorComparison(ProtocolModel):
    semantic: Literal["left_minus_right"] = "left_minus_right"
    left_scenario_id: str
    right_scenario_id: str
    left_belief_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    right_belief_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actions: list[AdvisorOfferDelta]
    root_value: AdvisorDeltaQuantity
    root_uncertainty: AdvisorDeltaQuantity


class AdvisorSemanticOffer(ProtocolModel):
    offer_id: int = Field(ge=0)
    actor: int = Field(ge=0)
    verb: str
    label: str


class AdvisorStrategyEvidence(ProtocolModel):
    schema_version: Literal[1] = 1
    policy_semantic: Literal["puct_visit_distribution/v1"]
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    viewer_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    advisor_identity: AdvisorIdentity
    offers: list[AdvisorSemanticOffer]
    scenarios: list[AdvisorScenarioEvidence]
    comparison: AdvisorComparison | None = None
    realized_compute: dict[str, int | float | str]
    prior_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AdviceResponse(ProtocolModel):
    """One advice response: real evidence on success, typed closed on failure."""

    contract: Literal["advice-v1"] = "advice-v1"
    serialization: Literal["advisor-canonical-json-v1"] = "advisor-canonical-json-v1"
    status: Literal["ok", "unavailable"]
    reason: AdviceUnavailableReason | None = None
    address: str | None = None
    frame: ExperienceFrame | None = None
    offers: list[InteractionOffer] = Field(default_factory=list)
    scenario: AdviceScenarioSummary | None = None
    evidence: DecisionEvidence | None = None
    deltas: dict[str, dict[str, float]] | None = None
    identity: AdviceIdentity | None = None
    request_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    strategy: AdvisorStrategyEvidence | None = None

    @model_validator(mode="after")
    def validate_versioned_envelope(self) -> "AdviceResponse":
        if self.status == "ok" and self.reason is not None:
            raise ValueError("available advice cannot carry an unavailable reason")
        if self.status == "unavailable" and self.reason is None:
            raise ValueError("unavailable advice requires a closed reason")
        if self.strategy is not None and self.status != "ok":
            raise ValueError("unavailable advice cannot carry strategy evidence")
        if self.status == "unavailable" and (
            self.evidence is not None
            or self.deltas is not None
            or self.strategy is not None
        ):
            raise ValueError("unavailable advice cannot carry partial evidence")
        if isinstance(self.identity, AdvisorIdentity):
            if self.request_sha256 is None:
                raise ValueError("versioned advice must bind its request SHA-256")
            if self.status == "ok" and (
                self.address is None
                or self.frame is None
                or not self.offers
                or self.strategy is None
            ):
                raise ValueError(
                    "available versioned advice requires complete evidence"
                )
        return self


class AdviceMeta(ProtocolModel):
    """The pinned decision's bootstrap: address, scenarios, expected identity."""

    address: str
    scenarios: list[AdviceScenarioSummary]
    identity: AdviceRequestIdentity
    contract: Literal["advice-v1"] = "advice-v1"
    versioned_request: dict[str, Any] | None = None


class AdviceRequest(ProtocolModel):
    """The sole POST /api/advice body, bridging legacy and advice-v1 callers."""

    address: str
    scenario_id: str | None = None
    identity: AdviceRequestIdentity | None = None
    contract: Literal["advice-v1"] | None = None
    viewer: ViewerIdentity | None = None
    scenario: BeliefScenario | None = None
    comparison_scenario: BeliefScenario | None = None
    advisor_identity: AdvisorIdentity | None = None

    @model_validator(mode="after")
    def validate_request_form(self) -> "AdviceRequest":
        legacy = self.scenario_id is not None or self.identity is not None
        versioned = any(
            value is not None
            for value in (
                self.contract,
                self.viewer,
                self.scenario,
                self.comparison_scenario,
                self.advisor_identity,
            )
        )
        if legacy and versioned:
            raise ValueError("advice request cannot mix legacy and advice-v1 fields")
        if legacy:
            if self.scenario_id is None or self.identity is None:
                raise ValueError("legacy advice requires scenario_id and identity")
            return self
        if (
            self.contract != "advice-v1"
            or self.viewer is None
            or self.scenario is None
            or self.advisor_identity is None
        ):
            raise ValueError(
                "advice-v1 requires contract, viewer, scenario, and advisor_identity"
            )
        try:
            parsed = ReplayDecisionAddress.parse(self.address)
        except InvalidAddressError as error:
            raise ValueError(
                "advice-v1 requires a canonical DecisionAddress"
            ) from error
        if parsed.viewer != self.viewer.rules_viewer:
            raise ValueError("decision address viewer differs from ViewerIdentity")
        if self.comparison_scenario is not None and (
            self.comparison_scenario.id == self.scenario.id
        ):
            raise ValueError("comparison scenarios must be distinct")
        return self

    @property
    def is_legacy(self) -> bool:
        return self.identity is not None

    @property
    def parsed_address(self) -> ReplayDecisionAddress:
        return ReplayDecisionAddress.parse(self.address)


@functools.cache
def load_advice_fixture() -> AdviceArtifact:
    """Load and validate the advice fixture once.

    The ``artifact`` validates as a ``StudyArtifact`` (which enforces
    viewer-safety, bindings, and opponent-hand privacy) and the scenarios
    cross-reference landmarks by id.
    """
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return AdviceArtifact.model_validate(payload)


def _expected_identity(artifact: AdviceArtifact) -> AdviceRequestIdentity:
    identity = artifact.artifact.identity
    return AdviceRequestIdentity(
        source_replay_id=identity.source_replay_id,
        match_id=identity.match_id,
        advisor_id=identity.model.id,
        compute_id=identity.analysis_budget.id,
    )


def advice_meta() -> AdviceMeta:
    """Return the pinned decision address, scenario summaries, and expected identity.

    Both the live play page and the replay/Study page bootstrap from this so
    they request advice for the same pinned ``erd1`` decision with the same
    identity, then POST ``/api/advice`` per scenario.
    """
    artifact = load_advice_fixture()
    landmark = artifact.artifact.landmarks[0]
    versioned_request = None
    if VERSIONED_FIXTURE_PATH.is_file():
        versioned_request = load_versioned_advice_fixture().request.model_dump(
            mode="json"
        )
    return AdviceMeta(
        address=landmark.decision_id,
        scenarios=[
            AdviceScenarioSummary(
                landmark_id=scenario.landmark_id,
                label=scenario.label,
                description=scenario.description,
                inferred_range=scenario.inferred_range,
                belief_kind=scenario.belief_kind,
            )
            for scenario in artifact.scenarios
        ],
        identity=_expected_identity(artifact),
        versioned_request=versioned_request,
    )


def compute_deltas(
    left: DecisionEvidence, right: DecisionEvidence
) -> dict[str, dict[str, float]]:
    """Per-action signed deltas (left - right) for policy, value, and uncertainty.

    The two scenarios share one action vocabulary, so alternatives align by id.
    """
    left_policy = {row.alternative: row.probability for row in left.policy_mass}
    right_policy = {row.alternative: row.probability for row in right.policy_mass}
    left_value = {
        row.alternative: row.expected_match_points for row in left.search_value
    }
    right_value = {
        row.alternative: row.expected_match_points for row in right.search_value
    }
    left_unc = {row.alternative: row.standard_error for row in left.uncertainty}
    right_unc = {row.alternative: row.standard_error for row in right.uncertainty}
    deltas: dict[str, dict[str, float]] = {}
    for alternative in left_policy:
        deltas[alternative] = {
            "policy_mass": left_policy[alternative] - right_policy[alternative],
            "search_value": left_value[alternative] - right_value[alternative],
            "uncertainty": left_unc[alternative] - right_unc[alternative],
        }
    return deltas


def _evidence_sha256(evidence_core: dict[str, Any]) -> str:
    """Content-address the evidence core, mirroring the fixture generator."""
    encoded = json.dumps(evidence_core, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def int13_condition_to_decision_evidence(
    condition: ConditionResult,
    alternative_ids: list[str],
    viewer: int,
    producer: str,
    generated_at: str,
) -> dict[str, Any]:
    """INT-13 seam: map one ``ConditionResult`` to the ``DecisionEvidence`` shape.

    This is the explicit binding point where INT-13's determinized-PUCT search
    evidence (``manabot.sim.conditional_search.ConditionResult``) becomes the
    per-action ``DecisionEvidence`` the shared live/Study surface renders. The
    prototype serves checked conditional-PUCT evidence from the static fixture
    at runtime. When INT-13's live search publishes, the ``POST /api/advice``
    server can route a live-address request to this mapping instead, without
    changing the surface or the ``DecisionEvidence`` contract.

    Field mapping (``ConditionResult`` -> ``DecisionEvidence``):

    - ``visit_counts`` -> ``policy_mass`` (normalized to a per-action
      distribution) and ``visits`` (raw per-action counts).
    - ``q_values`` -> ``search_value.expected_match_points`` (INT-13 q-values
      are mean per-action win-probability estimates).
    - ``world_q_values`` -> per-action ``uncertainty.standard_error`` using the
      standard error across searched worlds, and
      ``sampled_world_robustness.favorable_worlds`` by counting worlds whose
      per-action q-value exceeds 0.5. These are real per-world search values,
      not scalar broadcasts or aggregate proxies.

    ``producer`` is the INT-13 provenance string (e.g.
    ``"int-13-determinized-puct:v1"``). The returned dict validates as a
    ``DecisionEvidence`` and carries a content-addressed ``evidence_sha256``.
    """
    visit_counts = list(condition.visit_counts)
    q_values = list(condition.q_values)
    num_actions = len(alternative_ids)
    if len(visit_counts) != num_actions or len(q_values) != num_actions:
        raise ValueError("int13 condition action count does not match alternative_ids")
    world_q_values = [list(row) for row in condition.world_q_values]
    if len(world_q_values) != int(condition.sampled_worlds) or any(
        len(row) != num_actions for row in world_q_values
    ):
        raise ValueError("int13 condition world q-values do not match sampled worlds")
    total_visits = sum(visit_counts)
    denom = total_visits if total_visits > 0 else 1
    policy_mass = [
        {"alternative": alt, "probability": float(count) / float(denom)}
        for alt, count in zip(alternative_ids, visit_counts, strict=True)
    ]
    search_value = [
        {
            "alternative": alt,
            "perspective": viewer,
            "expected_match_points": float(q),
        }
        for alt, q in zip(alternative_ids, q_values, strict=True)
    ]
    visits = [
        {"alternative": alt, "visits": int(count)}
        for alt, count in zip(alternative_ids, visit_counts, strict=True)
    ]
    sampled = len(world_q_values)
    sampled_world_robustness = [
        {
            "alternative": alt,
            "favorable_worlds": sum(
                1 for world_values in world_q_values if world_values[index] > 0.5
            ),
            "sampled_worlds": sampled,
        }
        for index, alt in enumerate(alternative_ids)
    ]
    uncertainty = []
    for index, alt in enumerate(alternative_ids):
        samples = [float(world_values[index]) for world_values in world_q_values]
        standard_error = (
            statistics.stdev(samples) / math.sqrt(sampled) if sampled > 1 else 0.0
        )
        uncertainty.append(
            {
                "alternative": alt,
                "standard_error": standard_error,
                "method": "int13-world-q-spread",
            }
        )
    core = {
        "policy_mass": policy_mass,
        "search_value": search_value,
        "visits": visits,
        "sampled_world_robustness": sampled_world_robustness,
        "uncertainty": uncertainty,
    }
    return {
        **core,
        "provenance": {
            "producer": producer,
            "producer_version": "1",
            "generated_at": generated_at,
            "evidence_sha256": _evidence_sha256(core),
        },
    }


def int13_result_to_surface_deltas(
    result: ConditionalStrategyResult,
    left_condition_id: str,
    right_condition_id: str,
    alternative_ids: list[str],
    viewer: int,
    producer: str,
    generated_at: str,
) -> dict[str, dict[str, float]]:
    """INT-13 seam: reconcile two conditions into the surface's per-action deltas.

    The surface compares two belief scenarios; INT-13 produces one
    ``ConditionalStrategyResult`` for five aligned conditions (True, Has, Lacks,
    Q, Not(Q)) at one root. This maps two of those conditions (e.g. Has vs
    Lacks) into ``DecisionEvidence`` and returns the same per-action
    ``{policy_mass, search_value, uncertainty}`` delta dict the fixture-backed
    ``compute_deltas`` produces, so the surface renders INT-13 deltas unchanged.
    """
    by_id = result.condition_by_id
    if left_condition_id not in by_id or right_condition_id not in by_id:
        raise ValueError(
            f"int13 result missing condition {left_condition_id!r} or "
            f"{right_condition_id!r}"
        )
    left = DecisionEvidence.model_validate(
        int13_condition_to_decision_evidence(
            by_id[left_condition_id], alternative_ids, viewer, producer, generated_at
        )
    )
    right = DecisionEvidence.model_validate(
        int13_condition_to_decision_evidence(
            by_id[right_condition_id], alternative_ids, viewer, producer, generated_at
        )
    )
    return compute_deltas(left, right)


def _scenario_summary(
    artifact: AdviceArtifact, landmark_id: str
) -> AdviceScenarioSummary:
    scenario = next(
        record for record in artifact.scenarios if record.landmark_id == landmark_id
    )
    return AdviceScenarioSummary(
        landmark_id=scenario.landmark_id,
        label=scenario.label,
        description=scenario.description,
        inferred_range=scenario.inferred_range,
        belief_kind=scenario.belief_kind,
    )


def _unavailable(
    reason: AdviceUnavailableReason,
    identity: AdviceRequestIdentity,
) -> AdviceResponse:
    return AdviceResponse(status="unavailable", reason=reason, identity=identity)


def request_advice(
    address: str,
    scenario_id: str,
    identity: AdviceRequestIdentity,
) -> AdviceResponse:
    """Return advice for one scenario at one decision, or fail closed.

    Fail-closed cases (no evidence is ever returned on failure):
    - ``legacy_identity_incomplete``: the legacy identity does not have an
      exact checked upgrade to the fixture's full advisor identity.
    - ``scenario_not_found``: the scenario id is unknown.
    - ``decision_not_found``: the address does not match the fixture's decision.
    """
    artifact = load_advice_fixture()
    expected = _expected_identity(artifact)
    if identity != expected:
        return _unavailable("legacy_identity_incomplete", identity)

    landmark = next(
        (lm for lm in artifact.artifact.landmarks if lm.id == scenario_id), None
    )
    if landmark is None:
        return _unavailable("scenario_not_found", identity)
    if landmark.decision_id != address:
        return _unavailable("decision_not_found", identity)

    other = next(lm for lm in artifact.artifact.landmarks if lm.id != scenario_id)
    deltas = compute_deltas(landmark.evidence, other.evidence)
    return AdviceResponse(
        status="ok",
        address=landmark.decision_id,
        frame=landmark.frame,
        offers=list(landmark.frame.offers),
        scenario=_scenario_summary(artifact, scenario_id),
        evidence=landmark.evidence,
        deltas=deltas,
        identity=identity,
    )


def landmark_alternative_ids(landmark: StudyLandmark) -> list[str]:
    """The shared action vocabulary for one landmark, in alternative-id order."""
    return [alternative.id for alternative in landmark.alternatives]


def policy_mass_is_non_uniform(landmark: StudyLandmark) -> bool:
    """True if the landmark's policy mass is not uniform across alternatives."""
    probabilities = [row.probability for row in landmark.evidence.policy_mass]
    if not probabilities:
        return False
    first = probabilities[0]
    return any(not math.isclose(p, first, abs_tol=1e-9) for p in probabilities[1:])


# ---------------------------------------------------------------------------
# advice-v1 belief-conditioned provider
# ---------------------------------------------------------------------------


class BeliefDistributionPayload(ProtocolModel):
    """Canonical weights resolved separately from Game-owned scenario metadata."""

    space_identity: str
    belief_model_id: str
    weights: list[float]
    distribution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_kind: Literal["player_authored", "model_inferred"]
    provenance_identity: str

    @model_validator(mode="after")
    def validate_weights_and_digest(self) -> "BeliefDistributionPayload":
        if not self.weights:
            raise ValueError("belief distribution weights cannot be empty")
        if any(not math.isfinite(value) or value < 0.0 for value in self.weights):
            raise ValueError(
                "belief distribution weights must be finite and non-negative"
            )
        if sum(self.weights) <= 0.0:
            raise ValueError("belief distribution weights have zero mass")
        expected = belief_distribution_sha256(
            self.space_identity,
            self.belief_model_id,
            self.weights,
            self.provenance_identity,
        )
        if self.distribution_sha256 != expected:
            raise ValueError("belief distribution SHA-256 mismatch")
        return self


def belief_distribution_sha256(
    space_identity: str,
    belief_model_id: str,
    weights: Sequence[float],
    provenance_identity: str,
) -> str:
    return canonical_sha256(
        {
            "space_identity": space_identity,
            "belief_model_id": belief_model_id,
            "weights": [float(value) for value in weights],
            "provenance_identity": provenance_identity,
        }
    )


class AdvisorUnavailable(RuntimeError):
    """Typed fail-closed provider state with no authority-private detail."""

    def __init__(self, reason: AdviceUnavailableReason) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ResolvedAdvisorDecision:
    """Injected Game capability consumed by the Intelligence provider."""

    address: ReplayDecisionAddress
    frame: ExperienceFrame
    semantic_frame: DecisionFrame
    root: Any
    world_space: PossibleWorldSpace
    source_replay_sha256: str
    content_sha256: str


class AdvisorDecisionResolver(Protocol):
    def resolve(
        self,
        address: ReplayDecisionAddress,
        viewer: ViewerIdentity,
    ) -> ResolvedAdvisorDecision: ...


class BeliefDistributionResolver(Protocol):
    def resolve(
        self,
        scenario: BeliefScenario,
        decision: ResolvedAdvisorDecision,
        viewer: ViewerIdentity,
    ) -> BeliefDistributionPayload: ...


class LiveAdvisorRootProvider(Protocol):
    def resolve_live_advisor_decision(
        self,
        raw_address: str,
        authorized_viewer: int,
    ) -> Any: ...


def _resolved_advisor_decision(
    resolved: Any,
    address: ReplayDecisionAddress,
    viewer: ViewerIdentity,
    source_replay_sha256: str,
) -> ResolvedAdvisorDecision:
    try:
        semantic = DecisionFrame.from_json(resolved.root.semantic_decision_frame_json())
        world_space = PossibleWorldSpace.from_engine(resolved.root, viewer.rules_viewer)
    except (PossibleWorldError, ValueError) as error:
        raise AdvisorUnavailable("decision_root_unavailable") from error
    restored = resolved.restored
    if (
        restored.address != address.serialize()
        or restored.viewer != viewer.rules_viewer
        or restored.revision != address.revision
        or restored.frame.revision != address.revision
        or semantic.actor != viewer.rules_viewer
    ):
        raise AdvisorUnavailable("decision_identity_mismatch")
    replay_offer_ids = [int(offer.id) for offer in restored.frame.offers]
    semantic_offer_ids = [int(offer["id"]) for offer in semantic.offers]
    if replay_offer_ids != semantic_offer_ids:
        raise AdvisorUnavailable("action_identity_mismatch")
    try:
        content_manifest = resolved.root.content_pack_manifest()
    except Exception:
        content_manifest = {}
    return ResolvedAdvisorDecision(
        address=address,
        frame=restored.frame.model_copy(deep=True),
        semantic_frame=semantic,
        root=resolved.root,
        world_space=world_space,
        source_replay_sha256=source_replay_sha256,
        content_sha256=canonical_sha256(content_manifest),
    )


@dataclass(frozen=True)
class LiveAdvisorDecisionResolver:
    """Adapter for a committed decision retained by a live Game session."""

    provider: LiveAdvisorRootProvider
    source_replay_sha256: str

    def resolve(
        self,
        address: ReplayDecisionAddress,
        viewer: ViewerIdentity,
    ) -> ResolvedAdvisorDecision:
        try:
            resolved = self.provider.resolve_live_advisor_decision(
                address.serialize(), viewer.rules_viewer
            )
        except DecisionNotFoundError as error:
            raise AdvisorUnavailable("decision_not_found") from error
        except StudyBranchUnavailableError as error:
            raise AdvisorUnavailable("decision_root_unavailable") from error
        return _resolved_advisor_decision(
            resolved, address, viewer, self.source_replay_sha256
        )


@dataclass(frozen=True)
class StudyAdvisorDecisionResolver:
    """Narrow adapter over Game's existing replay/fork authority."""

    provider: StudyForkProvider
    source_replay_sha256: str

    def resolve(
        self,
        address: ReplayDecisionAddress,
        viewer: ViewerIdentity,
    ) -> ResolvedAdvisorDecision:
        try:
            resolved = self.provider.resolve_advisor_decision(
                address.serialize(), viewer.rules_viewer
            )
        except DecisionNotFoundError as error:
            raise AdvisorUnavailable("decision_not_found") from error
        except StudyBranchUnavailableError as error:
            raise AdvisorUnavailable("decision_root_unavailable") from error
        return _resolved_advisor_decision(
            resolved, address, viewer, self.source_replay_sha256
        )


@dataclass(frozen=True)
class StaticBeliefDistributionResolver:
    """Checked fixture resolver for authored or retained model payloads."""

    payloads: Mapping[str, BeliefDistributionPayload]

    def resolve(
        self,
        scenario: BeliefScenario,
        decision: ResolvedAdvisorDecision,
        viewer: ViewerIdentity,
    ) -> BeliefDistributionPayload:
        del decision, viewer
        payload = self.payloads.get(scenario.id)
        if payload is None:
            reason = (
                "belief_artifact_unavailable"
                if scenario.provenance.kind == "model_inferred"
                else "belief_distribution_unavailable"
            )
            raise AdvisorUnavailable(reason)
        return payload


@dataclass(frozen=True)
class RegisteredAdvisor:
    identity: AdvisorIdentity
    source_paths: tuple[Path, ...]

    def verify_artifact(self) -> None:
        artifact = self.identity.artifact
        if isinstance(artifact, CodeSourceArtifact):
            if source_bundle_sha256(self.source_paths) != artifact.source_bundle_sha256:
                raise AdvisorUnavailable("advisor_artifact_mismatch")
            return
        self._checkpoint_registration(artifact)

    def checkpoint_evaluator(self) -> Any | None:
        """Build the sole registered checkpoint evaluator, if selected."""

        artifact = self.identity.artifact
        if isinstance(artifact, CodeSourceArtifact):
            return None
        registration = self._checkpoint_registration(artifact)
        compute = self.identity.compute
        if (
            compute.simulations_per_scenario != registration.simulations
            or compute.sampled_worlds != registration.sampled_worlds
            or compute.c_puct != registration.c_puct
            or compute.max_steps != registration.max_steps
            or compute.branch_driver_id != registration.branch_driver_id
        ):
            raise AdvisorUnavailable("compute_mismatch")
        if self.identity.evaluator_id != "checkpoint_policy_neutral_value":
            raise AdvisorUnavailable("evaluator_mismatch")

        from manabot.sim.flat_mc import load_checkpoint_agent
        from manabot.sim.mcts import AgentLeafEvaluator
        from manabot.sim.search_runtime import (
            model_action_abi_sha256,
            model_observation_abi_sha256,
        )

        agent, observation_space = load_checkpoint_agent(
            str(registration.checkpoint_path)
        )
        parameter_count = sum(parameter.numel() for parameter in agent.parameters())
        if parameter_count != registration.parameter_count:
            raise AdvisorUnavailable("advisor_artifact_mismatch")
        if (
            model_observation_abi_sha256(observation_space)
            != artifact.observation_abi.sha256
        ):
            raise AdvisorUnavailable("observation_abi_mismatch")
        if model_action_abi_sha256(observation_space) != artifact.action_abi.sha256:
            raise AdvisorUnavailable("action_abi_mismatch")
        return AgentLeafEvaluator(
            agent,
            observation_space,
            value_mode=artifact.value_mode,
        )

    def _checkpoint_registration(self, artifact: CheckpointArtifact) -> Any:
        from manabot.sim.search_runtime import (
            RetainedCheckpointMismatchError,
            RetainedCheckpointUnavailableError,
            retained_int7_policy_only_checkpoint,
        )

        try:
            registration = retained_int7_policy_only_checkpoint(
                int(artifact.training_seed)
            )
        except RetainedCheckpointUnavailableError as exc:
            raise AdvisorUnavailable("advisor_artifact_unavailable") from exc
        except RetainedCheckpointMismatchError as exc:
            raise AdvisorUnavailable("advisor_artifact_mismatch") from exc
        expected_observation_abi = AbiIdentity(
            name="manabot_observation",
            version=registration.world_id,
            sha256=registration.observation_abi_sha256,
        )
        expected_action_abi = AbiIdentity(
            name="manabot_action",
            version=registration.world_id,
            sha256=registration.action_abi_sha256,
        )
        if (
            artifact.checkpoint_id != registration.checkpoint_id
            or artifact.checkpoint_sha256 != registration.checkpoint_sha256
            or artifact.checkpoint_bytes != registration.checkpoint_bytes
            or artifact.manifest_sha256 != registration.manifest_sha256
            or artifact.observation_abi != expected_observation_abi
            or artifact.action_abi != expected_action_abi
            or artifact.value_mode != registration.value_mode
            or self.identity.world_id != registration.world_id
        ):
            raise AdvisorUnavailable("advisor_artifact_mismatch")
        return registration


ADVISOR_SOURCE_PATHS = (
    REPO_ROOT / "etude" / "advice.py",
    REPO_ROOT / "etude" / "advice_identity.py",
    REPO_ROOT / "etude" / "study_branch.py",
    REPO_ROOT / "manabot" / "belief" / "range.py",
    REPO_ROOT / "manabot" / "sim" / "conditional_search.py",
    REPO_ROOT / "manabot" / "sim" / "mcts.py",
    REPO_ROOT / "manabot" / "sim" / "search_branch.py",
    REPO_ROOT / "manabot" / "sim" / "search_runtime.py",
    REPO_ROOT / "managym" / "decision.py",
    REPO_ROOT / "managym" / "possible_worlds.py",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_advisor_abis() -> tuple[AbiIdentity, AbiIdentity, AbiIdentity]:
    """Return current checked Observation/action/world ABI identities."""

    return (
        AbiIdentity(
            name="experience_frame",
            version="experience-v1",
            sha256=_file_sha256(REPO_ROOT / "protocol" / "experience-v1.schema.json"),
        ),
        AbiIdentity(
            name="semantic_decision_frame",
            version=str(SEMANTIC_DECISION_VERSION),
            sha256=_file_sha256(REPO_ROOT / "managym" / "decision.py"),
        ),
        AbiIdentity(
            name="possible_world_space",
            version=str(POSSIBLE_WORLD_SPACE_VERSION),
            sha256=_file_sha256(REPO_ROOT / "managym" / "possible_worlds.py"),
        ),
    )


def player_authored_provenance_identity(scenario: BeliefScenario) -> str:
    provenance = scenario.provenance
    if provenance.kind != "player_authored":
        raise AdvisorUnavailable("belief_provenance_mismatch")
    return (
        f"player_authored:{scenario.author_viewer_id}:"
        f"{provenance.created_at_table_revision}"
    )


def _scenario_provenance_identity(scenario: BeliefScenario) -> str:
    provenance = scenario.provenance
    if provenance.kind == "player_authored":
        return player_authored_provenance_identity(scenario)
    return (
        f"model_inferred:{provenance.belief_model_id}:"
        f"{provenance.checkpoint_sha256}:{provenance.artifact_manifest_sha256}:"
        f"{provenance.viewer_history_sha256}"
    )


def _canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def advice_request_sha256(request: AdviceRequest) -> str:
    return hashlib.sha256(_canonical_bytes(request)).hexdigest()


def serialize_advice_response(response: AdviceResponse) -> bytes:
    """Canonical public serializer shared verbatim by live and Study paths."""

    core = response.model_dump(mode="json", exclude={"response_sha256"})
    digest = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    if response.response_sha256 is not None and response.response_sha256 != digest:
        raise ValueError("advice response SHA-256 mismatch")
    complete = response.model_copy(update={"response_sha256": digest})
    return _canonical_bytes(complete)


def parse_advice_response_bytes(payload: bytes) -> AdviceResponse:
    response = AdviceResponse.model_validate_json(payload)
    serialize_advice_response(response)
    return response


def _identity_mismatch_reason(
    actual: AdvisorIdentity,
    expected: AdvisorIdentity,
) -> AdviceUnavailableReason | None:
    checks = (
        (actual.source_replay_id == expected.source_replay_id, "replay_mismatch"),
        (
            actual.source_replay_sha256 == expected.source_replay_sha256,
            "replay_mismatch",
        ),
        (actual.match_id == expected.match_id, "match_mismatch"),
        (actual.world_id == expected.world_id, "world_mismatch"),
        (actual.content_sha256 == expected.content_sha256, "content_mismatch"),
        (
            actual.observation_abi == expected.observation_abi,
            "observation_abi_mismatch",
        ),
        (actual.action_abi == expected.action_abi, "action_abi_mismatch"),
        (
            actual.possible_world_abi == expected.possible_world_abi,
            "possible_world_abi_mismatch",
        ),
        (
            actual.information_boundary == expected.information_boundary,
            "information_boundary_mismatch",
        ),
        (actual.planner_id == expected.planner_id, "planner_mismatch"),
        (actual.evaluator_id == expected.evaluator_id, "evaluator_mismatch"),
        (actual.artifact == expected.artifact, "advisor_artifact_mismatch"),
        (actual.compute == expected.compute, "compute_mismatch"),
        (actual.seed == expected.seed, "seed_plan_mismatch"),
    )
    return next((reason for matches, reason in checks if not matches), None)


def _quantity_delta(
    left: AdviceQuantity,
    right: AdviceQuantity,
) -> AdvisorDeltaQuantity:
    if isinstance(left, AvailableQuantity) and isinstance(right, AvailableQuantity):
        return AdvisorDeltaQuantity(
            status="available", value=float(left.value - right.value)
        )
    return AdvisorDeltaQuantity(
        status="unavailable", reason="comparison_quantity_unavailable"
    )


def _scenario_audience_is_visible(
    scenario: BeliefScenario,
    viewer: ViewerIdentity,
) -> bool:
    if scenario.audience.kind == "personal":
        return scenario.author_viewer_id == viewer.viewer_id
    return scenario.audience.table_id == viewer.table_id


class AdviceProvider:
    """One identity-pinned provider used by both live and Study adapters."""

    def __init__(
        self,
        *,
        registered: RegisteredAdvisor,
        decision_resolver: AdvisorDecisionResolver,
        belief_resolver: BeliefDistributionResolver,
    ) -> None:
        self.registered = registered
        self.decision_resolver = decision_resolver
        self.belief_resolver = belief_resolver
        self._cache: dict[str, bytes] = {}
        self.last_timings_ms: dict[str, float] = {}

    def advise(self, request: AdviceRequest, *, recompute: bool = False) -> bytes:
        if request.is_legacy:
            raise ValueError("AdviceProvider accepts only advice-v1 requests")
        started = time.perf_counter_ns()
        request_sha = advice_request_sha256(request)
        if not recompute and request_sha in self._cache:
            artifact_started = time.perf_counter_ns()
            assert request.advisor_identity is not None
            mismatch = _identity_mismatch_reason(
                request.advisor_identity, self.registered.identity
            )
            if mismatch is not None:
                raise ValueError("cached advice identity changed")
            self.registered.verify_artifact()
            finished = time.perf_counter_ns()
            self.last_timings_ms = {
                "request_validation": (artifact_started - started) / 1_000_000,
                "artifact_verification": (finished - artifact_started) / 1_000_000,
                "total": (finished - started) / 1_000_000,
            }
            return self._cache[request_sha]
        assert request.advisor_identity is not None
        try:
            response = self._compute(request, request_sha)
        except AdvisorUnavailable as unavailable:
            response = AdviceResponse(
                status="unavailable",
                reason=unavailable.reason,
                address=request.address,
                identity=request.advisor_identity,
                request_sha256=request_sha,
            )
        serialization_started = time.perf_counter_ns()
        payload = serialize_advice_response(response)
        finished = time.perf_counter_ns()
        self.last_timings_ms["serialization"] = (
            finished - serialization_started
        ) / 1_000_000
        self.last_timings_ms["total"] = (finished - started) / 1_000_000
        if not recompute:
            self._cache[request_sha] = payload
        return payload

    def _compute(self, request: AdviceRequest, request_sha: str) -> AdviceResponse:
        assert request.viewer is not None
        assert request.scenario is not None
        assert request.advisor_identity is not None
        identity = request.advisor_identity
        mismatch = _identity_mismatch_reason(identity, self.registered.identity)
        if mismatch is not None:
            raise AdvisorUnavailable(mismatch)
        if (
            request.parsed_address.replay_id != identity.source_replay_id
            or request.parsed_address.match_id != identity.match_id
        ):
            raise AdvisorUnavailable("decision_identity_mismatch")
        artifact_started = time.perf_counter_ns()
        self.registered.verify_artifact()
        artifact_finished = time.perf_counter_ns()
        decision = self.decision_resolver.resolve(
            request.parsed_address, request.viewer
        )
        decision_finished = time.perf_counter_ns()
        if decision.source_replay_sha256 != identity.source_replay_sha256:
            raise AdvisorUnavailable("replay_mismatch")
        if decision.content_sha256 != identity.content_sha256:
            raise AdvisorUnavailable("content_mismatch")
        if decision.world_space.identity == "" or identity.world_id != "w2":
            raise AdvisorUnavailable("world_mismatch")
        runtime_abis = runtime_advisor_abis()
        if identity.observation_abi != runtime_abis[0]:
            raise AdvisorUnavailable("observation_abi_mismatch")
        if identity.action_abi != runtime_abis[1]:
            raise AdvisorUnavailable("action_abi_mismatch")
        if identity.possible_world_abi != runtime_abis[2]:
            raise AdvisorUnavailable("possible_world_abi_mismatch")

        scenarios = [request.scenario]
        if request.comparison_scenario is not None:
            scenarios.append(request.comparison_scenario)
        beliefs: dict[str, BeliefState] = {}
        receipts: dict[str, BeliefNormalizationReceipt] = {}
        for scenario in scenarios:
            belief, receipt = self._resolve_belief(
                scenario, request, decision, request.viewer
            )
            beliefs[scenario.id] = belief
            receipts[scenario.id] = receipt
        normalization_finished = time.perf_counter_ns()

        from manabot.sim.conditional_search import (
            conditional_determinized_puct_beliefs,
            project_viewer_safe_result,
            validate_result,
        )

        compute = identity.compute
        evaluator = self.registered.checkpoint_evaluator()
        root_observation = (
            decision.root.observation_for_player(request.viewer.rules_viewer)
            if evaluator is not None
            else None
        )
        result = conditional_determinized_puct_beliefs(
            decision.root,
            beliefs=beliefs,
            simulations=compute.simulations_per_scenario,
            worlds=compute.sampled_worlds,
            seed=identity.seed.root_seed,
            c_puct=compute.c_puct,
            max_steps=compute.max_steps,
            evaluator=evaluator,
            root_observation=root_observation,
            branch_driver_id=compute.branch_driver_id,
            branch_audit=True,
            branch_match_id=identity.match_id,
        )
        validate_result(result, expected_condition_ids=tuple(beliefs))
        search_finished = time.perf_counter_ns()
        offer_ids = [int(offer["id"]) for offer in decision.semantic_frame.offers]
        projected = project_viewer_safe_result(result, offer_ids=offer_ids)
        evidence = self._strategy_evidence(decision, identity, projected, receipts)
        projection_finished = time.perf_counter_ns()
        self.last_timings_ms = {
            "artifact_verification": (artifact_finished - artifact_started) / 1_000_000,
            "decision_resolution": (decision_finished - artifact_finished) / 1_000_000,
            "normalization": (normalization_finished - decision_finished) / 1_000_000,
            "materialization_search": (search_finished - normalization_finished)
            / 1_000_000,
            "projection": (projection_finished - search_finished) / 1_000_000,
        }
        public_payload = evidence.model_dump(mode="json")
        forbidden = {
            "authority_private",
            "world_q_values",
            "world_root_values",
            "world_visit_counts",
            "branch_receipt",
            "sampled_indexes",
            "opponent_hand",
            "actual_query_truth",
            "rng_tapes",
        }
        if _contains_forbidden_key(public_payload, forbidden):
            raise AdvisorUnavailable("private_projection_failure")
        return AdviceResponse(
            status="ok",
            address=request.address,
            frame=decision.frame,
            offers=list(decision.frame.offers),
            identity=identity,
            request_sha256=request_sha,
            strategy=evidence,
        )

    def _resolve_belief(
        self,
        scenario: BeliefScenario,
        request: AdviceRequest,
        decision: ResolvedAdvisorDecision,
        viewer: ViewerIdentity,
    ) -> tuple[BeliefState, BeliefNormalizationReceipt]:
        assert request.advisor_identity is not None
        if not _scenario_audience_is_visible(scenario, viewer):
            raise AdvisorUnavailable("belief_viewer_mismatch")
        if scenario.source.decision_address != request.address:
            raise AdvisorUnavailable("belief_decision_mismatch")
        if isinstance(scenario.source.advice_identity, AdviceRequestIdentity):
            raise AdvisorUnavailable("legacy_identity_incomplete")
        if scenario.source.advice_identity != request.advisor_identity:
            raise AdvisorUnavailable("belief_provenance_mismatch")
        payload = self.belief_resolver.resolve(scenario, decision, viewer)
        if payload.space_identity != decision.world_space.identity:
            raise AdvisorUnavailable("belief_space_mismatch")
        expected_provenance = _scenario_provenance_identity(scenario)
        if (
            payload.provenance_kind != scenario.provenance.kind
            or payload.provenance_identity != expected_provenance
        ):
            raise AdvisorUnavailable("belief_provenance_mismatch")
        expected_distribution_sha256 = belief_distribution_sha256(
            payload.space_identity,
            payload.belief_model_id,
            payload.weights,
            payload.provenance_identity,
        )
        if payload.distribution_sha256 != expected_distribution_sha256:
            raise AdvisorUnavailable("belief_distribution_mismatch")
        from manabot.belief.range import BeliefError, BeliefState

        try:
            belief = BeliefState.from_probabilities(
                decision.world_space,
                payload.belief_model_id,
                payload.weights,
            )
        except BeliefError as error:
            raise AdvisorUnavailable("belief_invalid") from error
        return belief, BeliefNormalizationReceipt(
            scenario_id=scenario.id,
            space_identity=decision.world_space.identity,
            belief_model_id=payload.belief_model_id,
            distribution_sha256=payload.distribution_sha256,
            normalized_belief_sha256=belief.digest,
            positive_support=belief.positive_support_size,
            normalization_error=belief.normalization_error,
            provenance_kind=payload.provenance_kind,
            provenance_identity=payload.provenance_identity,
        )

    @staticmethod
    def _strategy_evidence(
        decision: ResolvedAdvisorDecision,
        identity: AdvisorIdentity,
        projected: Mapping[str, Any],
        receipts: Mapping[str, BeliefNormalizationReceipt],
    ) -> AdvisorStrategyEvidence:
        semantic_offers = [
            AdvisorSemanticOffer(
                offer_id=int(offer["id"]),
                actor=int(offer["actor"]),
                verb=str(offer["verb"]),
                label=str(offer.get("label") or offer.get("description") or ""),
            )
            for offer in decision.semantic_frame.offers
        ]
        scenario_rows: list[AdvisorScenarioEvidence] = []
        for raw_condition in projected["conditions"]:
            scenario_id = str(raw_condition["condition_id"])
            actions = [
                AdvisorOfferEvidence(
                    offer_id=int(action["offer_id"]),
                    label=str(action["label"]),
                    probability=float(action["probability"]),
                    visits=int(action["visits"]),
                    q=action["q"],
                    robustness=action["robustness"],
                    uncertainty=action["uncertainty"],
                )
                for action in raw_condition["actions"]
            ]
            if [action.offer_id for action in actions] != [
                offer.offer_id for offer in semantic_offers
            ]:
                raise AdvisorUnavailable("action_identity_mismatch")
            if not math.isclose(
                sum(action.probability for action in actions), 1.0, abs_tol=1e-9
            ):
                raise AdvisorUnavailable("policy_mass_invalid")
            scenario_rows.append(
                AdvisorScenarioEvidence(
                    scenario_id=scenario_id,
                    belief=receipts[scenario_id],
                    condition_mass=float(raw_condition["condition_mass"]),
                    support=int(raw_condition["support"]),
                    sampled_worlds=int(raw_condition["sampled_worlds"]),
                    actions=actions,
                    root_value=raw_condition["root_value"],
                    root_uncertainty=raw_condition["root_uncertainty"],
                    simulations=int(raw_condition["simulations"]),
                    cap_hits=int(raw_condition["cap_hits"]),
                    tree_nodes=int(raw_condition["tree_nodes"]),
                )
            )
        comparison = (
            _build_comparison(scenario_rows[0], scenario_rows[1])
            if len(scenario_rows) == 2
            else None
        )
        return AdvisorStrategyEvidence(
            policy_semantic=projected["policy_semantic"],
            decision_sha256=decision.address.decision_sha256,
            viewer_frame_sha256=decision.frame.frame_hash,
            advisor_identity=identity,
            offers=semantic_offers,
            scenarios=scenario_rows,
            comparison=comparison,
            realized_compute=dict(projected["realized_compute"]),
            prior_sha256=projected["prior_sha256"],
            plan_sha256=projected["plan_sha256"],
        )


def _build_comparison(
    left: AdvisorScenarioEvidence,
    right: AdvisorScenarioEvidence,
) -> AdvisorComparison:
    if [action.offer_id for action in left.actions] != [
        action.offer_id for action in right.actions
    ]:
        raise AdvisorUnavailable("action_identity_mismatch")
    actions = [
        AdvisorOfferDelta(
            offer_id=left_action.offer_id,
            policy_probability=(left_action.probability - right_action.probability),
            q=_quantity_delta(left_action.q, right_action.q),
            robustness=_quantity_delta(left_action.robustness, right_action.robustness),
            uncertainty=_quantity_delta(
                left_action.uncertainty, right_action.uncertainty
            ),
        )
        for left_action, right_action in zip(left.actions, right.actions, strict=True)
    ]
    return AdvisorComparison(
        left_scenario_id=left.scenario_id,
        right_scenario_id=right.scenario_id,
        left_belief_sha256=left.belief.normalized_belief_sha256,
        right_belief_sha256=right.belief.normalized_belief_sha256,
        actions=actions,
        root_value=_quantity_delta(left.root_value, right.root_value),
        root_uncertainty=_quantity_delta(left.root_uncertainty, right.root_uncertainty),
    )


def _contains_forbidden_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in forbidden or _contains_forbidden_key(child, forbidden)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(child, forbidden) for child in value)
    return False


class VersionedAdviceFixture(ProtocolModel):
    """Checked public request/response pair consumed by Game's prototype."""

    contract: Literal["advice-fixture-v1"] = "advice-fixture-v1"
    request: AdviceRequest
    response: AdviceResponse

    @model_validator(mode="after")
    def validate_identity(self) -> "VersionedAdviceFixture":
        if self.request.is_legacy:
            raise ValueError("versioned fixture cannot contain a legacy request")
        if self.response.status != "ok" or self.response.strategy is None:
            raise ValueError(
                "versioned fixture must contain complete strategy evidence"
            )
        if self.request.advisor_identity != self.response.identity:
            raise ValueError("fixture request and response identities differ")
        if self.response.request_sha256 != advice_request_sha256(self.request):
            raise ValueError("fixture response does not bind its exact request")
        serialize_advice_response(self.response)
        return self


@functools.cache
def load_versioned_advice_fixture() -> VersionedAdviceFixture:
    payload = json.loads(VERSIONED_FIXTURE_PATH.read_text(encoding="utf-8"))
    return VersionedAdviceFixture.model_validate(payload)


@functools.cache
def load_checkpoint_versioned_advice_fixture() -> VersionedAdviceFixture:
    payload = json.loads(CHECKPOINT_VERSIONED_FIXTURE_PATH.read_text(encoding="utf-8"))
    return VersionedAdviceFixture.model_validate(payload)


@functools.cache
def load_flip_versioned_advice_fixture() -> VersionedAdviceFixture:
    payload = json.loads(FLIP_VERSIONED_FIXTURE_PATH.read_text(encoding="utf-8"))
    return VersionedAdviceFixture.model_validate(payload)


@dataclass(frozen=True)
class VersionedAdviceFixtureRegistration:
    """One checked response paired with its exact artifact verification seam."""

    fixture_id: str
    path: Path
    loader: Callable[[], VersionedAdviceFixture]
    source_paths: tuple[Path, ...]

    def load(self) -> VersionedAdviceFixture:
        return self.loader()


VERSIONED_ADVICE_FIXTURE_REGISTRY = (
    VersionedAdviceFixtureRegistration(
        fixture_id="int-12-belief-conditioned-v1",
        path=VERSIONED_FIXTURE_PATH,
        loader=load_versioned_advice_fixture,
        source_paths=ADVISOR_SOURCE_PATHS,
    ),
    VersionedAdviceFixtureRegistration(
        fixture_id="int-5-checkpoint-policy-v1",
        path=CHECKPOINT_VERSIONED_FIXTURE_PATH,
        loader=load_checkpoint_versioned_advice_fixture,
        source_paths=ADVISOR_SOURCE_PATHS,
    ),
    VersionedAdviceFixtureRegistration(
        fixture_id="int-15-belief-conditioned-flip-v1",
        path=FLIP_VERSIONED_FIXTURE_PATH,
        loader=load_flip_versioned_advice_fixture,
        source_paths=ADVISOR_SOURCE_PATHS,
    ),
)


def request_versioned_fixture_advice(request: AdviceRequest) -> bytes:
    """Serve the checked v1 slice or fail closed without adapting identity."""

    if request.is_legacy:
        raise ValueError("versioned fixture adapter requires advice-v1")
    registered_fixtures = tuple(
        (registration, registration.load())
        for registration in VERSIONED_ADVICE_FIXTURE_REGISTRY
        if registration.path.is_file()
    )
    if not registered_fixtures:
        raise RuntimeError("no versioned advice fixtures are registered")
    matched_registration = next(
        (
            (registration, fixture)
            for registration, fixture in registered_fixtures
            if request == fixture.request
        ),
        None,
    )
    if matched_registration is not None:
        registration, matched_fixture = matched_registration
        assert matched_fixture.request.advisor_identity is not None
        try:
            RegisteredAdvisor(
                matched_fixture.request.advisor_identity,
                registration.source_paths,
            ).verify_artifact()
        except AdvisorUnavailable as unavailable:
            return serialize_advice_response(
                AdviceResponse(
                    status="unavailable",
                    reason=unavailable.reason,
                    address=request.address,
                    identity=request.advisor_identity,
                    request_sha256=advice_request_sha256(request),
                )
            )
        return serialize_advice_response(matched_fixture.response)
    fixture = next(
        (
            candidate
            for _, candidate in registered_fixtures
            if candidate.request.advisor_identity == request.advisor_identity
        ),
        registered_fixtures[0][1],
    )
    assert request.advisor_identity is not None
    assert fixture.request.advisor_identity is not None
    mismatch = _identity_mismatch_reason(
        request.advisor_identity, fixture.request.advisor_identity
    )
    if mismatch is not None:
        reason = mismatch
    elif request.address != fixture.request.address:
        reason = "decision_identity_mismatch"
    elif request.viewer != fixture.request.viewer:
        reason = "belief_viewer_mismatch"
    else:
        reason = "decision_root_unavailable"
    return serialize_advice_response(
        AdviceResponse(
            status="unavailable",
            reason=reason,
            address=request.address,
            identity=request.advisor_identity,
            request_sha256=advice_request_sha256(request),
        )
    )


ADVICE_RESPONSE_ADAPTER = TypeAdapter(AdviceResponse)
ADVICE_FIXTURE_ADAPTER = TypeAdapter(VersionedAdviceFixture)


def advice_schema() -> dict[str, Any]:
    """Draft 2020-12 schema for the sole versioned request/response fixture."""

    return ADVICE_FIXTURE_ADAPTER.json_schema()
