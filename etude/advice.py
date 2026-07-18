"""Belief-to-strategy comparison adapter for the shared decision surface.

This is the first AI-assisted play slice: a narrow, pure-Python consumer of a
checked, identity-pinned advice fixture. It does NOT invoke search at runtime
-- the fixture is the evidence. It validates the request identity against the
fixture's pinned identity, exposes the two belief scenarios and their real
flat-MC evidence at one ``erd1`` decision, and computes the explicit per-action
deltas between them. Identity mismatch, unknown scenario, or wrong address fail
closed to a typed ``unavailable`` state with no evidence.

The adapter carries two explicit seams, each a named, typed binding point so
the upstream waves plug in without duplicating the StudyArtifact substrate:

- **GAM-4 seam** (live decision address + retry/compare): ``AdviceRequestIdentity``
  and ``request_advice`` are the request contract. GAM-4 publishes the live
  ``erd1`` decision address and the fork/Retry/return controls; both the live
  play page and the Study page call the same ``POST /api/advice`` with this
  identity, so GAM-4's live-address and retry/compare contracts bind here.
- **INT-13 seam** (search evidence): ``int13_condition_to_decision_evidence``
  maps one INT-13 ``ConditionResult`` (determinized-PUCT search output from
  ``manabot.sim.conditional_search``) into the per-action ``DecisionEvidence``
  this surface renders, and ``int13_result_to_surface_deltas`` reconciles two
  INT-13 conditions into the surface's per-action delta dict. The prototype
  serves fixture-backed flat-MC evidence at runtime; these mappings are the
  typed binding point INT-13's live search fills when it publishes.

This is a prototype advisor (``flat-mc-search-v1``), not a production advisor:
the footer states "advisory only", and no strength or latency claim is made.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from etude.experience_protocol import ExperienceFrame, InteractionOffer, ProtocolModel
from etude.study_protocol import DecisionEvidence, StudyArtifact, StudyLandmark

if TYPE_CHECKING:
    # INT-13 search-evidence contracts (manabot.sim.conditional_search). The
    # seam is typed against these so the binding point is explicit and
    # type-checked, but etude never imports manabot at runtime: the prototype
    # serves fixture-backed evidence, and the live/Study surface never crosses
    # the player/training boundary at request time.
    from manabot.sim.conditional_search import (
        ConditionalStrategyResult,
        ConditionResult,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "protocol" / "fixtures" / "advice-curated-decision.json"


class AdviceRequestIdentity(ProtocolModel):
    """The advisor + compute identity a request claims to incorporate.

    The GAM-4 seam: this is the request contract GAM-4's live decision-address
    and retry/compare controls bind to. Mirrors the fixture's pinned
    ``StudyIdentity`` -- the source replay and match must match, the advisor id
    must match the pinned ``model.id``, and the compute id must match the pinned
    ``analysis_budget.id``. When GAM-4 publishes the live ``erd1`` address, the
    same identity flows through ``POST /api/advice`` unchanged.
    """

    source_replay_id: str
    match_id: str
    advisor_id: str
    compute_id: str


class AdviceScenarioSummary(ProtocolModel):
    """Viewer-safe presentation metadata for one belief scenario."""

    landmark_id: str
    label: str
    description: str
    inferred_range: str
    belief_kind: str


class AdviceScenarioRecord(AdviceScenarioSummary):
    """The fixture's scenario metadata, including the pinned seed family.

    ``seed_family`` is generator provenance, not a request field; the adapter
    cross-references each record to a landmark by ``landmark_id``.
    """

    seed_family: str


class AdviceArtifact(ProtocolModel):
    """Wrapper around a ``StudyArtifact`` plus a prototype presentation layer.

    The ``artifact`` is a standard ``StudyArtifact`` (two landmarks at the same
    ``erd1`` decision, each with real flat-MC evidence) and validates through
    both the Pydantic model and the Rust-owned ``study-v1`` schema. The
    ``scenarios`` array is a prototype presentation layer validated here, not by
    the Rust schema.
    """

    artifact: StudyArtifact
    scenarios: list[AdviceScenarioRecord]


class AdviceResponse(ProtocolModel):
    """One advice response: real evidence on success, typed closed on failure."""

    status: Literal["ok", "unavailable"]
    reason: str | None = None
    address: str | None = None
    frame: ExperienceFrame | None = None
    offers: list[InteractionOffer] = Field(default_factory=list)
    scenario: AdviceScenarioSummary | None = None
    evidence: DecisionEvidence | None = None
    deltas: dict[str, dict[str, float]] | None = None
    identity: AdviceRequestIdentity | None = None


class AdviceMeta(ProtocolModel):
    """The pinned decision's bootstrap: address, scenarios, expected identity."""

    address: str
    scenarios: list[AdviceScenarioSummary]
    identity: AdviceRequestIdentity


class AdviceRequest(ProtocolModel):
    """The POST /api/advice request body: one decision, one scenario, one identity."""

    address: str
    scenario_id: str
    identity: AdviceRequestIdentity


@functools.cache
def load_advice_fixture() -> AdviceArtifact:
    """Load and validate the advice fixture once.

    The ``artifact`` validates as a ``StudyArtifact`` (which enforces
    viewer-safety, bindings, and opponent-hand privacy) and the scenarios
    cross-reference landmarks by id.
    """
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    artifact = AdviceArtifact.model_validate(payload)
    landmark_ids = {landmark.id for landmark in artifact.artifact.landmarks}
    scenario_ids = {scenario.landmark_id for scenario in artifact.scenarios}
    if scenario_ids != landmark_ids:
        raise ValueError("advice scenarios do not cover the artifact landmarks")
    if len({landmark.decision_id for landmark in artifact.artifact.landmarks}) != 1:
        raise ValueError("advice landmarks must share one decision address")
    return artifact


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
    prototype serves fixture-backed flat-MC evidence at runtime; this mapping is
    not on the fixture request path. When INT-13's live search publishes, the
    ``POST /api/advice`` server can route a live-address request to this mapping
    instead of the static fixture, without changing the surface or the
    ``DecisionEvidence`` contract.

    Field mapping (``ConditionResult`` -> ``DecisionEvidence``):

    - ``visit_counts`` -> ``policy_mass`` (normalized to a per-action
      distribution) and ``visits`` (raw per-action counts).
    - ``q_values`` -> ``search_value.expected_match_points`` (INT-13 q-values
      are mean per-action win-probability estimates).
    - ``uncertainty`` (one scalar) -> ``uncertainty.standard_error`` per action,
      broadcast with ``method = "int13-scalar-broadcast"``. INT-13 does not yet
      expose per-action standard error; the broadcast is a documented gap, not a
      per-action precision claim.
    - ``condition_mass`` / ``sampled_worlds`` -> ``sampled_world_robustness``
      with ``favorable_worlds`` per action set to ``sampled_worlds`` when that
      action's mean q exceeds 0.5 and 0 otherwise (a viewer-safe per-action
      favorability proxy; the exact per-action favorable-world count is an
      INT-13 future field).

    ``producer`` is the INT-13 provenance string (e.g.
    ``"int-13-determinized-puct:v1"``). The returned dict validates as a
    ``DecisionEvidence`` and carries a content-addressed ``evidence_sha256``.
    """
    visit_counts = list(condition.visit_counts)
    q_values = list(condition.q_values)
    num_actions = len(alternative_ids)
    if len(visit_counts) != num_actions or len(q_values) != num_actions:
        raise ValueError("int13 condition action count does not match alternative_ids")
    total_visits = sum(visit_counts)
    denom = total_visits if total_visits > 0 else 1
    policy_mass = [
        {"alternative": alt, "probability": count / denom}
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
    sampled = int(condition.sampled_worlds)
    sampled_world_robustness = [
        {
            "alternative": alt,
            "favorable_worlds": sampled if float(q) > 0.5 else 0,
            "sampled_worlds": sampled,
        }
        for alt, q in zip(alternative_ids, q_values, strict=True)
    ]
    uncertainty = [
        {
            "alternative": alt,
            "standard_error": float(condition.uncertainty),
            "method": "int13-scalar-broadcast",
        }
        for alt in alternative_ids
    ]
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


def _unavailable(reason: str, identity: AdviceRequestIdentity) -> AdviceResponse:
    return AdviceResponse(status="unavailable", reason=reason, identity=identity)


def request_advice(
    address: str,
    scenario_id: str,
    identity: AdviceRequestIdentity,
) -> AdviceResponse:
    """Return advice for one scenario at one decision, or fail closed.

    Fail-closed cases (no evidence is ever returned on failure):
    - ``identity_mismatch``: the request identity does not match the fixture's
      pinned identity (source replay, match, advisor, or compute).
    - ``scenario_not_found``: the scenario id is unknown.
    - ``decision_not_found``: the address does not match the fixture's decision.
    """
    artifact = load_advice_fixture()
    expected = _expected_identity(artifact)
    if identity != expected:
        return _unavailable("identity_mismatch", identity)

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
