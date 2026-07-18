import type { ExperienceFrame, InteractionOffer } from './types';
import {
  assertViewerSafeStudyArtifact,
  type DecisionEvidence,
  type StudyArtifact,
} from './study-protocol';

// This module is the TypeScript twin of the Python adapter in etude/advice.py.
// Two explicit seams cross the wire here, matching the Python adapter:
//   - GAM-4 seam: AdviceRequestIdentity + AdviceRequest are the request
//     contract GAM-4's live decision address and retry/compare controls bind
//     to. Both the live play page and the Study page POST /api/advice with
//     this shape.
//   - INT-13 seam: the DecisionEvidence carried in AdviceResponse.evidence is
//     the per-action search-evidence shape int13_condition_to_decision_evidence
//     produces on the server; the surface renders it unchanged whether the
//     evidence is fixture-backed (today) or INT-13 live (later).

export interface AdviceRequestIdentity {
  source_replay_id: string;
  match_id: string;
  advisor_id: string;
  compute_id: string;
}

export interface AdviceScenarioSummary {
  landmark_id: string;
  label: string;
  description: string;
  inferred_range: string;
  belief_kind: string;
}

export interface AdviceScenarioRecord extends AdviceScenarioSummary {
  condition_id: string;
  seed_plan: string;
}

export interface AdviceArtifact {
  artifact: StudyArtifact;
  scenarios: AdviceScenarioRecord[];
}

export interface AdviceResponse {
  status: 'ok' | 'unavailable';
  reason: string | null;
  address: string | null;
  frame: ExperienceFrame | null;
  offers: InteractionOffer[];
  scenario: AdviceScenarioSummary | null;
  evidence: DecisionEvidence | null;
  deltas: Record<string, Record<string, number>> | null;
  identity: AdviceRequestIdentity | null;
}

export interface AdviceMeta {
  address: string;
  scenarios: AdviceScenarioSummary[];
  identity: AdviceRequestIdentity;
}

export interface AdviceRequest {
  address: string;
  scenario_id: string;
  identity: AdviceRequestIdentity;
}

/**
 * The advice fixture is a StudyArtifact wrapper plus a prototype presentation
 * layer. The artifact is validated through the existing viewer-safe study
 * validator (opponent-hand privacy, bindings); the scenarios are a prototype
 * presentation layer validated here. This is the TypeScript twin of the
 * Python adapter in etude/advice.py.
 */
export function assertViewerSafeAdviceArtifact(advice: AdviceArtifact): void {
  assertViewerSafeStudyArtifact(advice.artifact);
  if (advice.artifact.landmarks.length !== 2 || advice.scenarios.length !== 2) {
    throw new Error('advice comparison must contain exactly two scenarios');
  }
  const landmarkIds = new Set(advice.artifact.landmarks.map((lm) => lm.id));
  const scenarioIds = new Set(advice.scenarios.map((s) => s.landmark_id));
  if (scenarioIds.size !== landmarkIds.size || [...scenarioIds].some((id) => !landmarkIds.has(id))) {
    throw new Error('advice scenarios do not cover the artifact landmarks');
  }
  const decisionIds = new Set(advice.artifact.landmarks.map((lm) => lm.decision_id));
  if (decisionIds.size !== 1) {
    throw new Error('advice landmarks must share one decision address');
  }
  if (new Set(advice.scenarios.map((s) => s.condition_id)).size !== 2) {
    throw new Error('advice scenarios must bind distinct conditions');
  }
  if (new Set(advice.scenarios.map((s) => s.seed_plan)).size !== 1) {
    throw new Error('advice scenarios must share one paired seed plan');
  }
  const [first, second] = advice.artifact.landmarks;
  if (
    JSON.stringify(second.frame) !== JSON.stringify(first.frame) ||
    JSON.stringify(second.offer) !== JSON.stringify(first.offer) ||
    JSON.stringify(second.played) !== JSON.stringify(first.played) ||
    JSON.stringify(second.alternatives) !== JSON.stringify(first.alternatives)
  ) {
    throw new Error('advice landmarks must share facts, command, and action vocabulary');
  }
  if (JSON.stringify(first.evidence) === JSON.stringify(second.evidence)) {
    throw new Error('advice conditions must produce distinct evidence');
  }
  if (
    new Set(advice.artifact.landmarks.map((lm) => lm.evidence.provenance.producer)).size !== 1 ||
    new Set(advice.artifact.landmarks.map((lm) => lm.evidence.provenance.generated_at)).size !== 1
  ) {
    throw new Error('advice scenarios must share one producer identity');
  }
}

export function expectedIdentity(advice: AdviceArtifact): AdviceRequestIdentity {
  const identity = advice.artifact.identity;
  return {
    source_replay_id: identity.source_replay_id,
    match_id: identity.match_id,
    advisor_id: identity.model.id,
    compute_id: identity.analysis_budget.id,
  };
}

export function adviceMeta(advice: AdviceArtifact): AdviceMeta {
  return {
    address: advice.artifact.landmarks[0].decision_id,
    scenarios: advice.scenarios.map((s) => ({
      landmark_id: s.landmark_id,
      label: s.label,
      description: s.description,
      inferred_range: s.inferred_range,
      belief_kind: s.belief_kind,
    })),
    identity: expectedIdentity(advice),
  };
}

function sameIdentity(a: AdviceRequestIdentity, b: AdviceRequestIdentity): boolean {
  return (
    a.source_replay_id === b.source_replay_id &&
    a.match_id === b.match_id &&
    a.advisor_id === b.advisor_id &&
    a.compute_id === b.compute_id
  );
}

function unavailable(reason: string, identity: AdviceRequestIdentity): AdviceResponse {
  return {
    status: 'unavailable',
    reason,
    address: null,
    frame: null,
    offers: [],
    scenario: null,
    evidence: null,
    deltas: null,
    identity,
  };
}

function scenarioSummary(advice: AdviceArtifact, landmarkId: string): AdviceScenarioSummary {
  const record = advice.scenarios.find((s) => s.landmark_id === landmarkId);
  if (!record) {
    throw new Error(`missing scenario record for landmark ${landmarkId}`);
  }
  return {
    landmark_id: record.landmark_id,
    label: record.label,
    description: record.description,
    inferred_range: record.inferred_range,
    belief_kind: record.belief_kind,
  };
}

export function computeDeltas(
  left: DecisionEvidence,
  right: DecisionEvidence,
): Record<string, Record<string, number>> {
  const leftPolicy = new Map(left.policy_mass.map((r) => [r.alternative, r.probability]));
  const rightPolicy = new Map(right.policy_mass.map((r) => [r.alternative, r.probability]));
  const leftValue = new Map(left.search_value.map((r) => [r.alternative, r.expected_match_points]));
  const rightValue = new Map(right.search_value.map((r) => [r.alternative, r.expected_match_points]));
  const leftUnc = new Map(left.uncertainty.map((r) => [r.alternative, r.standard_error]));
  const rightUnc = new Map(right.uncertainty.map((r) => [r.alternative, r.standard_error]));
  const deltas: Record<string, Record<string, number>> = {};
  for (const alternative of leftPolicy.keys()) {
    deltas[alternative] = {
      policy_mass: (leftPolicy.get(alternative) ?? 0) - (rightPolicy.get(alternative) ?? 0),
      search_value: (leftValue.get(alternative) ?? 0) - (rightValue.get(alternative) ?? 0),
      uncertainty: (leftUnc.get(alternative) ?? 0) - (rightUnc.get(alternative) ?? 0),
    };
  }
  return deltas;
}

/**
 * Pure TypeScript mirror of the Python request_advice fail-closed logic.
 * Used by the unit test to exercise the adapter without a server. The live
 * page and replay page go through postAdvice below, which hits POST /api/advice.
 */
export function selectAdvice(
  advice: AdviceArtifact,
  address: string,
  scenarioId: string,
  identity: AdviceRequestIdentity,
): AdviceResponse {
  if (!sameIdentity(identity, expectedIdentity(advice))) {
    return unavailable('identity_mismatch', identity);
  }
  const landmark = advice.artifact.landmarks.find((lm) => lm.id === scenarioId);
  if (!landmark) {
    return unavailable('scenario_not_found', identity);
  }
  if (landmark.decision_id !== address) {
    return unavailable('decision_not_found', identity);
  }
  const other = advice.artifact.landmarks.find((lm) => lm.id !== scenarioId);
  if (!other) {
    return unavailable('scenario_not_found', identity);
  }
  return {
    status: 'ok',
    reason: null,
    address: landmark.decision_id,
    frame: landmark.frame,
    offers: [...landmark.frame.offers],
    scenario: scenarioSummary(advice, scenarioId),
    evidence: landmark.evidence,
    deltas: computeDeltas(landmark.evidence, other.evidence),
    identity,
  };
}

export async function fetchAdviceMeta(): Promise<AdviceMeta> {
  const response = await fetch('/api/advice');
  if (!response.ok) {
    throw new Error(`Failed to load advice meta (${response.status})`);
  }
  return (await response.json()) as AdviceMeta;
}

export async function postAdvice(request: AdviceRequest): Promise<AdviceResponse> {
  const response = await fetch('/api/advice', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new Error(`Failed to request advice (${response.status})`);
  }
  return (await response.json()) as AdviceResponse;
}
