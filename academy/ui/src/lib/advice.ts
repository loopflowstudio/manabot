import type { ExperienceFrame, InteractionOffer } from './types';
import type { DecisionEvidence } from './study-protocol';

// Client for POST /api/advice; academy/advice.py owns the advice logic.
// Two explicit seams cross the wire here:
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
