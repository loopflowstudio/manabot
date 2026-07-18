import type {
  CanonicalReplayProjectionV1,
  CanonicalReplayProjectionResponseV1,
  RestoredReplayDecision,
} from './replay-index';
import { canonicalReplayProjectionSha256 } from './replay-index';
import {
  assertViewerSafeStudyArtifact,
  type StudyArtifact,
  type StudyLandmark,
} from './study-protocol';
import type {
  Command,
  InteractionOffer,
  Observation,
  PresentationEvent,
} from './types';

export type StudyPlanKind = 'played' | 'policy' | 'search';

export interface StudyRetryResponse {
  attempt_id: string;
  trace_id: string;
  address: string;
  retry: {
    command: Command;
    projection: Observation;
    presentation: PresentationEvent[];
  };
  return_to: {
    address: string;
    ordinal: number;
    presentation_cursor: number;
  };
}

export interface StudyRevealResponse {
  attempt_id: string;
  artifact: StudyArtifact;
}

export interface StudyPlanPreviewResponse {
  attempt_id: string;
  plan: StudyPlanKind;
  alternative_id: string | null;
  command: Command;
  offer: InteractionOffer;
  projection: Observation;
  presentation: PresentationEvent[];
  return_to: {
    address: string;
    ordinal: number;
    presentation_cursor: number;
  };
}

export interface StudyDisplayPlan {
  kind: StudyPlanKind;
  label: string;
  command: Command;
  offer: InteractionOffer;
  alternativeId: string | null;
  sameAsPlayed: boolean;
  policyProbability: number | null;
  expectedMatchPoints: number | null;
  visits: number | null;
  favorableWorlds: number | null;
  sampledWorlds: number | null;
  standardError: number | null;
  uncertaintyMethod: string | null;
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, child]) => [key, canonical(child)]),
    );
  }
  return value;
}

function sameJson(left: unknown, right: unknown): boolean {
  return JSON.stringify(canonical(left)) === JSON.stringify(canonical(right));
}

function projectionWithoutAddresses(
  projection: CanonicalReplayProjectionResponseV1,
): CanonicalReplayProjectionV1 {
  const snapshot = JSON.parse(
    JSON.stringify(projection),
  ) as CanonicalReplayProjectionResponseV1;
  return {
    ...snapshot,
    decisions: snapshot.decisions.map(({ address: _address, ...row }) => row),
  };
}

export async function bindStudyArtifact(
  artifact: StudyArtifact,
  projection: CanonicalReplayProjectionResponseV1,
  restored: RestoredReplayDecision,
): Promise<StudyLandmark> {
  assertViewerSafeStudyArtifact(artifact);
  const digest = await canonicalReplayProjectionSha256(
    projectionWithoutAddresses(projection),
  );
  if (
    artifact.identity.source_replay_id !== projection.replay_id
    || artifact.identity.source_replay_sha256 !== digest
    || artifact.identity.match_id !== projection.match_id
    || artifact.identity.content_pack.content_hash !== projection.content_hash
    || artifact.identity.content_pack.asset_manifest_sha256
      !== projection.asset_manifest_hash
  ) {
    throw new Error('Study evidence identity differs from canonical replay.');
  }
  const landmarks = artifact.landmarks.filter(
    ({ decision_id }) => decision_id === restored.address,
  );
  if (landmarks.length !== 1) {
    throw new Error('Study evidence has no exact landmark for this decision.');
  }
  const landmark = landmarks[0];
  if (
    landmark.viewer !== restored.viewer
    || !sameJson(landmark.frame, restored.frame)
    || !sameJson(landmark.offer, restored.offer)
    || !sameJson(landmark.played, restored.command)
  ) {
    throw new Error('Study landmark differs from the restored decision.');
  }
  return landmark;
}

export function buildStudyPlans(landmark: StudyLandmark): StudyDisplayPlan[] {
  const alternatives = new Map(
    landmark.alternatives.map((alternative) => [alternative.id, alternative]),
  );
  const offerOrder = new Map(
    landmark.frame.offers.map((offer, index) => [offer.id, index]),
  );
  const policy = [...landmark.evidence.policy_mass].sort((left, right) => {
    const probability = right.probability - left.probability;
    if (probability !== 0) return probability;
    const leftOffer = alternatives.get(left.alternative)?.command.offer_id ?? 0;
    const rightOffer = alternatives.get(right.alternative)?.command.offer_id ?? 0;
    return (offerOrder.get(leftOffer) ?? 0) - (offerOrder.get(rightOffer) ?? 0);
  })[0];
  const visitByAlternative = new Map(
    landmark.evidence.visits.map((row) => [row.alternative, row.visits]),
  );
  const uncertaintyByAlternative = new Map(
    landmark.evidence.uncertainty.map((row) => [row.alternative, row.standard_error]),
  );
  const search = [...landmark.evidence.search_value].sort((left, right) => {
    const value = right.expected_match_points - left.expected_match_points;
    if (value !== 0) return value;
    const visits = (visitByAlternative.get(right.alternative) ?? 0)
      - (visitByAlternative.get(left.alternative) ?? 0);
    if (visits !== 0) return visits;
    const uncertainty = (uncertaintyByAlternative.get(left.alternative) ?? 0)
      - (uncertaintyByAlternative.get(right.alternative) ?? 0);
    if (uncertainty !== 0) return uncertainty;
    const leftOffer = alternatives.get(left.alternative)?.command.offer_id ?? 0;
    const rightOffer = alternatives.get(right.alternative)?.command.offer_id ?? 0;
    return (offerOrder.get(leftOffer) ?? 0) - (offerOrder.get(rightOffer) ?? 0);
  })[0];

  const alternativePlan = (
    kind: 'policy' | 'search',
    alternativeId: string,
  ): StudyDisplayPlan => {
    const alternative = alternatives.get(alternativeId);
    if (!alternative) throw new Error(`Study ${kind} alternative is missing.`);
    const offer = landmark.frame.offers.find(
      ({ id }) => id === alternative.command.offer_id,
    );
    if (!offer) throw new Error(`Study ${kind} offer is missing.`);
    const policyMetric = landmark.evidence.policy_mass.find(
      ({ alternative: id }) => id === alternativeId,
    );
    const searchMetric = landmark.evidence.search_value.find(
      ({ alternative: id }) => id === alternativeId,
    );
    const visitMetric = landmark.evidence.visits.find(
      ({ alternative: id }) => id === alternativeId,
    );
    const robustness = landmark.evidence.sampled_world_robustness.find(
      ({ alternative: id }) => id === alternativeId,
    );
    const uncertainty = landmark.evidence.uncertainty.find(
      ({ alternative: id }) => id === alternativeId,
    );
    return {
      kind,
      label: kind === 'policy' ? 'Policy' : 'Search',
      command: structuredClone(alternative.command),
      offer: structuredClone(offer),
      alternativeId,
      sameAsPlayed: alternative.command.offer_id === landmark.played.offer_id,
      policyProbability: policyMetric?.probability ?? null,
      expectedMatchPoints: searchMetric?.expected_match_points ?? null,
      visits: visitMetric?.visits ?? null,
      favorableWorlds: robustness?.favorable_worlds ?? null,
      sampledWorlds: robustness?.sampled_worlds ?? null,
      standardError: uncertainty?.standard_error ?? null,
      uncertaintyMethod: uncertainty?.method ?? null,
    };
  };

  return [
    {
      kind: 'played',
      label: 'Played',
      command: structuredClone(landmark.played),
      offer: structuredClone(landmark.offer),
      alternativeId: null,
      sameAsPlayed: true,
      policyProbability: null,
      expectedMatchPoints: null,
      visits: null,
      favorableWorlds: null,
      sampledWorlds: null,
      standardError: null,
      uncertaintyMethod: null,
    },
    alternativePlan('policy', policy.alternative),
    alternativePlan('search', search.alternative),
  ];
}
