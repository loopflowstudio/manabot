import type {
  Command,
  ExperienceFrame,
  InteractionOffer,
  PresentationEvent,
} from './types';
import { parseReplayDecisionAddress } from './replay-index';

export const STUDY_VERSION = 1 as const;
export type StudyVersion = typeof STUDY_VERSION;

export type KnowledgeScope = 'historical_viewer';

export interface ContentPackIdentity {
  id: string;
  version: string;
  content_hash: string;
  asset_manifest_sha256: string;
}

export interface EngineIdentity {
  version: string;
  build_sha256: string;
}

export interface ModelIdentity {
  id: string;
  checkpoint_sha256: string;
}

export interface AnalysisBudgetIdentity {
  id: string;
  max_nodes: number;
  sampled_worlds: number;
  rollouts_per_world: number;
}

export interface StudyIdentity {
  artifact_id: string;
  source_replay_id: string;
  source_replay_sha256: string;
  match_id: string;
  content_pack: ContentPackIdentity;
  engine: EngineIdentity;
  model: ModelIdentity;
  analysis_budget: AnalysisBudgetIdentity;
  knowledge_scope: KnowledgeScope;
}

export interface RecordedDecision {
  ordinal: number;
  event_cursor: number;
  automatic: boolean;
  frame: ExperienceFrame;
  offer: InteractionOffer;
  played: Command;
  presentation: PresentationEvent[];
}

export interface RecordedDecisionInput {
  version: StudyVersion;
  source_replay_id: string;
  decision_count: number;
  decisions: RecordedDecision[];
}

export type StudyDecisionKind =
  | 'priority'
  | 'targeting'
  | 'attack'
  | 'block'
  | 'other';

export const LANDMARK_REASONS = [
  'priority_commitment',
  'priority_response',
  'target_selection',
  'attack_declaration',
  'block_declaration',
  'branching_choice',
  'public_semantic_impact',
] as const;
export type LandmarkReason = (typeof LANDMARK_REASONS)[number];

export interface StudyDecision {
  id: string;
  ordinal: number;
  viewer: number;
  event_cursor: number;
  automatic: boolean;
  kind: StudyDecisionKind;
  frame: ExperienceFrame;
  offer: InteractionOffer;
  played: Command;
}

export interface RankedStudyLandmark {
  decision_id: string;
  rank: number;
  reasons: LandmarkReason[];
}

export interface StudyDecisionIndex {
  version: StudyVersion;
  identity: StudyIdentity;
  decisions: StudyDecision[];
  landmarks: RankedStudyLandmark[];
}

export interface DecisionAlternative {
  id: string;
  command: Command;
}

export interface PolicyMass {
  alternative: string;
  probability: number;
}

export interface SearchValue {
  alternative: string;
  perspective: number;
  expected_match_points: number;
}

export interface VisitCount {
  alternative: string;
  visits: number;
}

export interface SampledWorldRobustness {
  alternative: string;
  favorable_worlds: number;
  sampled_worlds: number;
}

export interface UncertaintyEvidence {
  alternative: string;
  standard_error: number;
  method: string;
}

export interface EvidenceProvenance {
  producer: string;
  producer_version: string;
  generated_at: string;
  evidence_sha256: string;
}

export interface DecisionEvidence {
  policy_mass: PolicyMass[];
  search_value: SearchValue[];
  visits: VisitCount[];
  sampled_world_robustness: SampledWorldRobustness[];
  uncertainty: UncertaintyEvidence[];
  provenance: EvidenceProvenance;
}

export interface StudyLandmark {
  id: string;
  decision_id: string;
  match_state_hash: string;
  viewer: number;
  prompt_id: number;
  offer_id: number;
  frame: ExperienceFrame;
  offer: InteractionOffer;
  played: Command;
  alternatives: DecisionAlternative[];
  evidence: DecisionEvidence;
}

export interface StudyArtifact {
  version: StudyVersion;
  identity: StudyIdentity;
  landmarks: StudyLandmark[];
}

function fail(message: string): never {
  throw new Error(`invalid study artifact: ${message}`);
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(canonical);
  }
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

function assertCommandBinding(
  command: Command,
  landmark: StudyLandmark,
  selected: boolean,
): void {
  if (
    command.match_id !== landmark.frame.match_id
    || command.expected_revision !== landmark.frame.revision
    || command.prompt_id !== landmark.prompt_id
    || (selected && command.offer_id !== landmark.offer_id)
    || !landmark.frame.offers.some(({ id }) => id === command.offer_id)
  ) {
    fail(`${landmark.id}: command identity drifted`);
  }
}

function assertMetricCover(
  landmark: StudyLandmark,
  label: string,
  values: readonly string[],
  alternatives: Set<string>,
): void {
  const actual = new Set(values);
  if (
    actual.size !== values.length
    || actual.size !== alternatives.size
    || [...actual].some((id) => !alternatives.has(id))
  ) {
    fail(`${landmark.id}: ${label} does not cover alternatives`);
  }
}

/**
 * Enforce semantic bindings and privacy invariants that JSON Schema cannot
 * express. Call this after schema validation at an external JSON boundary.
 */
export function assertViewerSafeStudyArtifact(artifact: StudyArtifact): void {
  if (artifact.identity.knowledge_scope !== 'historical_viewer') {
    fail('study v1 requires historical_viewer knowledge');
  }
  if (artifact.landmarks.length === 0) {
    fail('at least one landmark is required');
  }

  for (const landmark of artifact.landmarks) {
    const { frame } = landmark;
    const pack = artifact.identity.content_pack;
    let address: ReturnType<typeof parseReplayDecisionAddress>;
    try {
      address = parseReplayDecisionAddress(landmark.decision_id);
    } catch {
      fail(`${landmark.id}: decision_id is not an erd1 address`);
    }
    if (frame.match_id !== artifact.identity.match_id) {
      fail(`${landmark.id}: frame match does not match study identity`);
    }
    if (
      frame.content_hash !== pack.content_hash
      || frame.asset_manifest_hash !== pack.asset_manifest_sha256
    ) {
      fail(`${landmark.id}: frame content pack hashes drifted`);
    }
    if (
      frame.asset_pack !== null
      && frame.asset_pack !== undefined
      && (
        frame.asset_pack.id !== pack.id
        || frame.asset_pack.version !== pack.version
        || frame.asset_pack.manifest_sha256 !== pack.asset_manifest_sha256
      )
    ) {
      fail(`${landmark.id}: frame asset pack identity drifted`);
    }
    if (frame.projection.opponent.hand.length !== 0) {
      fail(`${landmark.id}: opponent-private hand identities are forbidden`);
    }
    if (
      frame.prompt === null
      || frame.prompt.id !== landmark.prompt_id
      || frame.prompt.actor !== landmark.viewer
      || landmark.offer.id !== landmark.offer_id
      || landmark.offer.actor !== landmark.viewer
    ) {
      fail(`${landmark.id}: viewer, prompt, or offer binding drifted`);
    }
    const frameOffer = frame.offers.find((offer) => offer.id === landmark.offer_id);
    if (frameOffer === undefined || !sameJson(frameOffer, landmark.offer)) {
      fail(`${landmark.id}: selected offer differs from frame offer`);
    }

    assertCommandBinding(landmark.played, landmark, true);
    if (
      address.replay_id !== artifact.identity.source_replay_id
      || address.match_id !== artifact.identity.match_id
      || address.viewer !== String(landmark.viewer)
      || address.revision !== String(frame.revision)
      || address.prompt_id !== String(landmark.prompt_id)
      || address.offer_id !== String(landmark.offer_id)
      || address.command_id !== landmark.played.command_id
    ) {
      fail(`${landmark.id}: replay decision address drifted`);
    }
    if (landmark.alternatives.length === 0) {
      fail(`${landmark.id}: no decision alternatives recorded`);
    }
    const alternatives = new Set(landmark.alternatives.map(({ id }) => id));
    if (alternatives.size !== landmark.alternatives.length) {
      fail(`${landmark.id}: duplicate alternative`);
    }
    for (const alternative of landmark.alternatives) {
      assertCommandBinding(alternative.command, landmark, false);
    }

    const evidence = landmark.evidence;
    assertMetricCover(
      landmark,
      'policy mass',
      evidence.policy_mass.map(({ alternative }) => alternative),
      alternatives,
    );
    assertMetricCover(
      landmark,
      'search value',
      evidence.search_value.map(({ alternative }) => alternative),
      alternatives,
    );
    assertMetricCover(
      landmark,
      'visits',
      evidence.visits.map(({ alternative }) => alternative),
      alternatives,
    );
    assertMetricCover(
      landmark,
      'sampled-world robustness',
      evidence.sampled_world_robustness.map(({ alternative }) => alternative),
      alternatives,
    );
    assertMetricCover(
      landmark,
      'uncertainty',
      evidence.uncertainty.map(({ alternative }) => alternative),
      alternatives,
    );

    const probabilitySum = evidence.policy_mass.reduce(
      (sum, { probability }) => sum + probability,
      0,
    );
    if (
      evidence.policy_mass.some(
        ({ probability }) =>
          !Number.isFinite(probability) || probability < 0 || probability > 1,
      )
      || Math.abs(probabilitySum - 1) > 1e-9
    ) {
      fail(`${landmark.id}: invalid policy mass`);
    }
    if (
      evidence.search_value.some(
        ({ perspective, expected_match_points }) =>
          perspective !== landmark.viewer || !Number.isFinite(expected_match_points),
      )
    ) {
      fail(`${landmark.id}: invalid search value`);
    }
    if (
      evidence.sampled_world_robustness.some(
        ({ favorable_worlds, sampled_worlds }) =>
          sampled_worlds === 0 || favorable_worlds > sampled_worlds,
      )
    ) {
      fail(`${landmark.id}: invalid sampled-world robustness`);
    }
    if (
      evidence.uncertainty.some(
        ({ standard_error }) =>
          !Number.isFinite(standard_error) || standard_error < 0,
      )
    ) {
      fail(`${landmark.id}: invalid uncertainty`);
    }
  }
}

function assertRecordedBinding(
  frame: ExperienceFrame,
  offer: InteractionOffer,
  played: Command,
  viewer: number,
  context: string,
): void {
  if (frame.prompt === null) {
    fail(`${context}: decision frame has no prompt`);
  }
  if (frame.prompt.actor !== viewer || offer.actor !== viewer) {
    fail(`${context}: viewer, prompt, or offer binding drifted`);
  }
  const frameOffer = frame.offers.find(({ id }) => id === offer.id);
  if (frameOffer === undefined || !sameJson(frameOffer, offer)) {
    fail(`${context}: selected offer differs from frame offer`);
  }
  if (
    played.match_id !== frame.match_id
    || played.expected_revision !== frame.revision
    || played.prompt_id !== frame.prompt.id
    || played.offer_id !== offer.id
  ) {
    fail(`${context}: played command identity drifted`);
  }
  if (frame.projection.opponent.hand.length !== 0) {
    fail(`${context}: opponent-private hand identities are forbidden`);
  }
}

/** Validate the closed canonical-decision consumer boundary after JSON Schema. */
export function assertViewerSafeRecordedDecisionInput(
  input: RecordedDecisionInput,
): void {
  if (input.decision_count !== input.decisions.length) {
    fail('declared decision count does not match input length');
  }
  let previousCursor: number | undefined;
  input.decisions.forEach((decision, ordinal) => {
    const context = `decision ${decision.ordinal}`;
    if (decision.ordinal !== ordinal) {
      fail(`${context}: ordinals must be contiguous and preserve source order`);
    }
    if (previousCursor !== undefined && decision.event_cursor <= previousCursor) {
      fail(`${context}: event cursors must strictly increase`);
    }
    previousCursor = decision.event_cursor;
    if (decision.frame.prompt === null) {
      fail(`${context}: decision frame has no prompt`);
    }
    assertRecordedBinding(
      decision.frame,
      decision.offer,
      decision.played,
      decision.frame.prompt.actor,
      context,
    );
  });
}

/** Validate the complete navigation index and separate landmark references. */
export function assertViewerSafeStudyDecisionIndex(index: StudyDecisionIndex): void {
  if (index.identity.knowledge_scope !== 'historical_viewer') {
    fail('study v1 requires historical_viewer knowledge');
  }

  const decisionIds = new Set<string>();
  let previousCursor: number | undefined;
  index.decisions.forEach((decision, ordinal) => {
    const context = `decision ${decision.ordinal}`;
    if (decision.ordinal !== ordinal) {
      fail(`${context}: ordinals must be contiguous and preserve source order`);
    }
    if (previousCursor !== undefined && decision.event_cursor <= previousCursor) {
      fail(`${context}: event cursors must strictly increase`);
    }
    previousCursor = decision.event_cursor;
    if (decisionIds.has(decision.id)) {
      fail(`${context}: duplicate study decision id`);
    }
    decisionIds.add(decision.id);
    if (decision.frame.match_id !== index.identity.match_id) {
      fail(`${context}: frame match does not match study identity`);
    }
    const pack = index.identity.content_pack;
    if (
      decision.frame.content_hash !== pack.content_hash
      || decision.frame.asset_manifest_hash !== pack.asset_manifest_sha256
    ) {
      fail(`${context}: frame content pack hashes drifted`);
    }
    if (
      decision.frame.asset_pack !== null
      && decision.frame.asset_pack !== undefined
      && (
        decision.frame.asset_pack.id !== pack.id
        || decision.frame.asset_pack.version !== pack.version
        || decision.frame.asset_pack.manifest_sha256 !== pack.asset_manifest_sha256
      )
    ) {
      fail(`${context}: frame asset pack identity drifted`);
    }
    assertRecordedBinding(
      decision.frame,
      decision.offer,
      decision.played,
      decision.viewer,
      context,
    );
  });

  if (index.landmarks.length > 7) {
    fail('study decision index cannot recommend more than seven landmarks');
  }
  const recommended = new Set<string>();
  index.landmarks.forEach((landmark, offset) => {
    if (landmark.rank !== offset + 1) {
      fail('landmark ranks must be contiguous and one-based');
    }
    const decision = index.decisions.find(({ id }) => id === landmark.decision_id);
    if (decision === undefined) {
      fail('landmark references a missing decision');
    }
    if (recommended.has(landmark.decision_id)) {
      fail('a decision can be recommended only once');
    }
    recommended.add(landmark.decision_id);
    if (decision.automatic || decision.frame.offers.length <= 1) {
      fail('automatic or forced decisions cannot be landmarks');
    }
    if (landmark.reasons.length === 0) {
      fail('landmark reasons cannot be empty');
    }
    const expectedReasons = [...new Set(landmark.reasons)].sort(
      (left, right) => LANDMARK_REASONS.indexOf(left) - LANDMARK_REASONS.indexOf(right),
    );
    if (!sameJson(landmark.reasons, expectedReasons)) {
      fail('landmark reasons must be unique and enum-ordered');
    }
  });
}
