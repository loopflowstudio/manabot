import type { AddressedReplayDecision } from './replay-index';
import type { Observation, PresentationEvent, RecordedPlayer } from './types';

export interface SearchConfig {
  algorithm: 'local_update';
  belief: 'physical_deal';
  worlds: number;
  depth: number;
  decision_seconds: number;
  alpha: number;
  beta: number;
}

export interface PendingSearch {
  code: string;
  message: string;
  match_id: string;
  expected_revision: number;
}

export function parseSearchConfigs(value: unknown): [SearchConfig | null, SearchConfig | null] {
  function valid(spec: unknown): spec is SearchConfig | null {
    if (spec === null) return true;
    if (typeof spec !== 'object' || spec === null) return false;
    const fields = spec as Record<string, unknown>;
    return fields.algorithm === 'local_update' && fields.belief === 'physical_deal'
      && typeof fields.worlds === 'number' && Number.isInteger(fields.worlds) && fields.worlds >= 1 && fields.worlds <= 64
      && typeof fields.depth === 'number' && Number.isInteger(fields.depth) && fields.depth >= 1 && fields.depth <= 40
      && typeof fields.decision_seconds === 'number' && fields.decision_seconds > 0 && fields.decision_seconds <= 30
      && typeof fields.alpha === 'number' && Number.isFinite(fields.alpha) && fields.alpha >= 0
      && typeof fields.beta === 'number' && Number.isFinite(fields.beta) && fields.beta > 0;
  }
  if (!Array.isArray(value) || value.length !== 2 || !valid(value[0]) || !valid(value[1])) {
    throw new Error('Invalid two-seat Search settings.');
  }
  return [value[0], value[1]];
}

export interface BotDecision {
  action: number;
  probabilities: number[];
  value: number | null;
  value_semantic: string | null;
  search_scores: (number | null)[];
  unavailable: string | null;
  factors: {chosen: number; probabilities: number[]; value: number}[];
  belief: {model: string; digest: string; condition_mass: number; sampled_worlds: number} | null;
  mirror?: {
    algorithm: string;
    worlds: number;
    depth: number;
    alpha: number;
    beta: number;
    rollouts: number;
    decision_seconds: number;
    seconds: number;
    seed: string;
    prior: number[];
    receipt_sha256: string;
  } | null;
}

export interface WatchDecision {
  mirror_retry?: PendingSearch | null;
  mirror?: (SearchConfig | null)[] | null;
  id: string;
  players: string[];
  source_kind?: string;
  identities: RecordedPlayer[];
  decision: AddressedReplayDecision;
  observation: Observation;
  outputs: BotDecision | null;
  presentation?: PresentationEvent[];
  count: number;
  finished: boolean;
  revealed: boolean;
  revision: number;
  winner: number | null;
}
