import type { AddressedReplayDecision } from './replay-index';
import type { Observation, RecordedPlayer } from './types';

export interface BotDecision {
  action: number;
  probabilities: number[];
  value: number | null;
  value_semantic: string | null;
  search_scores: (number | null)[];
  unavailable: string | null;
  factors: {chosen: number; probabilities: number[]; value: number}[];
  belief: {model: string; digest: string; condition_mass: number; sampled_worlds: number} | null;
}

export interface WatchDecision {
  id: string;
  players: string[];
  identities: RecordedPlayer[];
  decision: AddressedReplayDecision;
  observation: Observation;
  outputs: BotDecision | null;
  count: number;
  finished: boolean;
  revealed: boolean;
  revision: number;
  winner: number | null;
}
