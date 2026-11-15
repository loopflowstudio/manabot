import type { AddressedReplayDecision } from './replay-index';
import type { Observation, PresentationEvent, PresentationLabels, RecordedPlayer, TraceConfig } from './types';

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

export interface Position {
  key: string;
  kind: 'choice' | 'consequences' | 'state';
  turn: number | null;
  phase: string | null;
  step: string | null;
  actor: number | null;
  automatic: boolean;
  can_reveal: boolean;
}
export interface RecordingIndex {
  id: string;
  positions: Position[];
  total: number;
  next_offset: number | null;
  revision: number | null;
  status: 'finished' | 'waiting' | 'interrupted';
  can_produce: boolean;
  can_study: boolean;
  winner: number | null;
  players: RecordedPlayer[];
  source_kind: string | null;
  /** Seat 2's configured checkpoint, for records that predate recorded players. */
  opponent?: TraceConfig['opponent'];
  opponent_checkpoint?: TraceConfig['opponent_checkpoint'];
  watch: boolean;
  can_feedback: boolean;
}
export interface RecordedPosition extends Position {
  description: string | null;
  note_key: string;
  observation: Observation;
  decision: AddressedReplayDecision | null;
  outputs: BotDecision | null;
  viewer: number;
  revealed: boolean;
  history: PresentationEvent[];
  history_labels: PresentationLabels[];
  history_before: number | null;
  cursor: number;
  transition_start: number;
  consequences_recorded: boolean;
}

/** Seating changes the rendering only, never the information projection. */
export function orientBoard(observation: Observation, nearSeat: number): Observation {
  if (observation.agent.player_index === nearSeat) return observation;
  return { ...observation, agent: observation.opponent, opponent: observation.agent, won: observation.game_over ? !observation.won : observation.won };
}
