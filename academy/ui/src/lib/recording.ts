import type { SearchConfig, PendingSearch } from './search';
import type { AddressedReplayDecision } from './replay-index';
import type { Observation, PresentationEvent, PresentationLabels, RecordedPlayer, TraceConfig } from './types';

export interface BotDecision {
  action: number;
  probabilities: number[];
  value: number | null;
  update?: number | null;
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

export interface Position {
  key: string;
  kind: 'choice' | 'consequences' | 'state';
  /** The trace event a choice decides or a consequence follows; the final state's is the event count. */
  event?: number;
  turn: number | null;
  phase: string | null;
  step: string | null;
  actor: number | null;
  automatic: boolean;
  /** Both hands may show here: a Watch spectator view, or any position of a complete game you played. */
  can_reveal: boolean;
  /** The revision the position's decision or transition starts at; null at the latest state. */
  revision?: number | null;
  /** The chosen offer's label at a choice, the transition's narration after one. */
  label?: string | null;
  /** The chooser's recorded value at a choice, when one was retained. */
  value?: number | null;
  /** How an imported record re-dealt the position before this choice, in order. */
  branched?: BranchKind[] | null;
  /** The person chose here with a live analysis shown. */
  analysis_shown?: boolean;
}
/** A record intervention that branches the position from the game it came from. */
export type BranchKind = 'determinize' | 'shuffle_libraries' | 'reseed';
export interface RecordingIndex {
  id: string;
  positions: Position[];
  total: number;
  next_offset: number | null;
  revision: number | null;
  /** When the game started and, once over, ended (ISO 8601; absent from legacy records). */
  started_at?: string | null;
  ended_at?: string | null;
  status: 'finished' | 'waiting' | 'interrupted';
  can_produce: boolean;
  mirror?: (SearchConfig | null)[];
  mirror_retry?: PendingSearch | null;
  can_study: boolean;
  winner: number | null;
  players: RecordedPlayer[];
  source_kind: string | null;
  record_stop?: {status: 'complete' | 'interrupted' | 'diverged'; index: number; reason: string | null} | null;
  /** Seat 2's configured checkpoint, for records that predate recorded players. */
  opponent?: TraceConfig['opponent'];
  opponent_checkpoint?: TraceConfig['opponent_checkpoint'];
  watch: boolean;
  can_feedback: boolean;
  private_setup?: Record<string, unknown> | null;
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
  /** Why this board is the stored track's rather than your replayed view; null where it was replayed or never could be. */
  views_stopped?: string | null;
}

/**
 * Seating changes the rendering only, never the information projection.
 * After the game the near seat won only if it is the recorded winner: in a
 * draw (`winner` null) neither seat won.
 */
export function orientBoard(observation: Observation, nearSeat: number, winner: number | null): Observation {
  if (observation.agent.player_index === nearSeat) return observation;
  return { ...observation, agent: observation.opponent, opponent: observation.agent, won: observation.game_over ? winner === nearSeat : observation.won };
}
