/**
 * The opponent's turn as it happens, one beat at a time.
 *
 * Between two of the person's decisions the authority may play several moves
 * of the opponent's and deliver only the board after the last one. The table
 * draws that board at once (it is the authority's), then reads out what
 * happened on the way, one fact per beat, long enough to read, with what each
 * fact is about raised on the board. `beatsFor(facts, shown)` picks the facts a
 * person has not been shown yet that are not their own moves, when the opponent
 * made at least one of them (the consequences of the person's own move alone
 * are already on the board they are looking at); the table plays
 * them `BEAT_MS` apart and drops them all on Space, a click, a commit or a new
 * frame. Reduced motion plays none: the recap already lists every fact.
 *
 * A beat is presentation only. It never holds or reorders a decision: the
 * rail stays live while beats play, and acting ends them.
 */
import type { RecentFact } from './recent';

export interface Beat {
  seq: number;
  by: RecentFact['by'];
  text: string;
  /** Objects on the table the beat raises, by id. */
  subjects: number[];
}

/** How long each beat stays, in milliseconds: one short sentence, read once. */
export const BEAT_MS = 1100;

/** At most this many beats in a row; earlier facts stay in the recap and the move list. */
export const BEAT_LIMIT = 8;

/** The facts after `shown` (an event seq) that were not the person's own, newest `BEAT_LIMIT` of them. */
export function beatsFor(facts: readonly RecentFact[], shown: number): Beat[] {
  const unseen = facts.filter(fact => fact.seq > shown && fact.by !== 'you');
  if (!unseen.some(fact => fact.by === 'opponent')) return [];
  return unseen.slice(-BEAT_LIMIT).map(fact => ({ seq: fact.seq, by: fact.by, text: fact.text, subjects: fact.subjects }));
}
