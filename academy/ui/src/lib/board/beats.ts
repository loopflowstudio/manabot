/**
 * The opponent's turn as it happens, one beat at a time.
 *
 * Between two of the person's decisions the authority may play several moves
 * of the opponent's and deliver only the board after the last one. The board
 * draws that board at once (it is the authority's), then reads out what
 * happened on the way, one fact per beat, long enough to read, with what each
 * fact is about raised on the board. `beatsFor(facts, shown, names)` picks the
 * facts a person has not been shown yet that are not their own moves, when the
 * opponent made at least one of them (the consequences of the person's own
 * move alone are already on the board they are looking at). A busy turn of
 * more than `GROUP_ABOVE` facts reads as sentences instead: one player's
 * consecutive moves become one ("Heuristic played Plains and attacked with
 * Ally"), as do consecutive consequences of one kind. The board plays them one
 * pace apart (`BeatPace`, a per-browser setting) and drops them all at the
 * first input or a new frame. Reduced motion, or the pace "off", plays none:
 * the recap already lists every fact.
 *
 * A beat is presentation only: it never reorders, hides or chooses a decision.
 * The read-out comes before the decision, though: while a beat plays, the
 * decision's controls wait, and the first input ends the read-out and does
 * nothing else, so a person never answers a board they have not been shown.
 */
import type { PresentationEvent } from '$lib/types';
import type { RecentFact, Side } from './recent';

export interface Beat {
  seq: number;
  by: RecentFact['by'];
  text: string;
  /** Objects on the board the beat raises, by id. */
  subjects: number[];
}

/** At most this many beats in a row; earlier facts stay in the recap and the move list. */
export const BEAT_LIMIT = 8;

/** More facts than this read as grouped sentences. */
export const GROUP_ABOVE = 4;

function list(parts: string[]): string {
  return parts.length <= 1 ? parts.join('') : `${parts.slice(0, -1).join(', ')} and ${parts.at(-1)}`;
}

/** Consecutive moves of one player, or consecutive consequences of one kind, as one sentence each. */
function grouped(facts: readonly RecentFact[], names: Record<Side, string>): Beat[] {
  const runs: RecentFact[][] = [];
  for (const fact of facts) {
    const run = runs.at(-1);
    const joins = run && (fact.by !== null ? run[0].by === fact.by : run[0].by === null && run[0].kind === fact.kind);
    if (joins) run.push(fact);
    else runs.push([fact]);
  }
  return runs.map(run => {
    const by = run[0].by;
    const prefix = by !== null ? `${names[by]} ` : '';
    const parts = run.map((fact, i) => (i > 0 && prefix && fact.text.startsWith(prefix) ? fact.text.slice(prefix.length) : fact.text));
    return { seq: run.at(-1)!.seq, by, text: list(parts), subjects: run.flatMap(fact => fact.subjects) };
  });
}

/** The facts after `shown` (an event seq) that were not the person's own, newest `BEAT_LIMIT` beats of them. */
export function beatsFor(facts: readonly RecentFact[], shown: number, names: Record<Side, string>): Beat[] {
  const unseen = facts.filter(fact => fact.seq > shown && fact.by !== 'you');
  if (!unseen.some(fact => fact.by === 'opponent')) return [];
  const beats = unseen.length > GROUP_ABOVE ? grouped(unseen, names)
    : unseen.map(fact => ({ seq: fact.seq, by: fact.by, text: fact.text, subjects: fact.subjects }));
  return beats.slice(-BEAT_LIMIT);
}

/**
 * The turn the event `seq` happened in: the last turn start at or before it,
 * or null when none is retained. A beat can be drawn on a later turn's board,
 * since one engine step may run from an attack through damage into the next
 * turn, and the authority keeps a board only after each step.
 */
export function turnAt(events: readonly PresentationEvent[], seq: number): number | null {
  let turn: number | null = null;
  for (const event of events) {
    if (event.seq > seq) break;
    if (event.kind.kind === 'turn_started') turn = event.kind.turn_number;
  }
  return turn;
}

// ———————————————————————————————————————— Pace

export type BeatPace = 'off' | 'quick' | 'slow';

export const BEAT_PACES: readonly BeatPace[] = ['off', 'quick', 'slow'];

/** How long each beat stays, in milliseconds: one short sentence, read once. */
export const BEAT_MS: Record<Exclude<BeatPace, 'off'>, number> = { quick: 700, slow: 1400 };

const PACE_KEY = 'academy.table.beat-pace';

export function loadBeatPace(): BeatPace {
  const stored = typeof localStorage === 'undefined' ? null : localStorage.getItem(PACE_KEY);
  return BEAT_PACES.find(pace => pace === stored) ?? 'quick';
}

export function saveBeatPace(pace: BeatPace): void {
  localStorage.setItem(PACE_KEY, pace);
}
