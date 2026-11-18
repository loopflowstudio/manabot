/**
 * The opponent's turn as it happens, one beat at a time.
 *
 * Between two of the person's decisions the authority may play several moves
 * of the opponent's and deliver only the board after the last one. The table
 * draws that board at once (it is the authority's), then reads out what
 * happened on the way, one fact per beat, long enough to read, with what each
 * fact is about raised on the board. `beatsFor(facts, shown, names)` picks the
 * facts a person has not been shown yet that are not their own moves, when the
 * opponent made at least one of them (the consequences of the person's own
 * move alone are already on the board they are looking at). A busy turn of
 * more than `GROUP_ABOVE` facts reads as sentences instead: one player's
 * consecutive moves become one ("Heuristic played Plains and attacked with
 * Ally"), as do consecutive consequences of one kind. The table plays them one
 * pace apart (`BeatPace`, a per-browser setting) and drops them all on Space, a
 * click, a commit or a new frame. Reduced motion, or the pace "off", plays
 * none: the recap already lists every fact.
 *
 * A beat is presentation only. It never holds or reorders a decision: the
 * rail stays live while beats play, and acting ends them.
 */
import type { RecentFact, Side } from './recent';

export interface Beat {
  seq: number;
  by: RecentFact['by'];
  text: string;
  /** Objects on the table the beat raises, by id. */
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
