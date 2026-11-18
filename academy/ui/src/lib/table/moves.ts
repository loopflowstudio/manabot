/**
 * The log as a move list, grouped by turn as a chess score groups moves.
 *
 * `moveList(entries, players, owners, names)` turns the store's log into turns
 * with their moves: who made each one (a player's name, or automation, which is
 * never the person's), how many times in a row (consecutive identical automated
 * moves collapse to one line with a count), and the move's text split so card
 * names can be inspected. A run of the opponent's and automation's priority
 * passes folds into one quiet line that still counts each actor's passes; the
 * person's own passes stay moves. It reads only what the log says; it never
 * infers moves.
 */
import type { GameLogEntry } from '$lib/types';

export type MoveActor = 'you' | 'opponent' | 'auto' | 'note';

export interface Segment {
  text: string;
  /** Set when this segment is a card name the viewer can inspect. */
  card: string | null;
}

export interface Move {
  id: string;
  actor: MoveActor;
  /** The player name shown for this move, or "auto" for automation. */
  who: string;
  segments: Segment[];
  /** Consecutive identical moves collapsed into this one. */
  count: number;
  /** Set on a folded run of priority passes: each actor's passes, in first-pass order. */
  passes: PassCount[] | null;
}

export interface PassCount {
  actor: 'opponent' | 'auto';
  who: string;
  count: number;
}

export interface MoveTurn {
  turn: number | null;
  /** Whose turn it is, when known. */
  owner: string | null;
  moves: Move[];
}

const ACTORS: Record<GameLogEntry['actor'], MoveActor> = { hero: 'you', villain: 'opponent', auto: 'auto', system: 'note' };
const PASS = 'Pass priority';

/** Splits `text` on the longest known card names first, so "Cast Fire Nation Cadets" names one card. */
export function segments(text: string, names: readonly string[]): Segment[] {
  const sorted = [...new Set(names)].filter(name => name.length > 1).sort((a, b) => b.length - a.length);
  const out: Segment[] = [];
  let rest = text;
  while (rest) {
    let hit: { at: number; name: string } | null = null;
    for (const name of sorted) {
      const at = rest.indexOf(name);
      if (at >= 0 && (!hit || at < hit.at)) hit = { at, name };
    }
    if (!hit) { out.push({ text: rest, card: null }); break; }
    if (hit.at > 0) out.push({ text: rest.slice(0, hit.at), card: null });
    out.push({ text: hit.name, card: hit.name });
    rest = rest.slice(hit.at + hit.name.length);
  }
  return out;
}

export function moveList(
  entries: readonly GameLogEntry[],
  players: { you: string; opponent: string },
  owners: ReadonlyMap<number, 'you' | 'opponent'>,
  names: readonly string[],
): MoveTurn[] {
  const turns: MoveTurn[] = [];
  for (const entry of entries) {
    let turn = turns.at(-1);
    if (!turn || turn.turn !== entry.turn) {
      const owner = entry.turn === null ? undefined : owners.get(entry.turn);
      turn = { turn: entry.turn, owner: owner ? players[owner] : null, moves: [] };
      turns.push(turn);
    }
    const actor = ACTORS[entry.actor];
    const last = turn.moves.at(-1);
    if (entry.text === PASS && (actor === 'opponent' || actor === 'auto')) {
      const who = actor === 'opponent' ? players.opponent : 'auto';
      if (last?.passes) {
        const part = last.passes.find(candidate => candidate.actor === actor);
        if (part) part.count += 1;
        else last.passes.push({ actor, who, count: 1 });
      } else {
        turn.moves.push({ id: entry.id, actor: 'note', who: '', segments: [{ text: 'passed', card: null }], count: 1, passes: [{ actor, who, count: 1 }] });
      }
      continue;
    }
    if (last && actor === 'auto' && last.actor === 'auto' && last.segments.map(s => s.text).join('') === entry.text) {
      last.count += 1;
      continue;
    }
    const who = actor === 'you' ? players.you : actor === 'opponent' ? players.opponent : actor === 'auto' ? 'auto' : '';
    turn.moves.push({ id: entry.id, actor, who, segments: segments(entry.text, names), count: 1, passes: null });
  }
  return turns;
}
