/**
 * The opponent's turn on the table, one kept board stop at a time.
 *
 * The walk is the history's one cursor advancing on a timer (`BeatPace`) across
 * unread board stops: lines that happened at once in the rules (sharing one
 * kept `StepBoard`) form one stop, passes are never a stop, and the caption
 * reads the stop's sentence(s) from `tell` (carried on `HistoryLine.sentence`).
 * Clicking a scrubber tick or selecting a history line moves that same cursor
 * and holds; pace `off` or reduced motion never auto-walks, while the
 * scrubber, captions and "Show their turn again" still work by hand.
 */
import type { Observation, PresentationEvent, PresentationRule } from '$lib/types';
import { findBoard, type StepBoard } from './boards';
import { flattenHistory, turnAt, type HistoryTurn } from './history';
import { tellCounter } from './tell';

export { turnAt };

export interface StopDelta {
  kind: 'damage' | 'counters';
  text: string;
}

export interface StopDeltas {
  objects: Map<number, StopDelta[]>;
  players: Map<number, number>;
}

/** Whether `stop` owns cursor `seq` (as its representative seq or a folded event seq). */
export function stopHasSeq(stop: Pick<WalkStop, 'seq' | 'seqs'>, seq: number): boolean {
  return stop.seq === seq || stop.seqs.includes(seq);
}

/** Per-stop damage and counter badges drawn on cards and player plates while walking or looking back. */
export function computeStopDeltas(
  events: readonly PresentationEvent[],
  activeSeqs: ReadonlySet<number>,
  projection: Pick<Observation, 'agent' | 'opponent'>,
): StopDeltas {
  const objects = new Map<number, StopDelta[]>();
  const players = new Map<number, number>();
  if (activeSeqs.size === 0) return { objects, players };
  const objectDamage = new Map<number, number>();
  for (const event of events) {
    if (!activeSeqs.has(event.seq)) continue;
    if (event.kind.kind === 'damage') {
      if (event.kind.target.kind === 'object') {
        const id = event.kind.target.id.entity;
        objectDamage.set(id, (objectDamage.get(id) ?? 0) + event.kind.amount);
      } else if (event.kind.target.kind === 'player') {
        const id = event.kind.target.id;
        const idx = id === projection.agent.id || id === projection.agent.player_index
          ? projection.agent.player_index
          : id === projection.opponent.id || id === projection.opponent.player_index
            ? projection.opponent.player_index
            : null;
        if (idx !== null) players.set(idx, (players.get(idx) ?? 0) + event.kind.amount);
      }
    } else if (event.kind.kind === 'counters') {
      const id = event.kind.object.entity;
      const list = objects.get(id) ?? [];
      list.push({ kind: 'counters', text: `${tellCounter(event.kind.counter)} · ${event.kind.total}` });
      objects.set(id, list);
    }
  }
  for (const [id, amount] of objectDamage) {
    const list = objects.get(id) ?? [];
    list.unshift({ kind: 'damage', text: `−${amount}` });
    objects.set(id, list);
  }
  return { objects, players };
}

export interface WalkStop {
  /** Representative cursor seq for this stop (its first line's `boardSeq`). */
  seq: number;
  /** All event seqs folded into lines of this stop. */
  seqs: number[];
  /** `HistoryLine.id` values belonging to this stop (highlighted in the rail). */
  lineIds: string[];
  /** Turn number this stop belongs to. */
  turn: number | null;
  /** True when this stop starts a player move (drawn as a heavier scrubber tick). */
  moveStart: boolean;
  /** Sentence caption for this stop's line(s), including any rule. */
  caption: string;
  /** Rules named on lines in this stop. */
  rules: PresentationRule[];
  /** Board objects raised while the cursor rests on this stop. */
  subjects: number[];
  /** True when the opponent took an action or a non-draw-step event occurred on their turn. */
  opponentAction: boolean;
  /** True when this stop belongs solely to the person's own move or draw step. */
  yoursOnly: boolean;
  /** Kept board range key (`first:last`) when kept without fallback; null otherwise. */
  boardKey: string | null;
}

/**
 * Group selectable history lines into board stops: consecutive lines in the
 * same turn that share one exact kept `StepBoard` merge into a single stop.
 */
export function walkStops(
  turns: readonly HistoryTurn[],
  boards: readonly StepBoard[],
): WalkStop[] {
  const flat = flattenHistory(turns);
  const stops: WalkStop[] = [];
  for (const item of flat) {
    const { line, turn, ownerSide, parent, cursor } = item;
    const hit = findBoard(boards, cursor);
    const boardKey = hit && !hit.fallback ? `${hit.first}:${hit.last}` : null;
    const isOpponentMove = line.actor === 'opponent' || parent?.actor === 'opponent';
    const isYouMove = line.actor === 'you' || parent?.actor === 'you';
    const isDrawStep = line.rule === 'draw_step';
    const opponentAction = isOpponentMove || (!isYouMove && ownerSide === 'opponent' && !isDrawStep);
    const yoursOnly = isYouMove || (!opponentAction && ownerSide === 'you' && isDrawStep);
    const prev = stops.at(-1);
    if (prev && boardKey !== null && prev.boardKey === boardKey && prev.turn === turn) {
      for (const s of line.seqs) {
        if (!prev.seqs.includes(s)) prev.seqs.push(s);
      }
      if (!prev.lineIds.includes(line.id)) prev.lineIds.push(line.id);
      if (line.isMove) prev.moveStart = true;
      if (line.sentence && !prev.caption.includes(line.sentence)) {
        prev.caption = prev.caption ? `${prev.caption} · ${line.sentence}` : line.sentence;
      }
      if (line.rule && !prev.rules.includes(line.rule)) prev.rules.push(line.rule);
      for (const subject of line.subjects) {
        if (!prev.subjects.includes(subject)) prev.subjects.push(subject);
      }
      prev.opponentAction = prev.opponentAction || opponentAction;
      prev.yoursOnly = prev.yoursOnly && yoursOnly;
      continue;
    }
    stops.push({
      seq: cursor,
      seqs: [...line.seqs],
      lineIds: [line.id],
      turn,
      moveStart: line.isMove,
      caption: line.sentence,
      rules: line.rule ? [line.rule] : [],
      subjects: [...new Set(line.subjects)],
      opponentAction,
      yoursOnly,
      boardKey,
    });
  }
  return stops;
}

/**
 * Unread stops after `shownSeq` to walk automatically when the opponent acted.
 * Returns empty when the opponent did not act in the unread tail (the person's
 * own moves and draw step alone never trigger an automatic walk).
 */
export function unreadWalkStops(stops: readonly WalkStop[], shownSeq: number): WalkStop[] {
  const unseen = stops.filter(stop => stop.seqs.some(s => s > shownSeq));
  const firstOpponent = unseen.findIndex(stop => stop.opponentAction);
  if (firstOpponent < 0) return [];
  return unseen.slice(firstOpponent).filter(stop => !stop.yoursOnly);
}

/**
 * The turn's stops shown on the bottom scrubber: the cursor's own turn while
 * looking back, else the latest turn in which the opponent acted.
 */
export function scrubberStops(
  stops: readonly WalkStop[],
  cursor: number | null,
): { turn: number | null; stops: WalkStop[] } {
  let targetTurn: number | null = null;
  if (cursor !== null) {
    const active = stops.find(stop => stopHasSeq(stop, cursor));
    targetTurn = active?.turn ?? null;
  }
  if (targetTurn === null) {
    for (let i = stops.length - 1; i >= 0; i -= 1) {
      if (stops[i].opponentAction) {
        targetTurn = stops[i].turn;
        break;
      }
    }
  }
  if (targetTurn === null) return { turn: null, stops: [] };
  return {
    turn: targetTurn,
    stops: stops.filter(stop => stop.turn === targetTurn),
  };
}

/**
 * When a line shares a fallback board (dropped past one Command's position cap
 * or a recording without per-line `boards`), count how many later selectable
 * lines share that same board.
 */
export function sharedLaterCount(
  turns: readonly HistoryTurn[],
  cursor: number,
  lastSeq: number,
): number {
  const flat = flattenHistory(turns);
  let count = 0;
  for (const item of flat) {
    if (item.cursor > cursor && item.cursor <= lastSeq) count += 1;
  }
  return count;
}

// ———————————————————————————————————————— Pace

export type BeatPace = 'off' | 'quick' | 'slow';

export const BEAT_PACES: readonly BeatPace[] = ['off', 'quick', 'slow'];

/** How long each stop stays while walking, in milliseconds. */
export const BEAT_MS: Record<Exclude<BeatPace, 'off'>, number> = { quick: 700, slow: 1400 };

const PACE_KEY = 'academy.table.beat-pace';

export function loadBeatPace(): BeatPace {
  const stored = typeof localStorage === 'undefined' ? null : localStorage.getItem(PACE_KEY);
  return BEAT_PACES.find(pace => pace === stored) ?? 'quick';
}

export function saveBeatPace(pace: BeatPace): void {
  localStorage.setItem(PACE_KEY, pace);
}
