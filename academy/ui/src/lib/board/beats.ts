/**
 * The opponent's turn on the table, one kept board stop at a time.
 *
 * The walk is the history's one cursor advancing on a timer (`BEAT_MS`) across
 * unread board stops: lines that happened at once in the rules (sharing one
 * kept `StepBoard`) form one stop, passes are never a stop, and the caption
 * reads the stop's sentence(s) from `tell` (carried on `HistoryLine.sentence`).
 * Clicking a scrubber tick or selecting a history line moves that same cursor
 * and holds; reduced motion never auto-walks, while the scrubber, captions and
 * "Show their turn again" still work by hand.
 */
import type { Observation, PresentationEvent, PresentationRule } from '$lib/types';
import { findBoard, type StepBoard } from './boards';
import {
  flattenHistory,
  lineAtCursor,
  type CursorSelection,
  type FlatHistoryLine,
  type HistoryTurn,
} from './history';
import { playerIndex } from './recent';
import { tellCounter, tellPtDelta } from './tell';

export interface CursorView {
  stop: WalkStop | null;
  line: FlatHistoryLine | null;
  lineIds: ReadonlySet<string>;
  sourceSeq: number | null;
  subjects: Set<number>;
  seqs: ReadonlySet<number>;
  caption: string;
}

const EMPTY_CURSOR_VIEW: CursorView = {
  stop: null,
  line: null,
  lineIds: new Set(),
  sourceSeq: null,
  subjects: new Set(),
  seqs: new Set(),
  caption: '',
};

/**
 * Resolve what the board, rail and caption focus for `selection`.
 * `scope: 'stop'` (walk, scrubber tick, "Show their turn again") keeps every
 * line, subject, seq and caption phrase of the kept board stop intact even
 * when the walk is paused; `scope: 'line'` (clicking a history line, ← →, or
 * a Review sub-step) focuses that exact line.
 */
export function cursorView(
  turns: readonly HistoryTurn[],
  stops: readonly WalkStop[],
  selection: CursorSelection | null,
): CursorView {
  if (selection === null) return EMPTY_CURSOR_VIEW;
  const stop = stops.find(candidate => stopHasSeq(candidate, selection.seq)) ?? null;
  const line = lineAtCursor(turns, selection.seq);
  const useStop = selection.scope === 'stop' && stop !== null;
  const lineIds = useStop
    ? new Set(stop.lineIds)
    : line
      ? new Set([line.line.id])
      : new Set(stop?.lineIds ?? []);
  const subjects = new Set(
    useStop ? stop.subjects : (line?.line.subjects ?? stop?.subjects ?? []),
  );
  const seqs = useStop
    ? new Set(stop.seqs)
    : new Set(line?.line.seqs ?? stop?.seqs ?? [selection.seq]);
  const caption = (useStop ? stop.caption : line?.line.sentence) ?? stop?.caption ?? '';
  return {
    stop,
    line,
    lineIds,
    sourceSeq: line?.line.sourceSeq ?? null,
    subjects,
    seqs,
    caption,
  };
}

export interface StopDelta {
  kind: 'damage' | 'counters' | 'pump';
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
        const idx = playerIndex(projection, event.kind.target.id);
        if (idx !== null) players.set(idx, (players.get(idx) ?? 0) + event.kind.amount);
      }
    } else if (event.kind.kind === 'counters') {
      const id = event.kind.object.entity;
      const list = objects.get(id) ?? [];
      list.push({ kind: 'counters', text: `${tellCounter(event.kind.counter)} · ${event.kind.total}` });
      objects.set(id, list);
    } else if (event.kind.kind === 'pt_changed') {
      const text = tellPtDelta(event.kind.power, event.kind.toughness);
      const kind = event.kind.power <= 0 && event.kind.toughness <= 0 ? 'damage' : 'pump';
      for (const obj of event.kind.objects) {
        const list = objects.get(obj.entity) ?? [];
        list.push({ kind, text });
        objects.set(obj.entity, list);
      }
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
  /** True when every line in this stop is only a turn-based draw step. */
  drawStepOnly: boolean;
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
      prev.drawStepOnly = prev.drawStepOnly && isDrawStep;
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
      drawStepOnly: isDrawStep,
      boardKey,
    });
  }
  return stops;
}

/** Whether the track has at least one non-draw-step stop worth scrubbing or replaying. */
export function hasWalkableTurn(stops: readonly WalkStop[]): boolean {
  return stops.some(stop => !stop.drawStepOnly);
}

/**
 * Unread stops after `shownSeq` to walk automatically (both the person's own
 * moves and the opponent's moves, in order). A bare turn-start draw step with
 * no other moves or consequences does not trigger an automatic walk.
 */
export function unreadWalkStops(stops: readonly WalkStop[], shownSeq: number): WalkStop[] {
  const unseen = stops.filter(stop => stop.seqs.some(s => s > shownSeq));
  if (unseen.every(stop => stop.drawStepOnly)) return [];
  return unseen;
}

/**
 * The turn's stops shown on the bottom scrubber: the cursor's own turn while
 * looking back, else the latest turn with a non-draw-step move or consequence.
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
      if (!stops[i].drawStepOnly) {
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

export type WalkPace = 'study' | 'quick' | 'off';

export const PACE_MS: Record<WalkPace, number | null> = {
  study: 1400,
  quick: 700,
  off: null,
};

/** Default duration each stop stays while walking, in milliseconds (Study pace). */
export const BEAT_MS = PACE_MS.study!;

const PACE_STORAGE_KEY = 'academy.table.pace';

export function loadWalkPace(): WalkPace {
  try {
    const raw = typeof localStorage !== 'undefined' ? localStorage.getItem(PACE_STORAGE_KEY) : null;
    if (raw === 'study' || raw === 'quick' || raw === 'off') return raw;
  } catch { /* Default to Study pace when storage is unavailable. */ }
  return 'study';
}

export function saveWalkPace(pace: WalkPace): void {
  try {
    if (typeof localStorage !== 'undefined') localStorage.setItem(PACE_STORAGE_KEY, pace);
  } catch { /* Preference persistence is best-effort. */ }
}

export function paceMs(pace: WalkPace): number | null {
  return PACE_MS[pace];
}
