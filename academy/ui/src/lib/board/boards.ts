/**
 * Kept boards for live play and Review: every board the browser was sent this
 * session (frames and per-line boards, by seq range, structurally shared,
 * capped at `MAX_KEPT_BOARDS = 512`), plus the backward patcher.
 *
 * The server sends mid-step boards beside a frame (or on a recording's
 * `consequences` position) as patches chained back from the post-step
 * projection (`managym/academy/step_boards.py` defines the format): the newest
 * is its patch applied to the frame's board, each older one its patch applied
 * to the board after it. `stepBoards` rebuilds them once; `mergeBoards` keeps
 * them across the session up to 512; `findBoard` / `boardAt` look up any seq.
 */
import type { Observation, StepBoardPatch } from '$lib/types';

/** Latest boards kept in memory during one browser session. */
export const MAX_KEPT_BOARDS = 512;

export interface StepBoard {
  /** The presentation events (seqs) this board is the board after. */
  first: number;
  last: number;
  projection: Observation;
  /**
   * True when lines in `first..last` were dropped past one Command's position
   * cap (`MAX_STEP_BOARDS`) and fall back to the nearest later kept board.
   */
  fallback?: boolean;
}

export interface SeqRange {
  first: number;
  last: number;
}

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

/** One patch node: replace, edit an object, or rebuild an array of objects by id. */
export type PatchNode =
  | { '=': Json }
  | { o?: Record<string, PatchNode>; x?: string[] }
  | { a: ({ k: number | string; p?: PatchNode } | { '=': Json })[] };

export function applyPatch(base: Json, node: PatchNode): Json {
  if ('=' in node) return node['='];
  if ('a' in node) {
    const items = Array.isArray(base) ? base : [];
    const byId = new Map(items.map(item => [(item as { id: number | string }).id, item]));
    return node.a.map(entry => {
      if ('=' in entry) return entry['='];
      const kept = byId.get(entry.k);
      if (kept === undefined) throw new Error(`Board patch names no item ${entry.k}`);
      return entry.p ? applyPatch(kept, entry.p) : kept;
    });
  }
  const object = base !== null && typeof base === 'object' && !Array.isArray(base) ? base : {};
  const out: { [key: string]: Json } = { ...object };
  for (const key of node.x ?? []) delete out[key];
  for (const [key, child] of Object.entries(node.o ?? {})) out[key] = applyPatch(object[key] ?? null, child);
  return out;
}

/** The boards `patches` describe, oldest first, rebuilt back from the frame's `projection`. */
export function stepBoards(projection: Observation, patches: readonly StepBoardPatch[]): StepBoard[] {
  const boards: StepBoard[] = [];
  let after = projection as unknown as Json;
  for (let i = patches.length - 1; i >= 0; i -= 1) {
    after = applyPatch(after, patches[i].patch as PatchNode);
    boards.push({ first: patches[i].first, last: patches[i].last, projection: after as unknown as Observation });
  }
  return boards.reverse();
}

/**
 * Merge a frame's rebuilt mid-step boards and its own board for `range` (the
 * non-pass seqs delivered with this frame) into the session's kept boards,
 * retaining at most `cap` entries.
 */
export function mergeBoards(
  kept: readonly StepBoard[],
  projection: Observation,
  patches: readonly StepBoardPatch[],
  range: SeqRange | null,
  cap = MAX_KEPT_BOARDS,
): StepBoard[] {
  const incoming = patches.length ? stepBoards(projection, patches) : [];
  const batch: StepBoard[] = [];
  if (range !== null && incoming.length > 0 && range.first < incoming[0].first) {
    // Lines past the server's per-Command position cap share the oldest kept board.
    batch.push({
      first: range.first,
      last: incoming[0].first - 1,
      projection: incoming[0].projection,
      fallback: true,
    });
  }
  batch.push(...incoming);
  if (range !== null) {
    const tailFirst = incoming.length > 0 ? incoming.at(-1)!.last + 1 : range.first;
    if (tailFirst <= range.last) {
      batch.push({ first: tailFirst, last: range.last, projection });
    }
  }
  if (batch.length === 0) return kept.length > cap ? kept.slice(-cap) : [...kept];
  const minSeq = batch[0].first;
  const maxSeq = batch.at(-1)!.last;
  const survived = kept.filter(entry => entry.last < minSeq || entry.first > maxSeq);
  const merged = [...survived, ...batch].sort((a, b) => a.first - b.first);
  return merged.length > cap ? merged.slice(-cap) : merged;
}

/** The kept `StepBoard` covering `seq`, or null when that step's board was not kept. */
export function findBoard(boards: readonly StepBoard[], seq: number): StepBoard | null {
  for (let i = boards.length - 1; i >= 0; i -= 1) {
    const board = boards[i];
    if (board.first <= seq && seq <= board.last) return board;
  }
  return null;
}

/** The board the event `seq` was seen on, or null when not kept. */
export function boardAt(boards: readonly StepBoard[], seq: number): Observation | null {
  return findBoard(boards, seq)?.projection ?? null;
}
