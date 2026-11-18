/**
 * The person's board after each step of the opponent's turn, so each beat is
 * shown on the board it happened on.
 *
 * The server sends them beside a frame as patches chained back from the
 * frame's projection (`managym/academy/step_boards.py` defines the format):
 * the newest is its patch applied to the frame's board, each older one its
 * patch applied to the board after it. `stepBoards` rebuilds them once when
 * the frame arrives; `boardAt` names the board a presentation event was seen
 * on, or null when that is the frame's own board.
 */
import type { Observation, StepBoardPatch } from '$lib/types';

export interface StepBoard {
  /** The presentation events (seqs) this board is the board after. */
  first: number;
  last: number;
  projection: Observation;
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

/** The board the event `seq` was seen on, or null for the frame's own board. */
export function boardAt(boards: readonly StepBoard[], seq: number): Observation | null {
  return boards.find(board => board.first <= seq && seq <= board.last)?.projection ?? null;
}
