import { describe, expect, it } from 'vitest';
import type { Observation } from '$lib/types';
import { applyPatch, boardAt, findBoard, MAX_KEPT_BOARDS, mergeBoards, stepBoards } from './boards';

const board = (battlefield: { id: number; name: string; tapped: boolean }[], step = 'MAIN'): Observation =>
  ({ turn: { step }, agent: { battlefield }, opponent: { battlefield: [] } }) as unknown as Observation;

describe('step boards', () => {
  it('rebuild each board back from the frame, oldest first', () => {
    // The frame: the Ogre died in combat. Before that it attacked (tapped); before that it was untapped.
    const frame = board([{ id: 2, name: 'Elf', tapped: false }], 'END');
    const attacking = board([{ id: 1, name: 'Ogre', tapped: true }, { id: 2, name: 'Elf', tapped: false }], 'COMBAT');
    const declared = board([{ id: 1, name: 'Ogre', tapped: false }, { id: 2, name: 'Elf', tapped: false }], 'MAIN');
    const boards = stepBoards(frame, [
      { first: 3, last: 3, patch: { o: { turn: { o: { step: { '=': 'MAIN' } } }, agent: { o: { battlefield: { a: [{ k: 1, p: { o: { tapped: { '=': false } } } }, { k: 2 }] } } } } } },
      { first: 4, last: 6, patch: { o: { turn: { o: { step: { '=': 'COMBAT' } } }, agent: { o: { battlefield: { a: [{ '=': { id: 1, name: 'Ogre', tapped: true } }, { k: 2 }] } } } } } },
    ]);
    expect(boards.map(b => b.projection)).toEqual([declared, attacking]);
    // An unchanged item is the frame's own object, not a copy.
    expect(boards[1].projection.agent.battlefield[1]).toBe(frame.agent.battlefield[0]);
    expect(boardAt(boards, 5)).toEqual(attacking);
    expect(boardAt(boards, 3)).toEqual(declared);
    // Events no board covers return null.
    expect(boardAt(boards, 7)).toBeNull();
  });

  it('keeps boards across frames up to the 512 cap and marks capped prefixes as fallback', () => {
    const first = board([{ id: 1, name: 'Plains', tapped: false }], 'MAIN');
    const second = board([{ id: 1, name: 'Plains', tapped: true }], 'COMBAT');
    const third = board([{ id: 1, name: 'Plains', tapped: true }, { id: 2, name: 'Ally', tapped: false }], 'END');

    let kept = mergeBoards([], first, [], { first: 1, last: 2 });
    expect(boardAt(kept, 1)).toBe(first);
    expect(boardAt(kept, 2)).toBe(first);

    // Next update arrives with a patch for seq 5..5 and range 3..6:
    // seq 3..4 fell before the patch's first (per-Command cap) and fall back to seq 5's board.
    kept = mergeBoards(
      kept,
      third,
      [{ first: 5, last: 5, patch: { o: { turn: { o: { step: { '=': 'COMBAT' } } }, agent: { o: { battlefield: { a: [{ k: 1 }] } } } } } }],
      { first: 3, last: 6 },
    );
    expect(boardAt(kept, 1)).toBe(first);
    expect(findBoard(kept, 3)).toMatchObject({ first: 3, last: 4, fallback: true, projection: second });
    expect(findBoard(kept, 5)).toMatchObject({ first: 5, last: 5, projection: second });
    expect(findBoard(kept, 5)?.fallback).toBeUndefined();
    expect(boardAt(kept, 6)).toBe(third);

    // Enforces the 512 cap by dropping oldest boards first.
    let many = kept;
    for (let i = 0; i < MAX_KEPT_BOARDS + 10; i += 1) {
      const seq = 100 + i;
      many = mergeBoards(many, board([{ id: seq, name: `Card ${seq}`, tapped: false }]), [], { first: seq, last: seq });
    }
    expect(many).toHaveLength(MAX_KEPT_BOARDS);
    expect(boardAt(many, 1)).toBeNull();
    expect(boardAt(many, 100 + MAX_KEPT_BOARDS + 9)).not.toBeNull();
  });

  it('remove keys and replace values', () => {
    expect(applyPatch({ a: 1, b: [1, 2], c: 3 }, { o: { b: { '=': [2] } }, x: ['c'] })).toEqual({ a: 1, b: [2] });
  });

  it('refuse a patch that names an item the board does not have', () => {
    expect(() => stepBoards(board([]), [{ first: 0, last: 0, patch: { o: { agent: { o: { battlefield: { a: [{ k: 9 }] } } } } } }])).toThrow();
  });
});
