import { describe, expect, it } from 'vitest';
import type { Observation } from '$lib/types';
import { applyPatch, boardAt, stepBoards } from './boards';

const board = (battlefield: { id: number; name: string; tapped: boolean }[], step = 'MAIN'): Observation =>
  ({ turn: { step }, agent: { battlefield }, opponent: { battlefield: [] } }) as unknown as Observation;

describe('step boards', () => {
  it('rebuild each board back from the frame, oldest first', () => {
    // The frame: the Ogre died in combat. Before that it attacked (tapped); before that it was untapped.
    const frame = board([{ id: 2, name: 'Elf', tapped: false }], 'END');
    const attacking = board([{ id: 1, name: 'Ogre', tapped: true }, { id: 2, name: 'Elf', tapped: false }], 'COMBAT');
    const declared = board([{ id: 1, name: 'Ogre', tapped: false }, { id: 2, name: 'Elf', tapped: false }], 'MAIN');
    const boards = stepBoards(frame, [
      { first: 3, last: 3, patch: { o: { turn: { o: { step: { '=': 'MAIN' } } } , agent: { o: { battlefield: { a: [{ k: 1, p: { o: { tapped: { '=': false } } } }, { k: 2 }] } } } } } },
      { first: 4, last: 6, patch: { o: { turn: { o: { step: { '=': 'COMBAT' } } }, agent: { o: { battlefield: { a: [{ '=': { id: 1, name: 'Ogre', tapped: true } }, { k: 2 }] } } } } } },
    ]);
    expect(boards.map(b => b.projection)).toEqual([declared, attacking]);
    // An unchanged item is the frame's own object, not a copy.
    expect(boards[1].projection.agent.battlefield[1]).toBe(frame.agent.battlefield[0]);
    expect(boardAt(boards, 5)).toEqual(attacking);
    expect(boardAt(boards, 3)).toEqual(declared);
    // Events no board covers were seen on the frame's own board.
    expect(boardAt(boards, 7)).toBeNull();
  });

  it('remove keys and replace values', () => {
    expect(applyPatch({ a: 1, b: [1, 2], c: 3 }, { o: { b: { '=': [2] } }, x: ['c'] })).toEqual({ a: 1, b: [2] });
  });

  it('refuse a patch that names an item the board does not have', () => {
    expect(() => stepBoards(board([]), [{ first: 0, last: 0, patch: { o: { agent: { o: { battlefield: { a: [{ k: 9 }] } } } } } }])).toThrow();
  });
});
