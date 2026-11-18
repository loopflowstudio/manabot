import { describe, expect, it } from 'vitest';

import type { GameLogEntry } from '$lib/types';
import { moveList, segments } from './moves';

const players = { you: 'You', opponent: 'Search 64' };
let id = 0;
const entry = (actor: GameLogEntry['actor'], turn: number | null, text: string): GameLogEntry => ({ id: `e${(id += 1)}`, actor, turn, text });

describe('moveList', () => {
  it('groups moves by turn, names their players and owner, and keeps automation separate', () => {
    const owners = new Map([[3, 'you'], [4, 'opponent']] as const);
    const turns = moveList([
      entry('hero', 3, 'Play Mountain'),
      entry('auto', 3, 'Pass priority'),
      entry('auto', 3, 'Pass priority'),
      entry('villain', 4, 'Cast Water Tribe Rallier'),
      entry('auto', 4, 'Pass priority'),
    ], players, owners, ['Mountain', 'Water Tribe Rallier']);
    expect(turns.map(turn => [turn.turn, turn.owner])).toEqual([[3, 'You'], [4, 'Search 64']]);
    expect(turns[0].moves.map(move => [move.who, move.count])).toEqual([['You', 1], ['auto', 2]]);
    expect(turns[1].moves.map(move => move.actor)).toEqual(['opponent', 'auto']);
  });

  it('never folds a deliberate move into automation', () => {
    const turns = moveList([entry('hero', 2, 'Pass priority'), entry('auto', 2, 'Pass priority'), entry('hero', 2, 'Pass priority')],
      players, new Map(), []);
    expect(turns[0].moves.map(move => [move.actor, move.count])).toEqual([['you', 1], ['auto', 1], ['you', 1]]);
  });
});

describe('segments', () => {
  it('marks the longest card name and keeps the rest as text', () => {
    expect(segments('Cast Fire Nation Cadets targeting Forest', ['Fire Nation', 'Fire Nation Cadets', 'Forest'])).toEqual([
      { text: 'Cast ', card: null },
      { text: 'Fire Nation Cadets', card: 'Fire Nation Cadets' },
      { text: ' targeting ', card: null },
      { text: 'Forest', card: 'Forest' },
    ]);
    expect(segments('Pass priority', ['Forest'])).toEqual([{ text: 'Pass priority', card: null }]);
  });
});
