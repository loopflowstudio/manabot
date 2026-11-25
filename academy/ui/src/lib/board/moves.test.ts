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
    expect(turns[0].moves.map(move => [move.who, move.passes])).toEqual([['You', null], ['', [{ actor: 'auto', who: 'auto', count: 2 }]]]);
    expect(turns[1].moves.map(move => move.actor)).toEqual(['opponent', 'note']);
  });

  it("folds a run of the opponent's and automation's passes into one quiet line that counts each", () => {
    const turns = moveList([
      entry('villain', 5, 'Pass priority'), entry('auto', 5, 'Pass priority'), entry('villain', 5, 'Pass priority'),
      entry('auto', 5, 'Pass priority'), entry('villain', 5, 'Pass priority'), entry('villain', 5, 'Cast Kyoshi Warriors'),
      entry('villain', 5, 'Pass priority'),
    ], players, new Map(), []);
    expect(turns[0].moves.map(move => move.passes ?? move.segments.map(segment => segment.text).join(''))).toEqual([
      [{ actor: 'opponent', who: 'Search 64', count: 3 }, { actor: 'auto', who: 'auto', count: 2 }],
      'Cast Kyoshi Warriors',
      [{ actor: 'opponent', who: 'Search 64', count: 1 }],
    ]);
  });

  it('keeps the person\'s own passes as moves, never folded into automation', () => {
    const turns = moveList([entry('hero', 2, 'Pass priority'), entry('auto', 2, 'Pass priority'), entry('hero', 2, 'Pass priority')],
      players, new Map(), []);
    expect(turns[0].moves.map(move => [move.actor, move.passes?.length ?? 0])).toEqual([['you', 0], ['note', 1], ['you', 0]]);
  });

  it('collapses repeated automated and opponent moves, never the person\'s', () => {
    const turns = moveList([entry('auto', 2, 'Tap Island'), entry('auto', 2, 'Tap Island'),
      entry('villain', 2, 'Opponent decision'), entry('villain', 2, 'Opponent decision'),
      entry('hero', 2, 'Tap Island'), entry('hero', 2, 'Tap Island')], players, new Map(), []);
    expect(turns[0].moves.map(move => [move.actor, move.count])).toEqual([['auto', 2], ['opponent', 2], ['you', 1], ['you', 1]]);
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

  it('keeps mana symbols outside card-name segments so Moves renders them as pips', () => {
    expect(segments('Tap Kyoshi Village for {W}', ['Kyoshi Village'])).toEqual([
      { text: 'Tap ', card: null },
      { text: 'Kyoshi Village', card: 'Kyoshi Village' },
      { text: ' for {W}', card: null },
    ]);
    expect(segments('Tap Plains for {W}', ['Plains'])).toEqual([
      { text: 'Tap ', card: null },
      { text: 'Plains', card: 'Plains' },
      { text: ' for {W}', card: null },
    ]);
  });
});
