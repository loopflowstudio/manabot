import { describe, expect, it } from 'vitest';

import type { AnalysedDecision, Analysis } from './analysis';
import { agreement, duration, estimateArc, numberRows, numbersLine, winRate, type GameNumbers, type RecordRow } from './numbers';

const seat = (index: number, spells: number, attacks: number, damage: number | null = 3) => ({
  seat: index, lands_played: 4, spells_cast: spells, attacks, cards_drawn: 5, damage_dealt: damage, damage_taken: damage,
});
const game = (damage: number | null = 3): GameNumbers => ({ complete: true, turns: 10, seats: [seat(0, 1, 1, damage), seat(1, 2, 2, damage)] });

function decided(played: number, probabilities: number[], value: number | null): AnalysedDecision {
  return { event: 0, played, annotation: { probabilities, value }, semantic: 'signed_outcome', after: probabilities.map(() => null) };
}
const analysis = (decisions: AnalysedDecision[]): Analysis => ({
  id: 'a', seat: 0, policy: { name: 'Reader' } as Analysis['policy'], status: 'complete', reason: null, total: decisions.length, created_at: '', decisions,
});

describe('the game in numbers', () => {
  it('reads one line from your seat beside the opponent', () => {
    expect(numbersLine(game(), 0, { you: 'You', opponent: 'Heuristic' }))
      .toBe('10 turns · you cast 1 spell and attacked once; Heuristic cast 2 and attacked 2 times');
    expect(numbersLine(game(), 1, { you: 'You', opponent: 'Heuristic' }))
      .toBe('10 turns · you cast 2 spells and attacked 2 times; Heuristic cast 1 and attacked once');
  });

  it('leaves damage out of a game counted before damage was', () => {
    expect(numberRows(game(), 0).map(row => row.label)).toContain('Damage taken');
    expect(numberRows(game(null), 0).map(row => row.label)).toEqual(['Lands played', 'Spells cast', 'Attacks', 'Cards drawn']);
  });

  it('counts the decisions where you chose as the agent would, without grading the rest', () => {
    const read = analysis([decided(1, [0.2, 0.8], 0), decided(0, [0.3, 0.7], 0.5), decided(0, [], null)]);
    expect(agreement(read)).toEqual({ same: 1, of: 2 });
    expect(estimateArc(read)).toEqual({ start: '50', peak: '75', end: '75' });
  });

  it('says how long a game took', () => {
    expect(duration('2026-11-24T10:00:00Z', '2026-11-24T10:00:45Z')).toBe('45 s');
    expect(duration('2026-11-24T10:00:00Z', '2026-11-24T10:12:10Z')).toBe('12 min');
    expect(duration('2026-11-24T10:00:00Z', '2026-11-24T11:05:00Z')).toBe('1 h 05 min');
    expect(duration('2026-11-24T10:00:00Z', null)).toBeNull();
  });

  it('gives a win rate with its interval', () => {
    const row = { games: 10, wins: 7, interval: [0.397, 0.892] } as RecordRow;
    expect(winRate(row)).toBe('70% (40–89%)');
  });
});
