import { describe, expect, it } from 'vitest';
import { manaRuns } from './mana';

describe('manaRuns', () => {
  it('draws each run of braced symbols as one mana run', () => {
    expect(manaRuns('Pay {2}{R} for Firebending Lesson')).toEqual([
      { text: 'Pay ', mana: false }, { text: '{2}{R}', mana: true }, { text: ' for Firebending Lesson', mana: false },
    ]);
  });
  it('leaves words without mana as they are', () => {
    expect(manaRuns('No blocks')).toEqual([{ text: 'No blocks', mana: false }]);
    expect(manaRuns('{G}')).toEqual([{ text: '{G}', mana: true }]);
  });
});
