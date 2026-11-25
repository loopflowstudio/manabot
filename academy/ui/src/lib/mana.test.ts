import { describe, expect, it } from 'vitest';

import { manaRuns, pooledManaTerms, spendPhrase } from './mana';

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

describe('restricted mana wording', () => {
  it('says "only" for a positive class and not for an exclusion', () => {
    expect(spendPhrase('Lesson spells')).toBe('only for Lesson spells');
    expect(spendPhrase('anything but spells from hand')).toBe('for anything but spells from hand');
    expect(pooledManaTerms({ mana: [0, 0, 0, 0, 0, 1], restriction: 'anything but spells from hand', combat: false }))
      .toBe('for anything but spells from hand, normally expires at step end');
  });
});
