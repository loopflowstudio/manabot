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

  it('distinguishes step, end-of-combat, and end-of-turn mana expiry', () => {
    expect(pooledManaTerms({ mana: [0, 0, 0, 3, 3, 0], combat: false, expiry: 'end_of_turn' }))
      .toBe('normally expires at end of turn');
    expect(pooledManaTerms({ mana: [0, 0, 0, 2, 0, 0], combat: true, expiry: 'end_of_combat' }))
      .toBe('normally expires at end of combat');
    expect(pooledManaTerms({ mana: [0, 0, 0, 2, 0, 0], combat: true }))
      .toBe('normally expires at end of combat');
    expect(pooledManaTerms({ mana: [0, 0, 0, 0, 1, 0], restriction: 'creature spells', combat: false, expiry: 'step' }))
      .toBe('only for creature spells, normally expires at step end');
  });
});
