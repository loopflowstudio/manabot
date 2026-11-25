import { describe, expect, it } from 'vitest';

import { pooledManaText, spendPhrase } from './mana';

describe('restricted mana wording', () => {
  it('says "only" for a positive class and not for an exclusion', () => {
    expect(spendPhrase('Lesson spells')).toBe('only for Lesson spells');
    expect(spendPhrase('anything but spells from hand')).toBe('for anything but spells from hand');
    expect(pooledManaText({ mana: [0, 0, 0, 0, 0, 1], restriction: 'anything but spells from hand', combat: false } as Parameters<typeof pooledManaText>[0]))
      .toBe('1 {C} for anything but spells from hand, normally expires at step end');
  });
});
