import { describe, expect, it } from 'vitest';
import { battlefieldPageLayout, groupBattlefield } from './battlefield-layout';
import type { CardTypes, PermanentState } from './types';

function permanent(id: number, types?: Partial<CardTypes>, token = false): PermanentState {
  return { id, name: 'Test permanent', controller_id: 0, tapped: false, damage: 0,
    summoning_sick: false, power: 0, toughness: 0, base_power: 0,
    base_toughness: 0, plus1_counters: 0, types: types ? {
    is_creature: false, is_land: false, is_spell: false, is_artifact: false,
    is_enchantment: false, is_planeswalker: false, is_battle: false, ...types,
  } : undefined, is_token: token };
}

describe('battlefield lanes', () => {
  it('preserves every object once, with creatures ahead of land/artifact classification', () => {
    const cards = [permanent(1, { is_land: true }),
      permanent(2, { is_land: true, is_creature: true }),
      permanent(3, { is_artifact: true }, true),
      permanent(4, { is_artifact: true, is_creature: true }, true),
      permanent(5, { is_enchantment: true }), permanent(6)];
    const groups = groupBattlefield(cards);
    expect(groups.creatures.map(card => card.id)).toEqual([2, 4]);
    expect(groups.lands.map(card => card.id)).toEqual([1]);
    expect(groups.tokens.map(card => card.id)).toEqual([3]);
    expect(groups.other.map(card => card.id)).toEqual([5, 6]);
    expect(Object.values(groups).flat().map(card => card.id).sort()).toEqual([1, 2, 3, 4, 5, 6]);
  });
  it('compacts and then paginates instead of shrinking below a usable target', () => {
    const sparse = battlefieldPageLayout(500, 120, 2);
    const dense = battlefieldPageLayout(500, 120, 100);
    expect(sparse.cardWidth).toBeGreaterThan(dense.cardWidth);
    expect(dense.cardWidth).toBe(44);
    expect(dense.capacity).toBeGreaterThan(0);
    expect(dense.capacity).toBeLessThan(100);
    const corner = battlefieldPageLayout(90, 100, 1);
    expect(corner.cardWidth * 1.4).toBeLessThanOrEqual(90);
  });
});
