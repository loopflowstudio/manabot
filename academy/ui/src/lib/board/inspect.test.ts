import { describe, expect, it } from 'vitest';
import type { PermanentState } from '$lib/types';
import type { CardFace } from '$lib/card-faces.svelte';
import { keywordsOf, labelText, permissionTerms, ptProvenance, signed, statusesOf } from './inspect';

const face: CardFace = { cost: '{G}{W}', colors: ['G', 'W'], type_line: 'Creature — Human Soldier Ally', keywords: ['trample'], power: 1, toughness: 1 };
function permanent(extra: Partial<PermanentState>): PermanentState {
  return {
    id: 1, name: "Earth King's Lieutenant", controller_id: 0, tapped: false, damage: 0, summoning_sick: false,
    power: 1, toughness: 1, base_power: 1, base_toughness: 1, plus1_counters: 0, ...extra,
  };
}

describe('ptProvenance', () => {
  it('sums printed P/T, counters and named effects, and names what they leave unexplained', () => {
    const lieutenant = permanent({
      power: 6, toughness: 4, plus1_counters: 2,
      effects: [{ modification: 'ModifyPt', power: 2, toughness: 0, duration: 'EndOfTurn', source: 'Fire Nation Cadets', timestamp: 4 }],
    });
    expect(ptProvenance(lieutenant, face)?.parts).toEqual([
      { label: 'Printed', power: 1, toughness: 1, base: true },
      { label: '2 +1/+1 counters', power: 2, toughness: 2, base: false },
      { label: 'Fire Nation Cadets, until end of turn', power: 2, toughness: 0, base: false },
      { label: 'Other effects', power: 1, toughness: 1, base: false },
    ]);
  });

  it('starts from the latest base-setting effect instead of the printed card', () => {
    const plains = permanent({
      name: 'Plains', power: 2, toughness: 2, plus1_counters: 2,
      effects: [
        { modification: 'BecomeCreature', duration: 'Indefinite', source: 'Badgermole Cub', timestamp: 98 },
        { modification: 'SetBasePt', power: 0, toughness: 0, duration: 'Indefinite', source: 'Badgermole Cub', timestamp: 98 },
      ],
    });
    expect(ptProvenance(plains, null)?.parts).toEqual([
      { label: 'Set by Badgermole Cub', power: 0, toughness: 0, base: true },
      { label: '2 +1/+1 counters', power: 2, toughness: 2, base: false },
    ]);
  });

  it('does not invent a remainder when the printed values are unknown', () => {
    expect(ptProvenance(permanent({ power: 4, toughness: 4 }), null)?.parts).toEqual([]);
    expect(ptProvenance(permanent({ power: null, toughness: null }), face)).toBeNull();
  });

  it('gives a noncreature no power and toughness, though the projection reports 0/0', () => {
    const types = { is_creature: false, is_land: true, is_spell: false, is_artifact: false, is_enchantment: false, is_planeswalker: false, is_battle: false };
    expect(ptProvenance(permanent({ name: 'Island', power: 0, toughness: 0, types }), null)).toBeNull();
  });
});

describe('keywordsOf', () => {
  it("takes which keywords it has from the projection, and says why", () => {
    const granted = permanent({ keywords: ['flying', 'trample', 'vigilance'], effects: [
      { modification: 'AddKeywords', keywords: ['flying'], duration: 'EndOfTurn', source: 'Yip Yip!', timestamp: 2 },
    ] });
    // Field order from the projection; vigilance is granted by an ability the effect records do not name.
    expect(keywordsOf(granted, face)).toEqual([
      { keyword: 'flying', source: 'Yip Yip!, until end of turn' },
      { keyword: 'trample', source: null },
      { keyword: 'vigilance', source: 'Another ability' },
    ]);
  });

  it('shows no printed keyword the projection says it lost', () => {
    expect(keywordsOf(permanent({ abilities_lost: true }), face)).toEqual([]);
    expect(keywordsOf(permanent({ effects: [{ modification: 'LoseKeywords', keywords: ['trample'], duration: 'EndOfTurn', timestamp: 3 }] }), face)).toEqual([]);
  });
});

it('gives a saddled Mount its status in words', () => {
  expect(statusesOf(permanent({ effects: [
    { modification: 'Saddled', duration: 'EndOfTurn', source: 'Guidelight Pathmaker', timestamp: 7 },
    { modification: 'ModifyPt', power: 1, toughness: 1, duration: 'EndOfTurn', timestamp: 8 },
  ] }))).toEqual(['Saddled until end of turn']);
  expect(statusesOf(permanent({}))).toEqual([]);
});

it('signs P/T changes as rules text does', () => {
  expect(signed(2, 0)).toBe('+2/+0');
  expect(signed(-1, -1)).toBe('−1/−1');
});

describe('labelText', () => {
  it('reads a keyword in sentence case', () => {
    expect(labelText('first_strike')).toBe('First strike');
    expect(labelText('flying')).toBe('Flying');
  });
});

it('names who can cast before a source changes control', () => {
  const grant = { player_id: 1, source_permanent_id: 12 };
  const source = (id: number) => id === 12 ? 'Blossom-Blessed Angel' : 'other permanent';
  const player = (id: number) => id === 1 ? 'Alice' : 'Bob';
  expect(permissionTerms({ cast_permissions: [grant, { ...grant, follows_source_controller: true }] }, source, player)).toEqual([
    'Alice may cast this while they control Blossom-Blessed Angel.',
    'The controller of Blossom-Blessed Angel may cast this (currently Alice).',
  ]);
});
