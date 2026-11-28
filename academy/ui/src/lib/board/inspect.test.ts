import { describe, expect, it } from 'vitest';
import type { PermanentState } from '$lib/types';
import type { CardFace } from '$lib/card-faces.svelte';
import { counterTerms, labelText, permissionTerms, ptProvenance, signed } from './inspect';

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
  expect(permissionTerms({ cast_permission: { ...grant, flashback: true, this_turn: true } }, source, player)).toEqual([
    'Alice may cast this while they control Blossom-Blessed Angel.',
    'Alice may cast this with flashback this turn. Exile it when it leaves the stack.',
  ]);
});

it('lists every term of a grant, playing only with a play grant', () => {
  const player = (id: number) => id === 1 ? 'Alice' : 'Bob';
  expect(permissionTerms({
    cast_permissions: [
      { player_id: 1, cost: '{1}{R}', face_down: true, play_lands: true, condition: 'you control a Wizard' },
      { player_id: 1, cost: '{2}{R}', foretold: true, after_turn: 3 },
      { player_id: 1, cost: '{3}', own_turn_only: true, flash: true, any_mana: true, prepared: true },
      { player_id: 1, play_lands: true, source_permanent_id: 12 },
      { player_id: 1 },
    ],
  }, () => 'source', player)).toEqual([
    'Alice may play this for {1}{R}, face down, seen only while it may be played, only if you control a Wizard.',
    'Alice may cast this for {2}{R}, only after turn 3, foretold, face down.',
    'Alice may cast this for {3}, during their own turn only, as though it had flash, spending mana of any type, as a prepared copy kept while linked to its source.',
    'Alice may play this while they control source.',
    'Alice may cast this.',
  ]);
});

describe('counter terms', () => {
  it('names each positive counter kind, singular or plural', () => {
    expect(counterTerms({ stash: 1, double_strike: 2, stun: 0 })).toEqual(['1 stash counter', '2 double strike counters']);
    expect(counterTerms(undefined)).toEqual([]);
  });
});
