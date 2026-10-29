import { describe, expect, it } from 'vitest';
import { permanentNotes } from './permanent-notes';
import type { PermanentState } from './types';

function permanent(id: number, name: string, extra: Partial<PermanentState> = {}): PermanentState {
  return { id, name, controller_id: 0, tapped: false, damage: 0, summoning_sick: false, power: 2, toughness: 2,
    base_power: 2, base_toughness: 2, plus1_counters: 0, ...extra };
}

describe('permanent notes', () => {
  it('adds nothing for a permanent without effect-derived state', () => {
    const bear = permanent(1, 'Bear');
    expect(permanentNotes(bear, [bear])).toEqual([]);
  });

  it('names counters by kind beside +1/+1', () => {
    const bear = permanent(1, 'Bear', { plus1_counters: 1, counters: { stun: 2, double_strike: 1 } });
    expect(permanentNotes(bear, [bear])).toEqual(['2 stun counters', '1 double strike counter']);
  });

  it('states an attachment on both the attachment and its host, across seats', () => {
    const staff = permanent(2, 'Glider Staff', { attached_to: 9, power: null, toughness: null });
    const host = permanent(9, 'Bear', { controller_id: 1 });
    const twin = permanent(10, 'Bear', { controller_id: 1 });
    const table = [staff, host, twin];
    expect(permanentNotes(staff, table)).toEqual(['attached to Bear (copy 1 of 2)']);
    expect(permanentNotes(host, table)).toEqual(['Glider Staff attached']);
    expect(permanentNotes(twin, table)).toEqual([]);
  });

  it('states set creature types, a granted land and firebending', () => {
    const citizen = permanent(3, 'Citizen', { creature_types: ['Citizen'] });
    const land = permanent(4, 'Badgermole', { is_granted_land: true });
    const zuko = permanent(5, 'Zuko', { firebending: 2 });
    expect(permanentNotes(citizen, [citizen])).toEqual(['creature types set to Citizen']);
    expect(permanentNotes(land, [land])).toEqual(['land through an effect']);
    expect(permanentNotes(zuko, [zuko])).toEqual(['firebending 2']);
  });
});
