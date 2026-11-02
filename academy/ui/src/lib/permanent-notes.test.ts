import { describe, expect, it } from 'vitest';
import { effectNote, permanentNotes } from './permanent-notes';
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
    const zuko = permanent(5, 'Zuko', { granted_firebending: 2 });
    expect(permanentNotes(citizen, [citizen])).toEqual(['creature types set to Citizen']);
    expect(permanentNotes(land, [land])).toEqual(['land through an effect']);
    expect(permanentNotes(zuko, [zuko])).toEqual(['firebending 2 until end of turn']);
  });

  it('states a copy effect and its exceptions in the LLM text words', () => {
    const copy = permanent(6, 'Ran and Shaw', {
      copy_of: {
        name: 'Ran and Shaw',
        exceptions: [{ modification: 'Nonlegendary', duration: 'Indefinite', timestamp: 6 }],
      },
    });
    const airship = permanent(7, 'Phoenix Fleet Airship', { copy_of: { name: 'Phoenix Fleet Airship' } });
    expect(permanentNotes(copy, [copy])).toEqual(["copy of Ran and Shaw, except isn't legendary"]);
    expect(permanentNotes(airship, [airship])).toEqual(['copy of Phoenix Fleet Airship']);
  });

  it('states combat assignments, lost abilities and spent activation limits', () => {
    const attacker = permanent(6, 'Bear', { attacking: true });
    const blocker = permanent(7, 'Wall', { controller_id: 1, blocking: 6 });
    const blank = permanent(8, 'Turtle', { abilities_lost: true, used_abilities: [0, 2] });
    expect(permanentNotes(attacker, [attacker, blocker])).toEqual(['attacking']);
    expect(permanentNotes(blocker, [attacker, blocker])).toEqual(['blocking Bear']);
    expect(permanentNotes(blank, [blank])).toEqual(['has lost all abilities', 'used-up abilities 1, 3']);
  });

  it('states each continuous-effect record with its duration and source, in order', () => {
    const ogre = permanent(9, 'Ogre', {
      effects: [
        { modification: 'LoseAllAbilities', duration: 'EndOfTurn', source: 'Day of Black Sun', timestamp: 3 },
        { modification: 'ModifyPt', power: 2, toughness: 0, duration: 'EndOfTurn', timestamp: 4 },
        { modification: 'AddKeywords', keywords: ['flying', 'first_strike'], duration: 'EndOfTurn', timestamp: 5 },
        { modification: 'BecomeArtifactCreature', duration: 'Indefinite', timestamp: 6 },
      ],
    });
    expect(permanentNotes(ogre, [ogre])).toEqual([
      'until end of turn: loses all abilities (Day of Black Sun)',
      'until end of turn: +2/+0',
      'until end of turn: flying, first strike',
      'is an artifact creature',
    ]);
    expect(
      effectNote({ modification: 'KeepUnspentMana', color: 'Red', duration: 'EndOfTurn', timestamp: 1 }),
    ).toBe('until end of turn: keeps unspent red mana');
  });
});
