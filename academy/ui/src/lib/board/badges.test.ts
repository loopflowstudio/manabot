import { describe, expect, it } from 'vitest';
import type { PermanentState } from '$lib/types';
import { badgeReading, badgesOf, fitBadges, type Badge } from './badges';

function creature(fields: Partial<PermanentState> = {}): PermanentState {
  return { id: 1, name: 'Soldier', controller_id: 0, tapped: false, damage: 0, summoning_sick: false,
    power: 1, toughness: 1, base_power: 1, base_toughness: 1, plus1_counters: 0, ...fields };
}
const board = (p: PermanentState) => badgesOf(p).filter(b => !b.inspectOnly);
const face = { cost: '', type_line: 'Creature', power: 1, toughness: 1, colors: [], keywords: ['flash', 'defender'] };

describe('effective battlefield state', () => {
  it('uses public frame details even with no face, or a same-name face with other abilities', () => {
    const p = creature({ keywords: ['menace', 'flash'], keyword_grants: { menace: { source: 'Sokka' } },
      keyword_abilities: [{ label: 'Prowess', granted: true, source: 'Sokka' }] });
    expect(badgesOf(p, face)).toEqual(badgesOf(p, null));
    expect(board(p).map(b => [b.label, b.granted])).toEqual([['Menace', true], ['Prowess', true]]);
    expect(board(p).map(badgeReading)).toEqual(['Menace · Sokka', 'Prowess · Sokka']);
    expect(badgesOf(p).map(badgeReading)).toContain('Flash');
    expect(board(creature())).toEqual([]);
    expect(board(creature({ keyword_abilities: [{ label: 'Firebending 1' }] })).map(b => b.label)).toEqual(['Firebending 1']);
  });

  it('explains overridden Defender, and never restores a lost or hidden printed ability', () => {
    const bear = creature({ keywords: ['defender'], keyword_grants: {} });
    expect(board(bear).map(badgeReading)).toEqual(['Defender']);
    expect(board({ ...bear, attacks_despite_defender: true }).map(badgeReading)).toEqual(['Defender · can attack anyway']);
    expect(badgesOf(creature({ abilities_lost: true, face_down: true }), face).map(badgeReading)).toEqual(['No abilities', 'Untapped']);
    const grant = creature({ abilities_lost: true, keywords: ['flying'], keyword_grants: { flying: { effect_timestamp: 4 } },
      effects: [{ modification: 'LoseAllAbilities', duration: 'EndOfTurn', source: 'Frogify', timestamp: 1 },
        { modification: 'AddKeywords', keywords: ['flying'], duration: 'EndOfTurn', source: 'Yip Yip!', timestamp: 4 }] });
    expect(board(grant).map(badgeReading)).toEqual(['Lost base abilities · Frogify, until end of turn', 'Flying · Yip Yip!, until end of turn']);
  });

  it('preserves distinct Ward costs, source and duration, and repeated Prowess', () => {
    const p = creature({ keyword_abilities: [
      { label: 'Prowess' }, { label: 'Prowess' }, { label: 'Prowess', granted: true },
      { label: 'Ward {2}' }, { label: 'Ward—Pay 7 life' }, { label: 'Ward—Discard a card', granted: true, effect_timestamp: 6 },
    ], effects: [{ modification: 'GrantStatic', duration: 'UntilYourNextTurn', source: 'Shield', timestamp: 6 }] });
    expect(board(p).map(badgeReading)).toEqual(['Prowess ×2', 'Prowess · Another ability', 'Ward {2}', 'Ward—Pay 7 life',
      'Ward—Discard a card · Shield, until your next turn']);
  });

  it('shows the historical frame’s effective keywords, recorded restrictions and Firebending grant', () => {
    const p = creature({ granted_firebending: 3, keywords: ['flying'], effects: [
      { modification: 'Saddled', duration: 'EndOfTurn', source: 'Guidelight Pathmaker', timestamp: 1 },
      { modification: 'Firebending', amount: 3, duration: 'EndOfTurn', source: 'Fire Lord', timestamp: 2 },
    ] });
    expect(board(p).map(badgeReading)).toEqual(['Saddled · Guidelight Pathmaker, until end of turn', 'Flying',
      'Firebending 3 · Fire Lord, until end of turn']);
  });

  it('keeps restrictions once, in priority order, and combat roles in Inspect and speech', () => {
    const p = creature({ damage: 2, summoning_sick: true, first_strike_participant: true, plus1_counters: 2,
      blocking: 0, attacking: true, entered_this_turn: true, attacked_this_turn: true, counters: { charge: 1 },
      restrictions: [{ kind: 'cant_be_blocked' }, { kind: 'cant_block' }, { kind: 'cant_attack_or_block' },
        { kind: 'doesnt_untap', source: 'Watery Grasp' }, { kind: 'must_attack' }],
      effects: [{ modification: 'DoesntUntap', duration: 'Indefinite', source: 'Frost', timestamp: 1 }],
      keywords: ['first_strike', 'flash', 'future_keyword'] });
    expect(board(p).map(b => b.label)).toEqual(['2 damage', 'Summoning sick', "Can't be blocked", "Can't block",
      "Can't attack or block", "Doesn't untap", 'Must attack', 'Struck first', '+2/+2', 'Charge ×1', 'First strike', 'Future keyword']);
    expect(badgesOf(p, null, 'Badgermole').map(badgeReading)).toEqual(expect.arrayContaining([
      "Doesn't untap · Watery Grasp · Frost", "Must attack", 'first-strike participant', 'blocking Badgermole', 'attacking', 'Flash', 'entered this turn', 'attacked this turn',
    ]));
  });

  it('names a keyword counter as the source of its keyword, on one chip', () => {
    const mai = creature({ counters: { double_strike: 1 }, keywords: ['double_strike'], keyword_grants: { double_strike: { counter: true } },
      keyword_abilities: [{ label: 'Prowess' }] });
    expect(board(mai).map(b => [badgeReading(b), b.granted])).toEqual([['Double strike · 1 double strike counter', true], ['Prowess', undefined]]);
    expect(badgesOf(mai).map(badgeReading)).toContain('1 double strike counter');
    const printed = board({ ...mai, keyword_grants: {} })[0];
    expect(badgeReading(printed)).toBe('Double strike');
    expect(printed.granted).toBe(false);
    const shocked = creature({ counters: { stun: 1 } });
    expect(board(shocked).map(b => b.label)).toEqual(['Stun ×1']);
  });
});

describe('whole-chip fitting', () => {
  const labels = (...values: string[]): Badge[] => values.map(label => ({ label, kind: 'ability' }));
  const measure = (s: string) => s.length * 8 + 10;
  it('keeps a shorter ability beside overflow when a longer ability would displace it', () => {
    const badges = labels('First strike', 'Prowess', 'Ward {2}');
    const result = fitBadges(badges, [108], measure);
    expect(result.rows.map(row => row.map(chip => chip.label))).toEqual([['Prowess', '+2']]);
    expect(result.hidden.map(b => b.label)).toEqual(['First strike', 'Ward {2}']);
  });
  it('reserves space for the actual number of hidden chips', () => {
    const badges = labels('AA', 'BB', ...Array(9).fill('Exceeds width'));
    const result = fitBadges(badges, [60], measure);
    expect(result.rows[0].map(chip => chip.label)).toEqual(['+11']);
    expect(result.hidden).toEqual(badges);
  });
  it('places two full labels on a turned card and compacts only at narrow full widths', () => {
    const tyLee = board(creature({ restrictions: [{ kind: 'doesnt_untap' }], keyword_abilities: [{ label: 'Prowess' }] }));
    const textWidth = (s: string) => 6 * s.length + 10;
    expect(fitBadges(tyLee, [106, 70], textWidth).rows.map(row => row.map(c => c.label)))
      .toEqual([["Doesn't untap"], ['Prowess']]);
    const cadets = board(creature({ keyword_abilities: [{ label: 'Firebending 2' }] }));
    expect(fitBadges(cadets, [74, 42], textWidth).rows[0][0].label).toBe('Firebend 2');
    expect(fitBadges(cadets, [74], textWidth).rows[0][0].label).toBe('Firebend 2');
    expect(fitBadges(cadets, [106], textWidth).rows[0][0].label).toBe('Firebending 2');
  });
  it('fits the spiral at 16px; the name takes overflow when the plate fills the last row', () => {
    const sick: Badge = { kind: 'sick', label: 'Summoning sick' };
    const badges = [sick, ...labels('Ward {2}'), { kind: 'ability', label: 'Flash', inspectOnly: true } satisfies Badge];
    expect(fitBadges(badges, [74, 74], measure).rows.map(row => row.map(c => c.badge))).toEqual([[sick], [badges[1]]]);
    for (const widths of [[], [10]]) {
      expect(fitBadges(badges, widths, measure)).toEqual({ rows: [], hidden: badges.slice(0, 2) });
    }
  });
});
