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
    const p = creature({ keywords: ['menace', 'flash'], keyword_grants: { menace: null },
      keyword_abilities: [{ label: 'Prowess', granted: true }] });
    expect(badgesOf(p, face)).toEqual(badgesOf(p, null));
    expect(board(p).map(b => [b.label, b.granted])).toEqual([['Menace', true], ['Prowess', true]]);
    expect(badgesOf(p).map(badgeReading)).toContain('Flash');
    expect(board(creature())).toEqual([]);
    expect(board(creature({ keyword_abilities: [{ label: 'Firebending 1' }] })).map(b => b.label)).toEqual(['Firebending 1']);
  });

  it('explains overridden Defender, and never restores a lost or hidden printed ability', () => {
    const bear = creature({ keywords: ['defender'], keyword_grants: {} });
    expect(board(bear).map(badgeReading)).toEqual(['Defender']);
    expect(board({ ...bear, attacks_despite_defender: true }).map(badgeReading)).toEqual(['Defender · can attack anyway']);
    expect(badgesOf(creature({ abilities_lost: true, face_down: true }), face).map(badgeReading)).toEqual(['No abilities', 'Untapped']);
    const grant = creature({ abilities_lost: true, keywords: ['flying'], keyword_grants: { flying: 4 },
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
      restrictions: ['cant_be_blocked', 'cant_block', 'cant_attack_or_block', 'doesnt_untap', 'must_attack'],
      effects: [{ modification: 'DoesntUntap', duration: 'Indefinite', source: 'Frost', timestamp: 1 }],
      keywords: ['first_strike', 'flash', 'future_keyword'] });
    expect(board(p).map(b => b.label)).toEqual(['2 damage', 'Summoning sick', "Can't be blocked", "Can't block",
      "Can't attack or block", "Doesn't untap", 'Must attack', 'Struck first', '+2/+2', 'Charge ×1', 'First strike', 'Future keyword']);
    expect(badgesOf(p, null, 'Badgermole').map(badgeReading)).toEqual(expect.arrayContaining([
      "Doesn't untap · Frost", 'first-strike participant', 'blocking Badgermole', 'attacking', 'Flash', 'entered this turn', 'attacked this turn',
    ]));
  });

  it('names a keyword counter as the source of its keyword, on one chip', () => {
    const mai = creature({ counters: { double_strike: 1 }, keywords: ['double_strike'], keyword_grants: { double_strike: null },
      keyword_abilities: [{ label: 'Prowess' }] });
    expect(board(mai).map(b => [badgeReading(b), b.granted])).toEqual([['Double strike · 1 double strike counter', true], ['Prowess', undefined]]);
    expect(badgesOf(mai).map(badgeReading)).toContain('1 double strike counter');
    const shocked = creature({ counters: { stun: 1 } });
    expect(board(shocked).map(b => b.label)).toEqual(['Stun ×1']);
  });
});

describe('whole-chip fitting', () => {
  const labels = (...values: string[]): Badge[] => values.map(label => ({ label, kind: 'ability' }));
  const measure = (s: string) => s.length * 8 + 10;
  it('skips an oversized label, fitting a later one and the exact overflow count', () => {
    const result = fitBadges(labels('Ward—Discard a card', 'Prowess', 'Ward {2}'), 97, 1, measure);
    expect(result.visible.map(b => b.label)).toEqual(['Prowess']);
    expect(result.hidden).toBe(2);
  });
  it('makes room when the overflow count gains a digit', () => {
    const badges = labels('AA', 'BB', ...Array(9).fill('Exceeds width'));
    const result = fitBadges(badges, 60, 1, measure);
    expect(result.visible).toEqual([]);
    expect(result.hidden).toBe(11);
  });
  it('fits the spiral at 16px and uses the count alone when no row is available', () => {
    const sick: Badge = { kind: 'sick', label: 'Summoning sick' };
    const badges = [sick, ...labels('Ward {2}'), { kind: 'ability', label: 'Flash', inspectOnly: true } satisfies Badge];
    expect(fitBadges(badges, 74, 2, measure)).toEqual({ visible: badges.slice(0, 2), hidden: 0 });
    expect(fitBadges(badges, 74, 0, measure)).toEqual({ visible: [], hidden: 2 });
  });
});
