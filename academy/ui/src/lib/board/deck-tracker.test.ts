import { describe, expect, it } from 'vitest';
import type { PlayerState } from '$lib/types';
import {
  buildOpponentTracker,
  buildSelfTracker,
  parseManaValue,
  probAtLeastOne,
  type TrackerCardFace,
} from './deck-tracker';

function basePlayer(overrides: Partial<PlayerState> = {}): PlayerState {
  return {
    player_index: 0,
    id: 1,
    is_active: true,
    is_agent: true,
    life: 20,
    zone_counts: { HAND: 4, LIBRARY: 20, GRAVEYARD: 0, EXILE: 0 },
    library_count: 20,
    hand: [],
    battlefield: [],
    graveyard: [],
    exile: [],
    stack: [],
    ...overrides,
  };
}

const FACES: Record<string, TrackerCardFace> = {
  Island: { cost: '', colors: [], type_line: 'Basic Land — Island', keywords: [], power: null, toughness: null },
  Mountain: { cost: '', colors: [], type_line: 'Basic Land — Mountain', keywords: [], power: null, toughness: null },
  'Firebending Lesson': { cost: '{R}', colors: ['R'], type_line: 'Sorcery — Lesson', keywords: [], power: null, toughness: null },
  'Accumulate Wisdom': { cost: '{1}{U}', colors: ['U'], type_line: 'Instant — Lesson', keywords: [], power: null, toughness: null },
  'Airbending Lesson': { cost: '{1}{U}', colors: ['U'], type_line: 'Instant — Lesson', keywords: [], power: null, toughness: null },
};

describe('deck-tracker', () => {
  it('computes mana value and hypergeometric odds accurately', () => {
    expect(parseManaValue('{2}{U}{R}')).toBe(4);
    expect(parseManaValue('{X}{R}')).toBe(1);
    expect(probAtLeastOne(0, 20, 4)).toBe(0);
    expect(probAtLeastOne(2, 20, 1)).toBe(10);
    expect(probAtLeastOne(2, 10, 4)).toBe(67);
    expect(probAtLeastOne(5, 6, 3)).toBe(100);
  });

  it('builds self tracker with remaining counts, next-draw odds, land/spell summary, and sideboard', () => {
    const player = basePlayer({
      main_counts: { Island: 8, 'Firebending Lesson': 2, 'Accumulate Wisdom': 2 },
      unseen_counts: { Island: 6, 'Firebending Lesson': 2, 'Accumulate Wisdom': 0 },
      sideboard_counts: { 'Airbending Lesson': 1 },
      remaining_sideboard_counts: { 'Airbending Lesson': 1 },
    });
    const view = buildSelfTracker(player, FACES);
    expect(view).not.toBeNull();
    expect(view!.mode).toBe('self');
    expect(view!.totalRemaining).toBe(8);
    expect(view!.landsRemaining).toBe(6);
    expect(view!.landPercent).toBe(75);
    expect(view!.spellsRemaining).toBe(2);
    expect(view!.spellPercent).toBe(25);
    // Spells sorted by mana value before lands.
    expect(view!.rows.map(r => r.name)).toEqual(['Firebending Lesson', 'Accumulate Wisdom', 'Island']);
    expect(view!.rows[0].oddsPercent).toBe(25);
    expect(view!.rows[1].remaining).toBe(0);
    expect(view!.sideboard).toHaveLength(1);
    expect(view!.sideboard[0].name).toBe('Airbending Lesson');
  });

  it('builds open opponent tracker with P(in hand) odds and closed opponent tracker with seen cards only', () => {
    const openOpp = basePlayer({
      player_index: 1,
      id: 2,
      is_agent: false,
      hand_hidden_count: 4,
      main_counts: { Island: 8, 'Firebending Lesson': 2 },
      unseen_counts: { Island: 8, 'Firebending Lesson': 2 },
    });
    const openView = buildOpponentTracker(openOpp, FACES);
    expect(openView.mode).toBe('opponent-open');
    expect(openView.rows.find(r => r.name === 'Firebending Lesson')?.oddsPercent).toBe(67);

    const closedOpp = basePlayer({
      player_index: 1,
      id: 2,
      is_agent: false,
      hand_hidden_count: 4,
      battlefield: [
        { id: 10, name: 'Island', is_token: false } as PlayerState['battlefield'][number],
        { id: 11, name: 'Ally Token', is_token: true } as PlayerState['battlefield'][number],
      ],
      graveyard: [{ id: 12, name: 'Firebending Lesson' } as PlayerState['graveyard'][number]],
    });
    const closedView = buildOpponentTracker(closedOpp, FACES);
    expect(closedView.mode).toBe('opponent-closed');
    expect(closedView.totalSeen).toBe(2);
    expect(closedView.rows.map(r => r.name)).toEqual(['Firebending Lesson', 'Island']);
  });

  it('resolves numeric card definition IDs through observation definition_names', () => {
    const player = basePlayer({
      main_counts: { '23': 2, '370': 8 },
      unseen_counts: { '23': 1, '370': 7 },
    });
    const defNames = { '23': 'Airbender Ascension', '370': 'Plains' };
    const faces: Record<string, TrackerCardFace> = {
      'Airbender Ascension': { cost: '{1}{W}', colors: ['W'], type_line: 'Enchantment', keywords: [], power: null, toughness: null },
      Plains: { cost: '', colors: [], type_line: 'Basic Land — Plains', keywords: [], power: null, toughness: null },
    };
    const view = buildSelfTracker(player, faces, defNames);
    expect(view).not.toBeNull();
    expect(view!.rows.map(row => row.name)).toEqual(['Airbender Ascension', 'Plains']);
    expect(view!.landsRemaining).toBe(7);
    expect(view!.spellsRemaining).toBe(1);
    expect(view!.rows[0].cost).toBe('1W');
  });
});
