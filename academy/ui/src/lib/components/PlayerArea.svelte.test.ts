import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import type { PlayerState } from '$lib/types';
import PlayerArea from './PlayerArea.svelte';

function player(overrides: Partial<PlayerState>): PlayerState {
  return {
    player_index: 1,
    id: 2,
    is_active: false,
    is_agent: false,
    life: 20,
    zone_counts: { HAND: 7, LIBRARY: 33 },
    library_count: 33,
    hand: [],
    graveyard: [],
    exile: [],
    stack: [],
    battlefield: [],
    ...overrides,
  };
}

const names = { '4': 'Island', '9': 'Accumulate Wisdom' };

it('shows pooled amounts before qualifiers without repeating combat expiry', () => {
  const { body } = render(PlayerArea, {
    props: {
      label: 'You',
      tablePermanents: [],
      player: player({
        mana: [0, 0, 0, 2, 1, 0],
        combat_mana: 2,
        pooled_mana_classes: [
          { mana: [0, 0, 0, 2, 0, 0], combat: true },
          {
            mana: [0, 0, 0, 0, 1, 0], combat: false,
            restriction: 'creature spells', effect: 'a +1/+1 counter on a non-Human creature spell it casts',
          },
        ],
      }),
    },
  });
  expect(body).toContain('— including 2 {R}, expiring at end of combat');
  expect(body).toContain('; 1 {G} only for creature spells that puts a +1/+1 counter on a non-Human creature spell it casts, expiring at step end');
  expect(body.match(/at end of combat/g)).toHaveLength(1);
  expect(body).not.toContain('undefined');
});

it.each([undefined, []])('falls back to aggregate combat expiry without class rows (%s)', (classes) => {
  const { body } = render(PlayerArea, {
    props: {
      label: 'You', tablePermanents: [],
      player: player({ mana: [0, 0, 0, 2, 0, 0], combat_mana: 2, pooled_mana_classes: classes }),
    },
  });
  expect(body).toContain('2 mana expires at end of combat');
});

it("shows an open opponent's decklist with unseen counts", () => {
  const { body } = render(PlayerArea, {
    props: {
      label: 'Opponent',
      opponent: true,
      definitionNames: names,
      tablePermanents: [],
      player: player({
        main_counts: { '4': 17, '9': 23 },
        unseen_counts: { '4': 16, '9': 24 },
        sideboard_counts: { '9': 1 },
        remaining_sideboard_counts: { '9': 0 },
      }),
    },
  });
  expect(body).toContain('Decklist (40 cards, 40 unseen)');
  expect(body).toContain('Island: 16 unseen of 17');
  expect(body).toContain('Sideboard (0 remaining)');
  expect(body).not.toContain('Decklist closed');
});

it("says a closed opponent's decklist is closed and lists nothing", () => {
  const { body } = render(PlayerArea, {
    props: { label: 'Opponent', opponent: true, definitionNames: names, tablePermanents: [], player: player({}) },
  });
  expect(body).toContain('Decklist closed');
  expect(body).not.toContain('sideboard-summary');
  expect(body).not.toContain('Island');
});
