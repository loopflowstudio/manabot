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

it("shows an open opponent's decklist with unseen counts", () => {
  const { body } = render(PlayerArea, {
    props: {
      label: 'Opponent',
      opponent: true,
      definitionNames: names,
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
    props: { label: 'Opponent', opponent: true, definitionNames: names, player: player({}) },
  });
  expect(body).toContain('Decklist closed');
  expect(body).not.toContain('sideboard-summary');
  expect(body).not.toContain('Island');
});
