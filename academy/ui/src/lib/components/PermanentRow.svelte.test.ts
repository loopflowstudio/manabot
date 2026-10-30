import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import type { PermanentState } from '$lib/types';
import PermanentRow from './PermanentRow.svelte';

function permanent(overrides: Partial<PermanentState>): PermanentState {
  return {
    id: 7,
    name: 'Fire Nation Cadets',
    controller_id: 0,
    tapped: false,
    damage: 0,
    summoning_sick: false,
    power: 1,
    toughness: 2,
    base_power: 1,
    base_toughness: 2,
    plus1_counters: 0,
    ...overrides,
  };
}

it('prints and names ability-use and granted firebending statuses', () => {
  const { body } = render(PermanentRow, {
    props: {
      permanents: [
        permanent({ exhausted_abilities: 1, once_each_turn_used: 2, granted_firebending: 3 }),
      ],
    },
  });
  const notes = [
    'exhaust abilities used 1',
    'once-each-turn abilities used this turn 2',
    'firebending 3 until end of turn',
  ];
  expect(body).toContain(`aria-label="Fire Nation Cadets (${notes.join(', ')})"`);
  for (const note of notes) {
    expect(body).toContain(`data-testid="card-ability-status">${note}</span>`);
  }
});

it('omits the statuses when unused', () => {
  const { body } = render(PermanentRow, { props: { permanents: [permanent({})] } });
  expect(body).toContain('aria-label="Fire Nation Cadets"');
  expect(body).not.toContain('card-ability-status');
});
