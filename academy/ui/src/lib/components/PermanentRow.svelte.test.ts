import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import type { PermanentState } from '$lib/types';
import PermanentRow from './PermanentRow.svelte';

function permanent(id: number, name: string, extra: Partial<PermanentState> = {}): PermanentState {
  return { id, name, controller_id: 0, tapped: false, damage: 0, summoning_sick: false, power: 2, toughness: 2,
    base_power: 2, base_toughness: 2, plus1_counters: 0, ...extra };
}

it('shows a stunned creature and an equipped creature in their names and margins', () => {
  const stunned = permanent(1, 'Bear', { counters: { stun: 1 } });
  const equipped = permanent(2, 'Aang');
  const staff = permanent(3, 'Glider Staff', { attached_to: 2, power: null, toughness: null });
  const { body } = render(PermanentRow, { props: { permanents: [stunned, equipped, staff] } });
  expect(body).toContain('aria-label="Bear (1 stun counter)"');
  expect(body).toContain('aria-label="Aang (Glider Staff attached)"');
  expect(body).toContain('aria-label="Glider Staff (attached to Aang)"');
  expect(body).toMatch(/<li[^>]*>1 stun counter<\/li>/);
});

it('prints and names ability-use and granted firebending statuses', () => {
  const cadets = permanent(7, 'Fire Nation Cadets', {
    exhausted_abilities: 1,
    once_each_turn_used: 2,
    granted_firebending: 3,
  });
  const { body } = render(PermanentRow, { props: { permanents: [cadets] } });
  const notes = [
    'firebending 3 until end of turn',
    'exhaust abilities used 1',
    'once-each-turn abilities used this turn 2',
  ];
  expect(body).toContain(`aria-label="Fire Nation Cadets (${notes.join(', ')})"`);
  for (const note of notes) {
    expect(body).toMatch(new RegExp(`<li[^>]*>${note}</li>`));
  }
});

it('omits the statuses when unused', () => {
  const { body } = render(PermanentRow, { props: { permanents: [permanent(7, 'Fire Nation Cadets')] } });
  expect(body).toContain('aria-label="Fire Nation Cadets"');
  expect(body).not.toContain('<li');
});
