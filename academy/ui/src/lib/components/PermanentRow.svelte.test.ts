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
