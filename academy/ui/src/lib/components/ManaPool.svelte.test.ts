import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import type { PlayerState } from '$lib/types';
import ManaPool from './ManaPool.svelte';

it('explains pool parts only when the floating total has some', () => {
  const plain = { mana: [0, 0, 0, 1, 0, 0] } as PlayerState;
  const body = render(ManaPool, { props: { player: plain, label: 'You' } }).body;
  expect(body).toContain('You floating mana: 1 {R}');
  expect(body).not.toContain('Amounts below');
  const restricted = { mana: [0, 0, 0, 0, 0, 0], lesson_mana: [0, 1, 0, 0, 0, 0] } as PlayerState;
  const parts = render(ManaPool, { props: { player: restricted, label: 'You' } }).body;
  expect(parts).toContain('You floating mana: 1 {U}');
  expect(parts).toContain('Amounts below are part of that total');
});
