import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import type { Observation } from '$lib/types';
import fixture from '../../../../../protocol/fixtures/bolt-target.json';
import GameBoard from './GameBoard.svelte';

it('states the announced X on an ordered stack entry only when present', () => {
  const observation = structuredClone(fixture.recovery.frame.projection) as Observation;
  const entry = { id: 7, label: 'Crashing Wave', kind: 'spell', controller: 0 };
  observation.stack_objects = [{ ...entry, x: 3 }, { ...entry, id: 8 }];
  const ordered = render(GameBoard, { props: { observation } }).body
    .split('data-testid="ordered-stack"')[1].split('</ol>')[0];
  const [top, second] = ordered.split('<li>').slice(1);
  expect(top).toContain('X = 3');
  expect(second).not.toContain('X =');
});
