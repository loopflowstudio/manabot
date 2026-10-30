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

it('states kept red mana and hand-size freedom only for the player they apply to', () => {
  const observation = structuredClone(fixture.recovery.frame.projection) as Observation;
  observation.agent.mana = [0, 0, 0, 1, 0, 0];
  observation.agent.keeps_red_mana = true;
  observation.agent.no_maximum_hand_size = true;
  const body = render(GameBoard, { props: { observation } }).body;
  expect(body.match(/Keeps unspent red mana until end of turn/g)).toHaveLength(1);
  expect(body.match(/data-testid="no-maximum-hand-size"/g)).toHaveLength(1);
  delete observation.agent.keeps_red_mana;
  delete observation.agent.no_maximum_hand_size;
  const plain = render(GameBoard, { props: { observation } }).body;
  expect(plain).not.toContain('Keeps unspent red mana');
  expect(plain).not.toContain('No maximum hand size');
});

it('states turn history only for the player it belongs to', () => {
  const observation = structuredClone(fixture.recovery.frame.projection) as Observation;
  observation.agent.turn_history = { attacked: true, permanents_sacrificed: 2 };
  const body = render(GameBoard, { props: { observation } }).body;
  expect(body.match(/data-testid="turn-history"/g)).toHaveLength(1);
  expect(body).toContain('attacked · sacrificed 2 permanents');
  delete observation.agent.turn_history;
  expect(render(GameBoard, { props: { observation } }).body).not.toContain('turn-history');
});
