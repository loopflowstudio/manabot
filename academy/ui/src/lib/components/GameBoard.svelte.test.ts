import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import type { Observation } from '$lib/types';
import fixture from '../../../../../protocol/fixtures/bolt-target.json';
import GameBoard from './GameBoard.svelte';
import DecisionContext from './DecisionContext.svelte';

it.each([0, 1])('keeps viewer seat %i in place across delegated search, own response and expiry', (viewer) => {
  const observation = structuredClone(fixture.recovery.frame.projection) as Observation;
  const other = 1 - viewer;
  observation.agent.player_index = viewer;
  observation.agent.id = 40 + viewer;
  observation.opponent.player_index = other;
  observation.opponent.id = 40 + other;
  observation.agent.mana = [0, 3, 0, 0, 0, 0];
  observation.opponent.mana = [0, 1, 0, 0, 0, 0];
  observation.opponent.hand = [];
  observation.agent.inspected = [{
    ...observation.agent.hand[0], id: 900, name: 'Island',
    zone: 'LIBRARY', owner_id: observation.opponent.id,
  }];
  const instruction = `Seat ${viewer + 1} chooses for Seat ${other + 1} using that player's resources. Inspect the permitted library candidates.`;
  observation.decision_context = {
    actor: viewer, stage: 'private_search', instruction,
    recent_facts: [], source: null, targets: [], reserved_sacrifice: null,
    mana_cost: null, chosen_mode: null,
  };
  for (const whole_turn of [false, true]) {
    observation.player_controls = [{ player: other, controller: viewer, whole_turn, active: true, effective: true }];
    const board = render(GameBoard, { props: { observation, heroLabel: 'You' } }).body;
    const context = render(DecisionContext, { props: { observation } }).body;
    expect(context).toContain('Search cards');
    expect(context).not.toContain('Search your library');
    expect(context).toContain(instruction);
    expect(context).toContain(whole_turn ? 'for this turn' : 'for this combat phase');
    expect(board).toContain("Island · opponent's library · private inspection");
    expect(board).toContain('You mana: 3 {U}');
    expect(board).toContain('Opponent mana: 1 {U}');
    expect(board.indexOf('Opponent mana')).toBeLessThan(board.indexOf('You mana'));
  }
  // A control record can remain active while its chooser makes their own choice.
  observation.decision_context.stage = 'priority';
  observation.decision_context.instruction = 'Choose an explicit legal offer to commit one decision.';
  expect(render(DecisionContext, { props: { observation } }).body).not.toContain('chooses for');
  expect(render(GameBoard, { props: { observation, heroLabel: 'You' } }).body).toContain('You mana: 3 {U}');
  observation.player_controls = [];
  observation.agent.inspected = [];
  const expired = render(GameBoard, { props: { observation, heroLabel: 'You' } }).body;
  expect(expired).not.toContain('aria-label="Private inspection"');
  expect(expired).toContain('Opponent mana: 1 {U}');
  expect(render(DecisionContext, { props: { observation } }).body).not.toContain('aria-label="Player control"');
});

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

it('lists a player\'s continuous-effect records with their source', () => {
  const observation = structuredClone(fixture.recovery.frame.projection) as Observation;
  observation.agent.effects = [
    {
      modification: 'KeepUnspentMana',
      color: 'Red',
      duration: 'EndOfTurn',
      source: 'The Last Agni Kai',
      timestamp: 9,
    },
  ];
  const body = render(GameBoard, { props: { observation } }).body;
  expect(body.match(/data-testid="player-effect"/g)).toHaveLength(1);
  expect(body).toContain('until end of turn: keeps unspent red mana (The Last Agni Kai)');
  delete observation.agent.effects;
  expect(render(GameBoard, { props: { observation } }).body).not.toContain('player-effect');
});

it('renders opposite turn queues distinctly despite equal player counts', () => {
  const observation = structuredClone(fixture.recovery.frame.projection) as Observation;
  observation.agent.extra_turns = observation.opponent.extra_turns = 1;
  const hero = observation.agent.player_index;
  const villain = observation.opponent.player_index;
  observation.schedule = { extra_turns: [hero, villain], resume_with: villain, additional_combats: [] };
  const first = render(GameBoard, { props: { observation } }).body.split('aria-label="Scheduled turns"')[1].split('</ol>')[0];
  expect(first).toContain('Hero — extra turn');
  expect(first).toContain('Opponent — extra turn');
  expect(first.indexOf('Hero — extra turn')).toBeLessThan(first.indexOf('Opponent — extra turn'));
  expect(first).toContain('Opponent — normal turn resumes');
  observation.schedule.extra_turns.reverse();
  const second = render(GameBoard, { props: { observation } }).body.split('aria-label="Scheduled turns"')[1].split('</ol>')[0];
  expect(second).toContain('Hero — extra turn');
  expect(second).toContain('Opponent — extra turn');
  expect(second.indexOf('Opponent — extra turn')).toBeLessThan(second.indexOf('Hero — extra turn'));
});

it('keeps current restrictions with no pending combats and distinguishes historical unknowns', () => {
  const observation = structuredClone(fixture.recovery.frame.projection) as Observation;
  observation.schedule = { extra_turns: [], additional_combats: [], current_combat: {
    player_index: observation.agent.player_index, attackers: 'land creatures',
  } };
  const body = render(GameBoard, { props: { observation } }).body;
  expect(body).toContain('Current combat');
  expect(body).toContain('Only land creatures can attack.');
  expect(body).toContain('No extra combats pending.');
  delete observation.schedule.current_combat?.attackers;
  const unrestricted = render(GameBoard, { props: { observation } }).body;
  expect(unrestricted).toContain('No additional attacker restriction.');
  expect(unrestricted).not.toContain('Only land creatures');
  delete observation.schedule;
  const historical = render(GameBoard, { props: { observation } }).body;
  expect(historical).toContain('Schedule not recorded.');
  expect(historical).not.toContain('No extra turns scheduled.');
});

it('renders pending extra combats in order with native restrictions and viewer labels', () => {
  const observation = structuredClone(fixture.recovery.frame.projection) as Observation;
  observation.agent.player_index = 1;
  observation.schedule = { extra_turns: [], additional_combats: [
    { player_index: 1, attackers: 'land creatures' },
    { player_index: 1 },
  ] };
  const body = render(GameBoard, { props: { observation, heroLabel: 'You' } }).body;
  const combats = body.split('aria-label="Scheduled extra combats"')[1].split('</ol>')[0];
  expect(combats).toContain('You. Only land creatures can attack.');
  expect(combats).toContain('You. No additional attacker restriction.');
  expect(combats.indexOf('Only land creatures')).toBeLessThan(combats.indexOf('No additional attacker restriction'));
});
