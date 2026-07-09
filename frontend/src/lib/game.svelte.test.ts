import { describe, expect, it } from 'vitest';

import { createGameStore } from './game.svelte';
import type { Observation } from './types';

function makeObservation(): Observation {
  return {
    game_over: false,
    won: false,
    turn: {
      turn_number: 1,
      phase: 'PRECOMBAT_MAIN',
      step: 'PRIORITY',
      active_player_id: 10,
      agent_player_id: 10,
    },
    agent: {
      player_index: 0,
      id: 10,
      is_active: true,
      is_agent: true,
      life: 20,
      zone_counts: { HAND: 1, LIBRARY: 39, GRAVEYARD: 0, EXILE: 0, STACK: 0 },
      library_count: 39,
      hand_hidden_count: 0,
      hand: [
        {
          id: 101,
          registry_key: 1,
          name: 'Grey Ogre',
          zone: 'HAND',
          owner_id: 10,
          power: 2,
          toughness: 2,
          mana_value: 3,
          types: {
            is_creature: true,
            is_land: false,
            is_spell: true,
            is_artifact: false,
            is_enchantment: false,
            is_planeswalker: false,
            is_battle: false,
          },
        },
      ],
      graveyard: [],
      exile: [],
      stack: [],
      battlefield: [],
    },
    opponent: {
      player_index: 1,
      id: 20,
      is_active: false,
      is_agent: false,
      life: 20,
      zone_counts: { HAND: 1, LIBRARY: 39, GRAVEYARD: 0, EXILE: 0, STACK: 0 },
      library_count: 39,
      hand_hidden_count: 1,
      hand: [],
      graveyard: [],
      exile: [],
      stack: [],
      battlefield: [],
    },
  };
}

describe('GameStore', () => {
  it('applies observation payloads as full replacements', () => {
    const store = createGameStore();
    const observation = makeObservation();

    store.applyObservation(
      observation,
      [{ index: 0, type: 'PRIORITY_PASS_PRIORITY', focus: [10], description: 'Pass priority' }],
      'session-a',
      'token-a',
    );

    expect(store.observation?.turn.turn_number).toBe(1);
    expect(store.actions).toHaveLength(1);
    expect(store.sessionId).toBe('session-a');
    expect(store.resumeToken).toBe('token-a');
    expect(store.gameOver).toBe(false);
  });

  it('accumulates authoritative log entries from hero and villain actions', () => {
    const store = createGameStore();
    const observation = makeObservation();

    store.appendHeroAction('Play land: Mountain');
    store.applyObservation(observation, [], undefined, undefined, ['Villain: Pass priority']);

    expect(store.actionLog.map((entry) => entry.text)).toEqual([
      'Hero: Play land: Mountain',
      'Villain: Pass priority',
    ]);
  });

  it('transitions into game-over state', () => {
    const store = createGameStore();
    const observation = makeObservation();
    observation.game_over = true;

    store.applyGameOver(observation, 0);

    expect(store.gameOver).toBe(true);
    expect(store.winner).toBe(0);
    expect(store.actions).toEqual([]);
  });

  it('marks failed resume and clears stale session state', () => {
    const store = createGameStore();
    store.applyObservation(makeObservation(), [], 'session-a', 'token-a');

    store.markResumeFailed('expired');

    expect(store.resumeFailed).toBe(true);
    expect(store.sessionId).toBeNull();
    expect(store.resumeToken).toBeNull();
    expect(store.observation).toBeNull();
  });

  it('starts with default stops and toggles individual stop steps', () => {
    const store = createGameStore();
    expect(store.stops).toEqual({
      my: ['main1', 'main2'],
      opponent: ['end_step'],
      stop_on_stack: true,
      auto_pass: true,
    });

    store.toggleStop('my', 'upkeep');
    expect(store.stops.my).toEqual(['main1', 'main2', 'upkeep']);
    store.toggleStop('my', 'main1');
    expect(store.stops.my).toEqual(['main2', 'upkeep']);
    store.setStopOnStack(false);
    store.setAutoPass(false);
    expect(store.stops.stop_on_stack).toBe(false);
    expect(store.stops.auto_pass).toBe(false);
    store.resetStops();
    expect(store.stops.my).toEqual(['main1', 'main2']);
  });

  it('includes the stops config in new_game configs', () => {
    const store = createGameStore();
    store.toggleStop('opponent', 'declare_blockers');

    const config = store.newGameConfig();
    expect(config.stops).toEqual({
      my: ['main1', 'main2'],
      opponent: ['end_step', 'declare_blockers'],
    });
    expect(config.stop_on_stack).toBe(true);
    expect(config.auto_pass).toBe(true);
    expect(config.villain_type).toBe('search');
  });

  it('adopts the server stops echo from observation payloads', () => {
    const store = createGameStore();
    const echo = {
      my: ['end_step'],
      opponent: [],
      stop_on_stack: false,
      auto_pass: true,
    };

    store.applyObservation(makeObservation(), [], undefined, undefined, [], echo);

    expect(store.stops).toEqual(echo);
  });

  it('narrates auto-passed windows and clears the fast-forward flag', () => {
    const store = createGameStore();
    store.beginFastForward();
    expect(store.fastForwarding).toBe(true);

    store.applyObservation(
      makeObservation(),
      [],
      undefined,
      undefined,
      ['Villain: Cast Grey Ogre'],
      undefined,
      3,
    );

    expect(store.fastForwarding).toBe(false);
    expect(store.actionLog.map((entry) => entry.text)).toEqual([
      'Villain: Cast Grey Ogre',
      'Auto-passed 3 priority windows.',
    ]);
  });
});
