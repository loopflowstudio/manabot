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

    store.applyObservation(observation, [{ index: 0, type: 'PRIORITY_PASS_PRIORITY', card: null, focus: [10], description: 'Pass priority' }], 'session-a', 'token-a');

    expect(store.observation?.turn.turn_number).toBe(1);
    expect(store.actions).toHaveLength(1);
    expect(store.sessionId).toBe('session-a');
    expect(store.resumeToken).toBe('token-a');
    expect(store.gameOver).toBe(false);
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
});
