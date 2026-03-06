import { describe, expect, it } from 'vitest';

import { buildReplayFrames } from './replay';
import { createReplayStore } from './replay.svelte';
import type { Observation, Trace } from './types';

function makeObservation(turnNumber: number, gameOver = false): Observation {
  return {
    game_over: gameOver,
    won: false,
    turn: {
      turn_number: turnNumber,
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
      hand: [],
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

function makeTrace(): Trace {
  return {
    id: 'trace-1',
    config: {
      hero_deck: { Mountain: 12 },
      villain_deck: { Forest: 12 },
      villain_type: 'passive',
      seed: 1,
    },
    events: [
      {
        actor: 'hero',
        observation: makeObservation(1),
        actions: [],
        action: 0,
        action_description: 'Play land: Mountain',
        reward: 0,
      },
      {
        actor: 'villain',
        observation: makeObservation(2),
        actions: [],
        action: 0,
        action_description: 'Pass priority',
        reward: 0,
      },
    ],
    final_observation: makeObservation(3, true),
    winner: 0,
    end_reason: 'game_over',
    timestamp: '2026-03-06T12:00:00+00:00',
  };
}

describe('buildReplayFrames', () => {
  it('pairs annotations with the post-action board state', () => {
    const frames = buildReplayFrames(makeTrace());

    expect(frames).toHaveLength(3);
    expect(frames[0].observation.turn.turn_number).toBe(1);
    expect(frames[1].actionDescription).toBe('Play land: Mountain');
    expect(frames[1].observation.turn.turn_number).toBe(2);
    expect(frames[2].actionDescription).toBe('Pass priority');
    expect(frames[2].observation.game_over).toBe(true);
  });
});

describe('ReplayStore', () => {
  it('steps through frames and auto-pauses at the end', () => {
    const store = createReplayStore();
    store.loadTrace(makeTrace());

    store.togglePlaying();
    expect(store.playing).toBe(true);

    store.tick();
    expect(store.currentFrameIndex).toBe(1);
    expect(store.playing).toBe(true);

    store.tick();
    expect(store.currentFrameIndex).toBe(2);
    expect(store.playing).toBe(false);
  });
});
