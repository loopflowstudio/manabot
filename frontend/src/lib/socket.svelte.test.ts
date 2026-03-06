import { describe, expect, it } from 'vitest';

import { parseServerMessage } from './socket.svelte';

describe('parseServerMessage', () => {
  it('parses observation payloads', () => {
    const payload = JSON.stringify({
      type: 'observation',
      data: {
        game_over: false,
        won: false,
        turn: {
          turn_number: 1,
          phase: 'PRECOMBAT_MAIN',
          step: 'PRIORITY',
          active_player_id: 1,
          agent_player_id: 1,
        },
        agent: {
          player_index: 0,
          id: 1,
          is_active: true,
          is_agent: true,
          life: 20,
          zone_counts: {},
          library_count: 40,
          hand: [],
          graveyard: [],
          exile: [],
          stack: [],
          battlefield: [],
        },
        opponent: {
          player_index: 1,
          id: 2,
          is_active: false,
          is_agent: false,
          life: 20,
          zone_counts: {},
          library_count: 40,
          hand: [],
          graveyard: [],
          exile: [],
          stack: [],
          battlefield: [],
        },
      },
      actions: [],
      session_id: 'session-id',
      resume_token: 'token',
    });

    const parsed = parseServerMessage(payload);

    expect(parsed?.type).toBe('observation');
  });

  it('rejects malformed json', () => {
    expect(parseServerMessage('not-json')).toBeNull();
  });

  it('rejects unsupported message types', () => {
    const parsed = parseServerMessage(JSON.stringify({ type: 'ping' }));
    expect(parsed).toBeNull();
  });
});
