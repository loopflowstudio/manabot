import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { policyChoice, createGameStore, seedError } from './game.svelte';
import type { Command, Observation, RecoveryEnvelope } from './types';

interface BoltProtocolFixture {
  recovery: RecoveryEnvelope;
  command: Command;
}

const boltProtocolFixture = JSON.parse(readFileSync(
  new URL('../../../../protocol/fixtures/bolt-target.json', import.meta.url),
  'utf8',
)) as BoltProtocolFixture;

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
  it('projects the shared protocol-v1 Bolt fixture through the existing action table', () => {
    const store = createGameStore();
    const frame = boltProtocolFixture.recovery.frame;

    store.applyFrame(frame);

    expect(store.protocolFrame).toBe(frame);
    expect(store.observation).toBe(frame.projection);
    expect(store.actions).toEqual([
      {
        index: 0,
        type: 'CHOOSE_TARGET',
        focus: [0],
        description: 'Target Hero',
      },
      {
        index: 1,
        type: 'CHOOSE_TARGET',
        focus: [1],
        description: 'Target Villain',
      },
    ]);
    expect(store.actionSpaceKind).toBe('CHOOSE_TARGET');
    expect(boltProtocolFixture.command).toMatchObject({
      expected_revision: frame.revision,
      prompt_id: frame.prompt?.id,
      offer_id: 1,
    });
  });

  it('applies observation payloads as full replacements', () => {
    const store = createGameStore();
    const observation = makeObservation();

    store.applyObservation(
      observation,
      [{ index: 0, type: 'PRIORITY_PASS_PRIORITY', focus: [10], description: 'Pass priority' }],
    );

    expect(store.observation?.turn.turn_number).toBe(1);
    expect(store.actions).toHaveLength(1);
    expect(store.gameOver).toBe(false);
  });

  it('records client-local history notes anchored to the presentation track', () => {
    const store = createGameStore();
    store.applyObservation(makeObservation(), []);
    store.addHistoryNote('you', 1, 5, 'Pass turn');
    store.spendOnce('my', 'main1', 6);

    expect(store.historyNotes.map(({ actor, turn, afterSeq, text }) => ({ actor, turn, afterSeq, text }))).toEqual([
      { actor: 'you', turn: 1, afterSeq: 5, text: 'Pass turn' },
      { actor: 'auto', turn: 1, afterSeq: 6, text: 'Stopped once at First main on your turn; that stop is now off' },
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

  it('keeps the last board readable and takes no input once the game has stopped', () => {
    const store = createGameStore();
    const frame = structuredClone(boltProtocolFixture.recovery.frame);
    store.applyGame({ game: 'saved-game', opponent_label: 'Random' });
    store.applyFrame(frame);
    store.beginCommand();
    store.beginFastForward();
    store.markStopped('The opponent could not continue.');

    expect(store.stopped).toBe(true);
    expect(store.commandPending).toBe(false);
    expect(store.fastForwarding).toBe(false);
    expect(store.protocolFrame).toEqual(frame);
    expect(store.gameId).toBe('saved-game');
    expect(store.decisionStatus).toContain('Stopped');
  });

  it('names the game on the board and returns to setup without ending it', () => {
    const store = createGameStore();
    const setup = { villain_type: 'policy', opponent_sha256: 'b'.repeat(64), human_seat: 1 };
    store.applyGame({ game: 'game-a', opponent: { name: 'Pilot', sha256: 'a'.repeat(64), base_sha256: 'b'.repeat(64) }, opponent_label: 'Pilot · checkpoint', setup });
    store.applyFrame(structuredClone(boltProtocolFixture.recovery.frame));

    expect(store.gameId).toBe('game-a');
    expect(store.opponentChoice).toBe(policyChoice('b'.repeat(64)));
    expect(store.setup).toEqual(setup);

    // A choice made in setup is not overwritten by the same game's next reply.
    store.setOpponentChoice('random');
    store.applyGame({ game: 'game-a', opponent: { name: 'Pilot', sha256: 'a'.repeat(64) }, setup });
    expect(store.opponentChoice).toBe('random');

    store.leaveGame();
    expect(store.gameId).toBeNull();
    expect(store.observation).toBeNull();
    expect(store.setup).toBeNull();
  });

  it('keeps a complete u64 seed as decimal text and sends no identity with a start', () => {
    expect(seedError('18446744073709551615')).toBeNull();
    expect(seedError('18446744073709551616')).not.toBeNull();
    expect(seedError('1.5')).not.toBeNull();
    expect(seedError('-1')).not.toBeNull();
    const store = createGameStore();
    store.seed = '18446744073709551615'; store.dealMode = 'repeat';
    expect(store.newGameConfig().seed).toBe('18446744073709551615');
    expect(Object.keys(store.newGameConfig())).not.toEqual(expect.arrayContaining(['request_id', 'player_token']));
    store.dealMode = 'fresh';
    expect(store.newGameConfig().seed).toBeNull();
    store.seed = 'invalid';
    expect(store.seedError).toBeNull();
    store.dealMode = 'repeat';
    expect(store.seedError).not.toBeNull();
  });

  it('selects the setup a game was started with, imported decks and search included', () => {
    const store = createGameStore();
    const previousDecks = {...store.decks};
    const mirror = {algorithm:'local_update', belief:'physical_deal', worlds:4, depth:2, decision_seconds:1, alpha:0, beta:1};
    const draft = {human_seat:1, mana_floats:'usable', mana_payment:'auto', seed:'18446744073709551615', villain_type:'policy', opponent_sha256:'a'.repeat(64),
      mirror, decklists:'closed', stops:{my:['main1'], opponent:['end_step']}, auto_pass:false, stop_on_stack:true,
      deck_import:{hero:{source:'8 Island',filename:'a.txt'},villain:{source:'8 Mountain',filename:'b.txt'}}};
    store.applySetupDraft(draft);
    expect(store.decks).toEqual(previousDecks);
    expect(store.newGameConfig()).toMatchObject(draft);
  });

  it('keeps the opponent deck chosen here when a closed game names none', () => {
    const store = createGameStore();
    store.setHeroDeck('gw_allies');
    store.setVillainDeck('gw_allies');
    store.applySetupDraft({hero_deck:'ur_lessons', decklists:'closed', villain_type:'random'});
    expect(store.decks).toEqual({hero:'ur_lessons', villain:'gw_allies'});
    expect(store.decklists).toBe('closed');
  });

  it('does not invite a decision while waiting, disconnected or finished', () => {
    const store = createGameStore();
    store.connection = 'connected';
    expect(store.decisionStatus).toBe('Waiting for the next decision.');
    store.commandPending = true;
    expect(store.decisionStatus).toContain('being processed');
    store.connection = 'reconnecting';
    expect(store.decisionStatus).toContain('play is unavailable');
    store.gameOver = true;
    expect(store.decisionStatus).toBe('Game over — no next decision.');
  });
});

describe('stop changes in flight', () => {
  it('drops the echo of an older config, so fast presses keep their last state', () => {
    const store = createGameStore();
    const before = { ...store.stops, my: [...store.stops.my] };
    store.cycleStop('my', 'upkeep');
    store.noteStopsSent();
    const every = { ...store.stops, my: [...store.stops.my] };
    store.cycleStop('my', 'upkeep');
    store.noteStopsSent();
    expect(store.onceStops.my).toEqual(['upkeep']);
    // The frame that answers an earlier change, or one sent before any, still carries the old config.
    store.applyServerStops(before);
    expect(store.stops.my).toContain('upkeep');
    expect(store.onceStops.my).toEqual(['upkeep']);
    store.applyServerStops(every);
    expect(store.onceStops.my).toEqual(['upkeep']);
    // After the last echo the server's config holds again.
    store.applyServerStops({ ...before, my: ['main1'] });
    expect(store.stops.my).toEqual(['main1']);
  });

  it('does not persist passive server echoes to localStorage', () => {
    const backing = new Map<string, string>();
    (globalThis as { localStorage?: unknown }).localStorage = {
      getItem: (key: string) => backing.get(key) ?? null,
      setItem: (key: string, value: string) => backing.set(key, value),
      removeItem: (key: string) => backing.delete(key),
    };
    try {
      const store = createGameStore();
      store.applyServerStops({ my: ['main1'], opponent: [], stop_on_stack: true, auto_pass: true });
      expect(store.stops.my).toEqual(['main1']);
      expect(backing.size).toBe(0);

      store.toggleStop('my', 'upkeep');
      expect(backing.size).toBeGreaterThan(0);
    } finally {
      delete (globalThis as { localStorage?: unknown }).localStorage;
    }
  });
});
