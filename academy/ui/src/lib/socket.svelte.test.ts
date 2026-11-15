import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import type { SharedTableControlEvent } from './shared-table-protocol';
import { gameStore } from './game.svelte';
import { LIGHTNING_BOLT_PRESENTATION } from './presentation-fixtures';
import { presentationPlayer } from './presentation.svelte';
import { GameSocketController, parseServerMessage } from './socket.svelte';
import type {
  ClientMessage,
  Command,
  ExperienceFrame,
  RecoveryEnvelope,
  ServerMessage,
} from './types';

interface BoltProtocolFixture {
  recovery: RecoveryEnvelope;
  command: Command;
}

const boltProtocolFixture = JSON.parse(readFileSync(
  new URL('../../../../protocol/fixtures/bolt-target.json', import.meta.url),
  'utf8',
)) as BoltProtocolFixture;

function frameAt(revision: number): ExperienceFrame {
  return {
    ...structuredClone(boltProtocolFixture.recovery.frame),
    revision,
    frame_hash: `frame-${revision}`,
  };
}

function deliver(controller: GameSocketController, message: ServerMessage | SharedTableControlEvent): void {
  const testController = controller as unknown as {
    handleRawMessage(raw: string): void;
  };
  testController.handleRawMessage(JSON.stringify(message));
}

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

  it('parses protocol command outcomes', () => {
    const parsed = parseServerMessage(JSON.stringify({
      type: 'command_outcome',
      status: 'rejected',
      rejection: {
        command_id: 'command-a',
        code: 'stale_revision',
        message: 'Stale revision.',
        current_revision: 2,
        current_prompt: 3,
      },
    }));

    expect(parsed?.type).toBe('command_outcome');
    if (parsed?.type === 'command_outcome') {
      expect(parsed.status).toBe('rejected');
    }
  });

  it('rejects unsupported message types', () => {
    const parsed = parseServerMessage(JSON.stringify({ type: 'ping' }));
    expect(parsed).toBeNull();
  });
});

describe('GameSocketController offline gameplay', () => {
  it('allows only one outstanding game start and clears pending state on failure', () => {
    const controller = new GameSocketController();
    const queue = (controller as unknown as { outboundQueue: ClientMessage[] }).outboundQueue;
    gameStore.newGamePending = false;
    controller.sendNewGame({ demo: 'precision' });
    controller.sendNewGame({ demo: 'precision' });
    expect(gameStore.newGamePending).toBe(true);
    expect(queue.filter(message => message.type === 'new_game')).toHaveLength(1);
    deliver(controller, { type: 'error', message: 'Demo preparation failed.' });
    expect(gameStore.newGamePending).toBe(false);
    expect(gameStore.errorMessage).toBe('Demo preparation failed.');
  });
  it('retains the playable frame when a replacement is rejected', () => {
    const controller = new GameSocketController();
    const recovery = structuredClone(boltProtocolFixture.recovery);
    deliver(controller, { type: 'observation', data: recovery.frame.projection,
      actions: [], recovery } as ServerMessage);
    const before = gameStore.protocolFrame;
    controller.sendNewGame({ villain_type: 'policy', opponent_sha256: 'missing' });
    expect(gameStore.protocolFrame).toEqual(before);
    deliver(controller, { type: 'error', message: 'The selected opponent is unavailable.' });
    expect(gameStore.protocolFrame).toEqual(before);
    expect(gameStore.errorMessage).toContain('unavailable');
  });
  it('keeps the exact refused launch policy and decks for a deliberate retry', () => {
    const controller = new GameSocketController();
    deliver(controller, { type: 'error', message: 'Policy changed; restart and reselect.',
      launch: 'refused', launch_setup: { villain_type: 'policy', opponent_sha256: 'exact-id',
        hero_deck: 'gw_allies', villain_deck: 'ur_lessons', decklists: 'closed' } });
    expect(gameStore.opponentChoice).toBe('policy:exact-id');
    expect(gameStore.decks).toEqual({ hero: 'gw_allies', villain: 'ur_lessons' });
    controller.sendNewGame(gameStore.newGameConfig());
    const queue = (controller as unknown as { outboundQueue: ClientMessage[] }).outboundQueue;
    expect(queue.at(-1)).toMatchObject({ type: 'new_game', config: {
      villain_type: 'policy', opponent_sha256: 'exact-id', decklists: 'closed',
    } });
  });
  it('keeps the reason and does not redeal a launch the server ended', () => {
    const controller = new GameSocketController();
    const internals = controller as unknown as { launchId: string | null; outboundQueue: ClientMessage[] };
    const recovery = structuredClone(boltProtocolFixture.recovery);
    deliver(controller, { type: 'observation', data: recovery.frame.projection,
      actions: [], recovery } as ServerMessage);
    internals.launchId = 'launch-1';
    internals.outboundQueue.length = 0;
    const reason = 'The opponent could not continue this game: Policy returned a stale decision binding.';
    deliver(controller, { type: 'error', message: reason, session_ended: true });
    expect(gameStore.errorMessage).toBe(reason);
    expect(gameStore.protocolFrame).toEqual(recovery.frame);
    expect(gameStore.resumeFailed).toBe(true);
    expect(internals.launchId).toBeNull();
    expect(internals.outboundQueue.filter(message => message.type === 'new_game')).toHaveLength(0);
  });
  it('does not queue offer commands or F6 against a future recovered frame', () => {
    gameStore.prepareForNewGame();
    gameStore.applyFrame(boltProtocolFixture.recovery.frame);
    const controller = new GameSocketController();
    const queue = (controller as unknown as { outboundQueue: ClientMessage[] }).outboundQueue;

    expect(controller.sendAction(boltProtocolFixture.command.offer_id)).toBe(false);
    expect(controller.sendPassTurn()).toBe(false);
    expect(queue).toEqual([]);
    expect(gameStore.fastForwarding).toBe(false);

    deliver(controller, {
      type: 'observation',
      data: frameAt(43).projection,
      actions: [],
      recovery: {
        ...structuredClone(boltProtocolFixture.recovery),
        frame: frameAt(43),
      },
    });

    expect(gameStore.protocolFrame?.revision).toBe(43);
    expect(queue).toEqual([]);
  });
});

describe('unresolved Command recovery', () => {
  function pendingController(grant?: number) {
    const controller = new GameSocketController();
    const sent: string[] = [];
    const request = { type: 'command' as const,
      ...(grant === undefined ? {} : { grant_revision: grant }),
      command: structuredClone(boltProtocolFixture.command) };
    const state = controller as unknown as {
      pendingCommand: typeof request | null;
      pendingResume: boolean;
      socket: { send(raw: string): void; close(): void };
      reconcileCommand(recovery: RecoveryEnvelope): void;
    };
    state.pendingCommand = request;
    state.socket = { send: raw => sent.push(raw), close: () => {} };
    gameStore.prepareForNewGame();
    gameStore.table = null;
    gameStore.beginCommand();
    return { controller, state, request, sent };
  }

  it('resends exactly the original envelope only at the original decision', () => {
    const { state, request, sent } = pendingController();
    const recovery = structuredClone(boltProtocolFixture.recovery);
    recovery.accepted_commands = [];
    state.reconcileCommand(recovery);
    expect(sent.map(raw => JSON.parse(raw))).toEqual([request]);
    expect(state.pendingCommand).toEqual(request);
    expect(gameStore.commandPending).toBe(true);
  });

  it('retires a proven accepted submission without sending another choice', () => {
    const { state, request, sent } = pendingController();
    const recovery = structuredClone(boltProtocolFixture.recovery);
    recovery.accepted_commands = [{ command_id: request.command.command_id, actor: 0,
      accepted_at: request.command.expected_revision, resulting_revision: 43,
      resulting_frame_hash: 'committed' }];
    state.reconcileCommand(recovery);
    expect(sent).toEqual([]);
    expect(state.pendingCommand).toBeNull();
    expect(gameStore.commandPending).toBe(false);
  });

  it.each(['revision', 'prompt', 'match', 'grant'])('does not replay a stale %s or infer execution from a missing receipt', kind => {
    const { state, sent } = pendingController(kind === 'grant' ? 99 : undefined);
    const recovery = structuredClone(boltProtocolFixture.recovery);
    recovery.accepted_commands = [];
    if (kind === 'revision') recovery.frame.revision += 1;
    if (kind === 'prompt') recovery.frame.prompt!.id += 1;
    if (kind === 'match') recovery.frame.match_id = 'another-game';
    state.reconcileCommand(recovery);
    expect(sent).toEqual([]);
    expect(state.pendingCommand).toBeNull();
    expect(gameStore.errorMessage).toContain('stale or unknown outcome');
    expect(gameStore.commandPending).toBe(false);
  });

  it('interrupts an expired launched session without queuing a replacement', () => {
    const { controller, state, sent } = pendingController();
    gameStore.applyFrame(boltProtocolFixture.recovery.frame);
    state.pendingResume = true;
    deliver(controller, {type: 'error', message: 'Session expired.'});
    expect(sent).toEqual([]);
    expect(gameStore.protocolFrame?.match_id).toBe(boltProtocolFixture.command.match_id);
    expect(gameStore.resumeFailed).toBe(true);
    expect(gameStore.commandPending).toBe(false);
  });
});

describe('GameSocketController presentation seam', () => {
  it('accepts contiguous semantic sub-transitions inside one batched update', () => {
    gameStore.prepareForNewGame();
    presentationPlayer.clear();
    gameStore.applyFrame(frameAt(41));
    const controller = new GameSocketController();
    const stepped = structuredClone(LIGHTNING_BOLT_PRESENTATION.events);
    stepped[0].from_revision = 41;
    stepped[0].to_revision = 42;

    deliver(controller, {
      type: 'command_outcome',
      status: 'accepted',
      update: {
        base_revision: 41,
        frame: frameAt(43),
        presentation: stepped,
        receipt: null,
      },
    });

    expect(gameStore.protocolFrame?.revision).toBe(43);
    expect(gameStore.errorMessage).toBeNull();
    expect(presentationPlayer.events.map(({ seq }) => seq)).toEqual([900, 901, 902, 903, 904]);
  });

  it('keeps transient narration outside the canonical frame DTO', () => {
    gameStore.prepareForNewGame();
    presentationPlayer.clear();
    gameStore.applyFrame(frameAt(42));
    const controller = new GameSocketController();

    deliver(controller, {
      type: 'command_outcome',
      status: 'accepted',
      update: {
        base_revision: 42,
        frame: frameAt(43),
        presentation: LIGHTNING_BOLT_PRESENTATION.events,
        receipt: null,
        log: ['Villain: Pass priority'],
        auto_passed: 2,
      },
    });

    expect(gameStore.actionLog.map(({ text }) => text)).toContain('Villain: Pass priority');
    expect(gameStore.actionLog.map(({ text }) => text).join('\n')).toContain(
      'Auto-passed 2 priority windows.',
    );
  });

  it('commits the FrameUpdate before enqueueing its semantic events', () => {
    gameStore.prepareForNewGame();
    presentationPlayer.clear();
    gameStore.applyFrame(frameAt(42));
    const controller = new GameSocketController();

    deliver(controller, {
      type: 'command_outcome',
      status: 'accepted',
      update: {
        base_revision: 42,
        frame: frameAt(43),
        presentation: LIGHTNING_BOLT_PRESENTATION.events,
        receipt: null,
      },
    });

    expect(gameStore.protocolFrame?.revision).toBe(43);
    expect(presentationPlayer.currentEvent?.seq).toBe(902);
    expect(presentationPlayer.currentEvent?.to_revision).toBe(
      gameStore.protocolFrame?.revision,
    );
  });

  it('does not narrate protocol updates by diffing their snapshots', () => {
    gameStore.prepareForNewGame();
    presentationPlayer.clear();
    gameStore.applyFrame(frameAt(42));
    const controller = new GameSocketController();
    const next = frameAt(43);
    next.projection.agent.life -= 3;

    deliver(controller, {
      type: 'command_outcome',
      status: 'accepted',
      update: {
        base_revision: 42,
        frame: next,
        presentation: LIGHTNING_BOLT_PRESENTATION.events,
        receipt: null,
      },
    });

    expect(gameStore.actionLog.map((entry) => entry.text).join('\n')).not.toContain(
      'Hero life:',
    );
    expect(presentationPlayer.currentEvent?.kind.kind).toBe('resolved');
  });

  it('same-match recovery merges unseen tail events and never repeats seen motion', () => {
    gameStore.prepareForNewGame();
    presentationPlayer.clear();
    gameStore.applyFrame(frameAt(42));
    presentationPlayer.setScope(`live:${frameAt(42).match_id}`);
    presentationPlayer.enqueue(LIGHTNING_BOLT_PRESENTATION.events.slice(0, 3), LIGHTNING_BOLT_PRESENTATION.labels);
    const controller = new GameSocketController();
    const recovery = {
      ...structuredClone(boltProtocolFixture.recovery),
      frame: frameAt(43),
      presentation_cursor: 901,
      presentation_tail: LIGHTNING_BOLT_PRESENTATION.events.slice(1),
    };
    deliver(controller, { type: 'observation', data: recovery.frame.projection, actions: [], recovery });
    expect(gameStore.protocolFrame?.revision).toBe(43);
    expect(presentationPlayer.events.map(event => event.seq)).toEqual([900, 901, 902, 903, 904]);
    expect(presentationPlayer.currentEvent?.seq).toBe(903);
    deliver(controller, { type: 'observation', data: recovery.frame.projection, actions: [], recovery });
    expect(presentationPlayer.events).toHaveLength(5);
    expect(presentationPlayer.currentEvent).toBeNull();
  });

  it('keeps a committed frame when optional theater is malformed', () => {
    gameStore.prepareForNewGame();
    presentationPlayer.clear();
    gameStore.applyFrame(frameAt(42));
    const controller = new GameSocketController();
    const malformed = structuredClone(LIGHTNING_BOLT_PRESENTATION.events);
    malformed[0].kind = { kind: 'snapshot_changed' } as never;

    deliver(controller, {
      type: 'command_outcome',
      status: 'accepted',
      update: {
        base_revision: 42,
        frame: frameAt(43),
        presentation: malformed,
        receipt: null,
      },
    });

    expect(gameStore.protocolFrame?.revision).toBe(43);
    expect(presentationPlayer.currentEvent).toBeNull();
    expect(gameStore.errorMessage).toMatch(/Presentation stream rejected/);
  });

  it('commits a newer authority frame but rejects an event gap and requests recovery', () => {
    gameStore.prepareForNewGame();
    presentationPlayer.clear();
    gameStore.applyFrame(frameAt(42));
    const controller = new GameSocketController();
    (controller as unknown as { presentationCursor: number }).presentationCursor = 900;

    deliver(controller, {
      type: 'command_outcome',
      status: 'accepted',
      update: {
        base_revision: 42,
        frame: frameAt(43),
        presentation: LIGHTNING_BOLT_PRESENTATION.events.slice(1),
        receipt: null,
      },
    });

    expect(gameStore.protocolFrame?.revision).toBe(43);
    expect(presentationPlayer.currentEvent).toBeNull();
    expect(gameStore.errorMessage).toMatch(/Presentation cursor gap/);
  });

  it('produces the same frame and ordered event tail after refresh', () => {
    const recovery = {
      ...structuredClone(boltProtocolFixture.recovery),
      frame: frameAt(43),
    };

    gameStore.prepareForNewGame();
    presentationPlayer.clear();
    const first = new GameSocketController();
    deliver(first, {
      type: 'observation',
      data: recovery.frame.projection,
      actions: [],
      recovery,
    });
    const firstProjection = structuredClone(gameStore.protocolFrame);
    const firstEvents = presentationPlayer.events.map((event) => event.seq);
    const firstCursor = (first as unknown as { presentationCursor: number }).presentationCursor;

    gameStore.prepareForNewGame();
    presentationPlayer.clear();
    const refreshed = new GameSocketController();
    deliver(refreshed, {
      type: 'observation',
      data: recovery.frame.projection,
      actions: [],
      recovery,
    });

    expect(gameStore.protocolFrame).toEqual(firstProjection);
    expect(presentationPlayer.events.map((event) => event.seq)).toEqual(firstEvents);
    expect(
      (refreshed as unknown as { presentationCursor: number }).presentationCursor,
    ).toBe(firstCursor);
  });
});

it('draft validation never completes an outstanding submission or changes the frame', () => {
  gameStore.prepareForNewGame();
  gameStore.applyFrame(frameAt(42));
  gameStore.beginCommand();
  const controller = new GameSocketController();
  deliver(controller, { type: 'declaration_validation', request_id: 'draft.1',
    match_id: frameAt(42).match_id, revision: 42, prompt_id: frameAt(42).prompt!.id,
    status: 'capacity', message: 'Rules capacity reached.',
    capacity: { resource: 'combat_solver', limit: 1, visited: 2 }, table: null });
  expect(gameStore.commandPending).toBe(true);
  expect(gameStore.protocolFrame?.revision).toBe(42);
  expect(gameStore.declarationValidation?.capacity?.visited).toBe(2);
  gameStore.applyFrame(frameAt(43));
  expect(gameStore.declarationValidation).toBeNull();
});
