import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { gameStore } from './game.svelte';
import { LIGHTNING_BOLT_PRESENTATION } from './presentation';
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
  new URL('../../../protocol/fixtures/bolt-target.json', import.meta.url),
  'utf8',
)) as BoltProtocolFixture;

function frameAt(revision: number): ExperienceFrame {
  return {
    ...structuredClone(boltProtocolFixture.recovery.frame),
    revision,
    frame_hash: `frame-${revision}`,
  };
}

function deliver(controller: GameSocketController, message: ServerMessage): void {
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
    expect(presentationPlayer.currentEvent?.seq).toBe(900);
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
    expect(presentationPlayer.currentEvent?.kind.kind).toBe('cast');
  });

  it('recovery cancels current theater and replaces it with the recovery tail', () => {
    gameStore.prepareForNewGame();
    presentationPlayer.load(LIGHTNING_BOLT_PRESENTATION);
    const controller = new GameSocketController();

    deliver(controller, {
      type: 'observation',
      data: frameAt(43).projection,
      actions: [],
      recovery: {
        ...structuredClone(boltProtocolFixture.recovery),
        frame: frameAt(43),
        presentation_cursor: 903,
        presentation_tail: LIGHTNING_BOLT_PRESENTATION.events.slice(3),
      },
    });

    expect(gameStore.protocolFrame?.revision).toBe(43);
    expect(presentationPlayer.events.map((event) => event.seq)).toEqual([903, 904]);
    expect(presentationPlayer.currentEvent?.kind.kind).toBe('damage');
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
