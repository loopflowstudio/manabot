/**
 * Resume across reloads, reconnects and tabs: what this tab sends when its
 * socket opens, and what it keeps in browser storage for the game.
 */
import { readFileSync } from 'node:fs';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$app/environment', () => ({ browser: true }));

import { gameStore } from './game.svelte';
import { presentationPlayer } from './presentation.svelte';
import { GameSocketController, storedGame } from './socket.svelte';
import type { ClientMessage, Command, RecoveryEnvelope, ServerMessage } from './types';

const fixture = JSON.parse(readFileSync(
  new URL('../../../../protocol/fixtures/bolt-target.json', import.meta.url),
  'utf8',
)) as { recovery: RecoveryEnvelope; command: Command };
const GAME = fixture.recovery.frame.match_id;

class FakeSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static opened: FakeSocket[] = [];
  readyState = FakeSocket.CONNECTING;
  sent: ClientMessage[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((event: { code: number }) => void) | null = null;
  constructor(readonly url: string) { FakeSocket.opened.push(this); }
  send(raw: string): void { this.sent.push(JSON.parse(raw) as ClientMessage); }
  close(): void { this.readyState = 3; }
  open(): void { this.readyState = FakeSocket.OPEN; this.onopen?.(); }
  receive(message: ServerMessage): void { this.onmessage?.({ data: JSON.stringify(message) }); }
  drop(code = 1006): void { this.readyState = 3; this.onclose?.({ code }); }
}

function storage(): Storage {
  const saved = new Map<string, string>();
  return {
    getItem: (key: string) => saved.get(key) ?? null,
    setItem: (key: string, value: string) => { saved.set(key, value); },
    removeItem: (key: string) => { saved.delete(key); },
    clear: () => saved.clear(),
    key: () => null,
    length: 0,
  };
}

function board(game = GAME): ServerMessage {
  const recovery = structuredClone(fixture.recovery);
  recovery.frame.match_id = game;
  recovery.accepted_commands = [];
  return { type: 'observation', data: recovery.frame.projection, actions: [], recovery, game, opponent_label: 'Random', setup: { villain_type: 'random' } };
}

const latest = (): FakeSocket => FakeSocket.opened.at(-1)!;

beforeEach(() => {
  vi.useFakeTimers();
  FakeSocket.opened = [];
  const local = storage();
  const session = storage();
  vi.stubGlobal('WebSocket', FakeSocket);
  vi.stubGlobal('window', { location: { protocol: 'http:', host: 'academy.test' }, localStorage: local, sessionStorage: session });
  gameStore.leaveGame();
  gameStore.resumeRefused = null;
  gameStore.launchedGame = null;
  presentationPlayer.clear();
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('taking up a game by its id', () => {
  it('hands a launch that already started a game to that game\'s address, with no error', () => {
    const controller = new GameSocketController();
    controller.connect();
    latest().open();
    controller.sendNewGame({ launch: 'L1' });
    latest().receive({ type: 'error', message: 'game_finished: That game is over.', launch: 'L1', game: GAME } as ServerMessage);
    expect(gameStore.launchedGame).toBe(GAME);
    expect(gameStore.newGamePending).toBe(false);
    expect(gameStore.errorMessage).toBeNull();
  });

  it('asks for the game when the socket opens and remembers it for `/`', () => {
    const controller = new GameSocketController();
    controller.resume(GAME);
    latest().open();
    expect(latest().sent).toEqual([{ type: 'resume', game: GAME }]);
    expect(storedGame()).toBeNull();

    latest().receive(board());
    expect(gameStore.gameId).toBe(GAME);
    expect(gameStore.protocolFrame?.match_id).toBe(GAME);
    expect(gameStore.commandPending).toBe(false);
    expect(storedGame()).toBe(GAME);
  });

  it('resumes the same game after the connection drops, from where its history stopped', () => {
    const controller = new GameSocketController();
    controller.resume(GAME);
    latest().open();
    latest().receive(board());
    latest().drop();
    expect(gameStore.connection).toBe('disconnected');

    vi.advanceTimersByTime(1000);
    expect(FakeSocket.opened).toHaveLength(2);
    latest().open();
    // The cursor is past the last event this tab already holds.
    expect(latest().sent).toEqual([{ type: 'resume', game: GAME, presentation_cursor: fixture.recovery.presentation_tail.at(-1)!.seq + 1 }]);
  });

  it('leaves a finished game on its result card when the connection returns', () => {
    const controller = new GameSocketController();
    controller.resume(GAME);
    latest().open();
    const recovery = structuredClone(fixture.recovery);
    recovery.accepted_commands = [];
    recovery.frame.status = 'game_over';
    recovery.frame.prompt = null;
    recovery.frame.offers = [];
    recovery.frame.winner = 0;
    recovery.frame.projection.game_over = true;
    latest().receive({ type: 'game_over', data: recovery.frame.projection, winner: 0, recovery, game: GAME, opponent_label: 'Random', setup: { villain_type: 'random' } });
    expect(gameStore.gameOver).toBe(true);
    latest().drop();

    vi.advanceTimersByTime(1000);
    latest().open();
    // The server takes up no finished game; asking would trade the result for a refusal.
    expect(latest().sent).toEqual([]);
    expect(gameStore.gameId).toBe(GAME);
    expect(gameStore.gameOver).toBe(true);
  });

  it('gives the game up to another tab without fighting for it', () => {
    const controller = new GameSocketController();
    controller.resume(GAME);
    latest().open();
    latest().receive(board());
    latest().drop(4000);

    vi.advanceTimersByTime(10_000);
    expect(FakeSocket.opened).toHaveLength(1);
    expect(gameStore.errorMessage).toContain('another tab');
    // The game is still the one in progress; a reload takes it back.
    expect(storedGame()).toBe(GAME);
  });

  it('forgets a game the server refuses, so `/` opens setup', () => {
    window.localStorage.setItem('academy.game', GAME);
    const controller = new GameSocketController();
    controller.resume(GAME);
    latest().open();
    latest().receive({ type: 'error', message: 'record_diverged: This game no longer plays as recorded at step 12.' });

    expect(storedGame()).toBeNull();
    expect(gameStore.observation).toBeNull();
    expect(gameStore.resumeRefused).toEqual({ game: GAME, code: 'record_diverged', message: 'This game no longer plays as recorded at step 12.' });
    latest().drop();
    vi.advanceTimersByTime(1000);
    latest().open();
    expect(latest().sent).toEqual([]);
  });

  it('leaves a remembered game with the tab that holds it, and keeps it remembered', () => {
    window.localStorage.setItem('academy.game', GAME);
    const controller = new GameSocketController();
    controller.resume(GAME, false);
    latest().open();
    expect(latest().sent).toEqual([{ type: 'resume', game: GAME, take_over: false }]);
    latest().receive({ type: 'error', message: 'game_open_elsewhere: This game is open in another tab.' });

    expect(gameStore.resumeRefused).toEqual({ game: GAME, code: 'game_open_elsewhere', message: 'This game is open in another tab.' });
    expect(storedGame()).toBe(GAME);
  });

  it('takes its own game back on reconnect after a resume that would not take one over', () => {
    const controller = new GameSocketController();
    controller.resume(GAME, false);
    latest().open();
    latest().receive(board());
    latest().drop();
    vi.advanceTimersByTime(1000);
    latest().open();
    expect(latest().sent[0]).not.toHaveProperty('take_over');
  });

  it('opens setup once the game is over', () => {
    const controller = new GameSocketController();
    controller.resume(GAME);
    latest().open();
    latest().receive(board());
    const over = board();
    if (over.type !== 'observation' || !over.recovery) throw new Error('fixture');
    over.recovery.frame.projection.game_over = true;
    latest().receive({ type: 'game_over', data: over.recovery.frame.projection, winner: 0, recovery: over.recovery, game: GAME });
    expect(gameStore.gameOver).toBe(true);
    expect(storedGame()).toBeNull();
    expect(window.sessionStorage.getItem(`academy.tab.${GAME}`)).toBeNull();
  });
});

describe('what a tab keeps for its game', () => {
  it('keeps the Command in flight in this tab only, and resends that exact envelope after a reload', () => {
    const first = new GameSocketController();
    first.resume(GAME);
    latest().open();
    latest().receive(board());
    expect(first.sendAction(fixture.command.offer_id)).toBe(true);
    const sent = latest().sent.at(-1)!;
    expect(sent.type).toBe('command');

    // The shared pointer holds an id and nothing else.
    expect(window.localStorage.getItem('academy.game')).toBe(GAME);
    const kept = JSON.parse(window.sessionStorage.getItem(`academy.tab.${GAME}`)!);
    expect(kept.pending_command).toEqual(sent);
    expect(kept.last_frame.match_id).toBe(GAME);

    // A reload: a new controller, the same tab storage.
    gameStore.leaveGame();
    const reloaded = new GameSocketController();
    reloaded.resume(GAME);
    // The last board this tab drew stands in until the server answers.
    expect(gameStore.protocolFrame?.match_id).toBe(GAME);
    latest().open();
    latest().receive(board());
    expect(latest().sent).toEqual([{ type: 'resume', game: GAME }, sent]);
  });

  it('drops what it kept for the game it leaves, and keeps that game out of `/`', () => {
    const controller = new GameSocketController();
    controller.resume(GAME);
    latest().open();
    latest().receive(board());
    controller.leave();

    expect(storedGame()).toBeNull();
    expect(window.sessionStorage.getItem(`academy.tab.${GAME}`)).toBeNull();
    expect(gameStore.observation).toBeNull();
    // A fresh connection, attached to nothing.
    latest().open();
    expect(latest().sent).toEqual([]);
  });

  it('keeps the remembered game when an address asks for something else', () => {
    const controller = new GameSocketController();
    controller.resume(GAME);
    latest().open();
    latest().receive(board());
    controller.leave(false);
    expect(storedGame()).toBe(GAME);
  });
});

describe('starting a game', () => {
  it('starts before resuming, and plays the new game from then on', () => {
    const controller = new GameSocketController();
    controller.resume(GAME);
    latest().open();
    latest().receive(board());
    controller.sendNewGame({ villain_type: 'random' });
    expect(latest().sent.at(-1)).toEqual({ type: 'new_game', config: { villain_type: 'random' } });
    latest().receive(board('game-b'));

    expect(gameStore.gameId).toBe('game-b');
    expect(gameStore.newGamePending).toBe(false);
    expect(storedGame()).toBe('game-b');
    expect(window.sessionStorage.getItem(`academy.tab.${GAME}`)).toBeNull();
    latest().drop();
    vi.advanceTimersByTime(1000);
    latest().open();
    expect(latest().sent[0]).toMatchObject({ type: 'resume', game: 'game-b' });
  });

  it('sends a launch again after a lost reply, and never a plain start', () => {
    const launched = new GameSocketController();
    launched.sendNewGame({ launch: 'launch-1' });
    latest().open();
    latest().drop();
    vi.advanceTimersByTime(1000);
    latest().open();
    expect(latest().sent).toEqual([{ type: 'new_game', config: { launch: 'launch-1' } }]);
    latest().receive(board());
    expect(gameStore.newGamePending).toBe(false);

    gameStore.leaveGame();
    FakeSocket.opened = [];
    const plain = new GameSocketController();
    plain.sendNewGame({ villain_type: 'random' });
    latest().open();
    latest().drop();
    vi.advanceTimersByTime(1000);
    latest().open();
    expect(latest().sent).toEqual([]);
    expect(gameStore.newGamePending).toBe(false);
    expect(gameStore.errorMessage).toContain('History');
  });
});
