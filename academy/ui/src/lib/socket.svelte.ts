import type { PreviewAddress, PreviewResponse } from './payment-preview';
import type { CombatAddress, CombatPlan, CombatResponse } from './board/combat';
import { browser } from '$app/environment';

import { sanitizeDeckSelection } from './decks';
import { policyChoice, gameStore } from './game.svelte';
import { followShown, nextLiveAnalysis } from './analysis';
import { isMoveEvent } from './board/history';
import {
  mergePresentationLabels,
  presentationLabelsFromFrame,
  validatePresentationTail,
  validatePresentationUpdate,
} from './presentation';
import { presentationPlayer } from './presentation.svelte';
import type {
  ClientMessage,
  FrameUpdate,
  LaunchSetup,
  RecoveryEnvelope,
  ExperienceFrame,
  ServerMessage,
} from './types';

function applyLaunchSetup(setup: LaunchSetup): void {
  gameStore.searchConfig = setup.mirror ?? null;
  const decks = sanitizeDeckSelection({ hero: setup.hero_deck, villain: setup.villain_deck });
  if (decks) { gameStore.setHeroDeck(decks.hero); gameStore.setVillainDeck(decks.villain); }
  gameStore.deckImport = setup.deck_import ?? null;
  gameStore.setDecklists(setup.decklists);
  if (setup.villain_type === 'policy' && setup.opponent_sha256) gameStore.setOpponentChoice(policyChoice(setup.opponent_sha256));
  else if (setup.villain_type === 'search') gameStore.setOpponentChoice(`search-${setup.villain_sims ?? 64}`);
  else if (setup.villain_type === 'random' || setup.villain_type === 'passive') gameStore.setOpponentChoice(setup.villain_type);
}

// ———————————————— Browser storage
//
// One pointer shared by every tab names the game in progress, so `/` opens on
// it. What a tab is in the middle of (a Command it sent, the last frame it
// drew) is that tab's own, so two tabs on one game never overwrite each
// other's submission.

const GAME_STORAGE_KEY = 'academy.game';
const tabStorageKey = (game: string): string => `academy.tab.${game}`;

type PendingCommand = Extract<ClientMessage, { type: 'command' }>;
type StartMessage = Extract<ClientMessage, { type: 'new_game' | 'rematch' }>;

interface TabState {
  pending_command?: PendingCommand;
  last_frame?: ExperienceFrame;
}

/** The unfinished game this browser last played, if it remembers one. */
export function storedGame(): string | null {
  if (!browser) return null;
  try { return window.localStorage.getItem(GAME_STORAGE_KEY); } catch { return null; }
}

function rememberGame(game: string): void {
  try { window.localStorage.setItem(GAME_STORAGE_KEY, game); } catch { /* History still lists the game. */ }
}

/** Stop opening `/` on `game`; another game remembered since is left alone. */
export function forgetGame(game: string): void {
  try { if (window.localStorage.getItem(GAME_STORAGE_KEY) === game) window.localStorage.removeItem(GAME_STORAGE_KEY); } catch { /* Nothing was remembered. */ }
}

// When storage is disabled, this page can still retain exact envelopes in
// memory. A reload loses them, as the visible warning explains.
const volatileTabValues = new Map<string, string | null>();
const STORAGE_WARNING = 'Session storage is unavailable. A reload cannot recover a choice in flight; this game remains usable.';

function tabValue(key: string): string | null {
  if (!browser) return null;
  if (volatileTabValues.has(key)) return volatileTabValues.get(key) ?? null;
  try { return window.sessionStorage.getItem(key); }
  catch { gameStore.storageWarning = STORAGE_WARNING; return null; }
}

function writeTabValue(key: string, value: string | null): void {
  if (!browser) return;
  try {
    if (value === null) window.sessionStorage.removeItem(key);
    else window.sessionStorage.setItem(key, value);
    volatileTabValues.delete(key);
  } catch {
    volatileTabValues.set(key, value);
    gameStore.storageWarning = STORAGE_WARNING;
  }
}

function loadTabState(game: string): TabState {
  try {
    const parsed = JSON.parse(tabValue(tabStorageKey(game)) ?? '{}') as TabState;
    return {
      ...(parsed.last_frame?.match_id === game && parsed.last_frame.projection ? { last_frame: parsed.last_frame } : {}),
      ...(validPendingCommand(parsed.pending_command) ? { pending_command: parsed.pending_command } : {}),
    };
  } catch {
    return {};
  }
}

function validPendingCommand(value: unknown): value is PendingCommand {
  if (!value || typeof value !== 'object') return false;
  const request = value as Partial<PendingCommand>;
  const command = request.command;
  return request.type === 'command' && !!command
    && typeof command.command_id === 'string' && typeof command.match_id === 'string'
    && Number.isSafeInteger(command.expected_revision) && Number.isSafeInteger(command.prompt_id)
    && Number.isSafeInteger(command.offer_id) && Array.isArray(command.answers) && command.answers.length === 0;
}

// ———————————————— The wire

const VALID_SERVER_TYPES = new Set([
  'work_status',
  'analysis_state',
  'analysis',
  'observation',
  'game_over',
  'command_outcome',
  'error',
  'control_error',
]);

export function parseServerMessage(raw: string): ServerMessage | null {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object' || !('type' in parsed)) {
      return null;
    }

    const messageType = (parsed as { type?: unknown }).type;
    if (typeof messageType !== 'string' || !VALID_SERVER_TYPES.has(messageType)) {
      return null;
    }
    return parsed as ServerMessage;
  } catch {
    return null;
  }
}

/** A refusal is `code: sentence`; the sentence is what a person reads. */
function refusal(message: string): { code: string | null; text: string } {
  const match = /^([a-z][a-z0-9_]*): (.+)$/s.exec(message);
  return match ? { code: match[1], text: match[2] } : { code: null, text: message };
}

/** The other tab's `resume` closed this socket: the game is theirs now. */
const TAKEN_OVER = 4000;

/**
 * This tab's connection to `/ws/play`. It is attached to at most one game,
 * named by its id: `resume` takes one up, a start request begins one, and a
 * reconnect resumes the game it was attached to.
 */
export class GameSocketController {
  private socket: WebSocket | null = null;
  /** The game this tab plays; a reconnect resumes it. */
  private game: string | null = null;
  /** A start request whose reply has not arrived. */
  private starting: StartMessage | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private intentionallyClosed = false;
  private pendingResume = false;
  /** Whether the resume in flight takes the game from a tab that holds it. */
  private takeOver = true;
  private outboundQueue: ClientMessage[] = [];
  private pendingCommand: PendingCommand | null = null;
  private presentationCursor: number | null = null;

  connect(): void {
    if (!browser) {
      return;
    }

    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${protocol}://${window.location.host}/ws/play`;

    gameStore.setConnection(this.reconnectAttempts > 0 ? 'reconnecting' : 'connecting');

    this.intentionallyClosed = false;
    const socket = new WebSocket(url);
    this.socket = socket;

    socket.onopen = () => {
      this.reconnectAttempts = 0;
      gameStore.setConnection('connected');
      if (this.outboundQueue.some(message => message.type === 'new_game' || message.type === 'rematch')) {
        this.flushQueue();
        return;
      }
      const lost = this.starting;
      if (lost) {
        // The connection dropped before the start was answered. A launch
        // starts one game however often it is sent; anything else would deal
        // a second game, so it is not sent again.
        if (lost.type === 'new_game' && lost.config?.launch) { socket.send(JSON.stringify(lost)); return; }
        this.starting = null;
        gameStore.newGamePending = false;
        gameStore.setError('The connection dropped while the game was starting. If it started, it is in History.');
      }
      // A finished game is never taken up again: its result card stands as drawn.
      if (this.game && !(gameStore.gameOver && gameStore.gameId === this.game)) this.sendResume();
      else this.flushQueue();
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      if (this.socket === socket) this.handleRawMessage(event.data);
    };

    socket.onerror = () => {
      gameStore.setError('WebSocket error. Trying to reconnect.');
    };

    socket.onclose = (event: CloseEvent) => {
      if (this.socket !== socket) return;
      this.socket = null;
      if (gameStore.liveAnalysis.pending) gameStore.liveAnalysis = { ...gameStore.liveAnalysis, pending: false };
      gameStore.setConnection('disconnected');
      if (this.intentionallyClosed) return;
      if (event.code === TAKEN_OVER) {
        gameStore.endCommand();
        gameStore.endFastForward();
        gameStore.setError('This game is now open in another tab. Reload this tab to play it here.');
        return;
      }
      this.scheduleReconnect();
    };
  }

  disconnect(): void {
    this.intentionallyClosed = true;
    if (gameStore.liveAnalysis.pending) gameStore.liveAnalysis = { ...gameStore.liveAnalysis, pending: false };
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }

    gameStore.setConnection('disconnected');
  }

  /**
   * Take up `game` at its pending decision, in memory or from its stored
   * record. With `takeOver` false, a game another tab holds stays there and
   * the resume is refused (`game_open_elsewhere`).
   */
  resume(game: string, takeOver = true): void {
    const open = this.socket?.readyState === WebSocket.OPEN;
    if (this.game === game && (this.pendingResume || (open && gameStore.gameId === game))) return;
    if (this.game !== game) this.presentationCursor = null;
    this.game = game;
    const tab = loadTabState(game);
    // The last board this tab drew stands in until the server answers.
    if (tab.last_frame && gameStore.protocolFrame?.match_id !== game) {
      presentationPlayer.clear();
      gameStore.prepareForNewGame();
      gameStore.applyFrame(tab.last_frame);
    }
    this.pendingCommand = tab.pending_command ?? null;
    this.takeOver = takeOver;
    gameStore.resumeRefused = null;
    if (open) this.sendResume();
    else this.connect();
  }

  private sendResume(): void {
    if (!this.game || !this.socket) return;
    gameStore.beginCommand();
    this.pendingResume = true;
    // Only request a presentation delta while this controller still holds
    // the history before it; a reload needs the server's bounded tail.
    this.socket.send(JSON.stringify({
      type: 'resume',
      game: this.game,
      ...(this.presentationCursor === null ? {} : { presentation_cursor: this.presentationCursor }),
      ...(this.takeOver ? {} : { take_over: false }),
    } satisfies ClientMessage));
  }

  /**
   * Leave the board for setup. The game stays unfinished: History resumes it,
   * and so does `/` unless `forget` drops this browser's pointer to it.
   */
  leave(forget = true): void {
    // Closing the socket releases the game, so no late frame redraws its board.
    this.disconnect();
    this.resetGame(forget);
    this.connect();
  }

  sendNewGame(config: Record<string, unknown> = {}): void {
    this.start({ type: 'new_game', config });
  }

  /** Start a game from the setup of the one on the board, `config` applied over it. */
  sendRematch(intent: 'rematch' | 'swap_sides', config: Record<string, unknown> = {}): void {
    const game = gameStore.gameId;
    if (game && !gameStore.stopped) this.start({ type: 'rematch', game, intent, config });
  }

  private start(request: StartMessage): void {
    if (gameStore.newGamePending) return;
    gameStore.newGamePending = true;
    gameStore.resumeRefused = null;
    gameStore.setError(null);
    this.starting = request;
    this.send(request);
  }

  sendAction(offerId: number): boolean {
    const frame = gameStore.protocolFrame;
    const prompt = frame?.prompt;
    if (!frame || !prompt) return false;
    return this.submitCommand({
      type: 'command',
      command: {
        command_id: crypto.randomUUID(), match_id: frame.match_id,
        expected_revision: frame.revision, prompt_id: prompt.id,
        offer_id: offerId, answers: [],
      },
    });
  }

  private submitCommand(request: PendingCommand): boolean {
    if (this.pendingCommand || this.pendingResume || gameStore.stopped || gameStore.work?.state === 'running') return false;
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      gameStore.setError('Connection changed. Recovering the latest game state.');
      this.connect();
      return false;
    }
    // Every deliberate submission retains its exact envelope before send.
    // Recovery resends this envelope; it never creates another choice or ID.
    this.pendingCommand = request;
    this.persistTab();
    gameStore.beginCommand();
    this.socket.send(JSON.stringify(request));
    return true;
  }

  sendRetrySearch(): void {
    const pending = gameStore.searchRetry;
    if (!pending || gameStore.commandPending) return;
    gameStore.beginCommand();
    this.send({ type: 'retry_mirror', match_id: pending.match_id, expected_revision: pending.expected_revision });
  }

  sendSetStops(): void {
    const stops = gameStore.stops;
    gameStore.noteStopsSent();
    this.send({
      type: 'set_stops',
      stops: { my: [...stops.my], opponent: [...stops.opponent] },
      stop_on_stack: stops.stop_on_stack,
      auto_pass: stops.auto_pass,
    });
  }

  /** Show or hide a policy's live reading of your decisions; `policy` by id, else the server's default. */
  sendShowAnalysis(shown: boolean, policy: string | null = null): void {
    gameStore.liveAnalysis = { ...gameStore.liveAnalysis, pending: true };
    this.send({ type: 'show_analysis', shown, policy });
  }

  sendPassTurn(): boolean {
    if (this.pendingResume || gameStore.stopped || this.pendingCommand) return false;
    gameStore.beginFastForward();
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      gameStore.endFastForward();
      gameStore.setError('Connection changed. Recovering the latest game state.');
      this.connect();
      return false;
    }
    this.socket.send(JSON.stringify({ type: 'pass_turn' } satisfies ClientMessage));
    return true;
  }

  private send(message: ClientMessage): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
      return;
    }

    this.outboundQueue.push(message);
    this.connect();
  }

  private flushQueue(): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return;
    }

    while (this.outboundQueue.length > 0) {
      const message = this.outboundQueue.shift();
      if (!message) {
        break;
      }
      this.socket.send(JSON.stringify(message));
    }
  }

  private handleRawMessage(raw: string): void {
    const message = parseServerMessage(raw);
    if (message === null) {
      gameStore.newGamePending = false;
      gameStore.setError('Received invalid server payload.');
      return;
    }
    if (message.type === 'work_status') {
      const work = message.work;
      // While a game is starting, its work is the only work there is.
      const another = !gameStore.newGamePending && !!gameStore.gameId && !!message.game && message.game !== gameStore.gameId;
      if (gameStore.stopped || another) return;
      if (gameStore.work && work.sequence < gameStore.work.sequence) return;
      // A failed new-game load finishes in setup scope: end the wait without
      // treating terminal work as a new frame or submission receipt.
      if (work.state !== 'running') {
        gameStore.work = work;
        return;
      }
      if (work.scope === 'setup' && !gameStore.newGamePending) return;
      if (!gameStore.newGamePending && work.match_id && work.match_id !== gameStore.protocolFrame?.match_id) return;
      if (!gameStore.newGamePending && work.revision !== null && work.revision < (gameStore.protocolFrame?.revision ?? 0)) return;
      if (gameStore.work?.operation_id === work.operation_id && gameStore.work.state !== 'running' && work.state === 'running') return;
      gameStore.work = work;
      return;
    }
    if ('work' in message) gameStore.work = message.work ?? null;
    if ('mirror_retry' in message) gameStore.searchRetry = message.mirror_retry ?? null;
    if ('mirror' in message) {
      const search = message.mirror ?? null;
      gameStore.activeSearchConfig = search;
      if (!gameStore.observation || this.pendingResume || message.game !== gameStore.gameId) {
        gameStore.searchConfig = search;
      }
    }

    if ('launch_setup' in message && message.launch_setup) applyLaunchSetup(message.launch_setup);

    if (message.type === 'analysis_state' || message.type === 'analysis') {
      gameStore.liveAnalysis = nextLiveAnalysis(
        gameStore.liveAnalysis,
        message,
        presentationPlayer.events.filter(isMoveEvent).length,
      );
      return;
    }

    if (message.type === 'observation' || message.type === 'game_over') {
      this.starting = null;
      gameStore.newGamePending = false;
      this.pendingResume = false;
      // Once attached, the game is this tab's: its reconnects take it back.
      this.takeOver = true;
      gameStore.applyGame(message);
      if (message.game && message.game !== this.game) this.attach(message.game);
      if (!this.pendingCommand) gameStore.endCommand();
      if (message.recovery) {
        const sameMatch = gameStore.protocolFrame?.match_id === message.recovery.frame.match_id;
        const prevLastSeq = sameMatch ? (presentationPlayer.events.at(-1)?.seq ?? -1) : -1;
        this.applyRecovery(message.recovery);
        const patches = message.boards ?? [];
        // Recovery on reconnect/reload brings presentation_tail with no earlier
        // boards; only span the frame board across freshly delivered non-pass
        // events (never across missed pre-reconnect events).
        const freshNonPass = message.recovery.presentation_tail.filter(
          event => event.seq > prevLastSeq && event.kind.kind !== 'passed',
        );
        const isReconnect = message.recovery.reason === 'reconnect';
        const range = (!isReconnect || patches.length > 0) && freshNonPass.length > 0
          ? {
              first: isReconnect && patches.length > 0 ? patches[0].first : freshNonPass[0].seq,
              last: freshNonPass.at(-1)!.seq,
            }
          : null;
        gameStore.applyStepBoards(message.recovery.frame, patches, range);
        this.reconcileCommand(message.recovery);
      } else if (message.type === 'observation') {
        gameStore.applyObservation(
          message.data,
          message.actions,
          message.stops,
          message.deck_names,
          message.action_space ?? '',
        );
      } else {
        gameStore.applyGameOver(
          message.data,
          message.winner,
          message.stops,
          message.deck_names,
        );
      }
      // `/` opens on a game in progress, and on setup once it is over.
      if (this.game) {
        if (gameStore.gameOver) forgetGame(this.game);
        else rememberGame(this.game);
      }
      this.persistTab();
      // A first/replacement recovery resets match state. Its runner metadata
      // and live-analysis flag belong to the recovered frame and must survive that reset.
      if (message.type === 'observation') gameStore.liveAnalysis = followShown(gameStore.liveAnalysis, message.analysis?.policy ?? null);
      if ('work' in message) gameStore.work = message.work ?? null;
      if ('mirror_retry' in message) gameStore.searchRetry = message.mirror_retry ?? null;
      this.flushQueue();
      return;
    }

    if (message.type === 'command_outcome') {
      const outcomeId = message.status === 'accepted' ? message.update.receipt?.command_id
        : message.status === 'duplicate' ? message.receipt.command_id : message.rejection.command_id;
      if (this.pendingCommand && outcomeId !== this.pendingCommand.command.command_id) return;
      this.clearPendingCommand();
      gameStore.endCommand();
      if (message.status === 'accepted') {
        gameStore.submissionAccepted = true;
        this.applyUpdate(message.update);
      } else if (message.status === 'duplicate') {
        this.applyRecovery(message.recovery);
      } else {
        if (message.recovery) {
          this.applyRecovery(message.recovery);
        }
        gameStore.lastCommandRejection = message.rejection;
        gameStore.setError(message.rejection.message);
      }
      return;
    }

    // An error or a control error: a request was refused.
    const { code, text } = refusal(message.message);
    if (this.pendingResume) {
      this.refuseResume(code, text);
      return;
    }
    const launched = this.starting?.type === 'new_game' && this.starting.config?.launch !== undefined
      && message.type === 'error' ? message.game ?? null : null;
    this.starting = null;
    if (launched) {
      // A launch that already started a game refuses another: its game's address says what it is now.
      gameStore.newGamePending = false;
      gameStore.launchedGame = launched;
      return;
    }
    gameStore.endFastForward();
    this.clearPendingCommand();
    gameStore.endCommand();
    gameStore.newGamePending = false;
    if (gameStore.liveAnalysis.pending) gameStore.liveAnalysis = { ...gameStore.liveAnalysis, pending: false };
    if (message.type === 'error' && message.game_stopped) {
      this.stop(text);
      return;
    }
    if (message.type === 'error') gameStore.work = null;
    gameStore.setError(text);
  }

  /** This tab now plays `game`: what it kept for the game before is dropped. */
  private attach(game: string): void {
    if (this.game) writeTabValue(tabStorageKey(this.game), null);
    this.game = game;
  }

  /** Drop this tab's game state and per-tab storage; returns the game id cleared. */
  private resetGame(forget: boolean): string | null {
    const game = this.game;
    if (game && forget) forgetGame(game);
    if (game) writeTabValue(tabStorageKey(game), null);
    this.game = null;
    this.starting = null;
    this.pendingResume = false;
    this.pendingCommand = null;
    this.outboundQueue = [];
    this.presentationCursor = null;
    this.takeOver = true;
    presentationPlayer.clear();
    gameStore.leaveGame();
    return game;
  }

  /**
   * The server will not take the game up: no board, and the refusal for the
   * page to act on. A game another tab holds is still this browser's game.
   */
  private refuseResume(code: string | null, reason: string): void {
    const game = this.resetGame(code !== 'game_open_elsewhere');
    if (game) gameStore.resumeRefused = { game, code, message: reason };
  }

  /** The game broke while a request ran. Its last board stays readable; nothing resumes it. */
  private stop(reason: string): void {
    const game = this.game;
    if (game) { forgetGame(game); writeTabValue(tabStorageKey(game), null); }
    this.game = null;
    this.outboundQueue = [];
    this.pendingCommand = null;
    gameStore.markStopped(reason);
    this.disconnect();
  }

  private clearPendingCommand(): void {
    this.pendingCommand = null;
    this.persistTab();
  }

  private reconcileCommand(recovery: RecoveryEnvelope): void {
    const request = this.pendingCommand;
    if (!request) return;
    const command = request.command;
    if (recovery.frame.match_id === command.match_id
      && recovery.accepted_commands.some(receipt => receipt.command_id === command.command_id)) {
      this.clearPendingCommand();
      gameStore.endCommand();
      return;
    }
    const frame = recovery.frame;
    if (frame.match_id === command.match_id && frame.revision === command.expected_revision
      && frame.prompt?.id === command.prompt_id) {
      // No replacement ID or choice: the server deduplicates this exact envelope.
      gameStore.beginCommand();
      this.socket?.send(JSON.stringify(request));
      return;
    }
    this.clearPendingCommand();
    gameStore.endCommand();
    gameStore.setError('The unresolved submission has a stale or unknown outcome. Review the recovered decision before choosing again.');
  }

  private applyUpdate(update: FrameUpdate): void {
    const current = gameStore.protocolFrame;
    if (current && current.match_id === update.frame.match_id) {
      if (update.frame.revision <= current.revision) {
        return;
      }
      if (update.base_revision !== current.revision) {
        gameStore.setError('Game update gap detected. Reconnecting for recovery.');
        this.disconnect();
        this.connect();
        return;
      }
    }
    // Canonical truth lands first. Presentation is optional theater over the
    // committed frame and cannot delay or roll back authority.
    if (current?.match_id !== update.frame.match_id) this.presentationCursor = null;
    presentationPlayer.setScope(`live:${update.frame.match_id}`);
    const labels = mergePresentationLabels(
      ...(current?.match_id === update.frame.match_id ? [presentationLabelsFromFrame(current)] : []),
      presentationLabelsFromFrame(update.frame),
    );
    gameStore.applyFrame(update.frame);
    const nonPass = update.presentation.filter(event => event.kind.kind !== 'passed');
    const range = nonPass.length > 0
      ? { first: nonPass[0].seq, last: nonPass.at(-1)!.seq }
      : null;
    gameStore.applyStepBoards(update.frame, update.boards ?? [], range);
    try {
      validatePresentationUpdate(
        update.presentation,
        update.base_revision,
        update.frame.revision,
      );
      const expectedCursor = this.presentationCursor
        ?? update.presentation[0]?.seq
        ?? 0;
      this.presentationCursor = validatePresentationTail(
        update.presentation,
        expectedCursor,
      );
      presentationPlayer.enqueue(
        update.presentation,
        labels,
      );
      this.persistTab();
    } catch (error) {
      presentationPlayer.reject();
      this.rejectPresentation('Presentation stream rejected', error);
    }
  }

  private applyRecovery(recovery: RecoveryEnvelope): void {
    const current = gameStore.protocolFrame;
    const next = recovery.frame;
    if (!current || current.match_id !== next.match_id) {
      this.presentationCursor = null;
      presentationPlayer.clear();
      gameStore.prepareForNewGame();
    }
    if (
      current
      && current.match_id === next.match_id
      && next.revision < current.revision
    ) {
      return;
    }
    // Recovery always cancels any in-flight theater before resuming its
    // viewer-safe tail. The complete frame remains authoritative either way.
    presentationPlayer.setScope(`live:${next.match_id}`);
    const presentationLabels = mergePresentationLabels(
      ...(current?.match_id === next.match_id ? [presentationLabelsFromFrame(current)] : []),
      presentationLabelsFromFrame(next),
    );
    gameStore.applyFrame(next);
    try {
      this.presentationCursor = validatePresentationTail(
        recovery.presentation_tail,
        recovery.presentation_cursor,
      );
      presentationPlayer.recover(
        recovery.presentation_tail,
        presentationLabels,
        recovery.presentation_cursor,
      );
      this.persistTab();
    } catch (error) {
      presentationPlayer.reject();
      this.rejectPresentation('Recovery presentation rejected', error);
    }
  }

  /** Keep what a reload of this tab needs: the Command in flight and the last frame drawn. */
  private persistTab(): void {
    if (!this.game) return;
    const frame = gameStore.protocolFrame;
    const state: TabState = {
      ...(this.pendingCommand ? { pending_command: this.pendingCommand } : {}),
      ...(frame?.match_id === this.game && !gameStore.gameOver ? { last_frame: frame } : {}),
    };
    writeTabValue(tabStorageKey(this.game), Object.keys(state).length ? JSON.stringify(state) : null);
  }

  private rejectPresentation(prefix: string, error: unknown): void {
    const detail = error instanceof Error ? error.message : 'Unknown presentation error.';
    gameStore.setError(`${prefix}: ${detail}`);
    if (detail.startsWith('Presentation cursor gap:')) {
      this.disconnect();
      this.connect();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer !== null) {
      return;
    }

    const delay = Math.min(1000 * 2 ** this.reconnectAttempts, 5000);
    this.reconnectAttempts += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }
}

const controller = new GameSocketController();

export function connect(): void {
  controller.connect();
}

export function disconnect(): void {
  controller.disconnect();
}

export function resumeGame(game: string, takeOver = true): void {
  controller.resume(game, takeOver);
}

export function leaveGame(options: { forget: boolean } = { forget: true }): void {
  controller.leave(options.forget);
}

export function sendNewGame(config?: Record<string, unknown>): void {
  controller.sendNewGame(config);
}

export function sendRematch(intent: 'rematch' | 'swap_sides', config: Record<string, unknown> = {}): void {
  controller.sendRematch(intent, config);
}

export function sendAction(offerId: number): boolean {
  return controller.sendAction(offerId);
}

export function sendShowAnalysis(shown: boolean, policy: string | null = null): void {
  controller.sendShowAnalysis(shown, policy);
}

export function sendSetStops(): void {
  controller.sendSetStops();
}

export function sendPassTurn(): boolean {
  return controller.sendPassTurn();
}

export function sendRetrySearch(): void {
  controller.sendRetrySearch();
}

async function preview<T>(kind: 'payment' | 'combat', body: unknown, signal: AbortSignal): Promise<T> {
  const game = gameStore.gameId;
  if (!game) throw new Error('No game in progress');
  const response = await fetch(`/api/games/${encodeURIComponent(game)}/${kind}-preview`, {
    method: 'POST', signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(kind === 'payment' ? 'Payment preview unavailable' : 'Combat preview unavailable');
  return response.json();
}

export function fetchPaymentPreview(address: PreviewAddress, signal: AbortSignal): Promise<PreviewResponse> {
  return preview('payment', address, signal);
}

/** The native whole-declaration preview for a drafted attack or block; read-only. */
export function fetchCombatPreview(request: CombatAddress & { plan: CombatPlan }, signal: AbortSignal): Promise<CombatResponse> {
  return preview('combat', request, signal);
}
