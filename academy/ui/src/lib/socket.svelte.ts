import { currentGame, rememberCurrentGame } from './current-game.svelte';
import type { PlaySession } from './play-session';
import type { PreviewAddress, PreviewResponse } from './payment-preview';
import type { CombatAddress, CombatPlan, CombatResponse } from './board/combat';
import { browser } from '$app/environment';

import { sanitizeDeckSelection } from './decks';
import { playerToken, rememberRecordCredential } from './records';
import { policyChoice, gameStore } from './game.svelte';
import { followShown, nextLiveAnalysis } from './analysis';
import {
  mergePresentationLabels,
  presentationLabelsFromFrame,
  presentationLabelsFromObservation,
  validatePresentationTail,
  validatePresentationUpdate,
} from './presentation';
import { presentationPlayer, studyPresentationPlayer } from './presentation.svelte';
import type { RestoredReplayDecision } from './replay-index';
import type {
  BranchRetryPayload,
  SharedTableControlEvent,
  SharedTableRequest,
} from './shared-table-protocol';
import type {
  ClientMessage,
  FrameUpdate,
  LogLine,
  LaunchSetup,
  RecoveryEnvelope,
  ExperienceFrame,
  ServerMessage,
} from './types';

const LAUNCH_STORAGE_KEY = 'academy.launch';

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

const RESUME_STORAGE_KEY = 'etude.gui.resume';
const INTERRUPTED_STORAGE_KEY = 'academy.interrupted';

interface InterruptedSnapshot {
  launch: string | null;
  match_id: string;
  frame: ExperienceFrame;
  reason: string;
  recording_id: string | null;
  submission: 'unknown' | null;
  identity: typeof gameStore.interruptedIdentity;
}

type PendingCommand = Extract<SharedTableRequest, { type: 'command' }>;
const VALID_SERVER_TYPES = new Set([
  'work_status',
  'analysis_state',
  'analysis',
  'observation',
  'game_over',
  'command_outcome',
  'error',
  'table_snapshot',
  'belief_changed',
  'role_changed',
  'decision_restored',
  'branch_updated',
  'branch_returned',
  'control_error',
]);

type TableServerMessage = (ServerMessage | SharedTableControlEvent) & Pick<ServerMessage, 'mirror_retry' | 'mirror'>;

interface ResumeCredentials {
  session_id: string;
  resume_token: string;
  play_session_id?: string;
  presentation_cursor?: number;
  pending_command?: PendingCommand;
  last_frame?: ExperienceFrame;
  recording_id?: string;
  identity?: typeof gameStore.interruptedIdentity;
}

// When storage is disabled, this page can still retain exact envelopes and
// authority in memory. A reload loses them, as the visible warning explains.
const volatileTabValues = new Map<string, string | null>();

function tabValue(key: string): string | null {
  if (!browser) return null;
  if (volatileTabValues.has(key)) return volatileTabValues.get(key) ?? null;
  try { return window.sessionStorage.getItem(key); }
  catch { gameStore.storageWarning = 'Session storage is unavailable. Reload continuity is unavailable; this game remains usable.'; return null; }
}

function writeTabValue(key: string, value: string | null): void {
  if (!browser) return;
  try {
    if (value === null) window.sessionStorage.removeItem(key);
    else window.sessionStorage.setItem(key, value);
    volatileTabValues.delete(key);
  } catch {
    volatileTabValues.set(key, value);
    gameStore.storageWarning = 'Session storage is unavailable. Reload continuity is unavailable; this game remains usable.';
  }
}

function loadResumeCredentials(): ResumeCredentials | null {
  if (!browser) {
    return null;
  }

  const raw = tabValue(RESUME_STORAGE_KEY);
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as Partial<ResumeCredentials>;
    if (!parsed.session_id || !parsed.resume_token) {
      return null;
    }
    const presentationCursor = parsed.presentation_cursor;
    if (
      presentationCursor !== undefined
      && (!Number.isSafeInteger(presentationCursor) || presentationCursor < 0)
    ) {
      return null;
    }
    return {
      session_id: parsed.session_id,
      resume_token: parsed.resume_token,
      ...(parsed.identity ? { identity: parsed.identity } : {}),
      ...(typeof parsed.recording_id === 'string' ? { recording_id: parsed.recording_id } : {}),
      ...(parsed.last_frame?.match_id && parsed.last_frame?.projection ? { last_frame: parsed.last_frame } : {}),
      ...(validPendingCommand(parsed.pending_command) ? { pending_command: parsed.pending_command } : {}),
      ...(typeof parsed.play_session_id === 'string' ? {play_session_id: parsed.play_session_id} : {}),
      ...(presentationCursor === undefined
        ? {}
        : { presentation_cursor: presentationCursor }),
    };
  } catch {
    return null;
  }
}

function validPendingCommand(value: unknown): value is PendingCommand {
  if (!value || typeof value !== 'object') return false;
  const request = value as Partial<PendingCommand>;
  const command = request.command;
  return request.type === 'command' && !!command
    && typeof command.command_id === 'string' && typeof command.match_id === 'string'
    && Number.isSafeInteger(command.expected_revision) && Number.isSafeInteger(command.prompt_id)
    && Number.isSafeInteger(command.offer_id) && Array.isArray(command.answers) && command.answers.length === 0
    && (request.grant_revision === undefined || Number.isSafeInteger(request.grant_revision));
}

function saveResumeCredentials(credentials: ResumeCredentials): void {
  if (!browser) {
    return;
  }
  writeTabValue(RESUME_STORAGE_KEY, JSON.stringify({...credentials, ...(gameStore.playSession ? {play_session_id: gameStore.playSession.id} : {})}));
  rememberRecordCredential(credentials.resume_token);
}

function clearResumeCredentials(): void {
  if (!browser) {
    return;
  }
  writeTabValue(RESUME_STORAGE_KEY, null);
  writeTabValue('academy.continuation', null);
  writeTabValue(LAUNCH_STORAGE_KEY, null);
}

export function parseServerMessage(raw: string): TableServerMessage | null {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object' || !('type' in parsed)) {
      return null;
    }

    const messageType = (parsed as { type?: unknown }).type;
    if (typeof messageType !== 'string' || !VALID_SERVER_TYPES.has(messageType)) {
      return null;
    }
    return parsed as TableServerMessage;
  } catch {
    return null;
  }
}

export class GameSocketController {
  private socket: WebSocket | null = null;
  private launchId: string | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private intentionallyClosed = false;
  private pendingResume = false;
  private outboundQueue: SharedTableRequest[] = [];
  private pendingCommand: PendingCommand | null = null;
  private presentationCursor: number | null = null;
  private loadingHistory = false;

  connect(): void {
    if (!browser) {
      return;
    }

    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    if (!this.outboundQueue.some(message => message.type === 'new_game')
      && !window.location.hash && !loadResumeCredentials() && this.restoreInterrupted(this.launchId)) return;

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${protocol}://${window.location.host}/ws/play`;

    gameStore.setConnection(this.reconnectAttempts > 0 ? 'reconnecting' : 'connecting');

    this.intentionallyClosed = false;
    const socket = new WebSocket(url);
    this.socket = socket;

    socket.onopen = () => {
      this.reconnectAttempts = 0;
      gameStore.setConnection('connected');

      const watcherInvite = this.takeWatcherInvite();
      if (watcherInvite) {
        this.pendingResume = true;
        this.send({
          type: 'join_table',
          table_id: watcherInvite.tableId,
          invite_token: watcherInvite.token,
        });
        return;
      }

      if (!this.outboundQueue.some((message) => message.type === 'new_game')) {
        const credentials = loadResumeCredentials();
        if (credentials) {
          if (!gameStore.protocolFrame && credentials.last_frame) gameStore.applyFrame(credentials.last_frame);
          gameStore.recordingId = credentials.recording_id ?? gameStore.recordingId;
          this.pendingCommand = credentials.pending_command ?? this.pendingCommand;
          gameStore.beginCommand();
          this.loadingHistory = this.presentationCursor === null;
          this.pendingResume = true;
          // A persisted cursor outlives the in-memory history. Only request a
          // delta while this controller still owns that history; reloads need
          // the server's bounded tail as well as the authoritative frame.
          this.send({
            type: 'resume',
            session_id: credentials.session_id,
            resume_token: credentials.resume_token,
            ...(credentials.play_session_id ? {play_session_id: credentials.play_session_id} : {}),
            ...(this.presentationCursor === null ? {} : { presentation_cursor: this.presentationCursor }),
          });
          return;
        }
      }
      this.flushQueue();
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      if (this.socket === socket) this.handleRawMessage(event.data);
    };

    socket.onerror = () => {
      gameStore.setError('WebSocket error. Trying to reconnect.');
    };

    socket.onclose = () => {
      if (this.socket !== socket) return;
      this.socket = null;
      if (this.intentionallyClosed) {
        gameStore.setConnection('disconnected');
        return;
      }

      gameStore.setConnection('disconnected');
      this.scheduleReconnect();
    };
  }

  disconnect(): void {
    this.intentionallyClosed = true;
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

  // Reload resumes this tab's own launched game; anything else deals the launch.
  launch(id: string): void {
    this.launchId = id;
    if (this.restoreInterrupted(id)) return;
    const session = loadResumeCredentials()?.session_id;
    try {
      const started = JSON.parse(tabValue(LAUNCH_STORAGE_KEY) ?? 'null') as
        | { id: string; session_id: string; setup: LaunchSetup }
        | null;
      if (session && started?.id === id && started.session_id === session) {
        applyLaunchSetup(started.setup);
        this.connect();
        return;
      }
    } catch {
      writeTabValue(LAUNCH_STORAGE_KEY, null);
    }
    this.sendNewGame({ launch: id });
  }

  sendNewGame(config?: Record<string, unknown>): void {
    if (gameStore.newGamePending) return;
    writeTabValue(INTERRUPTED_STORAGE_KEY, null);
    if (!config?.launch) {
      this.launchId = null;
      if (browser) {
        writeTabValue(LAUNCH_STORAGE_KEY, null);
        const url = new URL(window.location.href);
        if (url.searchParams.has('launch')) {
          url.searchParams.delete('launch');
          window.history.replaceState(window.history.state, '', url);
        }
      }
    }
    gameStore.newGamePending = true;
    gameStore.setError(null);
    const grantRevision = gameStore.table?.access.grant_revision;
    this.send({
      type: 'new_game',
      ...(grantRevision === undefined ? {} : { grant_revision: grantRevision }),
      ...(config ? { config } : {}),
    });
  }

  sendContinuation(intent: 'rematch' | 'swap_sides' | 'change_opponent', config: Record<string, unknown> = {}): void {
    const session = gameStore.playSession;
    const table = gameStore.table;
    if (!session || !table?.attempt_id || gameStore.newGamePending || gameStore.resumeFailed) return;
    const request: SharedTableRequest = {
      type: 'rematch', intent, config: {...config, player_token: playerToken()}, request_id: crypto.randomUUID(),
      attempt_id: table.attempt_id, setup_revision: session.revision,
      grant_revision: table.access.grant_revision,
    };
    writeTabValue('academy.continuation', JSON.stringify({table: table.table_id, request}));
    gameStore.newGamePending = true;
    gameStore.setError(null);
    this.send(request);
  }

  sendSwapSides(): void {
    const table = gameStore.table;
    const frame = gameStore.protocolFrame;
    if (
      !table
      || !frame
      || gameStore.swapPending
      || gameStore.newGamePending
      || gameStore.gameOver
      || gameStore.resumeFailed
      || gameStore.restoredDecision
      || gameStore.searchRetry
      || !gameStore.hasCapability('configure_match')
    ) return;
    gameStore.swapPending = true;
    gameStore.swapError = null;
    gameStore.setError(null);
    this.send({
      type: 'swap_sides',
      grant_revision: table.access.grant_revision,
      request_id: crypto.randomUUID(),
      match_id: frame.match_id,
      expected_revision: frame.revision,
    });
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
    if (this.pendingCommand || this.pendingResume || gameStore.resumeFailed || gameStore.work?.state === 'running') return false;
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      gameStore.setError('Connection changed. Recovering the latest game state.');
      this.connect();
      return false;
    }
    if (browser && !loadResumeCredentials()) {
      gameStore.setError('This tab has no resume authority. Recover the game before choosing again.');
      return false;
    }
    if (gameStore.table) request = {...request, grant_revision: gameStore.table.access.grant_revision};
    // Every deliberate submission must retain its exact envelope before send.
    // Recovery resends this envelope; it never creates another choice or ID.
    this.pendingCommand = request;
    try { this.persistResumeState(); } catch {
      this.pendingCommand = null;
      gameStore.setError('Could not retain your submission in this tab. Free browser storage before trying again.');
      return false;
    }
    gameStore.beginCommand();
    this.socket.send(JSON.stringify(request));
    return true;
  }

  sendRetrySearch(): void {
    const pending = gameStore.searchRetry;
    if (!pending || gameStore.commandPending) return;
    gameStore.beginCommand();
    this.send({ type: 'retry_mirror', match_id: pending.match_id,
      expected_revision: pending.expected_revision,
      ...(gameStore.table ? { grant_revision: gameStore.table.access.grant_revision } : {}),
    });
  }

  sendSetStops(): void {
    const stops = gameStore.stops;
    gameStore.noteStopsSent();
    this.send({
      type: 'set_stops',
      ...(gameStore.table
        ? { grant_revision: gameStore.table.access.grant_revision }
        : {}),
      stops: { my: [...stops.my], opponent: [...stops.opponent] },
      stop_on_stack: stops.stop_on_stack,
      auto_pass: stops.auto_pass,
    });
  }

  /** Show or hide a policy's live reading of your decisions; `policy` by id, else the server's default. */
  sendShowAnalysis(shown: boolean, policy: string | null = null): void {
    gameStore.liveAnalysis = { ...gameStore.liveAnalysis, pending: true };
    this.send({
      type: 'show_analysis',
      ...(gameStore.table ? { grant_revision: gameStore.table.access.grant_revision } : {}),
      shown,
      policy,
    });
  }

  sendPassTurn(): boolean {
    if (this.pendingResume || gameStore.resumeFailed || this.pendingCommand) return false;
    gameStore.beginFastForward();
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      gameStore.endFastForward();
      gameStore.setError('Connection changed. Recovering the latest game state.');
      this.connect();
      return false;
    }
    this.socket.send(JSON.stringify({
      type: 'pass_turn',
      ...(gameStore.table
        ? { grant_revision: gameStore.table.access.grant_revision }
        : {}),
    } satisfies ClientMessage));
    return true;
  }

  sendTransferPilot(targetViewerId: string): void {
    const grantRevision = gameStore.table?.access.grant_revision;
    if (grantRevision === undefined) return;
    this.send({
      type: 'transfer_pilot',
      grant_revision: grantRevision,
      target_viewer_id: targetViewerId,
    });
  }

  sendRestoreDecision(address: string): void {
    const grantRevision = gameStore.table?.access.grant_revision;
    if (grantRevision === undefined) return;
    this.send({ type: 'restore_decision', grant_revision: grantRevision, address });
  }

  sendRetryDecision(offerId: number): void {
    const grantRevision = gameStore.table?.access.grant_revision;
    const restored = gameStore.restoredDecision;
    const prompt = restored?.frame.prompt;
    if (grantRevision === undefined || !restored || !prompt) return;
    this.send({
      type: 'retry_decision',
      grant_revision: grantRevision,
      address: restored.address,
      command: {
        command_id: `study.${crypto.randomUUID()}`,
        match_id: restored.frame.match_id,
        expected_revision: restored.revision,
        prompt_id: prompt.id,
        offer_id: offerId,
        answers: [],
      },
    });
  }

  sendReturnFromBranch(): void {
    const grantRevision = gameStore.table?.access.grant_revision;
    const attemptId = gameStore.branchAttemptId;
    if (grantRevision === undefined || !attemptId) return;
    this.send({
      type: 'return_from_branch',
      grant_revision: grantRevision,
      attempt_id: attemptId,
    });
  }

  sendReturnToLive(): void {
    const grantRevision = gameStore.table?.access.grant_revision;
    if (grantRevision === undefined) return;
    gameStore.returnToLive();
    this.send({ type: 'return_to_live', grant_revision: grantRevision });
  }

  private send(message: SharedTableRequest): void {
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
      if (gameStore.resumeFailed || (gameStore.sessionId && message.session_id !== gameStore.sessionId)) return;
      if (gameStore.work && work.sequence < gameStore.work.sequence) return;
      // A failed new-game load can finish in setup scope while other viewers
      // still show its initial match-scoped advance. End that table-wide wait
      // without treating terminal work as a new frame or submission receipt.
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
      if (!gameStore.observation || this.pendingResume || message.table?.attempt_id !== gameStore.table?.attempt_id) {
        gameStore.searchConfig = search;
      }
    }

    if ('play_session' in message && message.play_session) {
      gameStore.applyPlaySession(message.play_session as PlaySession);
    }

    if ('launch_setup' in message && message.launch_setup) applyLaunchSetup(message.launch_setup);

    if ('table' in message && message.table) {
      const matchChanged = message.table.table_id !== gameStore.table?.table_id || message.table.attempt_id !== gameStore.table?.attempt_id;
      if (matchChanged && !('play_session' in message && message.play_session)) gameStore.playSession = null;
      gameStore.applyTable(message.table);
    }

    if (message.type === 'analysis_state' || message.type === 'analysis') {
      gameStore.liveAnalysis = nextLiveAnalysis(gameStore.liveAnalysis, message, gameStore.actionLog.length);
      return;
    }

    if (message.type === 'table_snapshot') {
      if ('session_interrupted' in message && message.session_interrupted) {
        this.interrupt('The previous live session is unavailable. Choose New game explicitly to play again.');
        return;
      }
      this.pendingResume = false;
      gameStore.endCommand();
      this.persistControlCredentials(message);
      if (this.pendingCommand) {
        this.clearPendingCommand();
        gameStore.endCommand();
        gameStore.setError('The unresolved submission has an unknown outcome; this game has no live decision.');
      }
      return;
    }

    if (message.type === 'belief_changed') {
      gameStore.tableAnnouncement = message.belief.audience.kind === 'personal'
        ? 'Your read is private.'
        : 'A read was shared with the game.';
      return;
    }

    if (message.type === 'role_changed') {
      gameStore.tableAnnouncement = `You are now the ${message.table.access.role}.`;
      return;
    }

    if (message.type === 'decision_restored') {
      gameStore.restoreDecision(message.restored);
      this.restoreStudyHistory(message.restored);
      return;
    }

    if (message.type === 'branch_updated') {
      if (message.phase === 'retry') {
        const payload = message.payload as BranchRetryPayload;
        gameStore.applyBranchRetry(payload);
        studyPresentationPlayer.loadEvents(payload.retry.presentation, mergePresentationLabels(
          ...(gameStore.restoredDecision ? [presentationLabelsFromFrame(gameStore.restoredDecision.frame)] : []),
          presentationLabelsFromObservation(payload.retry.projection),
        ), `study:${payload.attempt_id}`);
      }
      return;
    }

    if (message.type === 'branch_returned') {
      const restored = message.restored as RestoredReplayDecision;
      gameStore.returnBranch(restored);
      this.restoreStudyHistory(restored);
      return;
    }

    if (message.type === 'control_error') {
      writeTabValue('academy.continuation', null);
      gameStore.newGamePending = false;
      if (gameStore.swapPending) {
        gameStore.swapPending = false;
        gameStore.swapError = message.message;
      }
      gameStore.endFastForward();
      this.clearPendingCommand();
      gameStore.endCommand();
      gameStore.setError(message.message);
      return;
    }

    if (message.type === 'observation' || message.type === 'game_over') {
      gameStore.tableUnavailable = false;
      const seatBefore = gameStore.observation?.agent.player_index ?? null;
      if (!this.pendingResume || !currentGame.href) rememberCurrentGame('/');
      if (this.pendingResume) {
        try {
          const pending = JSON.parse(tabValue('academy.continuation') ?? 'null');
          if (pending?.table === message.table?.table_id && pending.request.attempt_id === message.table?.attempt_id) this.send(pending.request);
          else writeTabValue('academy.continuation', null);
        } catch { /* No durable continuation to retry. */ }
      } else {
        writeTabValue('academy.continuation', null);
      }
      gameStore.newGamePending = false;
      if ('launch' in message && message.launch && message.launch_setup) {
        writeTabValue(LAUNCH_STORAGE_KEY, JSON.stringify({ id: message.launch, session_id: message.session_id, setup: message.launch_setup }));
      }
      this.pendingResume = false;
      if (message.type === 'observation') gameStore.liveAnalysis = followShown(gameStore.liveAnalysis, message.analysis?.policy ?? null);
      if (!this.pendingCommand) gameStore.endCommand();
      if (message.recovery) {
        const sessionId = message.type === 'observation' ? message.session_id : undefined;
        const resumeToken = message.type === 'observation' ? message.resume_token : undefined;
        this.applyRecovery(
          message.recovery,
          sessionId,
          resumeToken,
          message.log ?? [],
        );
        gameStore.applyStepBoards(message.recovery.frame, message.boards ?? []);
        this.reconcileCommand(message.recovery);
      } else if (message.type === 'observation') {
        gameStore.applyObservation(
          message.data,
          message.actions,
          message.session_id,
          message.resume_token,
          message.log ?? [],
          message.stops,
          message.deck_names,
          message.action_space ?? '',
        );
      } else {
        gameStore.applyGameOver(
          message.data,
          message.winner,
          message.log ?? [],
          message.stops,
          message.deck_names,
        );
      }
      if (message.session_id && message.resume_token) {
        this.persistResumeState(message.session_id, message.resume_token);
      }
      // A first/replacement recovery resets match state. Its runner metadata
      // belongs to the recovered frame and must survive that reset.
      if ('work' in message) gameStore.work = message.work ?? null;
      if ('mirror_retry' in message) gameStore.searchRetry = message.mirror_retry ?? null;
      // Live scores are read from one seat's view; after Swap sides the line
      // starts again from the new seat, whose earlier decisions arrive as history.
      const seatAfter = gameStore.observation?.agent.player_index ?? null;
      if (seatBefore !== null && seatAfter !== null && seatAfter !== seatBefore) {
        gameStore.liveAnalysis = { ...gameStore.liveAnalysis, points: [], reading: null };
      }
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

    if (this.pendingResume) {
      this.pendingResume = false;
      this.interrupt('The previous live session is unavailable. Choose New game explicitly to play again.');
      return;
    }

    writeTabValue('academy.continuation', null);
    gameStore.endFastForward();
    gameStore.endCommand();
    gameStore.newGamePending = false;
    if (gameStore.swapPending) {
      gameStore.swapPending = false;
      gameStore.swapError = message.message;
    }
    if (message.session_ended) {
      // The server stopped this game; resuming or redealing its launch would hide why.
      this.interrupt(message.message);
      this.launchId = null;
      return;
    }
    gameStore.work = null;
    gameStore.setError(message.message);
  }

  private interrupt(reason: string): void {
    const credentials = loadResumeCredentials();
    const frame = gameStore.protocolFrame ?? credentials?.last_frame;
    const submission = this.pendingCommand || credentials?.pending_command ? 'unknown' : null;
    if (frame) {
      const snapshot: InterruptedSnapshot = {
        launch: this.launchId, match_id: frame.match_id, frame, reason,
        recording_id: gameStore.recordingId ?? credentials?.recording_id ?? null,
        submission,
        identity: gameStore.table?.opponent || gameStore.table?.opponent_label
          ? {opponent: gameStore.table.opponent, opponent_label: gameStore.table.opponent_label}
          : credentials?.identity ?? gameStore.interruptedIdentity,
      };
      writeTabValue(INTERRUPTED_STORAGE_KEY, JSON.stringify(snapshot));
      gameStore.interruptedIdentity = snapshot.identity;
      if (!gameStore.protocolFrame) gameStore.applyFrame(frame);
    }
    clearResumeCredentials();
    this.pendingResume = false;
    this.outboundQueue = [];
    this.pendingCommand = null;
    gameStore.markResumeFailed(reason + (submission ? ' The unresolved submission has an unknown outcome.' : ''));
    this.disconnect();
  }

  private restoreInterrupted(launch: string | null): boolean {
    try {
      const snapshot = JSON.parse(tabValue(INTERRUPTED_STORAGE_KEY) ?? 'null') as InterruptedSnapshot | null;
      if (!snapshot || snapshot.launch !== launch || snapshot.frame?.match_id !== snapshot.match_id
        || !snapshot.frame.projection || typeof snapshot.reason !== 'string') return false;
      gameStore.applyFrame(snapshot.frame);
      gameStore.recordingId = snapshot.recording_id;
      gameStore.interruptedIdentity = snapshot.identity ?? null;
      gameStore.markResumeFailed(snapshot.reason + (snapshot.submission === 'unknown' ? ' The unresolved submission has an unknown outcome.' : ''));
      gameStore.setConnection('disconnected');
      return true;
    } catch { return false; }
  }

  private clearPendingCommand(): void {
    this.pendingCommand = null;
    this.persistResumeState();
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
    const sameGrant = request.grant_revision === gameStore.table?.access.grant_revision;
    if (frame.match_id === command.match_id && frame.revision === command.expected_revision
      && frame.prompt?.id === command.prompt_id && sameGrant
      && gameStore.hasCapability('submit_live_command')) {
      // No replacement ID or choice: the server deduplicates this exact envelope.
      gameStore.beginCommand();
      this.socket?.send(JSON.stringify(request));
      return;
    }
    this.clearPendingCommand();
    gameStore.endCommand();
    gameStore.setError('The unresolved submission has a stale or unknown outcome. Review the recovered decision before choosing again.');
  }

  private restoreStudyHistory(restored: RestoredReplayDecision): void {
    presentationPlayer.finishSequence();
    studyPresentationPlayer.loadPrefix(presentationPlayer.events.flatMap((event, index) =>
      event.to_revision <= restored.frame.revision
        ? [{ events: [event], labels: presentationPlayer.labelsAt(index) }]
        : [],
    ), `study:${restored.frame.match_id}:${restored.address}`, false);
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
    gameStore.applyFrame({
      ...update.frame,
      ...(update.log ? { log: update.log } : {}),
    });
    gameStore.applyStepBoards(update.frame, update.boards ?? []);
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
      this.persistResumeState();
    } catch (error) {
      presentationPlayer.reject();
      this.rejectPresentation('Presentation stream rejected', error);
    }
  }

  private applyRecovery(
    recovery: RecoveryEnvelope,
    sessionId?: string,
    resumeToken?: string,
    log: LogLine[] = [],
  ): void {
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
    gameStore.applyFrame(
      {
        ...next,
        ...(log.length > 0 ? { log } : {}),
      },
      sessionId,
      resumeToken,
    );
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
      // Initial recovery loads recorded history, not live motion from before
      // this page opened. Same-page reconnects animate only newly seen facts.
      if (this.loadingHistory) presentationPlayer.finishSequence();
      this.loadingHistory = false;
      this.persistResumeState(sessionId, resumeToken);
    } catch (error) {
      presentationPlayer.reject();
      this.rejectPresentation('Recovery presentation rejected', error);
    }
  }

  private persistResumeState(sessionId?: string, resumeToken?: string): void {
    const existing = loadResumeCredentials();
    const resolvedSessionId = sessionId ?? existing?.session_id;
    const resolvedResumeToken = resumeToken ?? existing?.resume_token;
    if (!resolvedSessionId || !resolvedResumeToken) {
      return;
    }
    saveResumeCredentials({
      session_id: resolvedSessionId,
      resume_token: resolvedResumeToken,
      ...(this.pendingCommand ? { pending_command: this.pendingCommand } : {}),
      ...(gameStore.protocolFrame ? { last_frame: gameStore.protocolFrame } : {}),
      ...(gameStore.recordingId ? { recording_id: gameStore.recordingId } : {}),
      identity: gameStore.table ? {opponent: gameStore.table.opponent, opponent_label: gameStore.table.opponent_label} : existing?.identity ?? gameStore.interruptedIdentity,
      ...(this.presentationCursor === null
        ? {}
        : { presentation_cursor: this.presentationCursor }),
    });
  }

  private persistControlCredentials(message: SharedTableControlEvent): void {
    const envelope = message as SharedTableControlEvent & {
      session_id?: string;
      resume_token?: string;
    };
    if (envelope.session_id && envelope.resume_token) {
      this.persistResumeState(envelope.session_id, envelope.resume_token);
    }
  }

  private takeWatcherInvite(): { tableId: string; token: string } | null {
    if (!browser || !window.location.hash) return null;
    const params = new URLSearchParams(window.location.hash.slice(1));
    const tableId = params.get('table');
    const token = params.get('watch');
    if (!tableId || !token) return null;
    window.history.replaceState(
      window.history.state,
      '',
      `${window.location.pathname}${window.location.search}`,
    );
    return { tableId, token };
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

export function connect(launchId?: string | null): void {
  if (launchId) controller.launch(launchId);
  else controller.connect();
}

export function disconnect(): void {
  controller.disconnect();
}

export function sendNewGame(config?: Record<string, unknown>): void {
  controller.sendNewGame(config);
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

export function sendTransferPilot(targetViewerId: string): void {
  controller.sendTransferPilot(targetViewerId);
}

export function sendRestoreDecision(address: string): void {
  controller.sendRestoreDecision(address);
}

export function sendRetryDecision(offerId: number): void {
  controller.sendRetryDecision(offerId);
}

export function sendReturnFromBranch(): void {
  controller.sendReturnFromBranch();
}

export function sendReturnToLive(): void {
  controller.sendReturnToLive();
}

export async function fetchPaymentPreview(address: PreviewAddress, signal: AbortSignal): Promise<PreviewResponse> {
  const credentials = loadResumeCredentials();
  if (!credentials) throw new Error('No live session');
  const response = await fetch(`/api/sessions/${encodeURIComponent(credentials.session_id)}/payment-preview`, {
    method: 'POST', signal, headers: { 'Content-Type': 'application/json', 'x-etude-participant-token': credentials.resume_token },
    body: JSON.stringify(address),
  });
  if (!response.ok) throw new Error('Payment preview unavailable');
  return response.json();
}

/** The native whole-declaration preview for a drafted attack or block; read-only. */
export async function fetchCombatPreview(request: CombatAddress & { plan: CombatPlan }, signal: AbortSignal): Promise<CombatResponse> {
  const credentials = loadResumeCredentials();
  if (!credentials) throw new Error('No live session');
  const response = await fetch(`/api/sessions/${encodeURIComponent(credentials.session_id)}/combat-preview`, {
    method: 'POST', signal, headers: { 'Content-Type': 'application/json', 'x-etude-participant-token': credentials.resume_token },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error('Combat preview unavailable');
  return response.json();
}

export function sendContinuation(intent: 'rematch' | 'swap_sides' | 'change_opponent', config: Record<string, unknown> = {}): void { controller.sendContinuation(intent, config); }

export function sendSwapSides(): void { controller.sendSwapSides(); }
