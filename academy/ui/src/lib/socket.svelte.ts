import { browser } from '$app/environment';

import { sanitizeDeckSelection } from './decks';
import { rememberRecordCredential } from './records';
import { policyChoice, gameStore } from './game.svelte';
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
  Command,
  FrameUpdate,
  LaunchSetup,
  RecoveryEnvelope,
  ExperienceFrame,
  ServerMessage,
} from './types';

const LAUNCH_STORAGE_KEY = 'academy.launch';

function applyLaunchSetup(setup: LaunchSetup): void {
  gameStore.mirrorSearch = setup.search?.[1] ?? null;
  const decks = sanitizeDeckSelection({ hero: setup.hero_deck, villain: setup.villain_deck });
  if (decks) { gameStore.setHeroDeck(decks.hero); gameStore.setVillainDeck(decks.villain); }
  gameStore.deckImport = setup.deck_import ?? null;
  gameStore.setDecklists(setup.decklists);
  if (setup.villain_type === 'policy' && setup.opponent_sha256) gameStore.setOpponentChoice(policyChoice(setup.opponent_sha256));
  else if (setup.villain_type === 'search') gameStore.setOpponentChoice(`search-${setup.villain_sims ?? 64}`);
  else if (setup.villain_type === 'random' || setup.villain_type === 'passive') gameStore.setOpponentChoice(setup.villain_type);
}

const RESUME_STORAGE_KEY = 'etude.gui.resume';
type PendingCommand = Extract<SharedTableRequest, { type: 'command' }>;
const VALID_SERVER_TYPES = new Set([
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

type TableServerMessage = (ServerMessage | SharedTableControlEvent) & Pick<ServerMessage, 'search_retry' | 'search'>;

interface ResumeCredentials {
  session_id: string;
  resume_token: string;
  presentation_cursor?: number;
  pending_command?: PendingCommand;
  last_frame?: ExperienceFrame;
  recording_id?: string;
}

function loadResumeCredentials(): ResumeCredentials | null {
  if (!browser) {
    return null;
  }

  const raw = window.sessionStorage.getItem(RESUME_STORAGE_KEY);
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
      ...(typeof parsed.recording_id === 'string' ? { recording_id: parsed.recording_id } : {}),
      ...(parsed.last_frame?.match_id && parsed.last_frame?.projection ? { last_frame: parsed.last_frame } : {}),
      ...(validPendingCommand(parsed.pending_command) ? { pending_command: parsed.pending_command } : {}),
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
  const command = request.command as Partial<Command> | undefined;
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
  window.sessionStorage.setItem(RESUME_STORAGE_KEY, JSON.stringify(credentials));
  rememberRecordCredential(credentials.resume_token);
}

function clearResumeCredentials(): void {
  if (!browser) {
    return;
  }
  window.sessionStorage.removeItem(RESUME_STORAGE_KEY);
  window.sessionStorage.removeItem(LAUNCH_STORAGE_KEY);
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
    const session = loadResumeCredentials()?.session_id;
    try {
      const started = JSON.parse(sessionStorage.getItem(LAUNCH_STORAGE_KEY) ?? 'null') as
        | { id: string; session_id: string; setup: LaunchSetup }
        | null;
      if (session && started?.id === id && started.session_id === session) {
        applyLaunchSetup(started.setup);
        this.connect();
        return;
      }
    } catch {
      sessionStorage.removeItem(LAUNCH_STORAGE_KEY);
    }
    this.sendNewGame({ launch: id });
  }

  sendNewGame(config?: Record<string, unknown>): void {
    if (gameStore.newGamePending) return;
    if (!config?.launch) {
      this.launchId = null;
      if (browser) {
        sessionStorage.removeItem(LAUNCH_STORAGE_KEY);
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

  sendAction(offerId: number): boolean {
    const frame = gameStore.protocolFrame;
    const prompt = frame?.prompt;
    if (!frame || !prompt || this.pendingCommand !== null || this.pendingResume || gameStore.resumeFailed) {
      return false;
    }
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      gameStore.setError('Connection changed. Recovering the latest game state.');
      this.connect();
      return false;
    }

    if (browser && !loadResumeCredentials()) {
      gameStore.setError('This tab has no resume authority. Recover the table before choosing again.');
      return false;
    }
    const request: PendingCommand = {
      type: 'command',
      ...(gameStore.table ? { grant_revision: gameStore.table.access.grant_revision } : {}),
      command: {
        command_id: crypto.randomUUID(), match_id: frame.match_id,
        expected_revision: frame.revision, prompt_id: prompt.id,
        offer_id: offerId, answers: [],
      },
    };
    this.pendingCommand = request;
    try {
      this.persistResumeState();
    } catch {
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
    this.send({ type: 'retry_search', match_id: pending.match_id,
      expected_revision: pending.expected_revision,
      ...(gameStore.table ? { grant_revision: gameStore.table.access.grant_revision } : {}),
    });
  }

  sendSetStops(): void {
    const stops = gameStore.stops;
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

  sendAuthorBelief(scenarioId: string): void {
    const grantRevision = gameStore.table?.access.grant_revision;
    if (grantRevision === undefined) return;
    this.send({ type: 'author_belief', grant_revision: grantRevision, scenario_id: scenarioId });
  }

  sendShareBelief(beliefId: string): void {
    const grantRevision = gameStore.table?.access.grant_revision;
    if (grantRevision === undefined) return;
    this.send({ type: 'share_belief', grant_revision: grantRevision, belief_id: beliefId });
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
    if ('search_retry' in message) gameStore.searchRetry = message.search_retry ?? null;
    if ('search' in message) {
      const search = message.search?.[1] ?? null;
      gameStore.activeMirrorSearch = search;
      if (!gameStore.observation || this.pendingResume || message.table?.attempt_id !== gameStore.table?.attempt_id) {
        gameStore.mirrorSearch = search;
      }
    }

    if ('launch_setup' in message && message.launch_setup) applyLaunchSetup(message.launch_setup);

    if ('table' in message && message.table) {
      gameStore.applyTable(message.table);
    }

    if (message.type === 'table_snapshot') {
      this.pendingResume = false;
      gameStore.endCommand();
      this.persistControlCredentials(message);
      if (this.pendingCommand) {
        this.clearPendingCommand();
        gameStore.endCommand();
        gameStore.setError('The unresolved submission has an unknown outcome; this table has no live decision.');
      }
      return;
    }

    if (message.type === 'belief_changed') {
      gameStore.tableAnnouncement = message.belief.audience.kind === 'personal'
        ? 'Your read is private.'
        : 'A read was shared with the table.';
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
      gameStore.newGamePending = false;
      gameStore.endFastForward();
      this.clearPendingCommand();
      gameStore.endCommand();
      gameStore.setError(message.message);
      return;
    }

    if (message.type === 'observation' || message.type === 'game_over') {
      gameStore.newGamePending = false;
      if ('launch' in message && message.launch && message.launch_setup) {
        sessionStorage.setItem(LAUNCH_STORAGE_KEY, JSON.stringify({ id: message.launch, session_id: message.session_id, setup: message.launch_setup }));
      }
      this.pendingResume = false;
      if (!this.pendingCommand) gameStore.endCommand();
      if (message.recovery) {
        const sessionId = message.type === 'observation' ? message.session_id : undefined;
        const resumeToken = message.type === 'observation' ? message.resume_token : undefined;
        this.applyRecovery(
          message.recovery,
          sessionId,
          resumeToken,
          message.log ?? [],
          message.auto_passed ?? 0,
        );
        this.reconcileCommand(message.recovery);
      } else if (message.type === 'observation') {
        gameStore.applyObservation(
          message.data,
          message.actions,
          message.session_id,
          message.resume_token,
          message.log ?? [],
          message.stops,
          message.auto_passed ?? 0,
          message.deck_names,
          message.action_space ?? '',
        );
      } else {
        gameStore.applyGameOver(
          message.data,
          message.winner,
          message.log ?? [],
          message.stops,
          message.auto_passed ?? 0,
          message.deck_names,
        );
      }
      if (message.session_id && message.resume_token) {
        this.persistResumeState(message.session_id, message.resume_token);
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
        this.applyUpdate(message.update);
      } else if (message.status === 'duplicate') {
        this.applyRecovery(message.recovery);
      } else {
        if (message.recovery) {
          this.applyRecovery(message.recovery);
        }
        gameStore.setError(message.rejection.message);
      }
      return;
    }

    if (this.pendingResume) {
      this.pendingResume = false;
      clearResumeCredentials();
      this.outboundQueue = [];
      this.pendingCommand = null;
      gameStore.newGamePending = false;
      gameStore.markResumeFailed('The previous live session is unavailable. Choose New game explicitly to play again.');
      this.disconnect();
      return;
    }

    gameStore.endFastForward();
    gameStore.endCommand();
    gameStore.newGamePending = false;
    if (message.session_ended) {
      // The server stopped this game; resuming or redealing its launch would hide why.
      this.pendingCommand = null;
      this.launchId = null;
      clearResumeCredentials();
      gameStore.markResumeFailed(message.message);
      return;
    }
    gameStore.setError(message.message);
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
      ...(update.auto_passed ? { auto_passed: update.auto_passed } : {}),
    });
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
    log: string[] = [],
    autoPassed = 0,
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
        ...(autoPassed > 0 ? { auto_passed: autoPassed } : {}),
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

export function sendAuthorBelief(scenarioId: string): void {
  controller.sendAuthorBelief(scenarioId);
}

export function sendShareBelief(beliefId: string): void {
  controller.sendShareBelief(beliefId);
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
