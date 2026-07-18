import { browser } from '$app/environment';

import { gameStore } from './game.svelte';
import {
  mergePresentationLabels,
  presentationLabelsFromFrame,
  validatePresentationTail,
  validatePresentationUpdate,
} from './presentation';
import { presentationPlayer } from './presentation.svelte';
import type { RestoredReplayDecision } from './replay-index';
import type {
  BranchRetryPayload,
  TestingHouseControlEvent,
  TestingHouseRequest,
} from './testing-house-protocol';
import type {
  ClientMessage,
  FrameUpdate,
  RecoveryEnvelope,
  ServerMessage,
} from './types';

const RESUME_STORAGE_KEY = 'etude.gui.resume';
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

type TableServerMessage = ServerMessage | TestingHouseControlEvent;

interface ResumeCredentials {
  session_id: string;
  resume_token: string;
  presentation_cursor?: number;
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
      ...(presentationCursor === undefined
        ? {}
        : { presentation_cursor: presentationCursor }),
    };
  } catch {
    return null;
  }
}

function saveResumeCredentials(credentials: ResumeCredentials): void {
  if (!browser) {
    return;
  }
  window.sessionStorage.setItem(RESUME_STORAGE_KEY, JSON.stringify(credentials));
}

function clearResumeCredentials(): void {
  if (!browser) {
    return;
  }
  window.sessionStorage.removeItem(RESUME_STORAGE_KEY);
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
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private intentionallyClosed = false;
  private pendingResume = false;
  private outboundQueue: TestingHouseRequest[] = [];
  private inFlightCommand: string | null = null;
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
          this.presentationCursor = credentials.presentation_cursor ?? null;
          this.pendingResume = true;
          this.send({ type: 'resume', ...credentials });
          return;
        }
      }
      this.flushQueue();
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      this.handleRawMessage(event.data);
    };

    socket.onerror = () => {
      gameStore.setError('WebSocket error. Trying to reconnect.');
    };

    socket.onclose = () => {
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

  sendNewGame(config?: Record<string, unknown>): void {
    this.inFlightCommand = null;
    this.presentationCursor = null;
    presentationPlayer.clear();
    gameStore.prepareForNewGame();
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
    if (!frame || !prompt || this.inFlightCommand !== null) {
      return false;
    }
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      gameStore.setError('Connection changed. Recovering the latest game state.');
      this.connect();
      return false;
    }

    const commandId = crypto.randomUUID();
    this.inFlightCommand = commandId;
    gameStore.beginCommand();
    this.socket.send(JSON.stringify({
      type: 'command',
      ...(gameStore.table
        ? { grant_revision: gameStore.table.access.grant_revision }
        : {}),
      command: {
        command_id: commandId,
        match_id: frame.match_id,
        expected_revision: frame.revision,
        prompt_id: prompt.id,
        offer_id: offerId,
        answers: [],
      },
    } satisfies ClientMessage));
    return true;
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

  private send(message: TestingHouseRequest): void {
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
      gameStore.setError('Received invalid server payload.');
      return;
    }

    if ('table' in message && message.table) {
      gameStore.applyTable(message.table);
    }

    if (message.type === 'table_snapshot') {
      this.pendingResume = false;
      this.persistControlCredentials(message);
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
      return;
    }

    if (message.type === 'branch_updated') {
      if (message.phase === 'retry') {
        gameStore.applyBranchRetry(message.payload as BranchRetryPayload);
      }
      return;
    }

    if (message.type === 'branch_returned') {
      gameStore.returnBranch(message.restored as RestoredReplayDecision);
      return;
    }

    if (message.type === 'control_error') {
      gameStore.endFastForward();
      gameStore.setError(message.message);
      return;
    }

    if (message.type === 'observation' || message.type === 'game_over') {
      this.pendingResume = false;
      this.inFlightCommand = null;
      gameStore.endCommand();
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
      this.inFlightCommand = null;
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
      presentationPlayer.clear();
      gameStore.markResumeFailed('Previous session expired or invalid. Start a new game.');
      this.disconnect();
      return;
    }

    gameStore.endFastForward();
    gameStore.endCommand();
    gameStore.setError(message.message);
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
    const labels = mergePresentationLabels(
      presentationPlayer.labels,
      ...(current ? [presentationLabelsFromFrame(current)] : []),
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
      presentationPlayer.clear();
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
    if (
      current
      && current.match_id === next.match_id
      && next.revision < current.revision
    ) {
      return;
    }
    // Recovery always cancels any in-flight theater before resuming its
    // viewer-safe tail. The complete frame remains authoritative either way.
    const presentationLabels = mergePresentationLabels(
      ...(current ? [presentationLabelsFromFrame(current)] : []),
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
      );
      this.persistResumeState(sessionId, resumeToken);
    } catch (error) {
      presentationPlayer.clear();
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
      ...(this.presentationCursor === null
        ? {}
        : { presentation_cursor: this.presentationCursor }),
    });
  }

  private persistControlCredentials(message: TestingHouseControlEvent): void {
    const envelope = message as TestingHouseControlEvent & {
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

export function connect(): void {
  controller.connect();
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
