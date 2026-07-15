import { browser } from '$app/environment';

import { gameStore } from './game.svelte';
import {
  presentationLabelsFromFrame,
  validatePresentationUpdate,
} from './presentation';
import { presentationPlayer } from './presentation.svelte';
import type {
  ClientMessage,
  FrameUpdate,
  RecoveryEnvelope,
  ServerMessage,
} from './types';

const RESUME_STORAGE_KEY = 'manabot.gui.resume';
const VALID_SERVER_TYPES = new Set(['observation', 'game_over', 'command_outcome', 'error']);

interface ResumeCredentials {
  session_id: string;
  resume_token: string;
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
    const parsed = JSON.parse(raw) as ResumeCredentials;
    if (!parsed.session_id || !parsed.resume_token) {
      return null;
    }
    return parsed;
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

export class GameSocketController {
  private socket: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private intentionallyClosed = false;
  private pendingResume = false;
  private outboundQueue: ClientMessage[] = [];
  private inFlightCommand: string | null = null;

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

      if (!this.outboundQueue.some((message) => message.type === 'new_game')) {
        const credentials = loadResumeCredentials();
        if (credentials) {
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
    presentationPlayer.clear();
    gameStore.prepareForNewGame();
    this.send(config ? { type: 'new_game', config } : { type: 'new_game' });
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
    this.socket.send(JSON.stringify({
      type: 'command',
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
      gameStore.setError('Received invalid server payload.');
      return;
    }

    if (message.type === 'observation' || message.type === 'game_over') {
      this.pendingResume = false;
      this.inFlightCommand = null;
      if (message.recovery) {
        const sessionId = message.type === 'observation' ? message.session_id : undefined;
        const resumeToken = message.type === 'observation' ? message.resume_token : undefined;
        this.applyRecovery(message.recovery, sessionId, resumeToken);
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
      if (message.type === 'observation' && message.session_id && message.resume_token) {
        saveResumeCredentials({
          session_id: message.session_id,
          resume_token: message.resume_token,
        });
      }
      this.flushQueue();
      return;
    }

    if (message.type === 'command_outcome') {
      this.inFlightCommand = null;
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
    gameStore.applyFrame(update.frame);
    try {
      validatePresentationUpdate(
        update.presentation,
        update.base_revision,
        update.frame.revision,
      );
      presentationPlayer.enqueue(
        update.presentation,
        presentationLabelsFromFrame(update.frame),
      );
    } catch (error) {
      presentationPlayer.clear();
      gameStore.setError(
        error instanceof Error
          ? `Presentation stream rejected: ${error.message}`
          : 'Presentation stream rejected.',
      );
    }
  }

  private applyRecovery(
    recovery: RecoveryEnvelope,
    sessionId?: string,
    resumeToken?: string,
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
    gameStore.applyFrame(next, sessionId, resumeToken);
    try {
      presentationPlayer.recover(
        recovery.presentation_tail,
        presentationLabelsFromFrame(next),
      );
    } catch (error) {
      presentationPlayer.clear();
      gameStore.setError(
        error instanceof Error
          ? `Recovery presentation rejected: ${error.message}`
          : 'Recovery presentation rejected.',
      );
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
