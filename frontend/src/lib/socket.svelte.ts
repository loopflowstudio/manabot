import { browser } from '$app/environment';

import { gameStore } from './game.svelte';
import type { ClientMessage, ServerMessage } from './types';

const RESUME_STORAGE_KEY = 'manabot.gui.resume';

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

    const message = parsed as { type?: string };
    if (!message.type) {
      return null;
    }
    if (message.type !== 'observation' && message.type !== 'game_over' && message.type !== 'error') {
      return null;
    }
    return parsed as ServerMessage;
  } catch {
    return null;
  }
}

class GameSocketController {
  private socket: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private intentionallyClosed = false;
  private pendingResume = false;
  private outboundQueue: ClientMessage[] = [];

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

      const queuedNewGame = this.outboundQueue.some((message) => message.type === 'new_game');
      let attemptedResume = false;
      if (!queuedNewGame) {
        const credentials = loadResumeCredentials();
        if (credentials) {
          this.pendingResume = true;
          attemptedResume = true;
          this.send({ type: 'resume', ...credentials });
        }
      }

      if (!attemptedResume) {
        this.flushQueue();
      }
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
    gameStore.prepareForNewGame();

    if (config) {
      this.send({ type: 'new_game', config });
    } else {
      this.send({ type: 'new_game' });
    }
  }

  sendAction(index: number): void {
    this.send({ type: 'action', index });
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

    if (message.type === 'observation') {
      this.pendingResume = false;
      gameStore.applyObservation(
        message.data,
        message.actions,
        message.session_id,
        message.resume_token,
      );
      if (message.session_id && message.resume_token) {
        saveResumeCredentials({
          session_id: message.session_id,
          resume_token: message.resume_token,
        });
      }
      this.flushQueue();
      return;
    }

    if (message.type === 'game_over') {
      this.pendingResume = false;
      gameStore.applyGameOver(message.data, message.winner);
      this.flushQueue();
      return;
    }

    if (this.pendingResume) {
      this.pendingResume = false;
      clearResumeCredentials();
      this.outboundQueue = [];
      gameStore.markResumeFailed('Previous session expired or invalid. Start a new game.');
      this.disconnect();
      return;
    }

    gameStore.setError(message.message);
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

export function sendAction(index: number): void {
  controller.sendAction(index);
}
