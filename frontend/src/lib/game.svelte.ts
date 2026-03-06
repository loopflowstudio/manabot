import type { ActionOption, ConnectionState, Observation } from './types';

export class GameStore {
  observation = $state<Observation | null>(null);
  actions = $state<ActionOption[]>([]);
  gameOver = $state(false);
  winner = $state<number | null>(null);
  errorMessage = $state<string | null>(null);
  connection = $state<ConnectionState>('disconnected');
  focusIds = $state<Set<number>>(new Set());
  sessionId = $state<string | null>(null);
  resumeToken = $state<string | null>(null);
  resumeFailed = $state(false);

  setConnection(next: ConnectionState): void {
    this.connection = next;
  }

  setError(message: string | null): void {
    this.errorMessage = message;
  }

  applyObservation(
    observation: Observation,
    actions: ActionOption[],
    sessionId?: string,
    resumeToken?: string,
  ): void {
    this.observation = observation;
    this.actions = actions;
    this.gameOver = observation.game_over;
    this.winner = null;
    this.errorMessage = null;
    this.resumeFailed = false;

    if (sessionId) {
      this.sessionId = sessionId;
    }
    if (resumeToken) {
      this.resumeToken = resumeToken;
    }
  }

  applyGameOver(observation: Observation, winner: number | null): void {
    this.observation = observation;
    this.actions = [];
    this.gameOver = true;
    this.winner = winner;
    this.errorMessage = null;
    this.resumeFailed = false;
    this.focusIds = new Set();
  }

  prepareForNewGame(): void {
    this.errorMessage = null;
    this.resumeFailed = false;
    this.winner = null;
    this.focusIds = new Set();
  }

  markResumeFailed(message: string): void {
    this.observation = null;
    this.actions = [];
    this.gameOver = false;
    this.winner = null;
    this.focusIds = new Set();
    this.resumeFailed = true;
    this.errorMessage = message;
    this.sessionId = null;
    this.resumeToken = null;
  }

  clearSession(): void {
    this.sessionId = null;
    this.resumeToken = null;
  }

  setFocus(ids: number[]): void {
    this.focusIds = new Set(ids);
  }

  clearFocus(): void {
    this.focusIds = new Set();
  }
}

export function createGameStore(): GameStore {
  return new GameStore();
}

export const gameStore = createGameStore();
