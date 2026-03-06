import { createLogEntry, deriveObservationNotes } from './log';
import type {
  ActionOption,
  ConnectionState,
  GameLogEntry,
  Observation,
} from './types';

export class GameStore {
  observation = $state<Observation | null>(null);
  actions = $state<ActionOption[]>([]);
  actionLog = $state<GameLogEntry[]>([]);
  gameOver = $state(false);
  winner = $state<number | null>(null);
  errorMessage = $state<string | null>(null);
  connection = $state<ConnectionState>('disconnected');
  focusIds = $state<Set<number>>(new Set());
  sessionId = $state<string | null>(null);
  resumeToken = $state<string | null>(null);
  resumeFailed = $state(false);
  selectedTargetId = $state<number | null>(null);
  villainType = $state<'passive' | 'random'>('passive');

  private logSequence = 0;

  setConnection(next: ConnectionState): void {
    this.connection = next;
  }

  setError(message: string | null): void {
    this.errorMessage = message;
  }

  setVillainType(next: 'passive' | 'random'): void {
    this.villainType = next;
  }

  applyObservation(
    observation: Observation,
    actions: ActionOption[],
    sessionId?: string,
    resumeToken?: string,
    log: string[] = [],
  ): void {
    const previous = this.observation;

    this.observation = observation;
    this.actions = actions;
    this.gameOver = observation.game_over;
    this.winner = null;
    this.errorMessage = null;
    this.resumeFailed = false;
    this.clearFocus();
    this.clearSelectedTarget();

    if (sessionId) {
      this.sessionId = sessionId;
    }
    if (resumeToken) {
      this.resumeToken = resumeToken;
    }

    this.appendVillainLog(log);
    this.appendDerivedNotes(previous, observation);
  }

  applyGameOver(
    observation: Observation,
    winner: number | null,
    log: string[] = [],
  ): void {
    const previous = this.observation;

    this.observation = observation;
    this.actions = [];
    this.gameOver = true;
    this.winner = winner;
    this.errorMessage = null;
    this.resumeFailed = false;
    this.clearFocus();
    this.clearSelectedTarget();

    this.appendVillainLog(log);
    this.appendDerivedNotes(previous, observation);
  }

  prepareForNewGame(): void {
    this.observation = null;
    this.actions = [];
    this.gameOver = false;
    this.errorMessage = null;
    this.resumeFailed = false;
    this.winner = null;
    this.focusIds = new Set();
    this.selectedTargetId = null;
    this.actionLog = [];
    this.logSequence = 0;
  }

  markResumeFailed(message: string): void {
    this.observation = null;
    this.actions = [];
    this.gameOver = false;
    this.winner = null;
    this.actionLog = [];
    this.clearFocus();
    this.clearSelectedTarget();
    this.resumeFailed = true;
    this.errorMessage = message;
    this.sessionId = null;
    this.resumeToken = null;
    this.logSequence = 0;
  }

  appendHeroAction(description: string): void {
    this.actionLog = [
      ...this.actionLog,
      this.createEntry('hero', `Hero: ${description}`),
    ];
  }

  selectTarget(objectId: number): void {
    this.selectedTargetId = objectId;
  }

  clearSelectedTarget(): void {
    this.selectedTargetId = null;
  }

  setFocus(ids: number[]): void {
    this.focusIds = new Set(ids);
  }

  clearFocus(): void {
    this.setFocus([]);
  }

  private appendVillainLog(log: string[]): void {
    if (log.length === 0) {
      return;
    }

    this.actionLog = [
      ...this.actionLog,
      ...log.map((line) => this.createEntry('villain', line)),
    ];
  }

  private appendDerivedNotes(
    previous: Observation | null,
    observation: Observation,
  ): void {
    const notes = deriveObservationNotes(previous, observation);
    if (notes.length === 0) {
      return;
    }

    this.actionLog = [
      ...this.actionLog,
      ...notes.map((note) => this.createEntry('system', note)),
    ];
  }

  private createEntry(
    actor: GameLogEntry['actor'],
    text: string,
    details: string[] = [],
  ): GameLogEntry {
    this.logSequence += 1;
    return createLogEntry(`log-${this.logSequence}`, actor, text, details);
  }
}

export function createGameStore(): GameStore {
  return new GameStore();
}

export const gameStore = createGameStore();
