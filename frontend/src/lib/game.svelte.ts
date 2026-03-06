import { deriveObservationNotes } from './log';
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

    this.appendLogLines('villain', log);
    this.appendLogLines('system', deriveObservationNotes(previous, observation));
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

    this.appendLogLines('villain', log);
    this.appendLogLines('system', deriveObservationNotes(previous, observation));
  }

  prepareForNewGame(): void {
    this.resetMatchState();
    this.errorMessage = null;
    this.resumeFailed = false;
  }

  markResumeFailed(message: string): void {
    this.resetMatchState();
    this.resumeFailed = true;
    this.errorMessage = message;
    this.sessionId = null;
    this.resumeToken = null;
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

  private resetMatchState(): void {
    this.observation = null;
    this.actions = [];
    this.gameOver = false;
    this.winner = null;
    this.actionLog = [];
    this.clearFocus();
    this.clearSelectedTarget();
    this.logSequence = 0;
  }

  private appendLogLines(actor: GameLogEntry['actor'], lines: string[]): void {
    if (lines.length === 0) {
      return;
    }

    this.actionLog = [
      ...this.actionLog,
      ...lines.map((line) => this.createEntry(actor, line)),
    ];
  }

  private createEntry(
    actor: GameLogEntry['actor'],
    text: string,
  ): GameLogEntry {
    this.logSequence += 1;
    return { id: `log-${this.logSequence}`, actor, text };
  }
}

export function createGameStore(): GameStore {
  return new GameStore();
}

export const gameStore = createGameStore();
