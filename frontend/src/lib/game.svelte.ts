import { deriveObservationNotes } from './log';
import { defaultStops, loadStoredStops, saveStoredStops } from './stops';
import type {
  ActionOption,
  ConnectionState,
  GameLogEntry,
  Observation,
  OpponentConfig,
  StopsConfig,
  StopSide,
} from './types';

export type OpponentChoice =
  | 'search-16'
  | 'search-64'
  | 'search-256'
  | 'random'
  | 'passive'
  | 'checkpoint';

export function buildOpponentConfig(
  choice: OpponentChoice,
  checkpointPath: string,
  checkpointDeterministic: boolean,
): OpponentConfig {
  if (choice === 'checkpoint') {
    return {
      villain_type: 'checkpoint',
      villain_checkpoint: checkpointPath.trim(),
      villain_deterministic: checkpointDeterministic,
    };
  }
  if (choice === 'random' || choice === 'passive') {
    return { villain_type: choice };
  }
  const sims = Number(choice.split('-')[1]);
  return { villain_type: 'search', villain_sims: sims };
}

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
  opponentChoice = $state<OpponentChoice>('search-64');
  checkpointPath = $state('');
  checkpointDeterministic = $state(false);
  // Priority stops (MTGO-style auto-pass). Loaded from localStorage, sent
  // with new_game, and overwritten by the server's effective-config echo.
  stops = $state<StopsConfig>(loadStoredStops());
  // True while a pass-turn (F6) request is in flight — the server is
  // fast-forwarding priority windows on our behalf.
  fastForwarding = $state(false);
  // Monotonic count of applied server updates (observation/game_over).
  // Surfaced in the DOM so tests can serialize on server responses.
  updateSeq = $state(0);

  private logSequence = 0;

  setConnection(next: ConnectionState): void {
    this.connection = next;
  }

  setError(message: string | null): void {
    this.errorMessage = message;
  }

  setOpponentChoice(next: OpponentChoice): void {
    this.opponentChoice = next;
  }

  setCheckpointPath(next: string): void {
    this.checkpointPath = next;
  }

  setCheckpointDeterministic(next: boolean): void {
    this.checkpointDeterministic = next;
  }

  opponentConfig(): OpponentConfig {
    return buildOpponentConfig(
      this.opponentChoice,
      this.checkpointPath,
      this.checkpointDeterministic,
    );
  }

  newGameConfig(): Record<string, unknown> {
    return {
      ...this.opponentConfig(),
      stops: { my: [...this.stops.my], opponent: [...this.stops.opponent] },
      stop_on_stack: this.stops.stop_on_stack,
      auto_pass: this.stops.auto_pass,
    };
  }

  toggleStop(side: StopSide, step: string): void {
    const steps = this.stops[side].includes(step)
      ? this.stops[side].filter((existing) => existing !== step)
      : [...this.stops[side], step];
    this.updateStops({ ...this.stops, [side]: steps });
  }

  setStopOnStack(value: boolean): void {
    this.updateStops({ ...this.stops, stop_on_stack: value });
  }

  setAutoPass(value: boolean): void {
    this.updateStops({ ...this.stops, auto_pass: value });
  }

  resetStops(): void {
    this.updateStops(defaultStops());
  }

  applyServerStops(stops: StopsConfig | undefined): void {
    if (stops) {
      this.updateStops(stops);
    }
  }

  beginFastForward(): void {
    this.fastForwarding = true;
  }

  endFastForward(): void {
    this.fastForwarding = false;
  }

  applyObservation(
    observation: Observation,
    actions: ActionOption[],
    sessionId?: string,
    resumeToken?: string,
    log: string[] = [],
    stops?: StopsConfig,
    autoPassed = 0,
  ): void {
    const previous = this.observation;

    this.updateSeq += 1;
    this.observation = observation;
    this.actions = actions;
    this.gameOver = observation.game_over;
    this.winner = null;
    this.errorMessage = null;
    this.resumeFailed = false;
    this.fastForwarding = false;
    this.clearFocus();
    this.clearSelectedTarget();
    this.applyServerStops(stops);

    if (sessionId) {
      this.sessionId = sessionId;
    }
    if (resumeToken) {
      this.resumeToken = resumeToken;
    }

    this.appendLogLines('villain', log);
    this.appendAutoPassNote(autoPassed);
    this.appendLogLines('system', deriveObservationNotes(previous, observation));
  }

  applyGameOver(
    observation: Observation,
    winner: number | null,
    log: string[] = [],
    stops?: StopsConfig,
    autoPassed = 0,
  ): void {
    const previous = this.observation;

    this.updateSeq += 1;
    this.observation = observation;
    this.actions = [];
    this.gameOver = true;
    this.winner = winner;
    this.errorMessage = null;
    this.resumeFailed = false;
    this.fastForwarding = false;
    this.clearFocus();
    this.clearSelectedTarget();
    this.applyServerStops(stops);

    this.appendLogLines('villain', log);
    this.appendAutoPassNote(autoPassed);
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

  private updateStops(next: StopsConfig): void {
    this.stops = next;
    saveStoredStops(next);
  }

  private appendAutoPassNote(autoPassed: number): void {
    if (autoPassed <= 0) {
      return;
    }
    const windows = autoPassed === 1 ? 'window' : 'windows';
    this.appendLogLines('system', [
      `Auto-passed ${autoPassed} priority ${windows}.`,
    ]);
  }

  private resetMatchState(): void {
    this.observation = null;
    this.actions = [];
    this.gameOver = false;
    this.winner = null;
    this.actionLog = [];
    this.fastForwarding = false;
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
