import { seedError, type PlaySession } from './play-session';
import { playerToken } from './records';
import { parseSearchConfigs, type SearchConfig, type PendingSearch } from './search';
import {
  loadStoredDeckSelection,
  sanitizeDeckSelection,
  saveStoredDeckSelection,
  type DeckChoice,
  type DeckSelection,
} from './decks';
import { deriveObservationNotes } from './log';
export { DECISION_PROMPTS } from './prompt-instructions';
import { defaultStops, loadStoredStops, saveStoredStops, sanitizeStops } from './stops';
import type { RestoredReplayDecision } from './replay-index';
import type {
  BranchRetryPayload,
  TableCapability,
  TableSnapshot,
} from './shared-table-protocol';
import type {
  ActionOption,
  WorkStatus,
  CommandRejection,
  ConnectionState,
  DeckNames,
  Decklists,
  ExperienceFrame,
  GameLogEntry,
  LogLine,
  Observation,
  OpponentConfig,
  StopsConfig,
  StopSide,
} from './types';

// Registered policies are chosen by complete execution identity: `policy:<id>`.
export type PlayerChoice =
  | `search-${number}`
  | 'random'
  | 'passive'
  | 'search'
  | 'greedy'
  | 'heuristic'
  | `policy:${string}`;

export function policyChoice(sha256: string): PlayerChoice {
  return `policy:${sha256}`;
}

// Search opponents sample hidden cards from the opponent's decklists, so the
// server refuses them in closed-decklist games.
export function isSearchChoice(choice: PlayerChoice): boolean {
  return choice.startsWith('search-');
}

export function buildOpponentConfig(choice: PlayerChoice): OpponentConfig {
  if (choice.startsWith('policy:')) {
    return {
      villain_type: 'policy',
      opponent_sha256: choice.slice('policy:'.length),
    };
  }
  if (choice === 'random' || choice === 'passive') {
    return { villain_type: choice };
  }
  const sims = Number(choice.split('-')[1]);
  return { villain_type: 'search', villain_sims: sims };
}

export class GameStore {
  constructor(private readonly persistPreferences = true) {}

  /** An isolated editor: nested source/search values must not mutate live setup. */
  editableSetup(draft: Record<string, unknown> = this.setupConfig()): GameStore {
    const editor = new GameStore(false);
    editor.applySetupDraft(JSON.parse(JSON.stringify(draft)));
    editor.watchMode = this.watchMode;
    editor.watchPlayerChoice = this.watchPlayerChoice;
    editor.watchSearchConfig = this.watchSearchConfig ? {...this.watchSearchConfig} : null;
    return editor;
  }

  playSession = $state<PlaySession | null>(null);
  storageWarning = $state('');
  tableUnavailable = $state(false);
  humanSeat = $state<0 | 1>(0);
  watchMode = $state(false);
  watchPlayerChoice = $state<PlayerChoice>('random');
  watchSearchConfig = $state<SearchConfig | null>(null);
  dealMode = $state<'fresh' | 'repeat'>('fresh');
  seed = $state('0');
  seedError = $derived(this.dealMode === 'repeat' ? seedError(this.seed) : null);

  applyPlaySession(session: PlaySession): void {
    const changed = session.game_number !== this.playSession?.game_number || session.id !== this.playSession?.id;
    this.playSession = session;
    if (!changed) return;
    this.watchMode = false;
    this.applySetupDraft(session.draft);
  }

  applySetupDraft(draft: Record<string, unknown>): void {
    this.humanSeat = draft.human_seat === 1 ? 1 : 0;
    this.dealMode = draft.seed == null ? 'fresh' : 'repeat';
    this.seed = String(draft.seed ?? '0');
    this.deckImport = (draft.deck_import as typeof this.deckImport) ?? null;
    const named = sanitizeDeckSelection({hero:draft.hero_deck, villain:draft.villain_deck});
    if (named) { this.setHeroDeck(named.hero); this.setVillainDeck(named.villain); }
    this.decklists = draft.decklists === 'closed' ? 'closed' : 'open';
    this.searchConfig = parseSearchConfigs([draft.mirror ?? null, null])[0];
    const stops = sanitizeStops({...((draft.stops ?? {}) as Record<string, unknown>), stop_on_stack:draft.stop_on_stack, auto_pass:draft.auto_pass});
    if (stops) this.stops = stops;
    if (draft.villain_type === 'policy' && typeof draft.opponent_sha256 === 'string') this.opponentChoice = policyChoice(draft.opponent_sha256);
    else if (draft.villain_type === 'search') this.opponentChoice = `search-${draft.villain_sims ?? 64}` as PlayerChoice;
    else if (draft.villain_type === 'random' || draft.villain_type === 'passive') this.opponentChoice = draft.villain_type;
  }

  table = $state<TableSnapshot | null>(null);
  // Frames and projections are the authority's immutable values, replaced
  // whole on every update: raw state, so reading one is a plain property
  // read rather than a deep proxy's (the table reads them thousands of times
  // a frame, and the resume record serializes the frame on every update).
  restoredDecision = $state.raw<RestoredReplayDecision | null>(null);
  branchAttemptId = $state<string | null>(null);
  branchProjection = $state.raw<Observation | null>(null);
  tableAnnouncement = $state('');
  protocolFrame = $state.raw<ExperienceFrame | null>(null);
  lastCommandRejection = $state<CommandRejection | null>(null);
  observation = $state.raw<Observation | null>(null);
  actions = $state<ActionOption[]>([]);
  // ActionSpaceEnum name for the current decision (PRIORITY, SCRY, ...).
  actionSpaceKind = $state<string>('');
  actionLog = $state<GameLogEntry[]>([]);
  gameOver = $state(false);
  winner = $state<number | null>(null);
  errorMessage = $state<string | null>(null);
  searchRetry = $state<PendingSearch | null>(null);
  searchConfig = $state<SearchConfig | null>(null);
  activeSearchConfig = $state<SearchConfig | null>(null);
  connection = $state<ConnectionState>('disconnected');
  focusIds = $state<Set<number>>(new Set());
  sessionId = $state<string | null>(null);
  resumeToken = $state<string | null>(null);
  resumeFailed = $state(false);
  recordingId = $state<string | null>(null);
  selectedTargetId = $state<number | null>(null);
  opponentChoice = $state<PlayerChoice>('search-64');
  decklists = $state<Decklists>('open');
  // Deck pickers (named decks; gui/server.py NAMED_DECKS). Loaded from
  // localStorage and sent with new_game.
  decks = $state<DeckSelection>(loadStoredDeckSelection());
  deckImport = $state<{
    hero: { source: string; filename: string };
    villain: { source: string; filename: string };
  } | null>(null);
  // Display names for the live matchup, echoed by the server on every
  // payload — what the game header renders.
  deckNames = $state<DeckNames | null>(null);
  // Priority stops (MTGO-style auto-pass). Loaded from localStorage, sent
  // with new_game, and overwritten by the server's effective-config echo.
  stops = $state<StopsConfig>(loadStoredStops());
  // True while a pass-turn (F6) request is in flight — the server is
  // fast-forwarding priority windows on our behalf.
  fastForwarding = $state(false);
  commandPending = $state(false);
  work = $state<WorkStatus | null>(null);
  interruptedIdentity = $state<Pick<TableSnapshot, 'opponent' | 'opponent_label'> | null>(null);
  submissionAccepted = $state(false);
  newGamePending = $state(false);
  // Monotonic count of applied server updates (observation/game_over).
  // Surfaced in the DOM so tests can serialize on server responses.
  updateSeq = $state(0);

  get decisionStatus(): string {
    if (this.resumeFailed) return 'Interrupted — the live session is unavailable.';
    if (this.restoredDecision) return 'Study position — return to live to play.';
    if (this.gameOver) return 'Game over — no next decision.';
    if (this.connection !== 'connected') return 'Waiting for connection — play is unavailable.';
    if (this.work?.state === 'running') return this.submissionAccepted ? 'Your choice was accepted. The table is working.' : 'The table is working.';
    if (this.commandPending || this.fastForwarding) return 'Waiting for the table — your action is being processed.';
    const prompt = this.protocolFrame?.prompt;
    if (!prompt) return 'Waiting for the next decision.';
    const hero = prompt.actor === this.observation?.agent.player_index;
    const actor = hero
      ? (this.hasCapability('submit_live_command') ? 'Your decision' : "Pilot's decision")
      : "Opponent's decision";
    return `${actor} — ${prompt.instruction || prompt.title}`;
  }

  private logSequence = 0;

  applyTable(table: TableSnapshot | null | undefined): void {
    if (!table) return;
    const previousRole = this.table?.access.role;
    if (table.opponent && table.attempt_id !== this.table?.attempt_id) {
      this.opponentChoice = policyChoice(table.opponent.base_sha256 ?? table.opponent.sha256);
    }
    this.table = table;
    this.recordingId = table.attempt_id ?? null;
    if (previousRole && previousRole !== table.access.role) {
      this.tableAnnouncement = `You are now the ${table.access.role}.`;
    }
  }

  hasCapability(capability: TableCapability): boolean {
    return this.table?.access.capabilities.includes(capability) ?? true;
  }

  restoreDecision(restored: RestoredReplayDecision): void {
    this.restoredDecision = restored;
    this.branchAttemptId = null;
    this.branchProjection = null;
    this.tableAnnouncement = `Decision ${restored.ordinal + 1} restored for isolated Study.`;
  }

  applyBranchRetry(payload: BranchRetryPayload): void {
    this.branchAttemptId = payload.attempt_id;
    this.branchProjection = payload.retry.projection;
    this.tableAnnouncement = 'Isolated line updated. The live table was not changed.';
  }

  returnBranch(restored: RestoredReplayDecision): void {
    this.restoredDecision = restored;
    this.branchAttemptId = null;
    this.branchProjection = null;
    this.tableAnnouncement = `Returned to recorded decision ${restored.ordinal + 1}.`;
  }

  returnToLive(): void {
    this.restoredDecision = null;
    this.branchAttemptId = null;
    this.branchProjection = null;
    this.tableAnnouncement = 'Returned to the live table.';
  }

  setConnection(next: ConnectionState): void {
    this.connection = next;
  }

  setError(message: string | null): void {
    this.errorMessage = message;
  }

  setOpponentChoice(next: PlayerChoice): void {
    this.opponentChoice = next;
  }

  setDecklists(next: Decklists): void {
    this.decklists = next;
  }

  setHeroDeck(next: DeckChoice): void {
    this.updateDecks({ ...this.decks, hero: next });
  }

  setVillainDeck(next: DeckChoice): void {
    this.updateDecks({ ...this.decks, villain: next });
  }

  opponentConfig(): OpponentConfig {
    return buildOpponentConfig(this.opponentChoice);
  }

  newGameConfig(): Record<string, unknown> {
    return {
      ...this.setupConfig(),
      request_id: crypto.randomUUID(),
      player_token: playerToken(),
    };
  }

  /** Setup data has no request identity or participant credential. */
  setupConfig(): Record<string, unknown> {
    return {
      ...this.opponentConfig(),
      human_seat: this.humanSeat,
      seed: this.dealMode === 'repeat' ? this.seed : null,
      ...(this.deckImport ? { deck_import: this.deckImport } : {}),
      hero_deck: this.decks.hero,
      villain_deck: this.decks.villain,
      stops: { my: [...this.stops.my], opponent: [...this.stops.opponent] },
      stop_on_stack: this.stops.stop_on_stack,
      auto_pass: this.stops.auto_pass,
      decklists: this.decklists,
      mirror: this.searchConfig,
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

  beginCommand(): void {
    this.lastCommandRejection = null;
    this.submissionAccepted = false;
    this.commandPending = true;
  }

  endCommand(): void {
    this.commandPending = false;
  }

  endFastForward(): void {
    this.fastForwarding = false;
  }

  applyObservation(
    observation: Observation,
    actions: ActionOption[],
    sessionId?: string,
    resumeToken?: string,
    log: LogLine[] = [],
    stops?: StopsConfig,
    deckNames?: DeckNames,
    actionSpaceKind = '',
    deriveNotes = true,
  ): void {
    const previous = this.observation;

    this.updateSeq += 1;
    this.observation = observation;
    this.actions = actions;
    this.actionSpaceKind = actionSpaceKind;
    this.gameOver = observation.game_over;
    this.winner = null;
    this.errorMessage = null;
    this.resumeFailed = false;
    this.fastForwarding = false;
    this.commandPending = false;
    this.clearFocus();
    this.clearSelectedTarget();
    this.applyServerStops(stops);
    this.applyDeckNames(deckNames);

    if (sessionId) {
      this.sessionId = sessionId;
    }
    if (resumeToken) {
      this.resumeToken = resumeToken;
    }

    this.appendServerLog(log);
    if (deriveNotes) {
      this.appendNotes(deriveObservationNotes(previous, observation));
    }
  }

  applyFrame(
    frame: ExperienceFrame,
    sessionId?: string,
    resumeToken?: string,
  ): void {
    this.lastCommandRejection = null;
    this.protocolFrame = frame;
    const actions = frame.offers.map((offer) => ({
      index: offer.id,
      type: offer.action_type,
      focus: [...offer.focus],
      description: offer.label,
    }));
    if (frame.projection.game_over) {
      this.applyGameOver(
        frame.projection,
        frame.winner,
        frame.log ?? [],
        frame.stops,
        frame.deck_names,
        false,
      );
    } else {
      this.applyObservation(
        frame.projection,
        actions,
        sessionId,
        resumeToken,
        frame.log ?? [],
        frame.stops,
        frame.deck_names,
        frame.action_space,
        false,
      );
    }
  }

  applyGameOver(
    observation: Observation,
    winner: number | null,
    log: LogLine[] = [],
    stops?: StopsConfig,
    deckNames?: DeckNames,
    deriveNotes = true,
  ): void {
    const previous = this.observation;

    this.updateSeq += 1;
    this.observation = observation;
    this.actions = [];
    this.actionSpaceKind = '';
    this.gameOver = true;
    this.winner = winner;
    this.errorMessage = null;
    this.resumeFailed = false;
    this.fastForwarding = false;
    this.clearFocus();
    this.clearSelectedTarget();
    this.applyServerStops(stops);
    this.applyDeckNames(deckNames);

    this.appendServerLog(log);
    if (deriveNotes) {
      this.appendNotes(deriveObservationNotes(previous, observation));
    }
  }

  prepareForNewGame(): void {
    this.resetMatchState();
    this.errorMessage = null;
    this.resumeFailed = false;
  }

  markResumeFailed(message: string): void {
    // Keep the last confirmed table readable, but retire every input authority.
    this.fastForwarding = false;
    this.commandPending = false;
    this.newGamePending = false;
    this.searchRetry = null;
    this.work = null;
    this.submissionAccepted = false;
    this.lastCommandRejection = null;
    if (this.table?.opponent || this.table?.opponent_label) this.interruptedIdentity = {opponent: this.table.opponent, opponent_label: this.table.opponent_label};
    this.table = null;
    this.restoredDecision = null;
    this.branchProjection = null;
    this.branchAttemptId = null;
    this.resumeFailed = true;
    this.errorMessage = message;
    this.sessionId = null;
    this.resumeToken = null;
  }

  /** The person's own move, in the turn of the decision it answered. */
  appendHeroAction(description: string): void {
    this.actionLog = [
      ...this.actionLog,
      this.createEntry('hero', this.observation?.turn.turn_number ?? null, description),
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
    if (this.persistPreferences) saveStoredStops(next);
  }

  private updateDecks(next: DeckSelection): void {
    this.decks = next;
    if (this.persistPreferences) saveStoredDeckSelection(next);
  }

  private applyDeckNames(deckNames: DeckNames | undefined): void {
    if (deckNames) {
      this.deckNames = deckNames;
    }
  }

  private resetMatchState(): void {
    this.interruptedIdentity = null;
    this.searchRetry = null;
    this.work = null;
    this.submissionAccepted = false;
    this.lastCommandRejection = null;
    this.protocolFrame = null;
    this.observation = null;
    this.actions = [];
    this.actionSpaceKind = '';
    this.gameOver = false;
    this.winner = null;
    this.actionLog = [];
    this.fastForwarding = false;
    this.commandPending = false;
    this.clearFocus();
    this.clearSelectedTarget();
    this.logSequence = 0;
    this.restoredDecision = null;
    this.branchAttemptId = null;
    this.branchProjection = null;
  }

  /** Opponent and automated moves, each in the turn the server says it was taken. */
  private appendServerLog(lines: LogLine[]): void {
    if (lines.length === 0) return;
    this.actionLog = [...this.actionLog, ...lines.map(line => this.createEntry(line.actor, line.turn, line.text))];
  }

  private appendNotes(notes: string[]): void {
    if (notes.length === 0) return;
    const turn = this.observation?.turn.turn_number ?? null;
    this.actionLog = [...this.actionLog, ...notes.map(note => this.createEntry('system', turn, note))];
  }

  private createEntry(actor: GameLogEntry['actor'], turn: number | null, text: string): GameLogEntry {
    this.logSequence += 1;
    return { id: `log-${this.logSequence}`, actor, turn, text };
  }
}

export function createGameStore(): GameStore {
  return new GameStore();
}

export const gameStore = createGameStore();
