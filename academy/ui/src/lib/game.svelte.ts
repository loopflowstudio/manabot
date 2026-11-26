import { parseSearchConfigs, type SearchConfig, type PendingSearch } from './search';
import {
  loadStoredDeckSelection,
  sanitizeDeckSelection,
  saveStoredDeckSelection,
  type DeckChoice,
  type DeckSelection,
} from './decks';
import { mergeBoards, type SeqRange, type StepBoard } from './board/boards';
import type { HistoryNote } from './board/history';
import type { ResumeRefusal } from './resume-refusal';
import { findKeptDeck, isKeptDeckChoice } from './kept-decks';
import { LIVE_ANALYSIS_OFF, type LiveAnalysis } from './analysis';
export { DECISION_PROMPTS } from './prompt-instructions';
import { defaultStops, loadStoredOnce, loadStoredStops, saveStoredOnce, saveStoredStops, sanitizeStops } from './stops';
import { nextStopState, pruneOnce, stopLabel, withStopState, type OnceStops } from './board/phases';
import type {
  ActionOption,
  WorkStatus,
  CommandRejection,
  ConnectionState,
  DeckNames,
  Decklists,
  ExperienceFrame,
  GameOpponent,
  StepBoardPatch,
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

/** A repeat-deal seed is a decimal unsigned 64-bit integer, kept as text so it never rounds. */
export function seedError(value: string): string | null {
  if (!/^\d+$/.test(value) || BigInt(value) > 18446744073709551615n) return 'Enter a decimal unsigned 64-bit seed (0–18446744073709551615).';
  return null;
}

/** What every `/ws/play` reply says about the game it belongs to. */
export interface GameReply {
  game?: string;
  opponent?: GameOpponent | null;
  opponent_label?: string | null;
  setup?: Record<string, unknown>;
}

/** Both players' imported lists, as setup sends them. */
type DeckImportSelection = {
  hero: { source: string; filename: string; name?: string };
  villain: { source: string; filename: string; name?: string };
};

export class GameStore {
  storageWarning = $state('');
  manaPayment = $state<'auto' | 'manual'>('manual');
  manaFloats = $state<'off' | 'usable' | 'all'>('off');
  humanSeat = $state<0 | 1>(0);
  turnOrder = $state<'play' | 'random' | 'draw'>('play');
  watchMode = $state(false);
  watchPlayerChoice = $state<PlayerChoice>('random');
  watchSearchConfig = $state<SearchConfig | null>(null);
  dealMode = $state<'fresh' | 'repeat'>('fresh');
  seed = $state('0');
  seedError = $derived(this.dealMode === 'repeat' ? seedError(this.seed) : null);

  applySetupDraft(draft: Record<string, unknown>): void {
    const payment = draft.mana_payment === undefined ? 'manual' : draft.mana_payment;
    if (payment !== 'manual' && payment !== 'auto') throw new Error('Invalid mana payment mode');
    this.manaPayment = payment;
    const mode = draft.mana_floats === undefined ? 'off' : draft.mana_floats;
    if (mode !== 'off' && mode !== 'usable' && mode !== 'all') throw new Error('Invalid mana tap mode');
    this.manaFloats = mode;
    this.humanSeat = draft.human_seat === 1 ? 1 : 0;
    this.turnOrder = draft.human_seat === 1 ? 'draw' : 'play';
    this.dealMode = draft.seed == null ? 'fresh' : 'repeat';
    this.seed = String(draft.seed ?? '0');
    // A draft carries both imported lists or none to select: a side dealt
    // from a pool is its pool, and a closed opponent's list is not sent until
    // the game is over (the opponent deck chosen here then stays as it is).
    const imported = draft.deck_import as { hero?: unknown; villain?: unknown } | undefined;
    this.deckImport = imported?.hero && imported.villain ? (imported as DeckImportSelection) : null;
    const named = sanitizeDeckSelection({
      hero: draft.hero_deck_pool ?? draft.hero_deck,
      villain: draft.villain_deck_pool ?? draft.villain_deck ?? this.decks.villain,
    });
    if (named) { this.setHeroDeck(named.hero); this.setVillainDeck(named.villain); }
    this.decklists = draft.decklists === 'closed' ? 'closed' : 'open';
    this.searchConfig = parseSearchConfigs([draft.mirror ?? null, null])[0];
    const stops = sanitizeStops({...((draft.stops ?? {}) as Record<string, unknown>), stop_on_stack:draft.stop_on_stack, auto_pass:draft.auto_pass});
    if (stops) this.stops = stops;
    if (draft.villain_type === 'policy' && typeof draft.opponent_sha256 === 'string') this.opponentChoice = policyChoice(draft.opponent_sha256);
    else if (draft.villain_type === 'search') this.opponentChoice = `search-${draft.villain_sims ?? 64}` as PlayerChoice;
    else if (draft.villain_type === 'random' || draft.villain_type === 'passive') this.opponentChoice = draft.villain_type;
  }

  /** The game on the board, named by its id: also its History row and its Replay. */
  gameId = $state<string | null>(null);
  opponent = $state.raw<GameOpponent | null>(null);
  opponentLabel = $state<string | null>(null);
  /** The setup the game on the board was started with: what Rematch, Swap sides and Play again send back. */
  setup = $state.raw<Record<string, unknown> | null>(null);
  // Frames and projections are the authority's immutable values, replaced
  // whole on every update: raw state, so reading one is a plain property
  // read rather than a deep proxy's (the board reads them thousands of times
  // a frame, and the tab record serializes the frame on every update).
  protocolFrame = $state.raw<ExperienceFrame | null>(null);
  /** Session-kept boards for `matchId` (frames and per-line boards, capped at 512). */
  keptBoards = $state.raw<{ matchId: string; boards: StepBoard[] } | null>(null);
  /** A policy's live reading of your decisions, while you show it; off at the start of every game. */
  liveAnalysis = $state.raw<LiveAnalysis>(LIVE_ANALYSIS_OFF);
  lastCommandRejection = $state<CommandRejection | null>(null);
  observation = $state.raw<Observation | null>(null);
  actions = $state<ActionOption[]>([]);
  // ActionSpaceEnum name for the current decision (PRIORITY, SCRY, ...).
  actionSpaceKind = $state<string>('');
  /** Client-local automation/yield notes anchored to `(turn, afterSeq)` on the track. */
  historyNotes = $state<HistoryNote[]>([]);
  gameOver = $state(false);
  winner = $state<number | null>(null);
  errorMessage = $state<string | null>(null);
  searchRetry = $state<PendingSearch | null>(null);
  searchConfig = $state<SearchConfig | null>(null);
  activeSearchConfig = $state<SearchConfig | null>(null);
  connection = $state<ConnectionState>('disconnected');
  focusIds = $state<Set<number>>(new Set());
  /** The game broke while a request ran: its last board stays readable and takes no input. */
  stopped = $state(false);
  /** A game the server would not take up again, and why; its Replay still opens. */
  resumeRefused = $state<ResumeRefusal | null>(null);
  selectedTargetId = $state<number | null>(null);
  opponentChoice = $state<PlayerChoice>('random');
  decklists = $state<Decklists>('open');
  // Deck pickers (named decks; gui/server.py NAMED_DECKS). Loaded from
  // localStorage and sent with new_game.
  decks = $state<DeckSelection>(loadStoredDeckSelection());
  deckImport = $state<DeckImportSelection | null>(null);
  // Display names for the live matchup, echoed by the server on every
  // payload — what the game header renders.
  deckNames = $state<DeckNames | null>(null);
  // Priority stops. Loaded from localStorage, sent with new_game, and updated
  // in memory by the server's effective-config echo.
  stops = $state<StopsConfig>(loadStoredStops());
  // Stops set to hold once: still ordinary stops in `stops`, cleared by the
  // client when the person leaves a window the game held there (`spendOnce`).
  onceStops = $state<OnceStops>(loadStoredOnce());
  // True while a pass-turn (F6) request is in flight — the server is
  // fast-forwarding priority windows on our behalf.
  fastForwarding = $state(false);
  commandPending = $state(false);
  work = $state<WorkStatus | null>(null);
  submissionAccepted = $state(false);
  newGamePending = $state(false);
  // Monotonic count of applied server updates (observation/game_over).
  // Surfaced in the DOM so tests can serialize on server responses.
  updateSeq = $state(0);

  get decisionStatus(): string {
    if (this.stopped) return 'Stopped — this game takes no more decisions.';
    if (this.gameOver) return 'Game over — no next decision.';
    if (this.connection !== 'connected') return 'Waiting for connection — play is unavailable.';
    if (this.work?.state === 'running') return this.submissionAccepted ? 'Your choice was accepted. The game is working.' : 'The game is working.';
    if (this.commandPending || this.fastForwarding) return 'Waiting for the game — your action is being processed.';
    const prompt = this.protocolFrame?.prompt;
    if (!prompt) return 'Waiting for the next decision.';
    const hero = prompt.actor === this.observation?.agent.player_index;
    return `${hero ? 'Your decision' : "Opponent's decision"} — ${prompt.instruction || prompt.title}`;
  }

  private logSequence = 0;

  /** Take the game a reply names; a new game selects its registered opponent in setup. */
  applyGame(reply: GameReply): void {
    if (!reply.game) return;
    if (reply.opponent && reply.game !== this.gameId) {
      this.opponentChoice = policyChoice(reply.opponent.base_sha256 ?? reply.opponent.sha256);
    }
    this.gameId = reply.game;
    this.opponent = reply.opponent ?? null;
    this.opponentLabel = reply.opponent_label ?? null;
    this.setup = reply.setup ?? null;
    this.resumeRefused = null;
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
    if (next === 'closed') {
      // Built-in rollout Search requires open decklists; switch away so the
      // selection cannot stay on a disabled option. Policy-backed `mirror`
      // search uses physical-deal belief and remains valid on closed lists.
      if (this.opponentChoice.startsWith('search-')) {
        this.opponentChoice = 'random';
      }
      if (this.watchPlayerChoice.startsWith('search-')) {
        this.watchPlayerChoice = 'random';
      }
    }
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
    const base = this.setupConfig();
    const resolvedSeat =
      this.turnOrder === 'random'
        ? (Math.random() < 0.5 ? 0 : 1)
        : this.turnOrder === 'draw'
          ? 1
          : this.humanSeat;
    return { ...base, human_seat: resolvedSeat };
  }

  setupConfig(): Record<string, unknown> {
    let resolvedImport: Record<string, unknown> | null = this.deckImport ? { ...this.deckImport } : null;
    if (!resolvedImport && (isKeptDeckChoice(this.decks.hero) || isKeptDeckChoice(this.decks.villain))) {
      const partial: Record<string, unknown> = {};
      const heroKept = findKeptDeck(this.decks.hero);
      const villainKept = findKeptDeck(this.decks.villain);
      if (heroKept) partial.hero = { source: heroKept.source, filename: heroKept.filename, name: heroKept.name };
      if (villainKept) partial.villain = { source: villainKept.source, filename: villainKept.filename, name: villainKept.name };
      if (partial.hero || partial.villain) resolvedImport = partial;
    }
    return {
      ...this.opponentConfig(),
      human_seat: this.turnOrder === 'draw' ? 1 : this.humanSeat,
      mana_payment: this.manaPayment,
      mana_floats: this.manaFloats,
      seed: this.dealMode === 'repeat' ? this.seed : null,
      ...(resolvedImport ? { deck_import: resolvedImport } : {}),
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

  /** The stop bar's press: off → every time → once → off. */
  cycleStop(side: StopSide, step: string): void {
    const state = !this.stops[side].includes(step) ? 'off' : this.onceStops[side].includes(step) ? 'once' : 'every';
    const next = withStopState(this.stops, this.onceStops, side, step, nextStopState(state));
    this.updateOnce(next.once);
    this.updateStops(next.stops);
  }

  /** The person left a window a once stop held: clear it, and anchor a client-local automation note. */
  spendOnce(side: StopSide, step: string, afterSeq = -1): void {
    const next = withStopState(this.stops, this.onceStops, side, step, 'off');
    this.updateOnce(next.once);
    this.updateStops(next.stops);
    const whose = side === 'my' ? 'your turn' : "the opponent's turn";
    this.addHistoryNote(
      'auto',
      this.observation?.turn.turn_number ?? null,
      afterSeq,
      `Stopped once at ${stopLabel(step)} on ${whose}; that stop is now off`,
    );
  }

  /** Record a client-local history note (a spent once stop or F6 yield). */
  addHistoryNote(actor: 'you' | 'auto', turn: number | null, afterSeq: number, text: string): void {
    this.logSequence += 1;
    this.historyNotes = [
      ...this.historyNotes,
      { id: `note-${this.logSequence}`, actor, turn, afterSeq, text },
    ];
  }

  /** Record client-side automation (e.g. Auto-tap) distinctly from deliberate moves. */
  appendAutoNote(text: string, afterSeq = -1): void {
    this.addHistoryNote(
      'auto',
      this.observation?.turn.turn_number ?? null,
      afterSeq,
      text,
    );
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

  /**
   * The server echoes its stop config on every frame, and answers each
   * `set_stops` with one frame, in order. While stop changes are in flight,
   * an echo that is not the last config sent predates it and is dropped;
   * after as many echoes as sends the server's config holds again.
   */
  private stopsInFlight: { config: string; echoes: number } | null = null;

  noteStopsSent(): void {
    this.stopsInFlight = { config: stopsKey(this.stops), echoes: (this.stopsInFlight?.echoes ?? 0) + 1 };
  }

  applyServerStops(stops: StopsConfig | undefined): void {
    if (!stops) return;
    const flight = this.stopsInFlight;
    if (flight) {
      const settled = stopsKey(stops) === flight.config || flight.echoes <= 1;
      this.stopsInFlight = settled ? null : { ...flight, echoes: flight.echoes - 1 };
      if (!settled) return;
    }
    this.updateStops(stops, false);
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
    stops?: StopsConfig,
    deckNames?: DeckNames,
    actionSpaceKind = '',
  ): void {
    this.updateSeq += 1;
    this.observation = observation;
    this.actions = actions;
    this.actionSpaceKind = actionSpaceKind;
    this.gameOver = observation.game_over;
    this.winner = null;
    this.errorMessage = null;
    this.stopped = false;
    this.fastForwarding = false;
    this.commandPending = false;
    this.clearFocus();
    this.clearSelectedTarget();
    this.applyServerStops(stops);
    this.applyDeckNames(deckNames);
  }

  /**
   * Merge the step boards (and frame board for `range`) that came with `frame`
   * into the session's kept boards (capped at 512). Malformed patches are
   * dropped without disturbing previously kept boards.
   */
  applyStepBoards(
    frame: ExperienceFrame,
    patches: readonly StepBoardPatch[],
    range: SeqRange | null = null,
  ): void {
    const existing = this.keptBoards?.matchId === frame.match_id ? this.keptBoards.boards : [];
    let next = existing;
    try {
      next = mergeBoards(existing, frame.projection, patches, range);
    } catch {
      next = existing;
    }
    this.keptBoards = { matchId: frame.match_id, boards: next };
  }

  applyFrame(frame: ExperienceFrame): void {
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
        frame.stops,
        frame.deck_names,
      );
    } else {
      this.applyObservation(
        frame.projection,
        actions,
        frame.stops,
        frame.deck_names,
        frame.action_space,
      );
    }
  }

  applyGameOver(
    observation: Observation,
    winner: number | null,
    stops?: StopsConfig,
    deckNames?: DeckNames,
  ): void {
    this.updateSeq += 1;
    this.observation = observation;
    this.actions = [];
    this.actionSpaceKind = '';
    this.gameOver = true;
    this.winner = winner;
    this.errorMessage = null;
    this.stopped = false;
    this.fastForwarding = false;
    this.clearFocus();
    this.clearSelectedTarget();
    this.applyServerStops(stops);
    this.applyDeckNames(deckNames);
  }

  prepareForNewGame(): void {
    this.resetMatchState();
    this.errorMessage = null;
    this.stopped = false;
  }

  /** Leave the board for setup: the game it showed stays unfinished in History. */
  leaveGame(): void {
    this.resetMatchState();
    this.gameId = null;
    this.opponent = null;
    this.opponentLabel = null;
    this.setup = null;
    this.stopped = false;
    this.errorMessage = null;
    this.newGamePending = false;
  }

  markStopped(message: string): void {
    // Keep the last confirmed board readable, but retire every input.
    this.fastForwarding = false;
    this.commandPending = false;
    this.newGamePending = false;
    this.searchRetry = null;
    this.work = null;
    this.submissionAccepted = false;
    this.lastCommandRejection = null;
    this.stopped = true;
    this.errorMessage = message;
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

  private updateStops(next: StopsConfig, persist = true): void {
    this.stops = next;
    if (persist) saveStoredStops(next);
    const once = pruneOnce(this.onceStops, next);
    if (once.my.length !== this.onceStops.my.length || once.opponent.length !== this.onceStops.opponent.length) this.updateOnce(once);
  }

  private updateOnce(next: OnceStops): void {
    this.onceStops = next;
    saveStoredOnce(next);
  }

  private updateDecks(next: DeckSelection): void {
    this.decks = next;
    saveStoredDeckSelection(next);
  }

  private applyDeckNames(deckNames: DeckNames | undefined): void {
    if (deckNames) {
      this.deckNames = deckNames;
    }
  }

  private resetMatchState(): void {
    this.liveAnalysis = LIVE_ANALYSIS_OFF;
    this.searchRetry = null;
    this.work = null;
    this.submissionAccepted = false;
    this.lastCommandRejection = null;
    this.protocolFrame = null;
    this.keptBoards = null;
    this.observation = null;
    this.actions = [];
    this.actionSpaceKind = '';
    this.gameOver = false;
    this.winner = null;
    this.historyNotes = [];
    this.fastForwarding = false;
    this.commandPending = false;
    this.clearFocus();
    this.clearSelectedTarget();
    this.logSequence = 0;
  }
}

function stopsKey(stops: StopsConfig): string {
  return JSON.stringify([[...stops.my].sort(), [...stops.opponent].sort(), stops.stop_on_stack, stops.auto_pass]);
}

export function createGameStore(): GameStore {
  return new GameStore();
}

export const gameStore = createGameStore();
