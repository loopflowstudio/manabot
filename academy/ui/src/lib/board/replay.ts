/**
 * A recorded game on the board.
 *
 * Replay, Watch and Review draw the same `Board` as live play, read-only. This
 * module turns the recording's read model into what the board takes:
 *
 * - `recordedFrame(position, …)`: the frame for one position. At a recorded
 *   choice it is the decision's own frame, with its offers and prompt, so the
 *   chooser's legal space is drawn exactly as it was offered; elsewhere it is
 *   the recorded board with no prompt and no offers.
 * - `bottomSeat(…)`: which seat sits at the bottom. Following the chooser keeps
 *   each decision in its chooser's acting view, hand included.
 * - `offerEvidence(…)`: the recorded policy, index-aligned with `frame.offers`,
 *   keyed by offer id for the cards and rows that draw each offer.
 * - `replayLog(…)` and `stripMarks(…)`: the move list and the decision strip,
 *   read from the index alone (each position's label, value and revision).
 * - `sinceRevision(…)`: where "since your last move" starts at a position: the
 *   chooser's previous decision, or the transition itself.
 * - `positionHeading(…)`: what the shown position is, in words.
 *
 * Nothing here reads a later position to describe an earlier one, and history
 * stays the acting viewer's recorded track: labels come from the index, never
 * from a spectator reveal.
 */
import { orientBoard, type BotDecision, type BranchKind, type Position, type RecordedPosition, type RecordingIndex, type SeatSwap } from '$lib/recording';
import { decisionMark, firstChoice, score, type AnalysedDecision, type DecisionMark, type LivePoint } from '$lib/analysis';
import { checkpointIdentity, playerIdentity, recordedIdentity, type PlayerIdentity } from '$lib/player-identity';
import type { ExperienceFrame, GameLogEntry, PresentationEvent } from '$lib/types';
import { mergeBoards, type StepBoard } from './boards';
import type { HistoryNote, HistoryTurn } from './history';
import { possessive } from './view';
import { PROTOCOL_VERSION } from '$lib/types';

// ———————————————————————————————————————— Perspective

/** Who sits at the bottom: a fixed seat, or whoever chose at this position. */
export type Perspective = 0 | 1 | 'chooser';

/** Native seat the human piloted at `eventIndex` (or at game end when null). */
export function humanSeatAt(startingSeat: 0 | 1, swaps: readonly SeatSwap[], eventIndex: number | null): 0 | 1 {
  if (!swaps.length) return startingSeat;
  const flips = eventIndex === null
    ? swaps.length
    : swaps.filter(swap => swap.event <= eventIndex).length;
  return ((startingSeat ^ (flips & 1)) as 0 | 1);
}

/** Native seat in `index.players` whose pilot controlled `seat` when `currentHeroSeat` was the human's seat. */
export function pilotSeatAt(seat: 0 | 1, finalHeroSeat: 0 | 1, currentHeroSeat: 0 | 1): 0 | 1 {
  return seat === currentHeroSeat ? finalHeroSeat : ((1 - finalHeroSeat) as 0 | 1);
}

/**
 * A seat's recorded identity; a record that predates recorded players falls back to
 * its configured opponent. In a Play game the bot's seat (whichever seat it finished in)
 * carries the checkpoint's producer and provenance.
 */
export function seatIdentity(
  index: Pick<RecordingIndex, 'players' | 'opponent' | 'opponent_checkpoint' | 'watch'> | null | undefined,
  seat: number,
): PlayerIdentity {
  const recorded = index?.players.find(player => player.seat === seat);
  const opponent = recorded
    ? (!index?.watch && recorded.kind === 'bot' ? index?.opponent : null)
    : (seat === 1 ? index?.opponent : null);
  const base = recorded ? recordedIdentity(recorded)
    : opponent ? checkpointIdentity(opponent) : playerIdentity({ name: `Seat ${seat + 1}` });
  return opponent ? { ...base, producer: opponent.producer, provenance: index?.opponent_checkpoint ?? undefined } : base;
}

/** Fixed-perspective dropdown label: names the pilot when unswapped, or the permanent native seat and deck when swapped. */
export function perspectiveLabel(seat: 0 | 1, name: string, deck: string | null, swapped: boolean): string {
  const base = `Seat ${seat + 1} at the bottom`;
  if (!swapped) return `${base} · ${name}`;
  return deck ? `${base} · ${deck}` : base;
}

/** The address's `near` value, or the recording's natural seat: chooser when swapped or watched, else the person. */
export function perspectiveOf(near: string | null, index: Pick<RecordingIndex, 'players' | 'seat_swaps'>): Perspective {
  if (near === '0' || near === '1') return Number(near) as 0 | 1;
  if (near === 'chooser') return 'chooser';
  if (index.seat_swaps?.length) return 'chooser';
  const human = index.players.find(player => player.kind === 'human');
  return human ? (human.seat === 1 ? 1 : 0) : 'chooser';
}

/** The seat drawn at the bottom at `selected`: a chooser stays at the bottom through its consequences. */
export function bottomSeat(perspective: Perspective, positions: readonly Position[], selected: number): 0 | 1 {
  if (perspective !== 'chooser') return perspective;
  for (let i = Math.min(selected, positions.length - 1); i >= 0; i--) {
    const actor = positions[i].actor;
    if (positions[i].kind === 'choice' && actor !== null) return actor === 1 ? 1 : 0;
  }
  return 0;
}

// ———————————————————————————————————————— Frames

/**
 * The frame the board draws at a recorded position, seen with `seat` at the
 * bottom. A choice keeps its frame's offers and prompt; a revealed choice
 * draws the spectator board under the same offers.
 */
export function recordedFrame(position: RecordedPosition, index: Pick<RecordingIndex, 'id' | 'revision' | 'winner'>, seat: 0 | 1): ExperienceFrame {
  const decision = position.kind === 'choice' ? position.decision : null;
  const observation = orientBoard(position.observation, seat, index.winner);
  if (decision) return { ...decision.frame, projection: observation };
  return {
    protocol: PROTOCOL_VERSION,
    match_id: index.id,
    // The board after a transition is one revision past the decision it follows.
    revision: position.kind === 'state' ? index.revision ?? 0 : (position.revision ?? 0) + 1,
    frame_hash: '',
    content_hash: '',
    asset_manifest_hash: '',
    status: observation.game_over ? 'game_over' : 'ready',
    prompt: null,
    projection: observation,
    offers: [],
    winner: index.winner,
    action_space: '',
    stops: { my: [], opponent: [], stop_on_stack: false, auto_pass: false },
  };
}

// ———————————————————————————————————————— Recorded policy

export interface OfferEvidence {
  /** The recorded probability of each offer, by offer id; absent when not retained. */
  probability: ReadonlyMap<number, number>;
  /** The offer the chooser committed. */
  chosen: number | null;
  /** The decision's recorded value and provenance, one line each, "not retained" where absent. */
  summary: string[];
  /** With search: each offer's raw policy, search score and updated policy, in offer order. */
  search: SearchRow[] | null;
  /** Search and belief provenance: worlds, belief model, seed and receipt. */
  provenance: string[];
  /** From an analysis: each offer's estimate after it, 0–1 from your seat, by offer id. */
  estimate: ReadonlyMap<number, number>;
  /** From an analysis: the policy's first choice, by offer id. */
  first: number | null;
  /** What the choice list is headed: "Analysis by Reader" for an analysis, else null. */
  heading: string | null;
}

export interface SearchRow {
  id: number;
  label: string;
  prior: number | null;
  score: number | null;
  updated: number | null;
}

const percent = (p: number): string => `${(p * 100).toFixed(p >= 0.995 || p < 0.001 ? 0 : 1)}%`;

/** Formats a recorded probability for a card or row: "37.5%". */
export function formatProbability(p: number): string { return percent(p); }

/**
 * What was recorded at a choice, aligned with its frame's offers. With search,
 * the cards show the updated policy (what was played from), and the summary
 * names the search that produced it.
 */
export function offerEvidence(frame: ExperienceFrame, outputs: BotDecision | null, chosen: number | null): OfferEvidence {
  const probability = new Map<number, number>();
  frame.offers.forEach((offer, i) => {
    const p = outputs?.probabilities[i];
    if (typeof p === 'number') probability.set(offer.id, p);
  });
  const summary: string[] = [];
  const value = outputs?.value;
  summary.push(typeof value === 'number' ? `Value ${value.toFixed(3)}${outputs?.value_semantic ? ` · ${outputs.value_semantic}` : ''}` : 'Value not retained');
  summary.push(outputs?.update != null ? `Update ${outputs.update}` : 'Update not retained');
  if (!outputs || !probability.size) summary.push('Policy not retained');
  if (outputs?.mirror) {
    const m = outputs.mirror;
    summary.push(`Search ${m.worlds} worlds × depth ${m.depth} · ${m.rollouts} rollouts · ${m.seconds.toFixed(2)} s`);
  }
  if (outputs?.unavailable) summary.push(outputs.unavailable);
  const scored = !!outputs && (!!outputs.mirror || outputs.search_scores.some(score => score !== null));
  const search = scored ? frame.offers.map((offer, i) => ({
    id: offer.id,
    label: offer.label,
    prior: outputs!.mirror?.prior[i] ?? null,
    score: outputs!.search_scores[i] ?? null,
    updated: outputs!.probabilities[i] ?? null,
  })) : null;
  const provenance: string[] = [];
  if (outputs?.belief) provenance.push(`Belief ${outputs.belief.model} · ${outputs.belief.sampled_worlds} sampled worlds · condition mass ${outputs.belief.condition_mass.toFixed(4)}`);
  if (outputs?.mirror) {
    const m = outputs.mirror;
    provenance.push(`α ${m.alpha}, β ${m.beta} · ${m.seconds.toFixed(3)} s of ${m.decision_seconds} s`);
    provenance.push('Uniform belief (physical deal): hidden cards are sampled with copy counts and public knowledge.');
    provenance.push(`Seed ${m.seed}`, `Receipt SHA-256 ${m.receipt_sha256}`);
  }
  return { probability, chosen, summary, search, provenance, estimate: new Map(), first: null, heading: null };
}

/**
 * An analysis of your choice, aligned with its frame's offers: the policy's
 * probability and its estimate after each offer, and which it would have
 * chosen. Stated as what the policy reads, never as a verdict on yours.
 */
export function analysedEvidence(frame: ExperienceFrame, decision: AnalysedDecision, chosen: number | null, policy: string): OfferEvidence {
  const probability = new Map<number, number>();
  const estimate = new Map<number, number>();
  frame.offers.forEach((offer, i) => {
    const p = decision.annotation.probabilities?.[i];
    if (typeof p === 'number') probability.set(offer.id, p);
    const after = decision.after[i];
    if (typeof after === 'number') estimate.set(offer.id, after);
  });
  const first = firstChoice(decision);
  const firstId = first === null ? null : frame.offers[first]?.id ?? null;
  const summary: string[] = [];
  const here = score(decision.annotation.value, decision.semantic);
  if (here !== null) summary.push(`${policy}'s estimate here: ${Math.round(here * 100)}`);
  else if (decision.annotation.value !== null) summary.push(`${policy}'s value here: ${decision.annotation.value.toFixed(3)} · ${decision.semantic ?? 'undeclared'}`);
  return { probability, chosen, summary, search: null, provenance: [], estimate, first: firstId, heading: `Analysis by ${policy}` };
}

// ———————————————————————————————————————— Move list and strip

/**
 * The recording as log entries, one per position a person would call a move,
 * with the position key as the entry id so a move jumps to its position. Every
 * move reads as the recording narrates it from the recorded person's view: the
 * person's own choices ("Cast Divide by Zero, targeting Tiger-Seal"), the
 * opponent's decisions in a Play game and automation. A step of a move that
 * completes later (a target or payment before the spell is on the stack) has
 * no narration and is no move of its own. Another seat's choice in a watched
 * game names its chosen offer. `heroSeat` is the seat
 * named as "you"; `seatName` names the players the narration calls "Seat N".
 */
const BRANCHES: Record<BranchKind, string> = {
  determinize: 'hidden cards re-dealt',
  shuffle_libraries: 'libraries reshuffled',
  reseed: 'randomness reseeded',
};
const BRANCH_ENTRY = 'branch:';
const SWAP_ENTRY = 'swap:';

/** The position a log entry jumps to: branch and swap notes jump to the position they precede. */
export function entryPosition(id: string): string {
  if (id.startsWith(BRANCH_ENTRY)) return id.slice(BRANCH_ENTRY.length);
  if (id.startsWith(SWAP_ENTRY)) {
    const second = id.indexOf(':', SWAP_ENTRY.length);
    return second >= 0 ? id.slice(second + 1) : id;
  }
  return id;
}

export function replayLog(
  positions: readonly Position[],
  heroSeat: 0 | 1,
  seatName?: (seat: 0 | 1, currentHero: 0 | 1) => string,
  swaps: readonly SeatSwap[] = [],
): GameLogEntry[] {
  const entries: GameLogEntry[] = [];
  // The first of the person's own steps in this step of the turn whose move
  // has not been named yet: the decision where that move began.
  let begun: Position | null = null;
  let swapIndex = 0;
  positions.forEach((position, i) => {
    const posEvent = position.event ?? Infinity;
    while (swapIndex < swaps.length && swaps[swapIndex].event <= posEvent) {
      const swap = swaps[swapIndex];
      entries.push({
        id: `${SWAP_ENTRY}${swapIndex}:${position.key}`,
        actor: 'system',
        turn: swap.turn ?? position.turn,
        text: swap.turn != null ? `Swapped sides · Turn ${swap.turn}` : 'Swapped sides',
      });
      begun = null;
      swapIndex += 1;
    }
    if (position.kind === 'state') return;
    const currentHero = (heroSeat ^ (swapIndex & 1)) as 0 | 1;
    // A neutral note where an imported record branched from its source game.
    if (position.kind === 'choice' && position.branched?.length) {
      entries.push({
        id: BRANCH_ENTRY + position.key,
        actor: 'system',
        turn: position.turn,
        text: `Branch · ${position.branched.map(kind => BRANCHES[kind]).join(', ')}`,
      });
    }
    if (position.kind === 'consequences' && positions[i - 1]?.kind === 'choice') return;
    if (position.kind === 'consequences' && position.label === null) return;
    // The recorded person's own choice reads as its narrated move, as the
    // opponent's does. A step of a move that completes later is no line; the
    // move jumps to the decision where it began.
    const after = positions[i + 1];
    const narrated = position.kind === 'choice' && position.actor === currentHero && after?.kind === 'consequences';
    const sameStep = begun !== null && begun.turn === position.turn && begun.step === position.step;
    if (narrated && after.label === null) {
      if (!sameStep) begun = position;
      return;
    }
    const anchor = narrated && sameStep && begun ? begun : position;
    begun = null;
    const auto = position.kind === 'consequences' && position.automatic;
    let text = ((narrated ? after.label : position.label) ?? 'Choice').replace(/^Auto: /, '');
    if (seatName) {
      const resolveSeat = (seat: 0 | 1): string => seatName(seat, currentHero);
      text = nameSeats(text, resolveSeat, position.actor === currentHero);
      // A recorded choice names players by setup role: Hero is the seat named "you".
      const chooser = position.actor === currentHero ? 'hero' : 'villain';
      text = text.replace(/^Target (Hero|Villain)$/, (_, role: string) => {
        const seat = (role === 'Hero' ? currentHero : 1 - currentHero) as 0 | 1;
        const name = resolveSeat(seat);
        return `Target ${name !== 'You' ? name : chooser === 'hero' ? 'yourself' : 'you'}`;
      });
    }
    entries.push({
      id: anchor.key,
      actor: auto ? 'auto' : position.actor === null ? 'villain' : position.actor === currentHero ? 'hero' : 'villain',
      turn: anchor.turn,
      text,
    });
  });
  return entries;
}

/**
 * Names the players a narration calls "Seat 1" and "Seat 2": the seat's name,
 * or the person: "you", and "yourself" in the person's own move.
 */
export function nameSeats(text: string, seatName: (seat: 0 | 1) => string, yours = false): string {
  return text.replace(/\bSeat ([12])\b/g, (_, seat: string) => {
    const name = seatName(seat === '1' ? 0 : 1);
    return name !== 'You' ? name : yours ? 'yourself' : 'you';
  });
}

/** Walk back past announcing steps (`label === null` consequences of the same actor, turn and step) to the choice where a move began. */
function moveStartKey(positions: readonly Position[], choiceIndex: number): string {
  let at = choiceIndex;
  const anchor = positions[at];
  while (at >= 2) {
    const prevAfter = positions[at - 1];
    const prevChoice = positions[at - 2];
    if (
      prevAfter.kind === 'consequences'
      && prevAfter.label === null
      && prevAfter.actor === anchor.actor
      && prevAfter.turn === anchor.turn
      && prevAfter.step === anchor.step
      && prevChoice.kind === 'choice'
      && prevChoice.actor === anchor.actor
      && prevChoice.turn === anchor.turn
      && prevChoice.step === anchor.step
    ) {
      at -= 2;
    } else {
      break;
    }
  }
  return positions[at].key;
}

/** The position key a track event jumps to in Review. */
export function positionKeyForEvent(
  positions: readonly Position[],
  event: Pick<PresentationEvent, 'from_revision'>,
  isMove: boolean,
): string | null {
  let fallback: string | null = null;
  const preferred = isMove ? 'choice' : 'consequences';
  for (let i = 0; i < positions.length; i++) {
    const position = positions[i];
    if (position.revision !== event.from_revision) continue;
    if (position.kind === preferred) {
      return preferred === 'choice' ? moveStartKey(positions, i) : position.key;
    }
    if (fallback === null) fallback = position.key;
  }
  return fallback;
}

export type ReviewJump =
  | { kind: 'subcursor'; seq: number }
  | { kind: 'seek'; index: number; pendingSubCursor: number | null };

/**
 * Decide how clicking a history line in Review moves: only stay on the current
 * `consequences` transition when `targetKey` names that exact transition; a
 * player move's `choice` key always seeks to `choice`.
 */
export function planReviewJump(
  positions: readonly Pick<Position, 'key' | 'kind'>[],
  selected: number,
  currentKey: string | null,
  targetKey: string,
  boardSeq: number | null,
): ReviewJump | null {
  const at = positions.findIndex(position => position.key === targetKey);
  if (at < 0) return null;
  if (positions[at].kind === 'consequences' && boardSeq !== null) {
    if (at === selected && currentKey === targetKey) {
      return { kind: 'subcursor', seq: boardSeq };
    }
    return { kind: 'seek', index: at, pendingSubCursor: boardSeq };
  }
  return { kind: 'seek', index: at, pendingSubCursor: null };
}

/**
 * Rebuild a Review `consequences` transition's per-line boards via `mergeBoards`
 * before orienting each board to `nearSeat`, preserving capped-prefix fallbacks
 * and surfacing patch reconstruction errors explicitly.
 */
export function reviewTransitionBoards(
  position: Pick<RecordedPosition, 'kind' | 'revision' | 'observation' | 'history' | 'boards'>,
  nearSeat: 0 | 1,
  winner: number | null,
): { boards: StepBoard[]; error: string | null } {
  if (position.kind !== 'consequences') return { boards: [], error: null };
  const nonPass = position.history.filter(
    event => event.from_revision === position.revision && event.kind.kind !== 'passed',
  );
  const range = nonPass.length > 0
    ? { first: nonPass[0].seq, last: nonPass.at(-1)!.seq }
    : null;
  const patches = position.boards ?? [];
  if (patches.length === 0 && range === null) return { boards: [], error: null };
  try {
    const raw = mergeBoards([], position.observation, patches, range);
    const boards = raw.map(board => ({
      ...board,
      projection: orientBoard(board.projection, nearSeat, winner),
      ...(patches.length === 0 && board.first < board.last ? { fallback: true } : {}),
    }));
    return { boards, error: null };
  } catch (reason) {
    return { boards: [], error: reason instanceof Error ? reason.message : 'Board patch failed' };
  }
}

/** The history jump key matching the shown Review position, or the latest before it. */
export function historyKeyAt(
  jumpKeys: ReadonlySet<string>,
  positions: readonly Position[],
  selected: number,
): string | null {
  for (let i = Math.min(selected, positions.length - 1); i >= 0; i--) {
    if (jumpKeys.has(positions[i].key)) return positions[i].key;
  }
  return null;
}

/** The move a position belongs to: the latest move at or before it. */
export function moveAt(entries: readonly GameLogEntry[], positions: readonly Position[], selected: number): string | null {
  return historyKeyAt(new Set(entries.map(entry => entry.id)), positions, selected);
}

export interface StripMark {
  /** Index into the recording's positions. */
  at: number;
  key: string;
  turn: number | null;
  /** The first position of its turn: the strip draws a tick and the number. */
  turnStart: boolean;
  /** A recorded decision, and whose: the seat named "you" in the move list, or the other. */
  decision: boolean;
  side: 'you' | 'opponent' | null;
  /** The chooser's recorded value, when retained; with an analysis, your 0–1 score. */
  value: number | null;
  /** An analysed decision where you chose otherwise than the policy. */
  differs?: boolean;
  /** A mid-game seat swap occurred right before this position; breaks the estimate line. */
  breakBefore?: boolean;
  label: string;
}

/** The decision strip as the board draws it: open under the board, or folded to its toggle in the stop bar's row. */
export interface StripLine {
  marks: StripMark[];
  /** The shown position in a recording; null in live play. */
  selected: number | null;
  open: boolean;
  ontoggle: () => void;
  /** Moves to a position; absent where positions cannot be opened (live play). */
  onjump?: (at: number) => void;
  /** Why positions cannot be opened, when they cannot. */
  note?: string | null;
  /** With an analysis: the line's caption; values are then 0–1 scores on a fixed 0–100 axis. */
  scale?: string | null;
}

/** Every position as a mark on one line: decisions stand out, turns are ticked. */
export function stripMarks(
  positions: readonly Position[],
  heroSeat: 0 | 1,
  swaps: readonly SeatSwap[] = [],
): StripMark[] {
  let swapIndex = 0;
  return positions.map((position, at) => {
    const posEvent = position.event ?? Infinity;
    let breakBefore = false;
    while (swapIndex < swaps.length && swaps[swapIndex].event <= posEvent) {
      breakBefore = true;
      swapIndex += 1;
    }
    const currentHero = (heroSeat ^ (swapIndex & 1)) as 0 | 1;
    return {
      at,
      key: position.key,
      turn: position.turn,
      turnStart: at === 0 || position.turn !== positions[at - 1].turn,
      decision: position.kind === 'choice',
      side: position.actor === null ? null : position.actor === currentHero ? 'you' : 'opponent',
      value: position.value ?? null,
      ...(breakBefore ? { breakBefore: true } : {}),
      label: `Turn ${position.turn ?? '?'} · ${position.label ?? (position.kind === 'state' ? 'Final position' : position.kind === 'choice' ? 'Choice' : 'Consequences')}`,
    };
  });
}

/**
 * The strip read against an analysis: your analysed decisions carry the
 * policy's score at them and whether you chose otherwise; every other value
 * is dropped, as it is on another scale.
 */
export function analysedMarks(marks: readonly StripMark[], positions: readonly Position[], decisions: ReadonlyMap<number, AnalysedDecision>): StripMark[] {
  return marks.map(mark => {
    const position = positions[mark.at];
    const decision = position.kind === 'choice' && position.event !== undefined ? decisions.get(position.event) : undefined;
    if (!decision) return { ...mark, value: null, differs: false };
    const first = firstChoice(decision);
    return { ...mark, value: score(decision.annotation.value, decision.semantic), differs: first !== null && decision.played !== first };
  });
}

function aggregateMoveMarks(
  choices: readonly [string, Position][],
  decisions: ReadonlyMap<number, AnalysedDecision>,
): Map<string, DecisionMark> {
  const marks = new Map<string, DecisionMark>();
  for (const [move, choice] of choices) {
    const decision = choice.event === undefined ? undefined : decisions.get(choice.event);
    const mark = decision ? decisionMark(decision) : null;
    if (!mark) continue;
    const marked = marks.get(move);
    marks.set(move, marked ? { ...marked, agreed: marked.agreed + mark.agreed, of: marked.of + mark.of } : mark);
  }
  return marks;
}

/**
 * Each of your analysed moves' mark, by move id: every analysed decision from
 * the position a move jumps to up to the next move counts toward it, so a
 * cast's target and payment choices are marked with the cast.
 */
export function moveMarks(entries: readonly GameLogEntry[], positions: readonly Position[], decisions: ReadonlyMap<number, AnalysedDecision>): Map<string, DecisionMark> {
  return aggregateMoveMarks(yourMoveChoices(entries, positions), decisions);
}

/**
 * Your moves chosen with a live analysis shown, by log entry id: a move is
 * marked when any of your choices from its entry up to the next entry was
 * committed with the analysis shown. A stored fact about the game, shown
 * whether or not an analysis is open.
 */
export function analysisShownMoves(entries: readonly GameLogEntry[], positions: readonly Position[]): Set<string> {
  return new Set(yourMoveChoices(entries, positions).filter(([, choice]) => choice.analysis_shown).map(([move]) => move));
}

/** Each of your choice positions with the id of the move entry it belongs to: the last entry at or before it. */
function yourMoveChoices(entries: readonly GameLogEntry[], positions: readonly Position[]): [string, Position][] {
  const ids = new Set(entries.map(entry => entry.id));
  const yours = new Set(entries.filter(entry => entry.actor === 'hero').map(entry => entry.id));
  const choices: [string, Position][] = [];
  let move: string | null = null;
  for (const position of positions) {
    if (ids.has(position.key)) move = position.key;
    if (move !== null && yours.has(move) && position.kind === 'choice') choices.push([move, position]);
  }
  return choices;
}

/** Turn starts far enough apart to number without overlap, as indices into `marks`. */
export function labelledTurns(marks: readonly StripMark[], minGap: number): StripMark[] {
  const span = Math.max(1, marks.length - 1);
  const shown: StripMark[] = [];
  for (const mark of marks) {
    if (!mark.turnStart) continue;
    const last = shown.at(-1);
    if (!last || (mark.at - last.at) / span >= minGap) shown.push(mark);
  }
  return shown;
}

/** Build Review history divider notes for mid-game seat swaps, anchored on `events` by revision. */
export function swapHistoryNotes(
  swaps: readonly SeatSwap[],
  positions: readonly Position[],
  events: readonly PresentationEvent[],
): HistoryNote[] {
  if (!swaps.length || !positions.length) return [];
  return swaps.flatMap((swap, swapIndex) => {
    const target = positions.find(position => (position.event ?? Infinity) >= swap.event);
    if (!target) return [];
    let maxPreRevision: number | null = null;
    for (const position of positions) {
      if ((position.event ?? Infinity) >= swap.event) break;
      if (position.revision != null) {
        maxPreRevision = maxPreRevision === null ? position.revision : Math.max(maxPreRevision, position.revision);
      }
    }
    let afterSeq = -1;
    if (maxPreRevision !== null) {
      for (const event of events) {
        if (event.from_revision <= maxPreRevision) afterSeq = event.seq;
        else break;
      }
    }
    const id = `${SWAP_ENTRY}${swapIndex}:${target.key}`;
    return [{
      id,
      actor: 'note' as const,
      turn: swap.turn ?? target.turn,
      afterSeq,
      text: swap.turn != null ? `Swapped sides · Turn ${swap.turn}` : 'Swapped sides',
      jumpKey: id,
    }];
  });
}

function historyYourMoveChoices(
  turns: readonly HistoryTurn[],
  positions: readonly Position[],
  heroSeat: 0 | 1,
  swaps: readonly SeatSwap[] = [],
): [string, Position][] {
  const indexByKey = new Map(positions.map((position, index) => [position.key, index]));
  const lineAt = new Map<number, { id: string; actor: HistoryTurn['lines'][number]['actor'] }>();
  let firstHero: string | null = null;
  for (const turn of turns) {
    for (const line of turn.lines) {
      if (line.passes !== null || !line.jumpKey) continue;
      const at = indexByKey.get(line.jumpKey);
      if (at === undefined) continue;
      if (line.actor === 'you' && firstHero === null) firstHero = line.id;
      const existing = lineAt.get(at);
      if (!existing || (existing.actor !== 'you' && line.actor === 'you')) {
        lineAt.set(at, { id: line.id, actor: line.actor });
      }
    }
  }
  const choices: [string, Position][] = [];
  let lastHero: string | null = null;
  positions.forEach((position, at) => {
    const line = lineAt.get(at);
    if (line?.actor === 'you') lastHero = line.id;
    const currentHero = humanSeatAt(heroSeat, swaps, position.event ?? null);
    if (position.kind !== 'choice' || position.actor !== currentHero) return;
    const owner = lastHero ?? firstHero;
    if (owner !== null) choices.push([owner, position]);
  });
  return choices;
}

/** Each of your analysed moves' mark, keyed by top-level `HistoryLine.id`. */
export function historyMoveMarks(
  turns: readonly HistoryTurn[],
  positions: readonly Position[],
  heroSeat: 0 | 1,
  decisions: ReadonlyMap<number, AnalysedDecision>,
  swaps: readonly SeatSwap[] = [],
): Map<string, DecisionMark> {
  return aggregateMoveMarks(historyYourMoveChoices(turns, positions, heroSeat, swaps), decisions);
}

/** Your moves chosen with a live analysis shown, keyed by top-level `HistoryLine.id`. */
export function historyAnalysisShown(
  turns: readonly HistoryTurn[],
  positions: readonly Position[],
  heroSeat: 0 | 1,
  swaps: readonly SeatSwap[] = [],
): Set<string> {
  return new Set(
    historyYourMoveChoices(turns, positions, heroSeat, swaps)
      .filter(([, choice]) => choice.analysis_shown)
      .map(([move]) => move),
  );
}

/**
 * Overlay a policy's live readings onto live strip marks (`trackMarks`). Each
 * reading becomes a point of its own, so a decision that emits no move still
 * carries its score.
 */
export function withLiveAnalysisPoints(marks: readonly StripMark[], points: readonly LivePoint[]): StripMark[] {
  const moveText = (mark: StripMark): string => mark.label.replace(/^Turn [^·]+ · /, '');
  const placed = points.map(live => {
    if (live.at !== null) return { live, at: live.at };
    const turnStart = marks.findIndex(mark => mark.turn !== null && mark.turn >= (live.turn ?? 0));
    return { live, at: turnStart === -1 ? marks.length : turnStart };
  }).sort((a, b) => a.at - b.at || a.live.event - b.live.event);
  const point = ({ live, at }: (typeof placed)[number]): StripMark => {
    const last = marks[at - 1];
    const turn = live.turn ?? last?.turn ?? marks[0]?.turn ?? null;
    const where = live.label ?? (last ? `after ${moveText(last)}` : 'before the first move');
    return { at: 0, key: `read-${live.event}`, turn, turnStart: false, decision: true, side: 'you', value: live.score, label: `Turn ${turn ?? '?'} · ${where}` };
  };
  const out: StripMark[] = [];
  let next = 0;
  marks.forEach((mark, index) => {
    while (next < placed.length && placed[next].at <= index) out.push(point(placed[next++]));
    out.push({ ...mark, value: null });
  });
  while (next < placed.length) out.push(point(placed[next++]));
  return out.map((mark, at) => ({ ...mark, at, turnStart: at === 0 || mark.turn !== out[at - 1].turn }));
}

/**
 * Where "since your last move" starts: at a choice, the same chooser's previous
 * decision, so the recap tells what happened in between; after a transition,
 * the transition itself. Null when nothing precedes it.
 */
export function sinceRevision(positions: readonly Position[], selected: number): number | null {
  const position = positions[selected];
  if (!position) return null;
  if (position.kind === 'consequences') return position.revision ?? null;
  for (let i = selected - 1; i >= 0; i--) {
    const earlier = positions[i];
    if (earlier.kind === 'choice' && (position.kind === 'state' || earlier.actor === position.actor)) return earlier.revision ?? null;
  }
  return null;
}

/** The context key under which the board gives its cards the recorded policy (a getter of `OfferEvidence | null`). */
export const EVIDENCE = Symbol('academy.board.evidence');

/** The recorded mass on a card's offers and whether the chooser played one of them; null when none was recorded. */
export function cardEvidence(evidence: OfferEvidence | null, offerIds: readonly number[]): { p: number | null; chosen: boolean } | null {
  if (!evidence || !offerIds.length) return null;
  const recorded = offerIds.map(id => evidence.probability.get(id)).filter((p): p is number => p !== undefined);
  return { p: recorded.length ? recorded.reduce((sum, p) => sum + p, 0) : null, chosen: evidence.chosen !== null && offerIds.includes(evidence.chosen) };
}

// ———————————————————————————————————————— Heading

/**
 * What a recorded position is, in words, for the rail's heading: "Your choice",
 * "After Heuristic's move", "Game over · Heuristic wins". `gameOver` is the
 * shown board's; `winner` the recording's (null for a draw).
 */
export function positionHeading(
  position: Pick<Position, 'kind' | 'actor' | 'automatic'>,
  seatName: (seat: 0 | 1) => string,
  gameOver: boolean,
  winner: number | null,
): string {
  if (gameOver) return winner === 0 || winner === 1 ? `Game over · ${seatName(winner)} ${seatName(winner) === 'You' ? 'win' : 'wins'}` : 'Game over · Draw';
  const seat = position.actor === 0 || position.actor === 1 ? position.actor : null;
  if (position.kind === 'choice') {
    if (seat === null) return 'A recorded choice';
    const owner = possessive(seatName(seat));
    return `${owner[0].toUpperCase()}${owner.slice(1)} choice`;
  }
  if (position.kind === 'consequences') {
    if (position.automatic) return 'After an automatic pass';
    return seat === null ? 'After a move' : `After ${possessive(seatName(seat))} move`;
  }
  return 'The latest position';
}
