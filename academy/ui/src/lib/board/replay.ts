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
import { orientBoard, type BotDecision, type Position, type RecordedPosition, type RecordingIndex } from '$lib/recording';
import type { ExperienceFrame, GameLogEntry } from '$lib/types';
import { possessive } from './view';
import { PROTOCOL_VERSION } from '$lib/types';

// ———————————————————————————————————————— Perspective

/** Who sits at the bottom: a fixed seat, or whoever chose at this position. */
export type Perspective = 0 | 1 | 'chooser';

/** The address's `near` value, or the recording's natural seat: the person, else the chooser. */
export function perspectiveOf(near: string | null, index: Pick<RecordingIndex, 'players'>): Perspective {
  if (near === '0' || near === '1') return Number(near) as 0 | 1;
  if (near === 'chooser') return 'chooser';
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
  return { probability, chosen, summary, search, provenance };
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
export function replayLog(positions: readonly Position[], heroSeat: 0 | 1, seatName?: (seat: 0 | 1) => string): GameLogEntry[] {
  const entries: GameLogEntry[] = [];
  // The first of the person's own steps in this step of the turn whose move
  // has not been named yet: the decision where that move began.
  let begun: Position | null = null;
  positions.forEach((position, i) => {
    if (position.kind === 'state') return;
    if (position.kind === 'consequences' && positions[i - 1]?.kind === 'choice') return;
    if (position.kind === 'consequences' && position.label === null) return;
    // The recorded person's own choice reads as its narrated move, as the
    // opponent's does. A step of a move that completes later is no line; the
    // move jumps to the decision where it began.
    const after = positions[i + 1];
    const narrated = position.kind === 'choice' && position.actor === heroSeat && after?.kind === 'consequences';
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
      text = nameSeats(text, seatName, position.actor === heroSeat);
      // A recorded choice names players by setup role: Hero is the seat named "you".
      const chooser = position.actor === heroSeat ? 'hero' : 'villain';
      text = text.replace(/^Target (Hero|Villain)$/, (_, role: string) => {
        const seat = (role === 'Hero' ? heroSeat : 1 - heroSeat) as 0 | 1;
        const name = seatName(seat);
        return `Target ${name !== 'You' ? name : chooser === 'hero' ? 'yourself' : 'you'}`;
      });
    }
    entries.push({
      id: anchor.key,
      actor: auto ? 'auto' : position.actor === null ? 'villain' : position.actor === heroSeat ? 'hero' : 'villain',
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

/** The move a position belongs to: the latest move at or before it. */
export function moveAt(entries: readonly GameLogEntry[], positions: readonly Position[], selected: number): string | null {
  const keys = new Set(entries.map(entry => entry.id));
  for (let i = Math.min(selected, positions.length - 1); i >= 0; i--) if (keys.has(positions[i].key)) return positions[i].key;
  return null;
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
  /** The chooser's recorded value, when retained. */
  value: number | null;
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
}

/** Every position as a mark on one line: decisions stand out, turns are ticked. */
export function stripMarks(positions: readonly Position[], heroSeat: 0 | 1): StripMark[] {
  return positions.map((position, at) => ({
    at,
    key: position.key,
    turn: position.turn,
    turnStart: at === 0 || position.turn !== positions[at - 1].turn,
    decision: position.kind === 'choice',
    side: position.actor === null ? null : position.actor === heroSeat ? 'you' : 'opponent',
    value: position.value ?? null,
    label: `Turn ${position.turn ?? '?'} · ${position.label ?? (position.kind === 'state' ? 'Final position' : position.kind === 'choice' ? 'Choice' : 'Consequences')}`,
  }));
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

/**
 * A live game's moves as strip marks, from its log: the person's and the
 * opponent's decisions, automation as plain positions. Live marks carry no
 * value and no position to jump to.
 */
export function logMarks(entries: readonly GameLogEntry[]): StripMark[] {
  return entries.map((entry, at) => ({
    at,
    key: entry.id,
    turn: entry.turn,
    turnStart: at === 0 || entry.turn !== entries[at - 1].turn,
    decision: entry.actor === 'hero' || entry.actor === 'villain',
    side: entry.actor === 'hero' ? 'you' : entry.actor === 'villain' ? 'opponent' : null,
    value: null,
    label: `Turn ${entry.turn ?? '?'} · ${entry.text}`,
  }));
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
