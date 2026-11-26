/**
 * The game in numbers: what each seat of a played game did
 * (`managym/academy/numbers.py`) and how it read beside an analysis.
 * Counts, never a rating or a grade.
 */
import { firstChoice, formatScore, score, type Analysis } from './analysis';

/** One native seat's counts (`managym.statistics.Behaviour` and `Totals`). */
export interface SeatNumbers {
  seat: number;
  lands_played: number;
  spells_cast: number;
  attacks: number;
  cards_drawn: number;
  /** Null for a game recorded before damage was counted. */
  damage_dealt: number | null;
  damage_taken: number | null;
}

/** A played game's numbers; `seats` in native order, seat 0 on the play. */
export interface GameNumbers {
  /** False for a game that stopped before game over: counts run to where it stopped. */
  complete: boolean;
  turns: number;
  seats: [SeatNumbers, SeatNumbers];
  /** Your decisions committed with a live analysis shown (0 in numbers saved before it was counted). */
  analysis_shown?: number;
  /** The step a game played on from another starts at; null or absent for an ordinary game. */
  played_from?: number | null;
}

const plural = (count: number, word: string): string => `${count} ${word}${count === 1 ? '' : 's'}`;

/** "On the play" for seat 0, else "On the draw". */
export function playOrDraw(seat: number): string {
  return seat === 0 ? 'On the play' : 'On the draw';
}

/** "45 s", "12 min", "1 h 05 min" between two ISO times; null without both. */
export function duration(start: string | null | undefined, end: string | null | undefined): string | null {
  if (!start || !end) return null;
  const seconds = Math.round((Date.parse(end) - Date.parse(start)) / 1000);
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  if (seconds < 60) return `${seconds} s`;
  const minutes = Math.round(seconds / 60);
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, '0')} min`;
}

/** "9 turns · you cast 6 spells and attacked 4 times; Heuristic cast 3 and attacked 2 times". */
export function numbersLine(numbers: GameNumbers, seat: number, names: { you: string; opponent: string }): string {
  const you = numbers.seats[seat];
  const them = numbers.seats[1 - seat];
  const times = (n: number): string => (n === 1 ? 'once' : `${n} times`);
  const who = names.you === 'You' ? 'you' : names.you;
  return `${plural(numbers.turns, 'turn')} · ${who} cast ${plural(you.spells_cast, 'spell')} and attacked ${times(you.attacks)}; `
    + `${names.opponent} cast ${them.spells_cast} and attacked ${times(them.attacks)}`;
}

export interface NumbersRow { label: string; you: number; opponent: number }

/** The table rows, your seat beside the opponent's; damage only where counted. */
export function numberRows(numbers: GameNumbers, seat: number): NumbersRow[] {
  const you = numbers.seats[seat];
  const them = numbers.seats[1 - seat];
  const rows: [string, keyof SeatNumbers][] = [
    ['Lands played', 'lands_played'], ['Spells cast', 'spells_cast'], ['Attacks', 'attacks'],
    ['Cards drawn', 'cards_drawn'], ['Damage dealt', 'damage_dealt'], ['Damage taken', 'damage_taken'],
  ];
  return rows.flatMap(([label, key]) => {
    const a = you[key];
    const b = them[key];
    return a === null || b === null ? [] : [{ label, you: a, opponent: b }];
  });
}

/** Decisions the analysis read with probabilities, and how many you chose as its first choice. */
export function agreement(analysis: Analysis): { same: number; of: number } {
  const read = analysis.decisions.filter(decision => firstChoice(decision) !== null);
  return { same: read.filter(decision => firstChoice(decision) === decision.played).length, of: read.length };
}

/** The analysis's estimate from your seat at its first, highest and last decision, 0–100. */
export function estimateArc(analysis: Analysis): { start: string; peak: string; end: string } | null {
  const scores = analysis.decisions
    .map(decision => score(decision.annotation.value, decision.semantic))
    .filter((value): value is number => value !== null);
  if (!scores.length) return null;
  return { start: formatScore(scores[0]), peak: formatScore(Math.max(...scores)), end: formatScore(scores[scores.length - 1]) };
}
