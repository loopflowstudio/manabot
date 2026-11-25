/**
 * What the result card says when a game ends: who won, how, and the life
 * totals, read from the final frame. It reads only facts the frame carries;
 * when the frame does not say why the game ended (a concession, an empty
 * library), it names the turn and life totals and nothing more. The game's
 * numbers come from its re-execution (`$lib/numbers`).
 */
import type { ExperienceFrame } from '$lib/types';

export interface GameResult {
  /** "You win", "Heuristic · scripted wins", "Draw". */
  headline: string;
  /** How it ended: "Heuristic · scripted went to −2 life on turn 9". */
  how: string;
  /** The final life totals: "You 14 · Heuristic · scripted −2". */
  life: string;
}

const signed = (life: number): string => (life < 0 ? `−${-life}` : String(life));

export function gameResult(
  frame: ExperienceFrame,
  names: { you: string; opponent: string },
  winner: number | null,
): GameResult {
  const { agent, opponent, turn } = frame.projection;
  const nameOf = (index: number): string => (index === agent.player_index ? names.you : names.opponent);
  const headline = winner === null ? 'Draw' : winner === agent.player_index ? (names.you === 'You' ? 'You win' : `${names.you} wins`) : `${names.opponent} wins`;
  const loser = winner === null ? null : winner === agent.player_index ? opponent : agent;
  const on = `on turn ${turn.turn_number}`;
  const how = loser === null ? `Neither player won, ${on}`
    : loser.life <= 0 ? `${nameOf(loser.player_index)} went to ${signed(loser.life)} life ${on}`
      : `${nameOf(loser.player_index)} lost ${on}`;
  const life = `${names.you} ${signed(agent.life)} · ${names.opponent} ${signed(opponent.life)}`;
  return { headline, how, life };
}
