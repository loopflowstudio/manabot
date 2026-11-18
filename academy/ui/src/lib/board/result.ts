/**
 * What the result card says when a game ends: who won, how, and one line
 * about the game, all read from the final frame and the move list. It reads
 * only facts the frame and the log carry; when the frame does not say why the
 * game ended (a concession, an empty library), it names the turn and life
 * totals and nothing more.
 */
import type { ExperienceFrame, GameLogEntry } from '$lib/types';

export interface GameResult {
  /** "You win", "Heuristic · scripted wins", "Draw". */
  headline: string;
  /** How it ended: "Heuristic · scripted went to −2 life on turn 9". */
  how: string;
  /** The final life totals: "You 14 · Heuristic · scripted −2". */
  life: string;
  /** One line from the move list: "9 turns · you cast 6 spells and attacked 4 times; …". */
  summary: string;
}

interface Tally { spells: number; attacks: number }

function tally(log: readonly GameLogEntry[], actor: GameLogEntry['actor']): Tally {
  const moves = log.filter(entry => entry.actor === actor);
  return {
    spells: moves.filter(entry => entry.text.startsWith('Cast ')).length,
    attacks: moves.filter(entry => entry.text.startsWith('Attack')).length,
  };
}

const plural = (count: number, word: string): string => `${count} ${word}${count === 1 ? '' : 's'}`;
const signed = (life: number): string => (life < 0 ? `−${-life}` : String(life));

function deeds(who: string, { spells, attacks }: Tally, short: boolean): string {
  const cast = short ? `cast ${spells}` : `cast ${plural(spells, 'spell')}`;
  return `${who} ${cast} and attacked ${attacks === 1 ? 'once' : `${attacks} times`}`;
}

export function gameResult(
  frame: ExperienceFrame,
  log: readonly GameLogEntry[],
  names: { you: string; opponent: string },
  winner: number | null,
): GameResult {
  const { agent, opponent, turn } = frame.projection;
  const nameOf = (index: number): string => (index === agent.player_index ? names.you : names.opponent);
  const headline = winner === null ? 'Draw' : winner === agent.player_index ? (names.you === 'You' ? 'You win' : `${names.you} wins`) : `${names.opponent} wins`;
  const loser = winner === null ? null : winner === agent.player_index ? opponent : agent;
  const on = `on turn ${turn.turn_number}`;
  const lowerYou = (who: string): string => (who === 'You' ? 'you' : who);
  const how = loser === null ? `Neither player won, ${on}`
    : loser.life <= 0 ? `${nameOf(loser.player_index)} went to ${signed(loser.life)} life ${on}`
      : `${nameOf(loser.player_index)} lost ${on}`;
  const life = `${names.you} ${signed(agent.life)} · ${names.opponent} ${signed(opponent.life)}`;
  const turns = Math.max(...log.map(entry => entry.turn ?? 0), turn.turn_number);
  const summary = `${plural(turns, 'turn')} · ${deeds(lowerYou(names.you), tally(log, 'hero'), false)}; ${deeds(names.opponent, tally(log, 'villain'), true)}`;
  return { headline, how, life, summary };
}
