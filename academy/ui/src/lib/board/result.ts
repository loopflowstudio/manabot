/**
 * What the result card says when a game ends: who won, how, and one line
 * about the game, read from the final frame and the presentation track.
 *
 * Counts `cast` and `attack_group` events directly on the track, falling back
 * to a flat `replayLog` only for older recordings written before moves were
 * recorded as events.
 */
import type { ExperienceFrame, GameLogEntry, PresentationEvent } from '$lib/types';
import { hasTrackHistory } from './history';

export interface GameResult {
  /** "You win", "Heuristic · scripted wins", "Draw". */
  headline: string;
  /** How it ended: "Heuristic · scripted went to −2 life on turn 9". */
  how: string;
  /** The final life totals: "You 14 · Heuristic · scripted −2". */
  life: string;
  /** One line from the history: "9 turns · you cast 6 spells and attacked 4 times; …". */
  summary: string;
}

interface Tally { spells: number; attacks: number }

function tallyTrack(events: readonly PresentationEvent[], seat: number): Tally {
  let spells = 0;
  let attacks = 0;
  for (const event of events) {
    if (event.kind.kind === 'cast' && event.kind.controller === seat) {
      spells += 1;
    } else if (
      event.kind.kind === 'attack_group'
      && event.kind.defender.kind === 'player'
      && event.kind.defender.id !== seat
    ) {
      attacks += 1;
    }
  }
  return { spells, attacks };
}

function tallyLegacyLog(log: readonly GameLogEntry[], actor: GameLogEntry['actor']): Tally {
  const phrases = log.filter(entry => entry.actor === actor).flatMap(entry => entry.text.split('; '));
  return {
    spells: phrases.filter(phrase => phrase.startsWith('Cast ')).length,
    attacks: phrases.filter(phrase => phrase.startsWith('Attack')).length,
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
  events: readonly PresentationEvent[],
  names: { you: string; opponent: string },
  winner: number | null,
  fallbackLog: readonly GameLogEntry[] = [],
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

  const useTrack =
    hasTrackHistory(events)
    || events.some(e => e.kind.kind === 'cast' || e.kind.kind === 'attack_group')
    || fallbackLog.length === 0;
  const youTally = useTrack ? tallyTrack(events, agent.player_index) : tallyLegacyLog(fallbackLog, 'hero');
  const oppTally = useTrack ? tallyTrack(events, opponent.player_index) : tallyLegacyLog(fallbackLog, 'villain');
  const trackTurns = events.flatMap(e => (e.kind.kind === 'turn_started' ? [e.kind.turn_number] : []));
  const logTurns = useTrack ? [] : fallbackLog.map(entry => entry.turn ?? 0);
  const turns = Math.max(...trackTurns, ...logTurns, turn.turn_number);
  const summary = `${plural(turns, 'turn')} · ${deeds(lowerYou(names.you), youTally, false)}; ${deeds(names.opponent, oppTally, true)}`;
  return { headline, how, life, summary };
}
