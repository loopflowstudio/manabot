import { describe, expect, it } from 'vitest';

import type { ExperienceFrame, GameLogEntry } from '$lib/types';
import { gameResult } from './result';

function frame(you: number, them: number, turn = 9): ExperienceFrame {
  return {
    projection: {
      turn: { turn_number: turn },
      agent: { player_index: 0, life: you },
      opponent: { player_index: 1, life: them },
    },
  } as unknown as ExperienceFrame;
}

const entry = (actor: GameLogEntry['actor'], text: string, turn = 1): GameLogEntry => ({ id: `${actor}${text}${turn}`, actor, turn, text });
const names = { you: 'You', opponent: 'Heuristic · scripted' };

describe('gameResult', () => {
  it('says who won by taking the other to zero, and on which turn', () => {
    const result = gameResult(frame(14, -2), [], names, 0);
    expect(result.headline).toBe('You win');
    expect(result.how).toBe('Heuristic · scripted went to −2 life on turn 9');
    expect(result.life).toBe('You 14 · Heuristic · scripted −2');
  });

  it('names the opponent as the winner and does not invent a reason', () => {
    expect(gameResult(frame(3, 20, 12), [], names, 1)).toMatchObject({
      headline: 'Heuristic · scripted wins',
      how: 'You lost on turn 12',
    });
  });

  it('summarizes each player\'s casts and attacks from the move list', () => {
    const log = [
      entry('hero', 'Cast Pop Quiz', 3), entry('hero', 'Attack with Tiger-Seal', 5), entry('hero', 'Pass priority', 5),
      entry('villain', 'Cast Allies at Last', 4), entry('villain', 'Attack with Badgermole', 6), entry('villain', 'Attack with Ally', 6),
      entry('auto', 'Pass priority', 10),
    ];
    expect(gameResult(frame(14, 0), log, names, 0).summary)
      .toBe('10 turns · you cast 1 spell and attacked once; Heuristic · scripted cast 1 and attacked 2 times');
  });

  it('calls a game nobody won a draw', () => {
    expect(gameResult(frame(0, 0), [], names, null)).toMatchObject({ headline: 'Draw', how: 'Neither player won, on turn 9' });
  });
});
