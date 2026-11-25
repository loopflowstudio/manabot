import { describe, expect, it } from 'vitest';

import type { ExperienceFrame, GameLogEntry, PresentationEvent } from '$lib/types';
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

function ev(seq: number, kind: PresentationEvent['kind']): PresentationEvent {
  return {
    seq,
    from_revision: seq,
    to_revision: seq + 1,
    caused_by: null,
    group: seq,
    importance: 'normal',
    suggested_ms: 400,
    sound: null,
    kind,
  };
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

  it('summarizes each player\'s casts and attacks directly from the presentation track', () => {
    const events: PresentationEvent[] = [
      ev(1, { kind: 'turn_started', player: 0, turn_number: 3 }),
      ev(2, { kind: 'cast', object: { entity: 1, incarnation: 1 }, controller: 0, stack: 1 }),
      ev(3, { kind: 'attack_group', attackers: [{ entity: 2, incarnation: 1 }], defender: { kind: 'player', id: 1 } }),
      ev(4, { kind: 'cast', object: { entity: 3, incarnation: 1 }, controller: 1, stack: 3 }),
      ev(5, { kind: 'attack_group', attackers: [{ entity: 4, incarnation: 1 }], defender: { kind: 'player', id: 0 } }),
      ev(6, { kind: 'attack_group', attackers: [{ entity: 5, incarnation: 1 }], defender: { kind: 'player', id: 0 } }),
      ev(7, { kind: 'turn_started', player: 0, turn_number: 10 }),
    ];
    expect(gameResult(frame(14, 0), events, names, 0).summary)
      .toBe('10 turns · you cast 1 spell and attacked once; Heuristic · scripted cast 1 and attacked 2 times');
  });

  it('falls back to legacy replayLog for old recordings without track events', () => {
    const log = [entry('villain', 'Cast Suki, Kyoshi Warrior; Cast Yip Yip!, targeting Earth Kingdom Jailer', 4)];
    expect(gameResult(frame(14, 0, 4), [], names, 0, log).summary)
      .toBe('4 turns · you cast 0 spells and attacked 0 times; Heuristic · scripted cast 2 and attacked 0 times');
  });

  it('calls a game nobody won a draw', () => {
    expect(gameResult(frame(0, 0), [], names, null)).toMatchObject({ headline: 'Draw', how: 'Neither player won, on turn 9' });
  });
});
