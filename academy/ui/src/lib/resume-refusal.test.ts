import { describe, expect, it } from 'vitest';
import { refusalAction } from './resume-refusal';

describe('refusalAction', () => {
  it('opens Replay for an addressed game with nothing left to play', () => {
    expect(refusalAction('game_finished', true)).toBe('replay');
    expect(refusalAction('game_is_watch', true)).toBe('replay');
  });

  it('opens setup silently when the remembered game is over, stopped or gone', () => {
    for (const code of ['game_finished', 'game_stopped', 'game_not_found', 'game_is_watch']) {
      expect(refusalAction(code, false)).toBe('setup');
    }
  });

  it('opens setup silently for a game that is not saved here, however it was named', () => {
    expect(refusalAction('game_not_found', true)).toBe('setup');
  });

  it('gives the reason for an addressed game that stopped, and for any game that cannot be rebuilt', () => {
    expect(refusalAction('game_stopped', true)).toBe('reason');
    expect(refusalAction('record_diverged', false)).toBe('reason');
    expect(refusalAction(null, false)).toBe('reason');
  });

  it('leaves a game another tab holds with that tab', () => {
    expect(refusalAction('game_open_elsewhere', false)).toBe('elsewhere');
  });
});
