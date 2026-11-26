/**
 * A game the server would not take up, and what Play does about it. The
 * server names the reason by code (`GameSession.resume`); the page acts on
 * the code and reads nothing else.
 */
export interface ResumeRefusal {
  game: string;
  /** `game_not_found`, `game_finished`, `game_stopped`, `game_is_watch`, `game_open_elsewhere`, `record_diverged`, or null for a failure with no code (a policy that would not load). */
  code: string | null;
  message: string;
}

/**
 * - `replay`: the address asked for a game with nothing left to play; open its Replay.
 * - `setup`: the game is not saved here, or `/` remembered one that is over or stopped; open setup and say nothing.
 * - `elsewhere`: another tab holds the game; open setup beside a link that takes it over.
 * - `reason`: show why, with a way to its Replay.
 */
export type RefusalAction = 'replay' | 'setup' | 'elsewhere' | 'reason';

const NOTHING_TO_RESUME = new Set(['game_finished', 'game_stopped', 'game_is_watch']);

/** `asked` is whether the address named the game (`/?game=<id>`) rather than `/` remembering it. */
export function refusalAction(code: string | null, asked: boolean): RefusalAction {
  if (code === 'game_open_elsewhere') return 'elsewhere';
  if (asked && (code === 'game_finished' || code === 'game_is_watch')) return 'replay';
  if (code === 'game_not_found') return 'setup';
  if (!asked && code !== null && NOTHING_TO_RESUME.has(code)) return 'setup';
  return 'reason';
}
