/**
 * The last matchup this browser started, so the home page can start it again
 * in one click: the setup the server echoed for that game (opponent, decks,
 * seat, deal mode, stops) and how to name it. A preference, kept in
 * `localStorage` under `academy.last-matchup`; it holds no table credential,
 * request identity or participant token, so "Play again" always starts a new
 * session, never resumes another tab's game.
 */
export const LAST_MATCHUP_KEY = 'academy.last-matchup';

export interface LastMatchup {
  /** The setup config to send with `new_game`, without request or participant identity. */
  config: Record<string, unknown>;
  opponent: string;
  decks: { hero: string; villain: string } | null;
}

const IDENTITY_KEYS = new Set(['request_id', 'player_token']);

export function matchupConfig(config: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(config).filter(([key]) => !IDENTITY_KEYS.has(key)));
}

export function loadLastMatchup(): LastMatchup | null {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(LAST_MATCHUP_KEY) ?? 'null');
    if (!value || typeof value !== 'object') return null;
    const record = value as Partial<LastMatchup>;
    if (!record.config || typeof record.config !== 'object' || typeof record.opponent !== 'string') return null;
    const decks = record.decks && typeof record.decks.hero === 'string' && typeof record.decks.villain === 'string' ? record.decks : null;
    return { config: matchupConfig(record.config), opponent: record.opponent, decks };
  } catch {
    return null;
  }
}

export function saveLastMatchup(matchup: LastMatchup): void {
  try {
    localStorage.setItem(LAST_MATCHUP_KEY, JSON.stringify({ ...matchup, config: matchupConfig(matchup.config) }));
  } catch { /* Play again falls back to the default setup. */ }
}

/** "You · UR Lessons, on the play · vs Heuristic · scripted · GW Allies" pieces for one line. */
export function matchupSeat(config: Record<string, unknown>): string {
  return config.human_seat === 1 ? 'on the draw' : 'on the play';
}
