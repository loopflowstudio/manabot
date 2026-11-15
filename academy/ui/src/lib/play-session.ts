/** Owner-private repeated-play setup. Table credentials remain in tab storage. */
export interface PlaySession {
  id: string;
  session_id: string;
  game_number: number;
  revision: number;
  draft: Record<string, unknown>;
  config: Record<string, unknown>;
}

export interface SessionResult {
  origin: string;
  human_seat: number;
  human_deck: string;
  opponent_deck: string;
  opponent_name: string;
  opponent_id: string;
  version: string;
  inference_json: string;
  matchup_json: string;
  decklists: string;
  completed: number;
  wins: number;
  draws: number;
  losses: number;
  active: number;
  stopped: number;
  interrupted: number;
  wall_elapsed_seconds: number | null;
  elapsed_samples: number;
}

export function seedError(value: string): string | null {
  if (!/^\d+$/.test(value) || BigInt(value) > 18446744073709551615n) return 'Enter a decimal unsigned 64-bit seed (0–18446744073709551615).';
  return null;
}
