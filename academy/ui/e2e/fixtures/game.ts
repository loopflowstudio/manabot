let game = 0;

/** What every Play reply says about its game, for a fresh game in controlled WebSocket fixtures. */
export function gameReply(): { game: string; opponent: null; opponent_label: null; mode: 'live'; setup: Record<string, unknown> } {
  return { game: `fixture-game-${++game}`, opponent: null, opponent_label: null, mode: 'live', setup: {} };
}
