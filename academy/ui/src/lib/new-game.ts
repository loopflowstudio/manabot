import type { GameStore } from './game.svelte';
import { watchStart } from './watch-start.svelte';

/**
 * Start the Watch game setup describes. Watch's seats are native seats: Seat 1
 * (the first bot and deck on the setup screen) is on the play, whatever seat
 * the person last played a game from.
 */
export async function startConfiguredWatch(setup: GameStore): Promise<string> {
  if (setup.seedError) throw new Error(setup.seedError);
  const result = await watchStart.start({
    players:[setup.watchPlayerChoice, setup.opponentChoice],
    mirror:[setup.watchSearchConfig, setup.searchConfig],
    hero_deck:setup.decks.hero,
    villain_deck:setup.decks.villain,
    ...(setup.deckImport ? {deck_import:setup.deckImport} : {}),
    decklists:setup.decklists,
    seed:setup.dealMode === 'repeat' ? setup.seed : null,
  });
  return String(result.id);
}
