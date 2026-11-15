import { rememberCurrentGame } from './current-game.svelte';
import type { GameStore } from './game.svelte';
import { watchStart } from './watch-start.svelte';

/** Map the retained human/opponent setup roles to Watch's two native seats. */
export async function startConfiguredWatch(setup: GameStore): Promise<string> {
  if (setup.seedError) throw new Error(setup.seedError);
  const humanFirst = setup.humanSeat === 0;
  const rolePlayers = [setup.watchPlayerChoice, setup.opponentChoice];
  const roleMirrors = [setup.watchSearchConfig, setup.searchConfig];
  const imported = setup.deckImport;
  const result = await watchStart.start({
    players:humanFirst ? rolePlayers : rolePlayers.toReversed(),
    mirror:humanFirst ? roleMirrors : roleMirrors.toReversed(),
    hero_deck:humanFirst ? setup.decks.hero : setup.decks.villain,
    villain_deck:humanFirst ? setup.decks.villain : setup.decks.hero,
    ...(imported ? {deck_import:humanFirst ? imported : {hero:imported.villain,villain:imported.hero}} : {}),
    decklists:setup.decklists,
    seed:setup.dealMode === 'repeat' ? setup.seed : null,
  });
  rememberCurrentGame(`/replay?trace=${encodeURIComponent(result.id)}`);
  return String(result.id);
}
