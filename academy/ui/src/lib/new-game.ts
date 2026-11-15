import { rememberCurrentGame } from './current-game.svelte';
import { gameStore, type GameStore } from './game.svelte';
import { recordFetch } from './records';

/** Map the retained human/opponent setup roles to Watch's two native seats. */
export async function startConfiguredWatch(setup: GameStore = gameStore): Promise<string> {
  if (setup.seedError) throw new Error(setup.seedError);
  const humanFirst = setup.humanSeat === 0;
  const rolePlayers = [setup.watchPlayerChoice, setup.opponentChoice];
  const roleMirrors = [setup.watchSearchConfig, setup.searchConfig];
  const imported = setup.deckImport;
  const response = await recordFetch('/api/watch', {
    method:'POST', headers:{'content-type':'application/json'},
    body:JSON.stringify({
      players:humanFirst ? rolePlayers : rolePlayers.toReversed(),
      mirror:humanFirst ? roleMirrors : roleMirrors.toReversed(),
      hero_deck:humanFirst ? setup.decks.hero : setup.decks.villain,
      villain_deck:humanFirst ? setup.decks.villain : setup.decks.hero,
      ...(imported ? {deck_import:humanFirst ? imported : {hero:imported.villain,villain:imported.hero}} : {}),
      decklists:setup.decklists,
      seed:setup.dealMode === 'repeat' ? setup.seed : null,
    }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : result.detail?.message ?? 'Game could not be started.');
  // The token stays in this tab. Reading the recording elsewhere grants no producer access.
  try { sessionStorage.setItem(`etude.watch.${result.id}`, result.token); }
  catch { throw new Error('This browser cannot retain the watched game credential. Enable session storage to watch a live game.'); }
  rememberCurrentGame(`/replay?trace=${encodeURIComponent(result.id)}`);
  return String(result.id);
}
