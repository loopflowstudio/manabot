import { rememberCurrentGame } from './current-game.svelte';
import { gameStore } from './game.svelte';
import { recordFetch } from './records';
import { seedError } from './play-session';

/** Map the retained human/opponent setup roles to Watch's two native seats. */
export async function startConfiguredWatch(): Promise<string> {
  if (gameStore.dealMode === 'repeat' && seedError(gameStore.seed)) throw new Error(seedError(gameStore.seed)!);
  const humanFirst = gameStore.humanSeat === 0;
  const rolePlayers = [gameStore.watchPlayerChoice, gameStore.opponentChoice];
  const roleMirrors = [gameStore.watchMirrorSearch, gameStore.mirrorSearch];
  const imported = gameStore.deckImport;
  const response = await recordFetch('/api/watch', {
    method:'POST', headers:{'content-type':'application/json'},
    body:JSON.stringify({
      players:humanFirst ? rolePlayers : rolePlayers.toReversed(),
      mirror:humanFirst ? roleMirrors : roleMirrors.toReversed(),
      hero_deck:humanFirst ? gameStore.decks.hero : gameStore.decks.villain,
      villain_deck:humanFirst ? gameStore.decks.villain : gameStore.decks.hero,
      ...(imported ? {deck_import:humanFirst ? imported : {hero:imported.villain,villain:imported.hero}} : {}),
      decklists:gameStore.decklists,
      seed:gameStore.dealMode === 'repeat' ? gameStore.seed : null,
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
