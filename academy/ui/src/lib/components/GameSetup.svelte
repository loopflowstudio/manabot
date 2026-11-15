<script lang="ts">
  import { gameStore } from '$lib/game.svelte';
  import { DECK_CHOICES, type DeckChoice } from '$lib/decks';
  import type { OpponentCatalog, TrainedOpponent } from '$lib/types';
  import PlayerPicker from './PlayerPicker.svelte';
  import DeckSelector from './DeckSelector.svelte';
  import DeckImport from './DeckImport.svelte';
  import DecklistsSelector from './DecklistsSelector.svelte';
  import { onMount } from 'svelte';
  let {onStart, disabled = false, opponentOnly = false, humanActionLabel = 'New Game'}: {humanActionLabel?:string; onStart: () => void; disabled?:boolean; opponentOnly?:boolean} = $props();
  let trained = $state<TrainedOpponent[]>([]);
  let catalogError = $state<string | null>(null);
  let extension = $state<string | null | undefined>(undefined);
  onMount(() => {
    void fetch('/api/opponents').then(async response => {
      if (!response.ok) throw new Error('Policies could not be loaded.');
      const catalog: OpponentCatalog = await response.json();
      trained = catalog.opponents; catalogError = catalog.catalog_error; extension = catalog.extension;
    }).catch(error => catalogError = error.message);
  });
  function controller(seat: number, human: boolean): void {
    if (human) {
      if (seat !== gameStore.humanSeat) {
        // Controllers belong to native seats. Reorient setup roles so editing
        // a controller keeps both native decks and the remaining bot in place.
        const decks = gameStore.decks;
        gameStore.setHeroDeck(decks.villain); gameStore.setVillainDeck(decks.hero);
        const imported = gameStore.deckImport;
        if (imported) gameStore.deckImport = {hero:imported.villain, villain:imported.hero};
        const opponent = gameStore.opponentChoice;
        gameStore.opponentChoice = gameStore.watchPlayerChoice; gameStore.watchPlayerChoice = opponent;
        const search = gameStore.mirrorSearch;
        gameStore.mirrorSearch = gameStore.watchMirrorSearch; gameStore.watchMirrorSearch = search;
      }
      gameStore.humanSeat = seat as 0 | 1; gameStore.watchMode = false;
    }
    else if (seat === gameStore.humanSeat) gameStore.watchMode = true;
  }
</script>
<section aria-label="Game setup" class="my-4 min-w-0 space-y-4">
  <div class="grid gap-4 md:grid-cols-2">
    {#each [0, 1] as seat}
      {@const hero = seat === gameStore.humanSeat}
      {#if !opponentOnly || !hero}
        <fieldset {disabled} class="min-w-0 space-y-3 rounded border border-line bg-field p-4">
          <legend class="type-rubric px-2">Seat {seat + 1} · {seat === 0 ? 'Play first' : 'Draw first'}</legend>
          {#if !opponentOnly}
            <label class="flex flex-col gap-1">Controller
              <select aria-label={`Seat ${seat + 1} controller`} class="rounded border border-line bg-panel p-2" value={hero && !gameStore.watchMode ? 'human' : 'bot'} onchange={e => controller(seat, e.currentTarget.value === 'human')}>
                <option value="human">You</option><option value="bot">Bot</option>
              </select>
            </label>
            <label class="flex flex-col gap-1">Deck
              {#if gameStore.deckImport}<span class="text-sm">Imported · {hero ? gameStore.deckImport.hero.filename : gameStore.deckImport.villain.filename}</span>
              {:else}<select aria-label={`Seat ${seat + 1} deck`} class="rounded border border-line bg-panel p-2" value={hero ? gameStore.decks.hero : gameStore.decks.villain} onchange={e => hero ? gameStore.setHeroDeck(e.currentTarget.value as DeckChoice) : gameStore.setVillainDeck(e.currentTarget.value as DeckChoice)}>
                {#each DECK_CHOICES as deck}<option value={deck.key}>{deck.label}</option>{/each}
              </select>{/if}
            </label>
          {/if}
          {#if !hero || gameStore.watchMode}
            <PlayerPicker label={`Seat ${seat + 1} bot`} value={hero ? gameStore.watchPlayerChoice : gameStore.opponentChoice} {trained} {extension} {catalogError} {disabled}
              decklists={gameStore.decklists} setup={{human_seat:gameStore.humanSeat,hero_deck:gameStore.decks.hero,villain_deck:gameStore.decks.villain,...(gameStore.deckImport ? {deck_import:gameStore.deckImport} : {})}}
              onChange={value => {if (hero) gameStore.watchPlayerChoice = value; else gameStore.setOpponentChoice(value);}}
              mirror={hero ? gameStore.watchMirrorSearch : gameStore.mirrorSearch}
              onMirrorChange={value => {if (hero) gameStore.watchMirrorSearch = value; else gameStore.mirrorSearch = value;}} />
          {:else}<p class="type-caption text-ink-2">Play your complete legal choices. Inspect cards at any time.</p>{/if}
        </fieldset>
      {/if}
    {/each}
  </div>
  {#if !opponentOnly}
    <div class="flex flex-wrap items-end gap-4">
      <label class="flex flex-col gap-1">Deal mode<select aria-label="Deal mode" class="rounded border border-line bg-field p-2" bind:value={gameStore.dealMode}><option value="fresh">Fresh deal</option><option value="repeat">Repeat seed</option></select></label>
      {#if gameStore.dealMode === 'repeat'}<label class="flex flex-col gap-1">Seed<input class="rounded border border-line bg-field p-2" inputmode="numeric" bind:value={gameStore.seed} /></label>{/if}
      <DecklistsSelector value={gameStore.decklists} opponent={gameStore.opponentChoice} {disabled} onChange={value => gameStore.setDecklists(value)} />
    </div>
    {#if !gameStore.deckImport}<details><summary class="cursor-pointer">Browse decks and cards</summary>
      <DeckSelector hero={gameStore.decks.hero} villain={gameStore.decks.villain} {disabled} heroLabel={`Seat ${gameStore.humanSeat + 1} deck`} villainLabel={`Seat ${2 - gameStore.humanSeat} deck`} onHeroChange={value => gameStore.setHeroDeck(value)} onVillainChange={value => gameStore.setVillainDeck(value)} />
    </details>{/if}
    {#key gameStore.humanSeat}<DeckImport heroLabel={gameStore.watchMode ? `Seat ${gameStore.humanSeat + 1} deck` : 'Your deck'} villainLabel={gameStore.watchMode ? `Seat ${2 - gameStore.humanSeat} deck` : 'Opponent deck'} />{/key}
    <button class="btn btn-primary" {disabled} onclick={onStart}>{gameStore.newGamePending ? 'Preparing game…' : gameStore.watchMode ? 'Start watching' : humanActionLabel}</button>
  {/if}
</section>
