<script lang="ts">
  import type { GameStore } from '$lib/game.svelte';
  import { DECK_CHOICES, isPoolDeckChoice, type DeckChoice, type DeckPoolSummary } from '$lib/decks';
  import type { OpponentCatalog, TrainedOpponent } from '$lib/types';
  import PlayerPicker from './PlayerPicker.svelte';
  import DeckSelector from './DeckSelector.svelte';
  import DeckImport from './DeckImport.svelte';
  import DecklistsSelector from './DecklistsSelector.svelte';
  import { onMount } from 'svelte';
  interface Props {
    setup: GameStore;
    onStart: () => void;
    disabled?: boolean;
    opponentOnly?: boolean;
    humanActionLabel?: string;
    showStart?: boolean;
  }
  let { setup, onStart, disabled = false, opponentOnly = false, humanActionLabel = 'Start game', showStart = true }: Props = $props();
  let trained = $state<TrainedOpponent[]>([]);
  let deckPools = $state<DeckPoolSummary[]>([]);
  let catalogError = $state<string | null>(null);
  let extension = $state<string | null | undefined>(undefined);
  onMount(() => {
    void fetch('/api/opponents').then(async response => {
      if (!response.ok) throw new Error('Policies could not be loaded.');
      const catalog: OpponentCatalog = await response.json();
      trained = catalog.opponents; catalogError = catalog.catalog_error; extension = catalog.extension;
    }).catch(error => catalogError = error.message);
    void fetch('/api/deck-pools').then(async response => {
      if (!response.ok) return;
      const data = await response.json() as { pools?: DeckPoolSummary[] };
      deckPools = data.pools ?? [];
    }).catch(() => {});
  });
  function controller(seat: number, human: boolean): void {
    if (human) {
      if (seat !== setup.humanSeat) {
        // Controllers belong to native seats. Reorient setup roles so editing
        // a controller keeps both native decks and the remaining bot in place.
        const decks = setup.decks;
        setup.setHeroDeck(decks.villain); setup.setVillainDeck(decks.hero);
        const imported = setup.deckImport;
        if (imported) setup.deckImport = {hero:imported.villain, villain:imported.hero};
        const opponent = setup.opponentChoice;
        setup.opponentChoice = setup.watchPlayerChoice; setup.watchPlayerChoice = opponent;
        const search = setup.searchConfig;
        setup.searchConfig = setup.watchSearchConfig; setup.watchSearchConfig = search;
      }
      setup.humanSeat = seat as 0 | 1; setup.watchMode = false;
    }
    else if (seat === setup.humanSeat) setup.watchMode = true;
  }
</script>
<section aria-label="Game setup" class="my-4 min-w-0 space-y-4">
  <div class="grid gap-4 md:grid-cols-2">
    {#each [0, 1] as seat}
      {@const hero = seat === setup.humanSeat}
      {#if !opponentOnly || !hero}
        <fieldset {disabled} class="min-w-0 space-y-3 rounded border border-line bg-field p-4">
          <legend class="type-label px-2">Seat {seat + 1} · {seat === 0 ? 'Play first' : 'Draw first'}</legend>
          {#if !opponentOnly}
            <label class="flex flex-col gap-1">Controller
              <select aria-label={`Seat ${seat + 1} controller`} class="rounded border border-line bg-panel p-2" value={hero && !setup.watchMode ? 'human' : 'bot'} onchange={e => controller(seat, e.currentTarget.value === 'human')}>
                <option value="human">You</option><option value="bot">Bot</option>
              </select>
            </label>
            <label class="flex flex-col gap-1">Deck
              {#if setup.deckImport}<span class="text-sm">Imported · {hero ? (setup.deckImport.hero.name ?? setup.deckImport.hero.filename) : (setup.deckImport.villain.name ?? setup.deckImport.villain.filename)}</span>
              {:else}<select aria-label={`Seat ${seat + 1} deck`} class="rounded border border-line bg-panel p-2" value={hero ? setup.decks.hero : setup.decks.villain} onchange={e => hero ? setup.setHeroDeck(e.currentTarget.value as DeckChoice) : setup.setVillainDeck(e.currentTarget.value as DeckChoice)}>
                {#if deckPools.length > 0}
                  <optgroup label="Deck pools">
                    {#each deckPools as pool}<option value={pool.key}>{pool.label} ({pool.count} decks)</option>{/each}
                  </optgroup>
                  <optgroup label="Bundled decks">
                    {#each DECK_CHOICES as deck}<option value={deck.key}>{deck.label}</option>{/each}
                  </optgroup>
                {:else}
                  {#each DECK_CHOICES as deck}<option value={deck.key}>{deck.label}</option>{/each}
                {/if}
              </select>{/if}
            </label>
          {/if}
          {#if !hero || setup.watchMode}
            <PlayerPicker label={`Seat ${seat + 1} bot`} value={hero ? setup.watchPlayerChoice : setup.opponentChoice} {trained} {extension} {catalogError} {disabled}
              decklists={setup.decklists} setup={{human_seat:setup.humanSeat,hero_deck:setup.decks.hero,villain_deck:setup.decks.villain,...(setup.deckImport ? {deck_import:setup.deckImport} : {})}}
              onChange={value => {if (hero) setup.watchPlayerChoice = value; else setup.setOpponentChoice(value);}}
              search={hero ? setup.watchSearchConfig : setup.searchConfig}
              onSearchChange={value => {if (hero) setup.watchSearchConfig = value; else setup.searchConfig = value;}} />
          {:else}
            <label class="flex items-center gap-2">
              Mana payment
              <select bind:value={setup.manaPayment} class="rounded border border-line bg-panel p-2">
                <option value="manual">Manual</option>
                <option value="auto">Automatic</option>
              </select>
            </label>
            <label class="flex items-center gap-2">
              Mana floats
              <select bind:value={setup.manaFloats} class="rounded border border-line bg-panel p-2">
                <option value="off">Off</option>
                <option value="usable">Only when the mana could be used</option>
                <option value="all">All</option>
              </select>
            </label>
          {/if}
        </fieldset>
      {/if}
    {/each}
  </div>
  {#if !opponentOnly}
    <div class="flex flex-wrap items-end gap-4">
      <label class="flex flex-col gap-1">Deal mode<select aria-label="Deal mode" class="rounded border border-line bg-field p-2" bind:value={setup.dealMode}><option value="fresh">Fresh deal</option><option value="repeat">Repeat seed</option></select></label>
      {#if setup.dealMode === 'repeat'}<label class="flex flex-col gap-1">Seed<input class="rounded border border-line bg-field p-2" inputmode="numeric" bind:value={setup.seed} /></label>{/if}
      <DecklistsSelector value={setup.decklists} opponent={setup.opponentChoice} {disabled} onChange={value => setup.setDecklists(value)} />
    </div>
    {#if !setup.deckImport && !isPoolDeckChoice(setup.decks.hero) && !isPoolDeckChoice(setup.decks.villain)}<details><summary class="cursor-pointer">Browse decks and cards</summary>
      <DeckSelector hero={setup.decks.hero} villain={setup.decks.villain} {disabled} heroLabel={`Seat ${setup.humanSeat + 1} deck`} villainLabel={`Seat ${2 - setup.humanSeat} deck`} onHeroChange={value => setup.setHeroDeck(value)} onVillainChange={value => setup.setVillainDeck(value)} />
    </details>{/if}
    {#key setup.humanSeat}<DeckImport {setup} heroLabel={setup.watchMode ? `Seat ${setup.humanSeat + 1} deck` : 'Your deck'} villainLabel={setup.watchMode ? `Seat ${2 - setup.humanSeat} deck` : 'Opponent deck'} />{/key}
  {/if}
  {#if showStart}
    <button class="btn btn-primary" {disabled} onclick={onStart}>{setup.newGamePending ? 'Preparing game…' : setup.watchMode ? 'Start watching' : humanActionLabel}</button>
  {/if}
</section>
