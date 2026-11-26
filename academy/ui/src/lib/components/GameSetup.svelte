<script lang="ts">
  import { onMount } from 'svelte';
  import { policyChoice, type GameStore, type PlayerChoice } from '$lib/game.svelte';
  import { DECK_CHOICES, ELEMENTAL_DECK_CHOICES, isPoolDeckChoice, type DeckChoice, type DeckPoolSummary } from '$lib/decks';
  import { isKeptDeckChoice, keptDeckChoice, loadKeptDecks, type KeptDeck } from '$lib/kept-decks';
  import type { OpponentCatalog, TrainedOpponent } from '$lib/types';
  import PlayerPicker from './PlayerPicker.svelte';
  import DeckSelector from './DeckSelector.svelte';
  import DeckImport from './DeckImport.svelte';
  import DecklistsSelector from './DecklistsSelector.svelte';

  interface Props {
    setup: GameStore;
    onStart: () => void;
    disabled?: boolean;
    applySmartDefaults?: boolean;
  }
  let {
    setup,
    onStart,
    disabled = false,
    applySmartDefaults = false,
  }: Props = $props();

  let trained = $state<TrainedOpponent[]>([]);
  let deckPools = $state<DeckPoolSummary[]>([]);
  let keptDecks = $state<KeptDeck[]>([]);
  let catalogError = $state<string | null>(null);
  let extension = $state<string | null | undefined>(undefined);
  let defaultOpponentId = $state<string | null>(null);
  let opponentEdited = false;
  let watchSeat1Edited = false;
  let deckEdited = false;
  let decklistsEdited = false;
  // What Start sends as `deck_import`, so the policy check admits kept decks too.
  const resolvedImport = $derived(setup.resolveDeckImport());
  const pickerSetup = (humanSeat: 0 | 1): Record<string, unknown> => ({
    human_seat: humanSeat,
    hero_deck: setup.decks.hero,
    villain_deck: setup.decks.villain,
    ...(resolvedImport ? { deck_import: resolvedImport } : {}),
  });

  function resolveCatalogChoice(id: string | null | undefined, list: TrainedOpponent[]): PlayerChoice | null {
    if (!id) return null;
    if (id.startsWith('policy:')) return id as PlayerChoice;
    if (id === 'random' || id === 'passive' || id.startsWith('search-')) return id as PlayerChoice;
    const match = list.find(entry => entry.available && (entry.aliases?.includes(id) || entry.sha256 === id));
    return match ? policyChoice(match.sha256) : null;
  }

  onMount(() => {
    let active = true;
    keptDecks = loadKeptDecks();
    const oppFetch = fetch('/api/opponents').then(async response => {
      if (!response.ok) throw new Error('Policies could not be loaded.');
      const catalog: OpponentCatalog = await response.json();
      if (!active) return null;
      trained = catalog.opponents;
      catalogError = catalog.catalog_error;
      extension = catalog.extension;
      defaultOpponentId = catalog.default_opponent ?? null;
      return catalog;
    }).catch(error => {
      if (active) catalogError = error.message;
      return null;
    });

    const poolFetch = fetch('/api/deck-pools').then(async response => {
      if (!response.ok) return [] as DeckPoolSummary[];
      const data = await response.json() as { pools?: DeckPoolSummary[] };
      const pools = data.pools ?? [];
      if (active) deckPools = pools;
      return pools;
    }).catch(() => [] as DeckPoolSummary[]);

    void Promise.all([oppFetch, poolFetch]).then(([catalog, pools]) => {
      if (!active) return;
      const preferred = resolveCatalogChoice(catalog?.default_opponent, catalog?.opponents ?? []);
      const heuristic = resolveCatalogChoice('heuristic', catalog?.opponents ?? []);
      if (!opponentEdited && preferred && (applySmartDefaults || setup.opponentChoice === 'random')) {
        setup.setOpponentChoice(preferred);
      }
      if (!watchSeat1Edited && setup.watchPlayerChoice === 'random') {
        const watchSeat1 = heuristic ?? preferred;
        if (watchSeat1) setup.watchPlayerChoice = watchSeat1;
      }
      if (applySmartDefaults && !deckEdited) {
        const hasHeldOut = pools.some(pool => pool.key === 'pool:held-out');
        if (hasHeldOut) {
          setup.setHeroDeck('pool:held-out');
          setup.setVillainDeck('pool:held-out');
          const chosen = setup.opponentChoice;
          if (!decklistsEdited && !chosen.startsWith('search-')) setup.setDecklists('closed');
        }
      }
    });

    return () => { active = false; };
  });

  function setMode(watch: boolean): void {
    setup.watchMode = watch;
    if (!watch) setup.humanSeat = setup.turnOrder === 'draw' ? 1 : 0;
  }

  function swapWatchBots(): void {
    opponentEdited = true;
    watchSeat1Edited = true;
    deckEdited = true;
    const prevOpp = setup.opponentChoice;
    setup.opponentChoice = setup.watchPlayerChoice;
    setup.watchPlayerChoice = prevOpp;
    const prevSearch = setup.searchConfig;
    setup.searchConfig = setup.watchSearchConfig;
    setup.watchSearchConfig = prevSearch;
    const prevHero = setup.decks.hero;
    setup.setHeroDeck(setup.decks.villain);
    setup.setVillainDeck(prevHero);
  }

  function applyPreset(kind: 'trophies' | 'elemental'): void {
    opponentEdited = true;
    deckEdited = true;
    decklistsEdited = true;
    setup.deckImport = null;
    if (kind === 'trophies') {
      setup.setHeroDeck('pool:held-out');
      setup.setVillainDeck('pool:held-out');
      const preferred = resolveCatalogChoice(defaultOpponentId, trained);
      if (preferred && !preferred.startsWith('search-')) setup.setOpponentChoice(preferred);
      else if (setup.opponentChoice.startsWith('search-')) setup.setOpponentChoice('random');
      setup.setDecklists('closed');
    } else {
      setup.setHeroDeck('elemental_ur');
      setup.setVillainDeck('elemental_gw');
      setup.setDecklists('open');
    }
  }

  function legacyDeckLabel(choice: DeckChoice): string | null {
    if (isPoolDeckChoice(choice) || isKeptDeckChoice(choice)) return null;
    if (ELEMENTAL_DECK_CHOICES.some(deck => deck.key === choice)) return null;
    return DECK_CHOICES.find(deck => deck.key === choice)?.label ?? null;
  }

  const hasTrophyPool = $derived(deckPools.some(pool => pool.key === 'pool:held-out'));
  const isPoolSelected = $derived(isPoolDeckChoice(setup.decks.hero) || isPoolDeckChoice(setup.decks.villain));
  const startLabel = $derived(
    setup.newGamePending
      ? 'Preparing game…'
      : setup.watchMode
        ? (isPoolSelected ? 'Deal & Watch' : 'Start watching')
        : (isPoolSelected ? 'Deal & Play' : 'Start game'),
  );
</script>

{#snippet deckField(side: 'hero' | 'villain', label: string, onSelect: (next: DeckChoice) => void)}
  {@const imported = setup.deckImport?.[side]}
  {@const current = setup.decks[side]}
  <label class="flex flex-col gap-1">
    <span class="type-label text-ink-2">Deck</span>
    {#if imported}
      <span class="text-sm">Imported · {imported.name ?? imported.filename}</span>
    {:else}
      <select aria-label={label} class="rounded border border-line bg-panel p-2" value={current}
        onchange={e => { deckEdited = true; onSelect(e.currentTarget.value as DeckChoice); }}>
        {#if deckPools.length > 0}
          <optgroup label="Deck pools">
            {#each deckPools as pool}<option value={pool.key}>{pool.label} ({pool.count} decks)</option>{/each}
          </optgroup>
        {/if}
        <optgroup label="Elemental decks">
          {#each ELEMENTAL_DECK_CHOICES as deck}<option value={deck.key}>{deck.label}</option>{/each}
        </optgroup>
        {#if legacyDeckLabel(current)}
          <option value={current}>{legacyDeckLabel(current)}</option>
        {/if}
        {#if keptDecks.length > 0}
          <optgroup label="Kept lists (this browser)">
            {#each keptDecks as kept (kept.id)}<option value={keptDeckChoice(kept.id)}>{kept.name}</option>{/each}
          </optgroup>
        {/if}
      </select>
    {/if}
  </label>
{/snippet}

<section aria-label="Game setup" class="my-4 min-w-0 space-y-4">
  <div class="flex flex-wrap items-center justify-between gap-3">
      <div class="inline-flex overflow-hidden rounded border border-line bg-field" role="group" aria-label="Match mode">
        <button type="button" class={`px-3 py-1.5 text-sm font-medium ${!setup.watchMode ? 'bg-action text-on-action' : 'text-ink-2 hover:text-ink'}`}
          aria-pressed={!setup.watchMode} {disabled} onclick={() => setMode(false)}>Play vs Bot</button>
        <button type="button" class={`border-l border-line px-3 py-1.5 text-sm font-medium ${setup.watchMode ? 'bg-action text-on-action' : 'text-ink-2 hover:text-ink'}`}
          aria-pressed={setup.watchMode} {disabled} onclick={() => setMode(true)}>Watch Bot vs Bot</button>
      </div>
      {#if hasTrophyPool}
        <div class="flex flex-wrap items-center gap-2">
          <span class="type-caption text-ink-3">Presets:</span>
          <button type="button" class="btn btn-secondary btn-sm" {disabled} onclick={() => applyPreset('trophies')}>TLA Trophies</button>
          <button type="button" class="btn btn-secondary btn-sm" {disabled} onclick={() => applyPreset('elemental')}>Elemental</button>
        </div>
      {/if}
  </div>

  <div class="grid gap-3 md:grid-cols-2">
      <fieldset {disabled} class="min-w-0 space-y-2.5 rounded border border-line bg-field p-3">
        <legend class="type-label px-2">{setup.watchMode ? 'Seat 1 · Bot 1' : 'Seat 1 · You'}</legend>
        {@render deckField('hero', 'Seat 1 deck', value => setup.setHeroDeck(value))}
        {#if setup.watchMode}
          <PlayerPicker label="Seat 1 bot" value={setup.watchPlayerChoice} {trained} {extension} {catalogError} {disabled}
            decklists={setup.decklists} manaPayment={setup.manaPayment} manaFloats={setup.manaFloats}
            setup={pickerSetup(0)}
            onChange={value => { watchSeat1Edited = true; setup.watchPlayerChoice = value; }}
            search={setup.watchSearchConfig}
            onSearchChange={value => { setup.watchSearchConfig = value; }} />
        {:else}
          <div class="flex flex-wrap gap-3">
            <label class="flex flex-col gap-1">
              <span class="type-label text-ink-2">Mana payment</span>
              <select bind:value={setup.manaPayment} class="rounded border border-line bg-panel p-2">
                <option value="manual">Manual</option>
                <option value="auto">Automatic</option>
              </select>
            </label>
            <label class="flex flex-col gap-1">
              <span class="type-label text-ink-2">Mana floats</span>
              <select bind:value={setup.manaFloats} class="rounded border border-line bg-panel p-2">
                <option value="off">Off</option>
                <option value="usable">Only when usable</option>
                <option value="all">All</option>
              </select>
            </label>
          </div>
        {/if}
      </fieldset>

    <fieldset {disabled} class="min-w-0 space-y-2.5 rounded border border-line bg-field p-3">
      <legend class="type-label px-2">{setup.watchMode ? 'Seat 2 · Bot 2' : 'Seat 2 · Opponent'}</legend>
      {@render deckField('villain', 'Seat 2 deck', value => setup.setVillainDeck(value))}
      <PlayerPicker label="Seat 2 bot" value={setup.opponentChoice} {trained} {extension} {catalogError} {disabled}
        decklists={setup.decklists} manaPayment={setup.manaPayment} manaFloats={setup.manaFloats}
        setup={pickerSetup(setup.watchMode ? 0 : setup.humanSeat)}
        onChange={value => { opponentEdited = true; setup.setOpponentChoice(value); }}
        search={setup.searchConfig}
        onSearchChange={value => { setup.searchConfig = value; }} />
    </fieldset>
  </div>

    <div class="flex flex-wrap items-end gap-4 rounded border border-line bg-panel p-3">
      {#if !setup.watchMode}
        <div class="flex flex-col gap-1">
          <span class="type-label text-ink-2">Your turn order</span>
          <div class="inline-flex overflow-hidden rounded border border-line bg-field" role="group" aria-label="Your turn order">
            {#each [['play', 'Play first'], ['random', 'Random'], ['draw', 'Draw']] as [value, text]}
              <button type="button"
                class={`px-2.5 py-1.5 text-sm ${setup.turnOrder === value ? 'bg-action text-on-action font-medium' : 'text-ink-2 hover:text-ink'} ${value !== 'play' ? 'border-l border-line' : ''}`}
                aria-pressed={setup.turnOrder === value} {disabled}
                onclick={() => {
                  setup.turnOrder = value as 'play' | 'random' | 'draw';
                  if (value === 'play') setup.humanSeat = 0;
                  if (value === 'draw') setup.humanSeat = 1;
                }}>{text}</button>
            {/each}
          </div>
        </div>
      {:else}
        <button type="button" class="btn btn-secondary btn-sm" {disabled} onclick={swapWatchBots}>⇄ Swap bots</button>
      {/if}
      <DecklistsSelector value={setup.decklists} players={setup.watchMode ? [setup.watchPlayerChoice, setup.opponentChoice] : [setup.opponentChoice]} {disabled} onChange={value => { decklistsEdited = true; setup.setDecklists(value); }} />
      <label class="flex flex-col gap-1">
        <span class="type-label text-ink-2">Deal mode</span>
        <select aria-label="Deal mode" class="rounded border border-line bg-field p-2" bind:value={setup.dealMode}>
          <option value="fresh">Fresh deal</option>
          <option value="repeat">Repeat seed</option>
        </select>
      </label>
      {#if setup.dealMode === 'repeat'}
        <label class="flex flex-col gap-1">
          <span class="type-label text-ink-2">Seed</span>
          <input class="rounded border border-line bg-field p-2" inputmode="numeric" bind:value={setup.seed} />
        </label>
      {/if}
        <button class="btn btn-primary ml-auto" {disabled} onclick={onStart}>{startLabel}</button>
    </div>
    {#if !setup.deckImport && !isPoolDeckChoice(setup.decks.hero) && !isPoolDeckChoice(setup.decks.villain) && !isKeptDeckChoice(setup.decks.hero) && !isKeptDeckChoice(setup.decks.villain)}
      <details><summary class="cursor-pointer type-caption text-ink-2">Browse bundled decks and cards</summary>
        <DeckSelector hero={setup.decks.hero} villain={setup.decks.villain} {disabled} heroLabel="Seat 1 deck" villainLabel="Seat 2 deck" onHeroChange={value => { deckEdited = true; setup.setHeroDeck(value); }} onVillainChange={value => { deckEdited = true; setup.setVillainDeck(value); }} />
      </details>
    {/if}
    <DeckImport {setup} heroLabel={setup.watchMode ? 'Seat 1 deck' : 'Your deck'} villainLabel={setup.watchMode ? 'Seat 2 deck' : 'Opponent deck'}
      onKeptChange={decks => { keptDecks = decks; }} />
</section>
