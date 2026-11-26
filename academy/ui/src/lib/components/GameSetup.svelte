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
    opponentOnly?: boolean;
    humanActionLabel?: string;
    showStart?: boolean;
    applySmartDefaults?: boolean;
  }
  let {
    setup,
    onStart,
    disabled = false,
    opponentOnly = false,
    humanActionLabel = 'Start game',
    showStart = true,
    applySmartDefaults = false,
  }: Props = $props();

  let trained = $state<TrainedOpponent[]>([]);
  let deckPools = $state<DeckPoolSummary[]>([]);
  let keptDecks = $state<KeptDeck[]>([]);
  let catalogError = $state<string | null>(null);
  let extension = $state<string | null | undefined>(undefined);
  let defaultOpponentId = $state<string | null>(null);
  let botRecords = $state<Record<string, { wins: number; losses: number }>>({});
  let opponentEdited = false;
  let watchSeat1Edited = false;
  let deckEdited = false;
  let decklistsEdited = false;

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
      if (!opponentEdited && preferred && (applySmartDefaults || (!setup.playSession && setup.opponentChoice === 'random'))) {
        setup.setOpponentChoice(preferred);
      }
      if (!watchSeat1Edited && !setup.playSession && setup.watchPlayerChoice === 'random') {
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

    // Tally bots' own W-L records from recent recorded games (never personal W-L).
    void fetch('/api/traces?limit=50').then(async response => {
      if (!response.ok) return;
      const data = await response.json() as {
        games?: {
          winner: number | null;
          players?: { seat: number; kind: string; name?: string; policy?: { id?: string } | null }[];
        }[];
      };
      if (!active) return;
      const counts: Record<string, { wins: number; losses: number }> = {};
      for (const game of data.games ?? []) {
        if (game.winner !== 0 && game.winner !== 1) continue;
        for (const player of game.players ?? []) {
          if (player.kind === 'human') continue;
          const key = player.policy?.id ?? player.name;
          if (!key) continue;
          const slot = counts[key] ?? { wins: 0, losses: 0 };
          if (player.seat === game.winner) slot.wins += 1;
          else slot.losses += 1;
          counts[key] = slot;
        }
      }
      botRecords = counts;
    }).catch(() => {});

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
        : (isPoolSelected && humanActionLabel === 'Start game' ? 'Deal & Play' : humanActionLabel),
  );
</script>

<section aria-label="Game setup" class="my-4 min-w-0 space-y-4">
  {#if !opponentOnly}
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
  {/if}

  <div class="grid gap-3 md:grid-cols-2">
    {#if !opponentOnly}
      <fieldset {disabled} class="min-w-0 space-y-2.5 rounded border border-line bg-field p-3">
        <legend class="type-label px-2">{setup.watchMode ? 'Seat 1 · Bot 1' : 'Seat 1 · You'}</legend>
        <label class="flex flex-col gap-1">
          <span class="type-label text-ink-2">Deck</span>
          {#if setup.deckImport?.hero}
            <span class="text-sm">Imported · {setup.deckImport.hero.name ?? setup.deckImport.hero.filename}</span>
          {:else}
            <select aria-label="Seat 1 deck" class="rounded border border-line bg-panel p-2" value={setup.decks.hero}
              onchange={e => { deckEdited = true; setup.setHeroDeck(e.currentTarget.value as DeckChoice); }}>
              {#if deckPools.length > 0}
                <optgroup label="Deck pools">
                  {#each deckPools as pool}<option value={pool.key}>{pool.label} ({pool.count} decks)</option>{/each}
                </optgroup>
              {/if}
              <optgroup label="Elemental decks">
                {#each ELEMENTAL_DECK_CHOICES as deck}<option value={deck.key}>{deck.label}</option>{/each}
              </optgroup>
              {#if legacyDeckLabel(setup.decks.hero)}
                <option value={setup.decks.hero}>{legacyDeckLabel(setup.decks.hero)}</option>
              {/if}
              {#if keptDecks.length > 0}
                <optgroup label="Kept lists (this browser)">
                  {#each keptDecks as kept (kept.id)}<option value={keptDeckChoice(kept.id)}>{kept.name}</option>{/each}
                </optgroup>
              {/if}
            </select>
          {/if}
        </label>
        {#if setup.watchMode}
          <PlayerPicker label="Seat 1 bot" value={setup.watchPlayerChoice} {trained} {extension} {catalogError} {disabled}
            decklists={setup.decklists} manaPayment={setup.manaPayment} manaFloats={setup.manaFloats} {botRecords}
            setup={{ human_seat: 0, hero_deck: setup.decks.hero, villain_deck: setup.decks.villain, ...(setup.deckImport ? { deck_import: setup.deckImport } : {}) }}
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
    {/if}

    <fieldset {disabled} class="min-w-0 space-y-2.5 rounded border border-line bg-field p-3">
      <legend class="type-label px-2">{setup.watchMode ? 'Seat 2 · Bot 2' : 'Seat 2 · Opponent'}</legend>
      {#if !opponentOnly}
        <label class="flex flex-col gap-1">
          <span class="type-label text-ink-2">Deck</span>
          {#if setup.deckImport?.villain}
            <span class="text-sm">Imported · {setup.deckImport.villain.name ?? setup.deckImport.villain.filename}</span>
          {:else}
            <select aria-label="Seat 2 deck" class="rounded border border-line bg-panel p-2" value={setup.decks.villain}
              onchange={e => { deckEdited = true; setup.setVillainDeck(e.currentTarget.value as DeckChoice); }}>
              {#if deckPools.length > 0}
                <optgroup label="Deck pools">
                  {#each deckPools as pool}<option value={pool.key}>{pool.label} ({pool.count} decks)</option>{/each}
                </optgroup>
              {/if}
              <optgroup label="Elemental decks">
                {#each ELEMENTAL_DECK_CHOICES as deck}<option value={deck.key}>{deck.label}</option>{/each}
              </optgroup>
              {#if legacyDeckLabel(setup.decks.villain)}
                <option value={setup.decks.villain}>{legacyDeckLabel(setup.decks.villain)}</option>
              {/if}
              {#if keptDecks.length > 0}
                <optgroup label="Kept lists (this browser)">
                  {#each keptDecks as kept (kept.id)}<option value={keptDeckChoice(kept.id)}>{kept.name}</option>{/each}
                </optgroup>
              {/if}
            </select>
          {/if}
        </label>
      {/if}
      <PlayerPicker label="Seat 2 bot" value={setup.opponentChoice} {trained} {extension} {catalogError} {disabled}
        decklists={setup.decklists} manaPayment={setup.manaPayment} manaFloats={setup.manaFloats} {botRecords}
        setup={{ human_seat: setup.humanSeat, hero_deck: setup.decks.hero, villain_deck: setup.decks.villain, ...(setup.deckImport ? { deck_import: setup.deckImport } : {}) }}
        onChange={value => { opponentEdited = true; setup.setOpponentChoice(value); }}
        search={setup.searchConfig}
        onSearchChange={value => { setup.searchConfig = value; }} />
    </fieldset>
  </div>

  {#if !opponentOnly}
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
      <DecklistsSelector value={setup.decklists} opponent={setup.opponentChoice} {disabled} onChange={value => { decklistsEdited = true; setup.setDecklists(value); }} />
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
      {#if showStart}
        <button class="btn btn-primary ml-auto" {disabled} onclick={onStart}>{startLabel}</button>
      {/if}
    </div>
    {#if !setup.deckImport && !isPoolDeckChoice(setup.decks.hero) && !isPoolDeckChoice(setup.decks.villain) && !isKeptDeckChoice(setup.decks.hero) && !isKeptDeckChoice(setup.decks.villain)}
      <details><summary class="cursor-pointer type-caption text-ink-2">Browse bundled decks and cards</summary>
        <DeckSelector hero={setup.decks.hero} villain={setup.decks.villain} {disabled} heroLabel="Seat 1 deck" villainLabel="Seat 2 deck" onHeroChange={value => { deckEdited = true; setup.setHeroDeck(value); }} onVillainChange={value => { deckEdited = true; setup.setVillainDeck(value); }} />
      </details>
    {/if}
    <DeckImport {setup} heroLabel={setup.watchMode ? 'Seat 1 deck' : 'Your deck'} villainLabel={setup.watchMode ? 'Seat 2 deck' : 'Opponent deck'}
      onKeptChange={decks => { keptDecks = decks; }} />
  {:else if showStart}
    <button class="btn btn-primary" {disabled} onclick={onStart}>{startLabel}</button>
  {/if}
</section>
