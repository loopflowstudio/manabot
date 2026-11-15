<script lang="ts">
  import { recordedGames } from '$lib/game-file';
  import { onMount } from 'svelte';
  import DeckSelector from './DeckSelector.svelte';
  import PlayerPicker from './PlayerPicker.svelte';
  import type { PlayerChoice } from '$lib/game.svelte';
  import type { OpponentCatalog, TrainedOpponent } from '$lib/types';
  import type { DeckChoice } from '$lib/decks';
  import { parseMirrors, type MirrorSearch } from '$lib/mirror';
  import { recordFetch } from '$lib/records';
  let { onOpen, disabled = false }: { onOpen: (id: string) => void; disabled?: boolean } = $props();
  let players = $state<PlayerChoice[]>(['random', 'search-16']);
  let hero = $state<DeckChoice>('ur_lessons');
  let villain = $state<DeckChoice>('gw_allies');
  let seed = $state('42');
  let mirrors = $state<(MirrorSearch | null)[]>([null, null]);
  let busy = $state(false);
  let error = $state('');
  let ready = $state(false);
  let importedGames = $state<string[]>([]);
  let selectedGame = $state(0);
  let choices = $state(['random', 'search']);
  let trained = $state<TrainedOpponent[]>([]);
  let catalogError = $state<string | null>(null);
  let extension = $state<string | null | undefined>(undefined);
  async function api<T>(url: string, body?: string): Promise<T> {
    const response = await recordFetch(url, { method: body === undefined ? 'GET' : 'POST', headers: {'content-type': 'application/json'}, ...(body === undefined ? {} : { body }) });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      if (typeof payload.detail?.message === 'string') throw new Error(payload.detail.message);
      throw new Error(typeof payload.detail === 'string' ? payload.detail.replaceAll('_', ' ') : `Request failed (${response.status})`);
    }
    return response.json() as Promise<T>;
  }
  onMount(() => {
    void api<OpponentCatalog>('/api/opponents').then(catalog => {
      choices = catalog.watch_players ?? ['random', 'search']; trained = catalog.opponents;
      catalogError = catalog.catalog_error ?? null; extension = catalog.extension; ready = true;
    }).catch(fail);
    try {
      const saved = JSON.parse(localStorage.getItem('etude.watch.setup') ?? 'null');
      if (saved && Array.isArray(saved.players) && saved.players.length === 2 && saved.players.every((p: unknown) => typeof p === 'string')) {
        players = saved.players; hero = saved.hero; villain = saved.villain; seed = saved.seed;
        if (saved.mirror) mirrors = parseMirrors(saved.mirror);
      }
    } catch { /* Invalid preferences do not select a game. */ }
    const params = new URLSearchParams(location.search);
    if (params.has('mirror')) {
      try { mirrors = parseMirrors(JSON.parse(params.get('mirror')!)); }
      catch (reason) { fail(reason); }
    }
    players = players.map((player, seat) => (params.get(`seat${seat + 1}`) ?? player) as PlayerChoice);
  });
  function fail(reason: unknown): void { error = reason instanceof Error ? reason.message : String(reason); }
  function open(result: { id: string; token?: string }): void {
    if (result.token) {
      sessionStorage.setItem(`etude.watch.${result.id}`, result.token);
    }
    onOpen(result.id);
  }
  async function start(): Promise<void> {
    if (busy) return;
    busy = true; error = '';
    try {
      const result = await api<{id: string; token: string}>('/api/watch', JSON.stringify({ players, hero_deck: hero, villain_deck: villain, seed, villain_sims: 16, mirror: mirrors }));
      localStorage.setItem('etude.watch.setup', JSON.stringify({players, hero, villain, seed, mirror: mirrors}));
      open(result);
    } catch (reason) { fail(reason); }
    finally { busy = false; }
  }
  async function importGame(event: Event): Promise<void> {
    const file = (event.currentTarget as HTMLInputElement).files?.[0];
    if (!file) return;
    busy = true; error = '';
    importedGames = [];
    try {
      const buffer = file.name.endsWith('.gz')
        ? await new Response(file.stream().pipeThrough(new DecompressionStream('gzip'))).text()
        : await file.text();
      try { importedGames = recordedGames(buffer); }
      catch { throw new Error(`${file.name} is not a game file: expected JSON or JSON Lines game records, such as an evaluation trace or an exported .game.jsonl.gz.`); }
      selectedGame = 0;
    } catch (reason) { fail(reason); }
    finally { busy = false; }
    if (importedGames.length === 1) await openImportedGame();
  }

  async function openImportedGame(): Promise<void> {
    busy = true; error = '';
    try {
      const result = await api<{id: string; token?: string}>('/api/watch/import', importedGames[selectedGame]);
      open(result);
    } catch (reason) { fail(reason); }
    finally { busy = false; }
  }
</script>
<section aria-label="Watch setup">
<h2 class="type-title">Watch two bots</h2>
      <fieldset disabled={!ready || busy || disabled} class="mt-6 flex flex-wrap items-end gap-4 border-b border-line pb-6">
        <legend class="sr-only">Bot matchup</legend>
        {#each [0, 1] as seat}
          <PlayerPicker label={`Seat ${seat + 1}`} value={players[seat]} {trained} {extension} {catalogError}
            setup={{hero_deck: hero, villain_deck: villain}}
            referencePlayers={choices} onChange={value => players[seat] = value}
            mirror={mirrors[seat]} onMirrorChange={value => mirrors[seat] = value} />
        {/each}
        <DeckSelector {hero} {villain} heroLabel="Seat 1 deck" villainLabel="Seat 2 deck" onHeroChange={v => hero = v} onVillainChange={v => villain = v} />
        <label class="flex flex-col gap-2 type-label">Deal seed<input class="w-32 rounded border border-line bg-field p-2" bind:value={seed} inputmode="numeric" /></label>
        <button class="btn btn-primary" onclick={() => void start()}>Start watching</button>
        {#if extension === null}<p class="type-caption">Trained bots need manabot installed on this server.</p>{/if}
        {#if extension}
          <div class="flex flex-col gap-2 type-caption">
            <label for="game-file">Open a game file</label>
            <input id="game-file" aria-describedby="game-file-caption" type="file" accept=".json,.jsonl,.gz" onchange={e => void importGame(e)} />
            <p id="game-file-caption">An exported training game or an evaluation trace</p>
          </div>
        {/if}
        {#if importedGames.length > 1}
          <label>Recorded game <select bind:value={selectedGame}>{#each importedGames as _, index}<option value={index}>Game {index + 1}</option>{/each}</select></label>
          <button class="btn btn-secondary" onclick={() => void openImportedGame()}>Open selected game</button>
        {/if}
      </fieldset>

{#if error}<p role="alert">{error}</p>{/if}
</section>
