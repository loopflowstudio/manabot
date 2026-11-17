<script lang="ts">
  import { goto } from '$app/navigation';
  import WorkStatus from './WorkStatus.svelte';
  import GameFileImport from './GameFileImport.svelte';
  import { onMount } from 'svelte';
  import GameSetup from './GameSetup.svelte';
  import { gameStore, type PlayerChoice } from '$lib/game.svelte';
  import { startConfiguredWatch } from '$lib/new-game';
  import { onDestroy } from 'svelte';
  onDestroy(() => watchStart.cancel());
  import { watchStart } from '$lib/watch-start.svelte';
  import { parseSearchConfigs } from '$lib/search';
  let { onOpen, disabled = false, setup = null }: { onOpen: (id: string) => void; disabled?: boolean; setup?: Record<string, unknown> | null } = $props();
  let setupReady = $state(false);
  let busy = $state(false);
  let error = $state('');
  onMount(() => {
    if (setup) {
      gameStore.applySetupDraft(setup);
      const players = setup.watch_players as string[] | null;
      if (players?.length === 2) { gameStore.watchPlayerChoice = players[0] as PlayerChoice; gameStore.opponentChoice = players[1] as PlayerChoice; }
      if (setup.watch_mirrors) { const mirrors = parseSearchConfigs(setup.watch_mirrors); gameStore.watchSearchConfig = mirrors[0]; gameStore.searchConfig = mirrors[1]; }
    }
    gameStore.watchMode = true;
    const params = new URLSearchParams(location.search);
    gameStore.watchPlayerChoice = (params.get('seat1') ?? gameStore.watchPlayerChoice) as PlayerChoice;
    gameStore.setOpponentChoice((params.get('seat2') ?? gameStore.opponentChoice) as PlayerChoice);
    const searchParam = params.get('search') ?? params.get('mirror');
    if (searchParam !== null) {
      try { const search = parseSearchConfigs(JSON.parse(searchParam)); gameStore.watchSearchConfig = search[0]; gameStore.searchConfig = search[1]; } catch (reason) { fail(reason); }
    }
    setupReady = true;
  });
  function fail(reason: unknown): void { error = reason instanceof Error ? reason.message : String(reason); }
  async function start(): Promise<void> {
    if (busy) return;
    if (!gameStore.watchMode) { await goto('/?setup=1'); return; }
    busy = true; error = '';
    try {
      onOpen(await startConfiguredWatch(gameStore));
    } catch (reason) { fail(reason); }
    finally { busy = false; }
  }
</script>
<section aria-label="Game setup for watching">
  {#if setupReady}<GameSetup setup={gameStore} onStart={() => void start()} disabled={busy || disabled} humanActionLabel="Continue to play" />{/if}
  <WorkStatus work={watchStart.work} connected={!watchStart.checking} />
  {#if watchStart.checking}<p role="status">Checking the same Watch setup; loading status is unavailable.</p>{/if}
  <GameFileImport {onOpen} disabled={busy || disabled} />
  {#if error}<p role="alert">{error}</p>{/if}
</section>
