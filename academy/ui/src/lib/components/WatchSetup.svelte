<script lang="ts">
  import { goto } from '$app/navigation';
  import GameFileImport from './GameFileImport.svelte';
  import { onMount } from 'svelte';
  import GameSetup from './GameSetup.svelte';
  import { gameStore, type PlayerChoice } from '$lib/game.svelte';
  import { startConfiguredWatch } from '$lib/new-game';
  import { parseMirrors } from '$lib/mirror';
  let { onOpen, disabled = false, setup = null }: { onOpen: (id: string) => void; disabled?: boolean; setup?: Record<string, unknown> | null } = $props();
  let setupReady = $state(false);
  let busy = $state(false);
  let error = $state('');
  onMount(() => {
    if (setup) {
      gameStore.applySetupDraft(setup);
      const players = setup.watch_players as string[] | null;
      if (players?.length === 2) { gameStore.watchPlayerChoice = players[0] as PlayerChoice; gameStore.opponentChoice = players[1] as PlayerChoice; }
      if (setup.watch_mirrors) { const mirrors = parseMirrors(setup.watch_mirrors); gameStore.watchMirrorSearch = mirrors[0]; gameStore.mirrorSearch = mirrors[1]; }
    }
    gameStore.watchMode = true;
    const params = new URLSearchParams(location.search);
    gameStore.watchPlayerChoice = (params.get('seat1') ?? gameStore.watchPlayerChoice) as PlayerChoice;
    gameStore.setOpponentChoice((params.get('seat2') ?? gameStore.opponentChoice) as PlayerChoice);
    if (params.has('mirror')) {
      try { const search = parseMirrors(JSON.parse(params.get('mirror')!)); gameStore.watchMirrorSearch = search[0]; gameStore.mirrorSearch = search[1]; } catch (reason) { fail(reason); }
    }
    setupReady = true;
  });
  function fail(reason: unknown): void { error = reason instanceof Error ? reason.message : String(reason); }
  async function start(): Promise<void> {
    if (busy) return;
    if (!gameStore.watchMode) { await goto('/?setup=1'); return; }
    busy = true; error = '';
    try {
      onOpen(await startConfiguredWatch());
    } catch (reason) { fail(reason); }
    finally { busy = false; }
  }
</script>
<section aria-label="Game setup for watching">
  {#if setupReady}<GameSetup onStart={() => void start()} disabled={busy || disabled} humanActionLabel="Continue to play" />{/if}
  <GameFileImport {onOpen} disabled={busy || disabled} />
  {#if error}<p role="alert">{error}</p>{/if}
</section>
