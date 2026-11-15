<script lang="ts">
  import { gameStore } from '$lib/game.svelte';
  import { sendContinuation } from '$lib/socket.svelte';
  let { onChangeOpponent, onEditNext }: { onChangeOpponent: () => void; onEditNext: () => void } = $props();
  const session = $derived(gameStore.playSession);
  function continueGame(intent: 'rematch' | 'swap_sides'): void {
    if (!gameStore.gameOver && !window.confirm('Stop the current game and start the next game in this session?')) return;
    sendContinuation(intent);
  }
</script>

{#if session}
  <section aria-label="Play session" class="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2 border-y border-line py-1 text-sm">
    <div class="flex flex-wrap items-center gap-x-4 gap-y-2">
      <span>Game {session.game_number} · {session.config.human_seat === 1 ? 'Draw first' : 'Play first'} · {session.draft.seed == null ? 'Fresh deal' : `Repeat seed ${session.draft.seed}`}</span>
      <a class="underline" href={`/games?session=${encodeURIComponent(session.id)}`}>Session history</a>
      {#if gameStore.hasCapability('configure_match')}
        <button class="btn btn-primary btn-sm" disabled={gameStore.newGamePending} onclick={() => continueGame('rematch')}>Rematch</button>
        <button class="btn btn-secondary btn-sm" disabled={gameStore.newGamePending} onclick={() => continueGame('swap_sides')}>Swap sides</button>
        <button class="btn btn-secondary btn-sm" disabled={gameStore.newGamePending} onclick={onEditNext}>Edit next game</button>
        <button class="btn btn-secondary btn-sm" disabled={gameStore.newGamePending} onclick={onChangeOpponent}>Change opponent</button>
      {/if}
    </div>
    <details class="min-w-0 max-w-full open:basis-full">
      <summary class="cursor-pointer">Exact game configuration</summary>
      <p class="mt-2">Seed {String(session.config.seed)} · Native human seat {String(session.config.human_seat)} · {String(session.config.decklists)} decklists. Wall elapsed includes pauses; response measurements unavailable.</p>
      <pre class="max-h-56 overflow-auto whitespace-pre-wrap break-all text-xs">{JSON.stringify(session.config, null, 2)}</pre>
    </details>
  </section>
{/if}
