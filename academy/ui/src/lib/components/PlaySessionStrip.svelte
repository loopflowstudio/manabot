<script lang="ts">
  import { gameStore } from '$lib/game.svelte';
  import { sendContinuation } from '$lib/socket.svelte';
  let { onChangeOpponent, onEditNext }: { onChangeOpponent: () => void; onEditNext: () => void } = $props();
  const session = $derived(gameStore.playSession);
  const heroPool = $derived(typeof session?.draft.hero_deck_pool === 'string' ? session.draft.hero_deck_pool : null);
  const villainPool = $derived(typeof session?.draft.villain_deck_pool === 'string' ? session.draft.villain_deck_pool : null);
  const hasDeckPool = $derived(heroPool !== null || villainPool !== null);
  function continueGame(intent: 'rematch' | 'swap_sides'): void {
    if (!gameStore.gameOver && !window.confirm('Stop the current game and start the next game in this session?')) return;
    sendContinuation(intent);
  }
  function dealNewDecks(): void {
    if (!session || !hasDeckPool) return;
    if (!gameStore.gameOver && !window.confirm('Stop the current game and deal new decks from the pool?')) return;
    sendContinuation('rematch', {
      hero_deck: heroPool ?? session.draft.hero_deck,
      villain_deck: villainPool ?? session.draft.villain_deck,
      seed: null,
    });
  }
</script>

{#if session}
  <section aria-label="Play session" class="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2 border-y border-line py-1 text-sm">
    <div class="flex flex-wrap items-center gap-x-4 gap-y-2">
      <span>Game {session.game_number} · {session.config.human_seat === 1 ? 'Draw first' : 'Play first'} · {session.draft.seed == null ? 'Fresh deal' : `Repeat seed ${session.draft.seed}`}</span>
      <a class="underline" href={`/games?session=${encodeURIComponent(session.id)}`}>Session history</a>
      <!-- An interrupted session no longer exists on the server; only New game starts again. -->
      {#if gameStore.hasCapability('configure_match') && !gameStore.resumeFailed}
        {#if hasDeckPool}<button data-testid="deal-new-decks" class="btn btn-primary btn-sm" disabled={gameStore.newGamePending} onclick={dealNewDecks}>Deal new decks</button>{/if}
        <button class={hasDeckPool ? 'btn btn-secondary btn-sm' : 'btn btn-primary btn-sm'} disabled={gameStore.newGamePending} onclick={() => continueGame('rematch')}>Rematch</button>
        <button class="btn btn-secondary btn-sm" disabled={gameStore.newGamePending} onclick={() => continueGame('swap_sides')}>Swap play/draw</button>
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
