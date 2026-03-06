<script lang="ts">
  import { onMount } from 'svelte';

  import { gameStore } from '$lib/game.svelte';
  import { connect, disconnect, sendAction, sendNewGame } from '$lib/socket.svelte';
  import type { CardState, PermanentState, PlayerState } from '$lib/types';

  const CARD_TYPE_LABELS: Array<[keyof CardState['types'], string]> = [
    ['is_land', 'Land'],
    ['is_creature', 'Creature'],
    ['is_spell', 'Spell'],
    ['is_artifact', 'Artifact'],
    ['is_enchantment', 'Enchantment'],
    ['is_planeswalker', 'Planeswalker'],
    ['is_battle', 'Battle'],
  ];

  onMount(() => {
    connect();
    return () => {
      disconnect();
    };
  });

  function formatTypes(card: CardState): string {
    const labels = CARD_TYPE_LABELS
      .filter(([key]) => card.types[key])
      .map(([, label]) => label);
    return labels.join(' · ') || 'Card';
  }

  function hasFocus(id: number): boolean {
    return gameStore.focusIds.has(id);
  }

  function stackCards(): CardState[] {
    if (!gameStore.observation) {
      return [];
    }
    return [...gameStore.observation.opponent.stack, ...gameStore.observation.agent.stack];
  }

  function permanentClass(permanent: PermanentState): string {
    const classes = [
      'w-36 rounded border bg-slate-800 p-2 text-xs transition-all duration-150',
      hasFocus(permanent.id) ? 'border-blue-400 shadow-[0_0_0_1px_rgb(96,165,250)]' : 'border-slate-700',
      permanent.tapped ? 'rotate-90' : '',
      permanent.summoning_sick ? 'opacity-70' : '',
    ];
    return classes.join(' ');
  }

  function playerClass(player: PlayerState): string {
    return [
      'rounded border p-3',
      hasFocus(player.id) ? 'border-blue-400' : 'border-slate-700',
    ].join(' ');
  }
</script>

<main class="min-h-screen bg-slate-900 text-slate-100">
  <div class="mx-auto flex w-full max-w-[1400px] flex-col gap-4 p-4">
    <section class="flex items-center justify-between rounded border border-slate-700 bg-slate-800 px-4 py-3">
      <div class="flex items-center gap-3">
        <span class="text-sm font-medium uppercase tracking-wide text-slate-300">Connection</span>
        <span
          class={`rounded px-2 py-1 text-xs font-semibold ${
            gameStore.connection === 'connected'
              ? 'bg-emerald-600/30 text-emerald-300'
              : gameStore.connection === 'reconnecting' || gameStore.connection === 'connecting'
                ? 'bg-amber-600/30 text-amber-300'
                : 'bg-slate-700 text-slate-300'
          }`}
        >
          {gameStore.connection}
        </span>
      </div>
      <button
        class="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-500"
        on:click={sendNewGame}
      >
        New Game
      </button>
    </section>

    {#if gameStore.errorMessage}
      <section class="rounded border border-rose-500/50 bg-rose-900/20 px-4 py-3 text-sm text-rose-200">
        {gameStore.errorMessage}
      </section>
    {/if}

    {#if gameStore.observation}
      <div class="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
        <section class="relative space-y-4 rounded border border-slate-700 bg-slate-800 p-4">
          <div class="rounded border border-slate-700 bg-slate-800 px-3 py-2 text-center text-sm font-semibold text-slate-200">
            Turn {gameStore.observation.turn.turn_number} · {gameStore.observation.turn.phase} · {gameStore.observation.turn.step}
          </div>

          <article class={playerClass(gameStore.observation.opponent)}>
            <div class="mb-2 flex items-center justify-between">
              <h2 class="text-sm uppercase tracking-wide text-slate-400">Opponent</h2>
              <div
                class={`rounded bg-slate-900 px-3 py-1 text-2xl font-bold ${
                  hasFocus(gameStore.observation.opponent.id) ? 'ring-1 ring-blue-400' : ''
                }`}
              >
                {gameStore.observation.opponent.life}
              </div>
            </div>
            <div class="mb-3 text-sm text-slate-300">Library: {gameStore.observation.opponent.library_count}</div>

            <div class="mb-3">
              <h3 class="mb-1 text-xs uppercase tracking-wide text-slate-400">Hand ({gameStore.observation.opponent.hand.length})</h3>
              <div class="flex flex-wrap gap-2">
                {#each gameStore.observation.opponent.hand as card}
                  <div
                    class={`h-14 w-10 rounded border bg-slate-900 ${hasFocus(card.id) ? 'border-blue-400' : 'border-slate-600'}`}
                    title="Hidden card"
                  ></div>
                {/each}
              </div>
            </div>

            <div>
              <h3 class="mb-1 text-xs uppercase tracking-wide text-slate-400">Graveyard ({gameStore.observation.opponent.graveyard.length})</h3>
              <div class="flex flex-wrap gap-2">
                {#each gameStore.observation.opponent.graveyard as card}
                  <div class={`rounded border bg-slate-800 px-2 py-1 text-xs ${hasFocus(card.id) ? 'border-blue-400' : 'border-slate-700'}`}>
                    {card.name}
                  </div>
                {/each}
              </div>
            </div>
          </article>

          <section class="rounded border border-slate-700 bg-slate-800 p-3">
            <h3 class="mb-2 text-xs uppercase tracking-wide text-slate-400">Battlefield</h3>
            <div class="space-y-4">
              <div class="min-h-16">
                <div class="mb-1 text-xs text-slate-500">Opponent</div>
                <div class="flex flex-wrap gap-3">
                  {#each gameStore.observation.opponent.battlefield as permanent}
                    <div class={permanentClass(permanent)}>
                      <div class="font-semibold">{permanent.name ?? 'Unknown Permanent'}</div>
                      {#if permanent.power !== null && permanent.toughness !== null}
                        <div class="text-slate-300">{permanent.power}/{permanent.toughness}</div>
                      {/if}
                      <div class="text-[11px] text-slate-400">{permanent.tapped ? 'Tapped' : 'Untapped'}{permanent.damage ? ` · ${permanent.damage} dmg` : ''}</div>
                    </div>
                  {/each}
                </div>
              </div>

              <div class="min-h-16">
                <div class="mb-1 text-xs text-slate-500">Hero</div>
                <div class="flex flex-wrap gap-3">
                  {#each gameStore.observation.agent.battlefield as permanent}
                    <div class={permanentClass(permanent)}>
                      <div class="font-semibold">{permanent.name ?? 'Unknown Permanent'}</div>
                      {#if permanent.power !== null && permanent.toughness !== null}
                        <div class="text-slate-300">{permanent.power}/{permanent.toughness}</div>
                      {/if}
                      <div class="text-[11px] text-slate-400">{permanent.tapped ? 'Tapped' : 'Untapped'}{permanent.damage ? ` · ${permanent.damage} dmg` : ''}</div>
                    </div>
                  {/each}
                </div>
              </div>
            </div>
          </section>

          <article class={playerClass(gameStore.observation.agent)}>
            <div class="mb-2 flex items-center justify-between">
              <h2 class="text-sm uppercase tracking-wide text-slate-400">Hero</h2>
              <div
                class={`rounded bg-slate-900 px-3 py-1 text-2xl font-bold ${
                  hasFocus(gameStore.observation.agent.id) ? 'ring-1 ring-blue-400' : ''
                }`}
              >
                {gameStore.observation.agent.life}
              </div>
            </div>
            <div class="mb-3 text-sm text-slate-300">Library: {gameStore.observation.agent.library_count}</div>

            <div class="mb-3">
              <h3 class="mb-1 text-xs uppercase tracking-wide text-slate-400">Hand ({gameStore.observation.agent.hand.length})</h3>
              <div class="flex flex-wrap gap-2">
                {#each gameStore.observation.agent.hand as card}
                  <div
                    class={`w-36 rounded border bg-slate-800 p-2 text-xs ${hasFocus(card.id) ? 'border-blue-400' : 'border-slate-700'}`}
                  >
                    <div class="font-semibold">{card.name}</div>
                    <div class="text-[11px] text-slate-400">{formatTypes(card)}</div>
                    {#if card.types.is_creature}
                      <div class="text-slate-300">{card.power}/{card.toughness}</div>
                    {/if}
                  </div>
                {/each}
              </div>
            </div>

            <div>
              <h3 class="mb-1 text-xs uppercase tracking-wide text-slate-400">Graveyard ({gameStore.observation.agent.graveyard.length})</h3>
              <div class="flex flex-wrap gap-2">
                {#each gameStore.observation.agent.graveyard as card}
                  <div class={`rounded border bg-slate-800 px-2 py-1 text-xs ${hasFocus(card.id) ? 'border-blue-400' : 'border-slate-700'}`}>
                    {card.name}
                  </div>
                {/each}
              </div>
            </div>
          </article>

          {#if stackCards().length > 0}
            <section class="rounded border border-indigo-500/40 bg-indigo-900/20 p-3">
              <h3 class="mb-2 text-xs uppercase tracking-wide text-indigo-200">Stack</h3>
              <div class="flex flex-wrap gap-2">
                {#each stackCards() as card}
                  <div class={`rounded border px-2 py-1 text-xs ${hasFocus(card.id) ? 'border-blue-400 bg-slate-800' : 'border-indigo-400/50 bg-slate-900/80'}`}>
                    {card.name}
                  </div>
                {/each}
              </div>
            </section>
          {/if}

          {#if gameStore.gameOver}
            <div class="absolute inset-0 grid place-items-center rounded bg-slate-950/80">
              <div class="rounded border border-slate-600 bg-slate-900 p-6 text-center shadow-xl">
                <h2 class="mb-2 text-2xl font-bold">Game Over</h2>
                <p class="mb-4 text-slate-300">
                  {#if gameStore.winner === null}
                    Draw
                  {:else if gameStore.winner === 0}
                    Hero wins
                  {:else}
                    Opponent wins
                  {/if}
                </p>
                <button
                  class="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-500"
                  on:click={sendNewGame}
                >
                  Play Again
                </button>
              </div>
            </div>
          {/if}
        </section>

        <aside class="rounded border border-slate-700 bg-slate-800 p-4">
          <h2 class="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-300">Actions</h2>
          <div class="space-y-2">
            {#if gameStore.actions.length === 0}
              <p class="text-sm text-slate-400">No actions available.</p>
            {:else}
              {#each gameStore.actions as action}
                <button
                  class="w-full rounded border border-slate-600 bg-slate-900 px-3 py-2 text-left text-sm transition hover:border-blue-400 hover:bg-slate-800"
                  on:mouseenter={() => gameStore.setFocus(action.focus)}
                  on:mouseleave={() => gameStore.clearFocus()}
                  on:click={() => sendAction(action.index)}
                  disabled={gameStore.gameOver}
                >
                  <div class="font-medium">{action.description}</div>
                  <div class="mt-1 text-xs text-slate-400">{action.type}</div>
                </button>
              {/each}
            {/if}
          </div>
        </aside>
      </div>
    {:else}
      <section class="rounded border border-slate-700 bg-slate-800 p-10 text-center text-slate-300">
        <p class="mb-4 text-lg">Start a game to begin.</p>
        <button
          class="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-500"
          on:click={sendNewGame}
        >
          New Game
        </button>
      </section>
    {/if}
  </div>
</main>
