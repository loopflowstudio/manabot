<script lang="ts">
  import { onMount } from 'svelte';

  import { buildClickableTargets, filterActionsForTarget, focusIdsForActionIndexes } from '$lib/action-map';
  import ActionPanel from '$lib/components/ActionPanel.svelte';
  import GameBoard from '$lib/components/GameBoard.svelte';
  import GameLog from '$lib/components/GameLog.svelte';
  import OpponentSelector from '$lib/components/OpponentSelector.svelte';
  import { gameStore } from '$lib/game.svelte';
  import { connect, disconnect, sendAction, sendNewGame } from '$lib/socket.svelte';
  import type { ActionOption } from '$lib/types';

  let hoveredTargetId: number | null = null;

  onMount(() => {
    connect();
    return () => {
      disconnect();
    };
  });

  $: clickableTargets = buildClickableTargets(gameStore.actions);
  $: filteredActions = filterActionsForTarget(
    gameStore.actions,
    clickableTargets,
    gameStore.selectedTargetId,
  );
  $: highlightedActionIndexes = new Set(
    hoveredTargetId === null ? [] : clickableTargets.get(hoveredTargetId)?.actionIndexes ?? [],
  );

  function restoreFocus(): void {
    if (gameStore.selectedTargetId !== null) {
      const actionIndexes = clickableTargets.get(gameStore.selectedTargetId)?.actionIndexes ?? [];
      gameStore.setFocus(focusIdsForActionIndexes(gameStore.actions, actionIndexes));
      return;
    }
    gameStore.clearFocus();
  }

  function startNewGame(): void {
    sendNewGame({ villain_type: gameStore.villainType });
  }

  function handleActionSelect(action: ActionOption): void {
    if (gameStore.gameOver) {
      return;
    }

    gameStore.appendHeroAction(action.description);
    gameStore.clearSelectedTarget();
    gameStore.clearFocus();
    hoveredTargetId = null;
    sendAction(action.index);
  }

  function handleActionHover(action: ActionOption | null): void {
    if (action) {
      gameStore.setFocus(action.focus);
      return;
    }
    restoreFocus();
  }

  function handleBoardTargetSelect(objectId: number): void {
    const actionIndexes = clickableTargets.get(objectId)?.actionIndexes ?? [];
    const matchingActions = gameStore.actions.filter((action) => actionIndexes.includes(action.index));

    if (matchingActions.length === 1) {
      handleActionSelect(matchingActions[0]);
      return;
    }

    gameStore.selectTarget(objectId);
    gameStore.setFocus(focusIdsForActionIndexes(gameStore.actions, actionIndexes));
  }

  function handleBoardTargetHover(objectId: number | null): void {
    hoveredTargetId = objectId;
    if (objectId === null) {
      restoreFocus();
      return;
    }

    const actionIndexes = clickableTargets.get(objectId)?.actionIndexes ?? [];
    gameStore.setFocus(focusIdsForActionIndexes(gameStore.actions, actionIndexes));
  }
</script>

<main class="mx-auto w-full max-w-[1600px] p-4">
  <div class="mb-4 flex flex-wrap items-center justify-between gap-3 rounded border border-slate-700 bg-slate-800 px-4 py-3">
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

    <div class="flex flex-wrap items-center gap-3">
      <OpponentSelector value={gameStore.villainType} onChange={(value) => gameStore.setVillainType(value)} />
      <button
        class="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-500"
        on:click={startNewGame}
      >
        New Game
      </button>
    </div>
  </div>

  {#if gameStore.errorMessage}
    <section class="mb-4 rounded border border-rose-500/50 bg-rose-900/20 px-4 py-3 text-sm text-rose-200">
      {gameStore.errorMessage}
    </section>
  {/if}

  {#if gameStore.observation}
    <div class="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_320px_320px]">
      <GameBoard
        observation={gameStore.observation}
        focusedIds={gameStore.focusIds}
        {clickableTargets}
        onSelectTarget={handleBoardTargetSelect}
        onHoverTarget={handleBoardTargetHover}
        winner={gameStore.winner}
        overlayActionLabel="Play Again"
        onOverlayAction={startNewGame}
      />

      <ActionPanel
        actions={filteredActions}
        selectedTargetId={gameStore.selectedTargetId}
        {highlightedActionIndexes}
        disabled={gameStore.gameOver}
        onHoverAction={handleActionHover}
        onSelectAction={handleActionSelect}
        onClearSelection={() => {
          gameStore.clearSelectedTarget();
          hoveredTargetId = null;
          gameStore.clearFocus();
        }}
      />

      <GameLog entries={gameStore.actionLog} />
    </div>
  {:else}
    <section class="rounded border border-slate-700 bg-slate-800 p-10 text-center text-slate-300">
      <p class="mb-4 text-lg">Start a game to begin.</p>
      <button
        class="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-500"
        on:click={startNewGame}
      >
        New Game
      </button>
    </section>
  {/if}
</main>
