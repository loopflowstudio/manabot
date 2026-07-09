<script lang="ts">
  import { onMount } from 'svelte';

  import { buildClickableTargets, filterActionsForTarget, focusIdsForActionIndexes } from '$lib/action-map';
  import ActionPanel from '$lib/components/ActionPanel.svelte';
  import GameBoard from '$lib/components/GameBoard.svelte';
  import GameLog from '$lib/components/GameLog.svelte';
  import OpponentSelector from '$lib/components/OpponentSelector.svelte';
  import StopsPanel from '$lib/components/StopsPanel.svelte';
  import { gameStore } from '$lib/game.svelte';
  import { connect, disconnect, sendAction, sendNewGame, sendPassTurn, sendSetStops } from '$lib/socket.svelte';
  import type { ActionOption, StopSide } from '$lib/types';

  let hoveredTargetId = $state<number | null>(null);

  onMount(() => {
    connect();
    return () => {
      disconnect();
    };
  });

  const clickableTargets = $derived(buildClickableTargets(gameStore.actions));
  const filteredActions = $derived(
    filterActionsForTarget(gameStore.actions, clickableTargets, gameStore.selectedTargetId),
  );
  const highlightedActionIndexes = $derived(
    new Set(hoveredTargetId === null ? [] : clickableTargets.get(hoveredTargetId) ?? []),
  );

  function targetActionIndexes(objectId: number): number[] {
    return clickableTargets.get(objectId) ?? [];
  }

  function restoreFocus(): void {
    if (gameStore.selectedTargetId !== null) {
      const actionIndexes = targetActionIndexes(gameStore.selectedTargetId);
      gameStore.setFocus(focusIdsForActionIndexes(gameStore.actions, actionIndexes));
      return;
    }
    gameStore.clearFocus();
  }

  function startNewGame(): void {
    const config = gameStore.opponentConfig();
    if (config.villain_type === 'checkpoint' && !config.villain_checkpoint) {
      gameStore.setError('Enter a checkpoint path (.pt) to play against a policy.');
      return;
    }
    sendNewGame(gameStore.newGameConfig());
  }

  const gameActive = $derived(gameStore.observation !== null && !gameStore.gameOver);
  const canPassTurn = $derived(gameActive && gameStore.actions.length > 0);

  function syncStopsToServer(): void {
    if (gameActive) {
      sendSetStops();
    }
  }

  function handleToggleStop(side: StopSide, step: string): void {
    gameStore.toggleStop(side, step);
    syncStopsToServer();
  }

  function handleStopOnStackChange(value: boolean): void {
    gameStore.setStopOnStack(value);
    syncStopsToServer();
  }

  function handleAutoPassChange(value: boolean): void {
    gameStore.setAutoPass(value);
    syncStopsToServer();
  }

  function handleResetStops(): void {
    gameStore.resetStops();
    syncStopsToServer();
  }

  function handlePassTurn(): void {
    if (!canPassTurn || gameStore.fastForwarding) {
      return;
    }
    gameStore.appendHeroAction('Pass turn (F6)');
    sendPassTurn();
  }

  function handleKeydown(event: KeyboardEvent): void {
    if (event.key === 'F6') {
      event.preventDefault();
      handlePassTurn();
    }
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
    const actionIndexes = targetActionIndexes(objectId);
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

    const actionIndexes = targetActionIndexes(objectId);
    gameStore.setFocus(focusIdsForActionIndexes(gameStore.actions, actionIndexes));
  }
</script>

<main class="mx-auto w-full max-w-[1600px] p-4" data-update-seq={gameStore.updateSeq}>
  <div class="mb-4 flex flex-wrap items-center justify-between gap-3 rounded border border-slate-700 bg-slate-800 px-4 py-3">
    <div class="flex items-center gap-3">
      <span class="text-sm font-medium uppercase tracking-wide text-slate-300">Connection</span>
      <span
        data-testid="connection-badge"
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
      <OpponentSelector
        value={gameStore.opponentChoice}
        checkpointPath={gameStore.checkpointPath}
        checkpointDeterministic={gameStore.checkpointDeterministic}
        onChange={(value) => gameStore.setOpponentChoice(value)}
        onCheckpointPathChange={(value) => gameStore.setCheckpointPath(value)}
        onCheckpointDeterministicChange={(value) => gameStore.setCheckpointDeterministic(value)}
      />
      <button
        class="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-500"
        onclick={startNewGame}
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

      <div class="flex flex-col gap-4">
        <ActionPanel
          actions={filteredActions}
          selectedTargetId={gameStore.selectedTargetId}
          {highlightedActionIndexes}
          disabled={gameStore.gameOver}
          fastForwarding={gameStore.fastForwarding}
          {canPassTurn}
          onHoverAction={handleActionHover}
          onSelectAction={handleActionSelect}
          onPassTurn={handlePassTurn}
          onClearSelection={() => {
            gameStore.clearSelectedTarget();
            hoveredTargetId = null;
            gameStore.clearFocus();
          }}
        />

        <StopsPanel
          stops={gameStore.stops}
          onToggleStop={handleToggleStop}
          onStopOnStackChange={handleStopOnStackChange}
          onAutoPassChange={handleAutoPassChange}
          onReset={handleResetStops}
        />
      </div>

      <GameLog entries={gameStore.actionLog} />
    </div>
  {:else}
    <div class="mx-auto flex max-w-xl flex-col gap-4">
      <section class="rounded border border-slate-700 bg-slate-800 p-10 text-center text-slate-300">
        <p class="mb-4 text-lg">Start a game to begin.</p>
        <button
          class="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-500"
          onclick={startNewGame}
        >
          New Game
        </button>
      </section>

      <StopsPanel
        stops={gameStore.stops}
        onToggleStop={handleToggleStop}
        onStopOnStackChange={handleStopOnStackChange}
        onAutoPassChange={handleAutoPassChange}
        onReset={handleResetStops}
      />
    </div>
  {/if}
</main>

<svelte:window onkeydown={handleKeydown} />
