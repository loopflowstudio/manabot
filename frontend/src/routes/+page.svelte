<script lang="ts">
  import { onMount } from 'svelte';

  import { buildClickableTargets, filterActionsForTarget, focusIdsForActionIndexes } from '$lib/action-map';
  import ActionPanel from '$lib/components/ActionPanel.svelte';
  import DecisionAdvice from '$lib/components/DecisionAdvice.svelte';
  import DeckIdentity from '$lib/components/DeckIdentity.svelte';
  import DeckSelector from '$lib/components/DeckSelector.svelte';
  import GameBoard from '$lib/components/GameBoard.svelte';
  import GameLog from '$lib/components/GameLog.svelte';
  import OpponentSelector from '$lib/components/OpponentSelector.svelte';
  import StopsPanel from '$lib/components/StopsPanel.svelte';
  import { fetchAdviceMeta, postAdvice, type AdviceMeta, type AdviceResponse } from '$lib/advice';
  import { gameStore } from '$lib/game.svelte';
  import { presentationPlayer } from '$lib/presentation.svelte';
  import { connect, disconnect, sendAction, sendNewGame, sendPassTurn, sendSetStops } from '$lib/socket.svelte';
  import type { ActionOption, StopSide } from '$lib/types';

  let hoveredTargetId = $state<number | null>(null);

  onMount(() => {
    connect();
    void loadAdviceMeta();
    return () => {
      disconnect();
    };
  });

  // The live advice surface loads the fixture's pinned completed-match
  // decision as the demonstration beside the ActionPanel. The live frame
  // during play has no erd1 address yet (the canonical replay finalizes at
  // game close), so live-address advice is a future GAM-4 integration point
  // via this same POST /api/advice seam.
  let adviceMetaState = $state<AdviceMeta | null>(null);
  let adviceResponse = $state<AdviceResponse | null>(null);
  let adviceSelectedScenario = $state('');
  let adviceStatus = $state<'ok' | 'unavailable' | 'loading'>('loading');

  async function loadAdviceMeta(): Promise<void> {
    try {
      const meta = await fetchAdviceMeta();
      adviceMetaState = meta;
      adviceSelectedScenario = meta.scenarios[0]?.landmark_id ?? '';
      await loadAdvice(adviceSelectedScenario);
    } catch {
      adviceStatus = 'unavailable';
    }
  }

  async function loadAdvice(scenarioId: string): Promise<void> {
    if (!adviceMetaState || !scenarioId) return;
    adviceSelectedScenario = scenarioId;
    adviceStatus = 'loading';
    try {
      const response = await postAdvice({
        address: adviceMetaState.address,
        scenario_id: scenarioId,
        identity: adviceMetaState.identity,
      });
      adviceResponse = response;
      adviceStatus = response.status;
    } catch {
      adviceStatus = 'unavailable';
    }
  }

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

  // The matchup names the players: "You (UR Lessons) vs Search 64 (GW
  // Allies)". Selectors hide while a game runs, so the choice is stable.
  const OPPONENT_LABELS: Record<string, string> = {
    'search-16': 'Search 16',
    'search-64': 'Search 64',
    'search-256': 'Search 256',
    checkpoint: 'Checkpoint',
    random: 'Random',
    passive: 'Passive',
  };
  const opponentLabel = $derived(OPPONENT_LABELS[gameStore.opponentChoice] ?? 'Opponent');

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
    if (!sendPassTurn()) {
      return;
    }
    // The log records game vocabulary, not affordances: no key hints.
    gameStore.appendHeroAction('Pass turn');
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

    if (!sendAction(action.index)) {
      return;
    }
    gameStore.appendHeroAction(action.description);
    gameStore.clearSelectedTarget();
    gameStore.clearFocus();
    hoveredTargetId = null;
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

<main class="mx-auto w-full max-w-[1400px] px-4 py-6" data-update-seq={gameStore.updateSeq}>
  <h1 class="sr-only">Play</h1>

  {#if gameStore.errorMessage}
    <section role="alert" aria-atomic="true" class="mb-4 rounded border border-mountain/50 bg-mountain/20 px-4 py-3 text-mountain-ink">
      {gameStore.errorMessage}
    </section>
  {/if}

  <!-- The sheet: one continuous leaf. Regions are ruled, never boxed. -->
  <div class="overflow-hidden rounded-sm border border-line bg-panel shadow-sheet">
    <div class="px-10 pb-10 pt-6 max-md:px-5 max-md:pb-6">
  <!-- The masthead, in levels: the matchup names the players, fields show
       only while they matter, one red action, and the connection is a
       whisper beside it. -->
  <div
    data-testid="game-header"
    class="flex flex-wrap items-end justify-between gap-x-8 gap-y-4 border-b border-line pb-4"
  >
    <div class="min-w-0 self-center">
      {#if gameStore.deckNames && gameStore.observation}
        <!-- The players live at their bars; this line is the summary for
             assistive tech and the proof suite. -->
        <span data-testid="deck-names" class="sr-only">
          You ({gameStore.deckNames.hero}) vs {opponentLabel} ({gameStore.deckNames.villain})
        </span>
      {:else}
        <span class="type-annotation text-ink-2">No game in progress</span>
      {/if}
    </div>

    <div class="flex flex-wrap items-end gap-x-5 gap-y-3">
      {#if !gameStore.observation || gameStore.gameOver}
        <DeckSelector
          hero={gameStore.decks.hero}
          villain={gameStore.decks.villain}
          onHeroChange={(value) => gameStore.setHeroDeck(value)}
          onVillainChange={(value) => gameStore.setVillainDeck(value)}
        />
        <OpponentSelector
          value={gameStore.opponentChoice}
          checkpointPath={gameStore.checkpointPath}
          checkpointDeterministic={gameStore.checkpointDeterministic}
          onChange={(value) => gameStore.setOpponentChoice(value)}
          onCheckpointPathChange={(value) => gameStore.setCheckpointPath(value)}
          onCheckpointDeterministicChange={(value) => gameStore.setCheckpointDeterministic(value)}
        />
      {/if}
      <div class="flex items-center gap-4">
        <div data-testid="connection-summary" class="flex items-center gap-1.5">
          <i
            aria-hidden="true"
            class={`inline-block h-1.5 w-1.5 rounded-full ${
              gameStore.connection === 'connected'
                ? 'bg-forest'
                : gameStore.connection === 'reconnecting' || gameStore.connection === 'connecting'
                  ? 'bg-plains'
                  : 'bg-swamp'
            }`}
          ></i>
          <span
            data-testid="connection-badge"
            data-connection-state={gameStore.connection}
            role="status"
            aria-live="polite"
            aria-atomic="true"
            aria-label={`Connection status: ${gameStore.connection}`}
            class="type-label uppercase text-ink-3"
          >
            {gameStore.connection}
          </span>
        </div>
        <button
          class="btn btn-primary"
          onclick={startNewGame}
        >
          New Game
        </button>
      </div>
    </div>
  </div>

  {#if gameStore.observation}
    <div class="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_300px]">
      <GameBoard
        observation={gameStore.observation}
        focusedIds={gameStore.focusIds}
        {clickableTargets}
        onSelectTarget={handleBoardTargetSelect}
        onHoverTarget={handleBoardTargetHover}
        winner={gameStore.winner}
        deckNames={gameStore.deckNames}
        heroLabel="You"
        villainLabel={opponentLabel}
        overlayActionLabel="Play Again"
        onOverlayAction={startNewGame}
        {presentationPlayer}
      />

      <!-- The margin column: actions, stops, and the log as annotations
           beside the score, behind a single vertical rule. -->
      <div
        class="flex min-w-0 flex-col gap-6 pt-4 max-xl:mt-2 max-xl:border-t max-xl:border-line xl:ml-8 xl:border-l xl:border-line xl:pl-8"
      >
        <ActionPanel
          actions={filteredActions}
          actionSpaceKind={gameStore.actionSpaceKind}
          selectedTargetId={gameStore.selectedTargetId}
          {highlightedActionIndexes}
          disabled={gameStore.gameOver}
          fastForwarding={gameStore.fastForwarding}
          {canPassTurn}
          focusKey={`${gameStore.updateSeq}:${gameStore.selectedTargetId ?? 'all'}`}
          onHoverAction={handleActionHover}
          onSelectAction={handleActionSelect}
          onPassTurn={handlePassTurn}
          onClearSelection={() => {
            gameStore.clearSelectedTarget();
            hoveredTargetId = null;
            gameStore.clearFocus();
          }}
        />

        <div class="border-t border-line pt-4">
          <DecisionAdvice
            mode="live"
            scenarios={adviceMetaState?.scenarios ?? []}
            selectedScenarioId={adviceSelectedScenario}
            frame={adviceResponse?.frame ?? null}
            offers={adviceResponse?.offers ?? []}
            evidence={adviceResponse?.evidence ?? null}
            deltas={adviceResponse?.deltas ?? null}
            status={adviceStatus}
            reason={adviceResponse?.reason ?? null}
            advisorId={adviceMetaState?.identity.advisor_id ?? ''}
            computeId={adviceMetaState?.identity.compute_id ?? ''}
            onSelectScenario={(id) => void loadAdvice(id)}
          />
        </div>

        <div class="border-t border-line pt-4">
          <StopsPanel
            stops={gameStore.stops}
            onToggleStop={handleToggleStop}
            onStopOnStackChange={handleStopOnStackChange}
            onAutoPassChange={handleAutoPassChange}
            onReset={handleResetStops}
          />
        </div>

        <div class="border-t border-line pt-4">
          <GameLog entries={gameStore.actionLog} />
        </div>
      </div>
    </div>
  {:else}
    <div class="mx-auto max-w-xl py-12 text-center">
      <p class="type-annotation mb-1 text-ink-2">The table is set.</p>
      <p class="type-caption mb-5 text-ink-2">Start a game to begin.</p>
      <button
        class="btn btn-primary"
        onclick={startNewGame}
      >
        New Game
      </button>
      <div class="mx-auto mt-10 max-w-sm border-t border-line pt-4 text-left">
        <StopsPanel
          stops={gameStore.stops}
          onToggleStop={handleToggleStop}
          onStopOnStackChange={handleStopOnStackChange}
          onAutoPassChange={handleAutoPassChange}
          onReset={handleResetStops}
        />
      </div>
    </div>
  {/if}
    </div>
  </div>
</main>

<svelte:window onkeydown={handleKeydown} />
