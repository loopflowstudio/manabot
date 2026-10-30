<script lang="ts">
  import { onMount, tick } from 'svelte';

  import { offerFocus, offerTargets } from '$lib/offer-map';
  import DecisionSurface from '$lib/components/DecisionSurface.svelte';
  import DecisionContext from '$lib/components/DecisionContext.svelte';
  import DecisionAdvice from '$lib/components/DecisionAdvice.svelte';
  import DeckIdentity from '$lib/components/DeckIdentity.svelte';
  import DeckSelector from '$lib/components/DeckSelector.svelte';
  import GameBoard from '$lib/components/GameBoard.svelte';
  import Feedback from '$lib/components/Feedback.svelte';
  import GameLog from '$lib/components/GameLog.svelte';
  import PresentationHistory from '$lib/components/PresentationHistory.svelte';
  import OpponentSelector from '$lib/components/OpponentSelector.svelte';
  import StopsPanel from '$lib/components/StopsPanel.svelte';
  import SharedTablePanel from '$lib/components/SharedTablePanel.svelte';
  import { fetchAdviceMeta, postAdvice, type AdviceMeta, type AdviceResponse } from '$lib/advice';
  import { checkpointChoice, gameStore } from '$lib/game.svelte';
  import { sanitizeDeckSelection } from '$lib/decks';
  import { presentationPlayer } from '$lib/presentation.svelte';
  import {
    connect,
    disconnect,
    sendAction,
    sendAuthorBelief,
    sendNewGame,
    sendPassTurn,
    sendRestoreDecision,
    sendRetryDecision,
    sendReturnFromBranch,
    sendReturnToLive,
    sendSetStops,
    sendShareBelief,
    sendTransferPilot,
  } from '$lib/socket.svelte';
  import type { Decklists, InteractionOffer, OpponentCatalog, StopSide, TrainedOpponent } from '$lib/types';

  let demoName = $state<string | null>(null);
  let learnDemo = $state(false);
  let showSetup = $state(false);

  let trainedOpponents = $state<TrainedOpponent[]>([]);
  let catalogError = $state<string | null>(null);
  async function loadOpponents(): Promise<void> {
    try {
      const response = await fetch('/api/opponents');
      if (!response.ok) throw new Error('Trained opponents could not be loaded.');
      const data: OpponentCatalog = await response.json();
      trainedOpponents = data.opponents;
      catalogError = data.catalog_error;
    } catch {
      catalogError = 'Trained opponents could not be loaded.';
    }
  }

  let hoveredTargetId = $state<number | null>(null);

  onMount(() => {
    const params = new URLSearchParams(window.location.search);
    demoName = params.get('demo');
    if (!demoName) {
      const decks = sanitizeDeckSelection({ hero: params.get('hero'), villain: params.get('villain') });
      if (decks) {
        gameStore.setHeroDeck(decks.hero);
        gameStore.setVillainDeck(decks.villain);
      }
      const opponent = params.get('opponent');
      if (opponent && /^[a-f0-9]{64}$/.test(opponent)) gameStore.setOpponentChoice(checkpointChoice(opponent));
    }
    learnDemo = demoName === 'learn';
    if (demoName) gameStore.setOpponentChoice(learnDemo ? 'random' : 'passive');
    connect();
    void loadAdviceMeta();
    void loadOpponents();
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
  let adviceOpen = $state(false);
  let advicePanel: HTMLDetailsElement | null = $state(null);

  async function compareBelief(scenarioId: string): Promise<void> {
    adviceOpen = true;
    void loadAdvice(scenarioId);
    await tick();
    advicePanel?.querySelector('summary')?.focus();
    advicePanel?.scrollIntoView({ block: 'start' });
  }

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

  const tableFrame = $derived(gameStore.restoredDecision?.frame ?? gameStore.protocolFrame);
  const offers = $derived(tableFrame?.offers ?? []);
  const clickableTargets = $derived(offerTargets(offers));
  const highlightedOfferIds = $derived(new Set(hoveredTargetId === null ? [] : clickableTargets.get(hoveredTargetId) ?? []));

  function focusIdsForOffers(ids: number[]): number[] {
    return offers.filter(offer => ids.includes(offer.id)).flatMap(offerFocus);
  }

  function targetOfferIds(objectId: number): number[] {
    return clickableTargets.get(objectId) ?? [];
  }

  function restoreFocus(): void {
    if (gameStore.selectedTargetId !== null) {
      const offerIds = targetOfferIds(gameStore.selectedTargetId);
      gameStore.setFocus(focusIdsForOffers(offerIds));
      return;
    }
    gameStore.clearFocus();
  }

  function startNewGame(): void {
    if (!gameStore.hasCapability('configure_match')) {
      gameStore.setError('Only the acting pilot can start or reset the match.');
      return;
    }
    if (demoName) {
      sendNewGame({ demo: demoName });
      return;
    }
    sendNewGame(gameStore.newGameConfig());
    showSetup = false;
  }

  const gameActive = $derived(gameStore.observation !== null && !gameStore.gameOver);
  const pilotCanAct = $derived(gameStore.hasCapability('submit_live_command'));
  const participantStudying = $derived(gameStore.restoredDecision !== null);
  const canPassTurn = $derived(
    gameActive && offers.length > 0 && pilotCanAct && !participantStudying && !gameStore.commandPending && gameStore.connection === 'connected',
  );
  const boardObservation = $derived(
    gameStore.branchProjection
      ?? gameStore.restoredDecision?.frame.projection
      ?? gameStore.observation,
  );

  // The matchup names the players: "You (UR Lessons) vs Search 64 (GW
  // Allies)". Selectors hide while a game runs, so the choice is stable.
  const OPPONENT_LABELS: Record<string, string> = {
    'search-16': 'Search 16',
    'search-64': 'Search 64',
    'search-256': 'Search 256',
    random: 'Random',
    passive: 'Passive',
  };
  const opponentLabel = $derived(
    gameStore.table?.opponent_label
      ?? OPPONENT_LABELS[gameStore.opponentChoice]
      ?? 'Opponent',
  );

  function syncStopsToServer(): void {
    if (gameActive && gameStore.hasCapability('configure_match')) {
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

  function handleActionSelect(action: InteractionOffer): void {
    if (participantStudying) {
      if (!gameStore.branchAttemptId && gameStore.hasCapability('explore_study')) sendRetryDecision(action.id);
      return;
    }
    if (gameStore.gameOver || !pilotCanAct) return;

    if (!sendAction(action.id)) {
      return;
    }
    gameStore.appendHeroAction(action.label);
    gameStore.clearSelectedTarget();
    gameStore.clearFocus();
    hoveredTargetId = null;
  }

  function handleActionHover(action: InteractionOffer | null): void {
    if (action) {
      gameStore.setFocus(offerFocus(action));
      return;
    }
    restoreFocus();
  }

  function handleBoardTargetSelect(objectId: number): void {
    const offerIds = targetOfferIds(objectId);
    gameStore.selectTarget(objectId);
    gameStore.setFocus(focusIdsForOffers(offerIds));
  }

  function handleBoardTargetHover(objectId: number | null): void {
    hoveredTargetId = objectId;
    if (objectId === null) {
      restoreFocus();
      return;
    }

    const offerIds = targetOfferIds(objectId);
    gameStore.setFocus(focusIdsForOffers(offerIds));
  }
</script>

<main
  class="mx-auto w-full max-w-[1400px] px-4 py-6"
  data-update-seq={gameStore.updateSeq}
  data-frame-revision={gameStore.protocolFrame?.revision ?? -1}
  data-table-revision={gameStore.table?.table_revision ?? -1}
>
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

    <div class="flex min-w-0 max-w-full flex-wrap items-end gap-x-5 gap-y-3">
      {#if learnDemo}<p class="type-caption">Learn demo · UR Lessons vs Random GW Allies. Automated checked setup; New Game repeats it.</p>
      {:else if demoName}<p class="type-caption" data-testid="demo-origin">Precision demo · custom stress decks · automated checked setup. Choose actions from the legal choices below. New Game repeats setup.</p>{/if}
      {#if !demoName && (!gameStore.observation || gameStore.gameOver || showSetup) && gameStore.hasCapability('configure_match')}
        <OpponentSelector
          value={gameStore.opponentChoice}
          trained={trainedOpponents}
          {catalogError}
          onChange={(value) => gameStore.setOpponentChoice(value)}
        />
      {/if}
      <div class="flex items-center gap-4">
        {#if !demoName && gameStore.observation && !gameStore.gameOver && gameStore.hasCapability('configure_match')}
          <button class="btn" aria-expanded={showSetup} onclick={() => showSetup = !showSetup}>
            {showSetup ? 'Return to game' : 'Choose decks'}
          </button>
        {/if}
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
          disabled={gameStore.newGamePending || !gameStore.hasCapability('configure_match')}
        >
          {gameStore.newGamePending ? 'Preparing game…' : gameStore.hasCapability('configure_match') ? 'New Game' : 'Pilot only'}
        </button>
      </div>
    </div>
  </div>

  {#if !gameStore.observation}
    <p class="mt-3 text-sm text-ink-2">New games appear in shared <a href="/games" class="underline">Games history</a>, where you can set your player name. Completed replays are shared; feedback stays private.</p>
  {/if}
  {#if !demoName && (!gameStore.observation || gameStore.gameOver || showSetup) && gameStore.hasCapability('configure_match')}
    {#if gameStore.observation && !gameStore.gameOver}
      <p class="type-caption mt-4 text-ink-2">Choose your next matchup. New Game ends this attempt and starts the selected decks; Return to game keeps playing.</p>
    {/if}
    <DeckSelector
      hero={gameStore.decks.hero}
      villain={gameStore.decks.villain}
      disabled={gameStore.newGamePending}
      onHeroChange={(value) => gameStore.setHeroDeck(value)}
      onVillainChange={(value) => gameStore.setVillainDeck(value)}
    />
    <label class="type-caption flex items-center gap-2 text-ink-2" data-testid="decklists-choice">
      Decklists
      <select
        class="rounded border border-line bg-panel px-2 py-1"
        value={gameStore.decklists}
        disabled={gameStore.newGamePending}
        onchange={(event) => gameStore.setDecklists(event.currentTarget.value as Decklists)}
      >
        <option value="open">Open — both main decks and sideboards are known</option>
        <option value="closed">Closed — neither player sees the other's lists</option>
      </select>
    </label>
  {/if}
  {#if gameStore.observation && boardObservation}
    <div class="grid grid-cols-1 items-start">
      <div id="table-board" class="min-w-0"><a href="#table-choices" class="type-caption underline">Choose a play</a>
      <GameBoard
        observation={boardObservation}
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
      </div>

      <!-- The margin column: actions, stops, and the log as annotations
           beside the score, behind a single vertical rule. -->
      <div
        id="table-choices" class="flex min-w-0 flex-col gap-6 border-t-2 border-action pt-6"
      >
        <a href="#table-board" class="type-caption underline">Inspect board</a>
        <p data-testid="current-decision" role="status" aria-live="polite" aria-atomic="true" class="text-ink">
          {gameStore.decisionStatus}
        </p>
        {#if gameStore.branchProjection}
          <DecisionContext observation={gameStore.branchProjection} actor={gameStore.branchProjection.decision_context?.actor ?? null} />
          <p class="type-caption">Alternate action played. Return to recorded decision to try another choice.</p>
        {:else if tableFrame}
        <DecisionSurface
          frame={tableFrame}
          navigationKey={String(gameStore.updateSeq)}
          selectedTargetId={gameStore.selectedTargetId}
          {highlightedOfferIds}
          disabled={(participantStudying ? !gameStore.hasCapability('explore_study') : gameStore.gameOver || !pilotCanAct) || gameStore.commandPending || gameStore.connection !== 'connected'}
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
        {/if}

        <PresentationHistory player={presentationPlayer} />

        <details bind:this={advicePanel} bind:open={adviceOpen} class="border-t border-line pt-4"><summary class="type-rubric cursor-pointer">Advice</summary>
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
        </details>

        <div class="border-t border-line pt-4">
          <fieldset disabled={!gameStore.hasCapability('configure_match')}>
            <legend class="sr-only">Pilot stop controls</legend>
            <StopsPanel
              stops={gameStore.stops}
              onToggleStop={handleToggleStop}
              onStopOnStackChange={handleStopOnStackChange}
              onAutoPassChange={handleAutoPassChange}
              onReset={handleResetStops}
            />
          </fieldset>
        </div>

        <details class="border-t border-line pt-4"><summary class="type-rubric cursor-pointer">Decision log</summary>
          <GameLog entries={gameStore.actionLog} />
        </details>
      </div>
    </div>
  {:else}
    <div class="mx-auto max-w-xl py-12 text-center">
      <p class="type-annotation mb-1 text-ink-2">The table is set.</p>
      <p role="status" class="type-caption mb-5 text-ink-2">{gameStore.newGamePending ? 'Preparing the checked demo position. This may take a moment…' : 'Start a game to begin.'}</p>
      <div class="mx-auto mt-10 max-w-sm border-t border-line pt-4 text-left">
        <fieldset disabled={!gameStore.hasCapability('configure_match')}>
          <legend class="sr-only">Pilot stop controls</legend>
          <StopsPanel
            stops={gameStore.stops}
            onToggleStop={handleToggleStop}
            onStopOnStackChange={handleStopOnStackChange}
            onAutoPassChange={handleAutoPassChange}
            onReset={handleResetStops}
          />
        </fieldset>
      </div>
    </div>
  {/if}
    </div>
  </div>
  {#if gameStore.table}
    <details class="mt-4 border-t border-line pt-3">
      <summary class="type-rubric cursor-pointer">Table and Study tools</summary>
      <SharedTablePanel
        table={gameStore.table}
        scenarios={adviceMetaState?.scenarios ?? []}
        selectedScenarioId={adviceSelectedScenario}
        restored={gameStore.restoredDecision}
        branchAttemptId={gameStore.branchAttemptId}
        announcement={gameStore.tableAnnouncement}
        onTransferPilot={sendTransferPilot}
        onAuthorBelief={sendAuthorBelief}
        onShareBelief={sendShareBelief}
        onSelectBelief={(scenarioId) => void compareBelief(scenarioId)}
        onRestoreDecision={sendRestoreDecision}
        onReturnFromBranch={sendReturnFromBranch}
        onReturnToLive={sendReturnToLive}
      />
    </details>
  {/if}
</main>

<svelte:window onkeydown={handleKeydown} />

{#if gameStore.table?.attempt_id && gameStore.hasCapability('configure_match')}
  {#key gameStore.table.attempt_id}<Feedback attemptId={gameStore.table.attempt_id} />{/key}
{/if}

