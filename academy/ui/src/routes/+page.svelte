<script lang="ts">
  import { currentGame } from '$lib/current-game.svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import GameSetup from '$lib/components/GameSetup.svelte';
  import { startConfiguredWatch } from '$lib/new-game';
  import { onDestroy } from 'svelte';
  onDestroy(() => watchStart.cancel());
  import { watchStart } from '$lib/watch-start.svelte';
  import { revealWithinScrollRegions } from '$lib/scroll-regions';
  import { onMount, tick, untrack, setContext } from 'svelte';

  import {
    dragMoved, dragOffer, loadTableMode, saveTableMode, selectForPlay, selectedOffers,
    selectionMatches, type HandDrag, type PlaySelection, type TableMode,
  } from '$lib/play-selection';
  import { visibleCardLabel } from '$lib/decision-labels';
  import { offerFocus, offerTargets } from '$lib/offer-map';
  import { PaymentPreviewController, PAYMENT_PREVIEW_CONTEXT, type PreviewState } from '$lib/payment-preview';
  import PaymentPreview from '$lib/components/PaymentPreview.svelte';
  import { fetchPaymentPreview } from '$lib/socket.svelte';
  import DecisionSurface from '$lib/components/DecisionSurface.svelte';
  import DecisionContext from '$lib/components/DecisionContext.svelte';
  import PlaySessionStrip from '$lib/components/PlaySessionStrip.svelte';
  import { recordFetch } from '$lib/records';
  import type { PlaySession } from '$lib/play-session';
  import DeckIdentity from '$lib/components/DeckIdentity.svelte';
  import { checkpointIdentity, playerIdentity } from '$lib/player-identity';
  import GameBoard from '$lib/components/GameBoard.svelte';
  import Feedback from '$lib/components/Feedback.svelte';
  import GameLog from '$lib/components/GameLog.svelte';
  import PresentationHistory from '$lib/components/PresentationHistory.svelte';
  import WorkStatus from '$lib/components/WorkStatus.svelte';
  import StopsPanel from '$lib/components/StopsPanel.svelte';
  import SharedTablePanel from '$lib/components/SharedTablePanel.svelte';
  import { policyChoice, gameStore, type GameStore } from '$lib/game.svelte';
  import { sanitizeDeckSelection } from '$lib/decks';
  import { presentationPlayer, studyPresentationPlayer } from '$lib/presentation.svelte';
  import {
    connect,
    disconnect,
    sendAction,
    sendNewGame,
    sendContinuation,
    sendPassTurn,
    sendRetrySearch,
    sendRestoreDecision,
    sendRetryDecision,
    sendReturnFromBranch,
    sendReturnToLive,
    sendSetStops,
    sendTransferPilot,
  } from '$lib/socket.svelte';
  import type { InteractionOffer, StopSide } from '$lib/types';

  let paymentState = $state<PreviewState | null>(null);
  // The decision whose payment slot is open. Only a proposal that concerns a
  // payment opens it; it then stays until the decision changes, so the offer
  // list above keeps its layout while the pointer moves.
  let paymentSlotDecision = $state<string | null>(null);
  const payments = new PaymentPreviewController(fetchPaymentPreview, state => {
    paymentState = state;
    if (state && !state.loading && state.preview?.state !== 'not_applicable') paymentSlotDecision = decisionKey;
  });
  setContext(PAYMENT_PREVIEW_CONTEXT, () => paymentState?.preview?.state === 'exact'
    ? paymentState.preview.sources.flatMap((source, index) => source.source.kind === 'object' ? [{ id: source.source.id.entity, order: index + 1 }] : []) : []);

  let demoName = $state<string | null>(null);
  let learnDemo = $state(false);
  type Editor = {kind:'closed'} | {kind:'new' | 'next'; setup:GameStore; opponentOnly:boolean; pendingAttempt?:string | null};
  let editor = $state<Editor>({kind:'closed'});
  let mounted = $state(false);
  let awaitingRecovery = $state(false);
  let setupStatus = $state('');

  let tableMode = $state<TableMode>('inspect');
  let selection = $state<PlaySelection | null>(null);
  let drag = $state<HandDrag | null>(null);
  let suppressClick = false;
  let stopsPending = $state<number | null>(null);
  let stopsOpen = $state(false);
  let stopsPanel: HTMLDetailsElement | null = $state(null);
  let hoveredTargetId = $state<number | null>(null);

  onMount(() => {
    tableMode = loadTableMode();
    const params = new URLSearchParams(window.location.search);
    demoName = params.get('demo');
    if (!demoName) {
      const decks = sanitizeDeckSelection({ hero: params.get('hero'), villain: params.get('villain') });
      if (decks) {
        gameStore.setHeroDeck(decks.hero);
        gameStore.setVillainDeck(decks.villain);
      }
      const opponent = params.get('opponent');
      if (opponent && /^[a-f0-9]{64}$/.test(opponent)) gameStore.setOpponentChoice(policyChoice(opponent));
    }
    learnDemo = demoName === 'learn';
    if (demoName) gameStore.setOpponentChoice(learnDemo ? 'random' : 'passive');
    try { awaitingRecovery = !gameStore.resumeToken && sessionStorage.getItem('etude.gui.resume') !== null; } catch { /* In-memory setup remains usable. */ }
    mounted = true;
    connect(params.get('launch'));

    return () => {
      payments.invalidate();
      disconnect();
    };
  });

  let secondaryOpen = $state(false);

  const tableFrame = $derived(gameStore.restoredDecision?.frame ?? gameStore.protocolFrame);
  const decisionKey = $derived(tableFrame ? `${tableFrame.match_id}:${tableFrame.revision}:${tableFrame.prompt?.id}` : null);
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

  function openEditor(kind: 'new' | 'next', opponentOnly = false): void {
    const draft = kind === 'next' ? gameStore.playSession?.draft : undefined;
    editor = {kind, opponentOnly, setup:gameStore.editableSetup(draft)};
    setupStatus = '';
    gameStore.setError(null);
  }

  function cancelEditor(): void {
    const destination = editor.kind === 'new' ? currentGame.href ?? '/' : '/';
    editor = {kind:'closed'};
    setupStatus = '';
    gameStore.setError(null);
    if (page.url.searchParams.has('setup')) void goto(destination);
  }

  function startNewGame(): void {
    if (gameStore.newGamePending) return;
    if (demoName) { sendNewGame({demo:demoName}); return; }
    if (editor.kind === 'closed') { openEditor('new'); return; }
    const setup = editor.setup;
    if (setup.seedError) { gameStore.setError(setup.seedError); return; }
    if (setup.watchMode) {
      gameStore.newGamePending = true;
      gameStore.setError(null);
      void startConfiguredWatch(setup).then(id => goto(`/replay?trace=${encodeURIComponent(id)}`))
        .catch(error => gameStore.setError(error.message)).finally(() => gameStore.newGamePending = false);
      return;
    }
    if (!gameStore.hasCapability('configure_match')) { gameStore.setError('Only the acting pilot can start or reset the match.'); return; }
    if (gameActive && !window.confirm('Stop the current game and start a new session?')) return;
    editor.pendingAttempt = gameStore.table?.attempt_id ?? null;
    sendNewGame(setup.newGameConfig());
  }

  function editorConfig(): Record<string, unknown> {
    if (editor.kind === 'closed') return {};
    const setup = editor.setup;
    return editor.opponentOnly ? {...setup.opponentConfig(), mirror:setup.searchConfig} : setup.setupConfig();
  }

  async function saveNextSetup(): Promise<void> {
    const session = gameStore.playSession;
    const table = gameStore.table;
    if (editor.kind !== 'next' || editor.setup.watchMode || !session || !table || !gameStore.resumeToken) return;
    if (editor.setup.seedError) { gameStore.setError(editor.setup.seedError); return; }
    gameStore.setError(null);
    try {
      const response = await recordFetch(`/api/play-sessions/${encodeURIComponent(session.id)}`, {
        method: 'PUT', headers: {'content-type':'application/json', 'x-etude-participant-token':gameStore.resumeToken},
        body: JSON.stringify({table_id:table.table_id, grant_revision:table.access.grant_revision, revision:session.revision, config:editorConfig()}),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Setup could not be saved.');
      gameStore.applyPlaySession(payload as PlaySession);
      setupStatus = 'Next-game setup saved. The current game is unchanged; admission is checked when starting.';
      gameStore.tableAnnouncement = setupStatus;
    } catch (error) { gameStore.setError(error instanceof Error ? error.message : 'Setup could not be saved.'); }
  }

  function continueWithSetup(): void {
    if (editor.kind !== 'next' || editor.setup.watchMode || gameStore.newGamePending) return;
    if (editor.setup.seedError) { gameStore.setError(editor.setup.seedError); return; }
    if (gameActive && !window.confirm('Stop the current game and start the next game in this session?')) return;
    editor.pendingAttempt = gameStore.table?.attempt_id ?? null;
    sendContinuation(editor.opponentOnly ? 'change_opponent' : 'rematch', editorConfig());
  }

  $effect(() => {
    if (awaitingRecovery && (gameStore.resumeToken || gameStore.resumeFailed || gameStore.tableUnavailable)) awaitingRecovery = false;
    if (!mounted || awaitingRecovery || demoName || page.url.searchParams.has('launch')) return;
    const wantsNew = page.url.searchParams.has('setup');
    untrack(() => {
      if (wantsNew) openEditor('new');
      else if (!gameStore.observation && editor.kind === 'closed') openEditor('new');
      else if (editor.kind === 'new') editor = {kind:'closed'};
    });
  });

  // A refusal leaves intent and edits intact. Only a new authoritative attempt closes it.
  $effect(() => {
    if (editor.kind !== 'closed' && editor.pendingAttempt !== undefined && !gameStore.newGamePending) {
      const attempt = gameStore.table?.attempt_id;
      if (attempt && attempt !== editor.pendingAttempt) untrack(cancelEditor);
      else editor.pendingAttempt = undefined;
    }
  });

  const gameActive = $derived(gameStore.observation !== null && !gameStore.gameOver && !gameStore.resumeFailed);
  const pilotCanAct = $derived(gameStore.hasCapability('submit_live_command'));
  const participantStudying = $derived(gameStore.restoredDecision !== null);
  const tablePresentation = $derived(participantStudying ? studyPresentationPlayer : presentationPlayer);
  const inputReady = $derived(!gameStore.commandPending && gameStore.work?.state !== 'running' && stopsPending === null && gameStore.connection === 'connected');
  // Both table modes submit through the same live authority gate. Play only
  // adds selection and gestures; Study retains its separate retry capability.
  const canSubmitLive = $derived(gameActive && pilotCanAct && !participantStudying
    && !gameStore.branchProjection && !gameStore.newGamePending && inputReady);
  const paymentSlot = $derived(canSubmitLive && decisionKey !== null && paymentSlotDecision === decisionKey);
  const canPassTurn = $derived(canSubmitLive && tableFrame?.action_space === 'PRIORITY' && offers.length > 0);
  const boardObservation = $derived(
    gameStore.branchProjection
      ?? gameStore.restoredDecision?.frame.projection
      ?? gameStore.observation,
  );

  const liveInput = $derived(tableMode === 'play' && canSubmitLive);
  const draggingOffer = $derived(drag && tableFrame ? dragOffer(drag.selection, tableFrame) : null);

  $effect(() => {
    gameStore.updateSeq;
    untrack(() => { payments.invalidate(); clearPlaySelection(); });
  });
  $effect(() => {
    if (!liveInput || (selection && !selectionMatches(selection, tableFrame))) untrack(clearPlaySelection);
  });
  $effect(() => {
    if (stopsPending !== null && (gameStore.updateSeq !== stopsPending || gameStore.connection !== 'connected')) stopsPending = null;
  });

  function clearPlaySelection(): void {
    payments.clear();
    if (drag?.moved) suppressClick = true;
    selection = null;
    drag = null;
    gameStore.clearSelectedTarget();
    hoveredTargetId = null;
    gameStore.clearFocus();
  }
  function cancelSelection(): void {
    const object = selection?.object;
    const restore = document.activeElement?.closest('.play-selection');
    clearPlaySelection();
    if (restore && object !== undefined) void tick().then(() => {
      document.querySelector<HTMLElement>(`[data-object-id="${object}"]`)?.focus({ preventScroll: true });
    });
  }
  function changeMode(mode: TableMode): void {
    clearPlaySelection();
    tableMode = mode;
    saveTableMode(mode);
  }
  function suppressDragClick(event: MouseEvent): void {
    if (suppressClick && event.detail > 0) {
      event.preventDefault();
      event.stopPropagation();
      suppressClick = false;
    }
  }
  function startHandDrag(event: PointerEvent): void {
    suppressClick = false;
    if (!liveInput || event.button !== 0 || !event.isPrimary || !(event.target instanceof Element) || !tableFrame) return;
    const card = event.target.closest<HTMLElement>('[data-object-id]');
    if (!card) return;
    const bound = selectForPlay(tableFrame, Number(card.dataset.objectId));
    if (!bound || !dragOffer(bound, tableFrame)) return;
    drag = { selection: bound, pointer: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
    card.setPointerCapture(event.pointerId);
  }
  function moveHandDrag(event: PointerEvent): void {
    if (!drag || drag.pointer !== event.pointerId) return;
    drag.moved = dragMoved(drag, event.clientX, event.clientY);
    if (drag.moved) { event.preventDefault(); selection = drag.selection; }
  }
  function endHandDrag(event: PointerEvent): void {
    const pending = drag;
    if (!pending || pending.pointer !== event.pointerId) return;
    const drop = document.querySelector('[data-testid="play-drop-area"]')?.getBoundingClientRect();
    const offer = liveInput && tableFrame ? dragOffer(pending.selection, tableFrame) : null;
    if (pending.moved) {
      suppressClick = true;
      if (drop && event.clientX >= drop.left && event.clientX <= drop.right && event.clientY >= drop.top && event.clientY <= drop.bottom && offer) handleActionSelect(offer);
      else clearPlaySelection();
    }
    drag = null;
  }
  async function editStops(): Promise<void> {
    secondaryOpen = true;
    stopsOpen = true;
    await tick();
    stopsPanel?.querySelector('summary')?.focus({ preventScroll: true });
    if (stopsPanel) revealWithinScrollRegions(stopsPanel);
  }

  // The matchup names the players: "You (UR Lessons) vs Search 64 (GW
  // Allies)". Selectors hide while a game runs, so the choice is stable.
  const OPPONENT_LABELS: Record<string, string> = {
    'search-16': 'Search 16',
    'search-64': 'Search 64',
    'search-256': 'Search 256',
    random: 'Random',
    passive: 'Passive',
  };
  const tableIdentity = $derived(gameStore.table ?? gameStore.interruptedIdentity);
  const opponentIdentity = $derived(tableIdentity?.opponent
    ? checkpointIdentity(tableIdentity.opponent)
    : playerIdentity({ name: tableIdentity?.opponent_label ?? OPPONENT_LABELS[gameStore.opponentChoice] ?? 'Opponent' }));

  function syncStopsToServer(): void {
    if (!gameStore.observation && editor.kind !== 'closed') {
      editor.setup.applyServerStops({ ...gameStore.stops, my: [...gameStore.stops.my], opponent: [...gameStore.stops.opponent] });
    }
    if (gameStore.observation && gameStore.hasCapability('configure_match')) {
      clearPlaySelection();
      stopsPending = gameStore.updateSeq;
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
    if (event.key === 'Escape') cancelSelection();
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
    if (!canSubmitLive) return;
    if (tableMode === 'play' && selection
      && !selectedOffers(selection, gameStore.protocolFrame).some(offer => offer.id === action.id)) return;

    payments.invalidate();
    if (!sendAction(action.id)) {
      return;
    }
    gameStore.appendHeroAction(action.label);
    clearPlaySelection();
  }

  $effect(() => {
    if (!canSubmitLive || gameStore.connection !== 'connected') untrack(() => payments.invalidate());
  });

  function dismissPaymentOnFocus(event: FocusEvent): void {
    if (event.target instanceof Element && !event.target.closest('[data-offer-id], [data-testid="payment-preview"]')) {
      payments.clear();
    }
  }

  function previewOffer(action: InteractionOffer): void {
    if (canSubmitLive && tableFrame && ['PRIORITY_CAST_SPELL', 'PRIORITY_ACTIVATE_ABILITY', 'CHOOSE_TARGET', 'PAY_COST', 'TAP_FOR_COST', 'CHOOSE_MODE'].includes(action.action_type ?? '')) {
      void payments.select(tableFrame, action.id, action.label);
    } else {
      payments.clear();
    }
  }

  function handleActionHover(action: InteractionOffer | null): void {
    if (action) {
      previewOffer(action);
      gameStore.setFocus(offerFocus(action));
      return;
    }
    restoreFocus();
  }

  function handleBoardTargetSelect(objectId: number): void {
    if (tableMode === 'play' && !participantStudying && pilotCanAct) {
      if (!liveInput || !tableFrame) return;
      selection = selectForPlay(tableFrame, objectId);
    }
    const offerIds = targetOfferIds(objectId);
    if (canSubmitLive && tableFrame) {
      const candidates = tableFrame.offers.filter(offer => offerIds.includes(offer.id));
      if (candidates.length === 1) previewOffer(candidates[0]);
      else payments.clear();
    }
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
  class:table-active={gameStore.observation !== null && editor.kind === 'closed'}
  class="play-page mx-auto w-full max-w-[1600px] px-4 py-4"
  data-update-seq={gameStore.updateSeq}
  data-frame-revision={gameStore.protocolFrame?.revision ?? -1}
  data-table-revision={gameStore.table?.table_revision ?? -1}
>
  <h1 class="sr-only">{!gameStore.observation || editor.kind !== 'closed' ? 'New game' : 'Current game'}</h1>

  {#if gameStore.errorMessage}
    <section role="alert" aria-atomic="true" class="mb-4 rounded border border-mountain/50 bg-mountain/20 px-4 py-3 text-mountain-ink">
      {#if gameStore.resumeFailed}<strong>Interrupted.</strong>{' '}{/if}{gameStore.errorMessage}
      {#if gameStore.resumeFailed && gameStore.recordingId}
        <a class="ml-2 underline" href={`/replay?trace=${encodeURIComponent(gameStore.recordingId)}`}>Open saved recording</a>
      {/if}
    </section>
  {/if}
  {#if gameStore.searchRetry}
    <section role="alert" class="mb-4 rounded border border-line bg-panel px-4 py-3">
      <p>{gameStore.submissionAccepted ? 'Your choice was accepted. ' : ''}{gameStore.searchRetry.message}</p>
      <button class="mt-2 rounded border border-line px-3 py-2" disabled={gameStore.commandPending || !pilotCanAct || gameStore.connection !== 'connected'} onclick={sendRetrySearch}>{(gameStore.searchRetry.code.startsWith('search_') || gameStore.searchRetry.code.startsWith('mirror_')) ? 'Retry search' : 'Retry bot decision'}</button>
    </section>
  {/if}
  <WorkStatus work={watchStart.work} connected={!watchStart.checking} />
  {#if watchStart.checking}<p role="status">Checking the same Watch setup; loading status is unavailable.</p>{/if}
  {#if gameStore.observation && gameStore.activeSearchConfig}
    <p class="type-caption mb-2 text-ink-2">Opponent: Search {gameStore.activeSearchConfig.worlds}×{gameStore.activeSearchConfig.depth} · Uniform belief (physical deal)</p>
  {/if}

  <!-- The sheet: one continuous leaf. Regions are ruled, never boxed. -->
  <div class="play-sheet overflow-hidden rounded-sm border border-line bg-panel shadow-sheet">
    <div class="sheet-content px-5 pb-4 pt-3">
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
          You ({gameStore.deckNames.hero}) vs {opponentIdentity.text} ({gameStore.deckNames.villain})
        </span>
      {:else}
        <span class="type-annotation text-ink-2">No game in progress</span>
      {/if}
      <!-- Bot work shows in the masthead's free side, so the board never shifts while a bot thinks. -->
      <WorkStatus work={gameStore.work} connected={gameStore.connection === 'connected'} inline />
    </div>

    <div class="flex min-w-0 max-w-full flex-wrap items-end gap-x-5 gap-y-3">
      {#if learnDemo}<p class="type-caption">Learn demo · UR Lessons vs Random GW Allies. Automated checked setup; Start game repeats it.</p>
      {:else if demoName}<p class="type-caption" data-testid="demo-origin">Precision demo · custom stress decks · automated checked setup. Choose actions from the legal choices below. Start game repeats setup.</p>{/if}
      {#if demoName}<button class="btn btn-primary" disabled={gameStore.newGamePending} onclick={startNewGame}>{gameStore.newGamePending ? 'Preparing game…' : 'Start game'}</button>{/if}
      <div class="flex items-center gap-4">
        <div role="group" aria-label="Table interaction mode" class="flex gap-1">
          <button class="btn btn-secondary btn-sm" aria-pressed={tableMode === 'play'} onclick={() => changeMode('play')}>Play</button>
          <button class="btn btn-secondary btn-sm" aria-pressed={tableMode === 'inspect'} onclick={() => changeMode('inspect')}>Inspect</button>
        </div>
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
            aria-label={gameStore.resumeFailed ? 'Session status: interrupted' : `Connection status: ${gameStore.connection}`}
            class="type-label uppercase text-ink-3"
          >
            {gameStore.resumeFailed ? 'Interrupted' : gameStore.connection}
          </span>
        </div>

      </div>
    </div>
  </div>

  {#if gameStore.tableUnavailable}<p role="status" class="mt-2 text-sm">The previous table is no longer live. Its recording is retained; start the next game explicitly.</p>{/if}
  {#if gameStore.storageWarning}<p role="status" class="mt-2 text-sm">{gameStore.storageWarning}</p>{/if}
  {#if !demoName}<PlaySessionStrip onEditNext={() => openEditor('next')} onChangeOpponent={() => openEditor('next', true)} />{/if}
  {#if !gameStore.observation}
    <p class="mt-3 text-sm text-ink-2">New games appear in shared <a href="/games" class="underline">Games history</a>, where you can set your player name. Completed replays are shared; feedback stays private.</p>
  {/if}
  {#if !demoName && editor.kind !== 'closed' && gameStore.hasCapability('configure_match')}
    <section aria-label={editor.kind === 'new' ? 'New session setup' : 'Next game setup'} class="mt-4 border-b border-line pb-4">
      <h2 class="type-title">{editor.kind === 'new' ? 'New session' : editor.opponentOnly ? 'Change opponent' : 'Edit next game'}</h2>
      <p class="type-caption mt-2">{editor.kind === 'new'
        ? gameStore.playSession
          ? 'Start a separate session. These edits do not change your current session draft.'
          : 'Choose your setup to start a new session.'
        : 'Setup for the next game in this Play session. Earlier games keep their recorded configuration.'}</p>
      {#key editor.setup}
        <GameSetup setup={editor.setup} onStart={startNewGame} disabled={gameStore.newGamePending}
          opponentOnly={editor.opponentOnly} showStart={editor.kind === 'new'} />
      {/key}
      {#if setupStatus}<p role="status" class="type-caption mt-2">{setupStatus}</p>{/if}
      {#if editor.kind === 'next'}
        {#if editor.setup.watchMode}
          <p class="type-caption">Two bots start a separate watched game. Your Play session and its current game stay available in Games.</p>
          <button class="btn btn-primary mt-3" disabled={gameStore.newGamePending} onclick={startNewGame}>Start new watched game</button>
        {:else}
          <button class="btn btn-secondary mt-3" disabled={gameStore.newGamePending} onclick={saveNextSetup}>Save next setup</button>
          <button class="btn btn-primary mt-3" disabled={gameStore.newGamePending} onclick={continueWithSetup}>Start next game</button>
        {/if}
      {/if}
      <button class="btn btn-secondary mt-3" disabled={gameStore.newGamePending} onclick={cancelEditor}>Cancel</button>
    </section>
  {/if}
  {#if gameStore.observation && boardObservation}
    <div class="play-workspace" data-testid="play-workspace">
      <div id="table-board" class="board-region" tabindex="-1"><a href="#table-choices" class="table-jump type-caption underline">Choose a play</a>
      <GameBoard
        workspace
        playMode={tableMode === 'play' && !participantStudying}
        canSelectForPlay={liveInput}
        selectedId={selection?.object ?? null}
        onHandClickCapture={suppressDragClick}
        onHandPointerDown={startHandDrag}
        onHandPointerMove={moveHandDrag}
        onHandPointerUp={endHandDrag}
        onHandPointerCancel={clearPlaySelection}
        gestureActive={drag?.moved ?? false}
        dragLabel={draggingOffer?.confirm_label ?? draggingOffer?.label ?? null}
        observation={boardObservation}
        focusedIds={gameStore.focusIds}
        {clickableTargets}
        onSelectTarget={handleBoardTargetSelect}
        onHoverTarget={handleBoardTargetHover}
        winner={gameStore.winner}
        deckNames={gameStore.deckNames}
        heroIdentity={playerIdentity({ name: 'You', kind: 'human' })}
        villainIdentity={opponentIdentity}
        overlayActionLabel={demoName ? 'Play Again' : 'Rematch'}
        onOverlayAction={demoName ? startNewGame : () => sendContinuation('rematch')}
        presentationPlayer={tablePresentation}
      />
      </div>

      <!-- The margin column: actions, stops, and the log as annotations
           beside the score, behind a single vertical rule. -->
      <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard access to the scrolling choices rail) -->
      <div
        id="table-choices" tabindex="0" role="region" aria-label="Current choices and game controls" class="choices-region"
      >
        <a href="#table-board" class="table-jump type-caption underline">Inspect board</a>
        <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard access to long decision status) -->
        <p tabindex="0" data-testid="current-decision" role="status" aria-live="polite" aria-atomic="true" class="text-ink">
          {gameStore.decisionStatus}
        </p>
        {#if tableMode === 'play' && selection && !participantStudying}
          <div class="play-selection border-b border-line pb-2" role="region" aria-label="Selected card actions">
            <p class="type-caption">Selected: {visibleCardLabel(boardObservation, selection.object) ?? 'player'}. Choose an action below.</p>
            <button class="btn btn-ghost btn-sm" onclick={cancelSelection}>Cancel selection</button>
          </div>
        {/if}
        {#if gameStore.branchProjection}
          <DecisionContext observation={gameStore.branchProjection} actor={gameStore.branchProjection.decision_context?.actor ?? null} />
          <p class="type-caption">Alternate action played. Return to recorded decision to try another choice.</p>
        {:else if tableFrame}
        <DecisionSurface
          workspace
          interrupted={gameStore.resumeFailed}
          frame={tableFrame}
          navigationKey={String(gameStore.updateSeq)}
          selectedTargetId={gameStore.selectedTargetId}
          {highlightedOfferIds}
          disabled={participantStudying ? !gameStore.hasCapability('explore_study') || !inputReady : !canSubmitLive}
          fastForwarding={gameStore.fastForwarding}
          {canPassTurn}
          onHoverAction={handleActionHover}
          onSelectAction={handleActionSelect}
          onPassTurn={handlePassTurn}
          onClearSelection={clearPlaySelection}
        />
        {#if paymentSlot}
          <PaymentPreview
            state={paymentState}
            observation={tableFrame.projection}
            chosenSourceFirst={tableFrame.offers.some(offer => offer.id === paymentState?.offer && offer.action_type === 'TAP_FOR_COST')}
            onClose={() => payments.clear()}
            onRetry={() => { if (tableFrame && paymentState) void payments.select(tableFrame, paymentState.offer, paymentState.label); }}
          />
        {/if}
        {/if}

        <p data-testid="automation-status" class="type-caption text-ink-2">Priority-stop automation: {gameStore.stops.auto_pass ? `on · your turn ${gameStore.stops.my.length} / opponent ${gameStore.stops.opponent.length} · stack ${gameStore.stops.stop_on_stack ? 'on' : 'off'}` : 'off · every priority choice'}{gameStore.fastForwarding ? ' · yielding this turn' : ''}{stopsPending !== null ? ' · updating…' : ''} <button class="underline" onclick={editStops}>Edit stops</button></p>
        <details bind:open={secondaryOpen} class="secondary-panels">
          <summary class="type-rubric cursor-pointer">History and settings</summary>
          <div class="secondary-content space-y-4" tabindex="0" role="region" aria-label="History and settings">
            <PresentationHistory player={tablePresentation} observation={boardObservation} />
            {#if tableFrame && !gameStore.branchProjection}
              <DecisionContext observation={tableFrame.projection} actor={tableFrame.prompt?.actor ?? null} interrupted={gameStore.resumeFailed} />
            {/if}

            <div class="border-t border-line pt-4">
              <fieldset disabled={!gameStore.hasCapability('configure_match') || participantStudying || !inputReady}>
                <legend class="sr-only">Pilot stop controls</legend>
                <StopsPanel
                  bind:open={stopsOpen}
                  bind:element={stopsPanel}
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
        restored={gameStore.restoredDecision}
        branchAttemptId={gameStore.branchAttemptId}
        announcement={gameStore.tableAnnouncement}
        onTransferPilot={sendTransferPilot}
        onRestoreDecision={sendRestoreDecision}
        onReturnFromBranch={sendReturnFromBranch}
        onReturnToLive={sendReturnToLive}
      />
    </details>
  {/if}
</main>

<svelte:window onkeydown={handleKeydown} onfocusin={dismissPaymentOnFocus} />

{#if gameStore.table?.attempt_id && gameStore.hasCapability('configure_match')}
  {#key gameStore.table.attempt_id}<Feedback attemptId={gameStore.table.attempt_id} />{/key}
{/if}


<style>
  [aria-label="Table interaction mode"] button[aria-pressed="true"] { border-color: var(--accent); background: var(--accent-soft); }
  .play-selection { max-height: 120px; overflow-y: auto; flex-shrink: 0; }
  .play-workspace { display: grid; gap: 1rem; min-width: 0; }
  .board-region { min-width: 0; }
  .choices-region { display: flex; flex-direction: column; gap: 1rem; min-width: 0; border-top: 2px solid var(--accent); padding-top: 1rem; }
  @media (min-width: 1024px) and (min-height: 650px) {
    .table-active { height: calc(100dvh - 64px); display: flex; flex-direction: column; }
    .table-active > :global(*) { flex-shrink: 0; }
    .table-active .play-sheet { flex: 1; min-height: 0; }
    .table-active .sheet-content { height: 100%; display: flex; flex-direction: column; }
    .table-active .sheet-content > :global(*) { flex-shrink: 0; }
    .table-active .play-workspace { flex: 1; min-height: 0; }
    .play-workspace { grid-template-columns: minmax(0, 1fr) clamp(19rem, 26vw, 23rem); height: calc(100dvh - 190px); min-height: 0; padding-top: .5rem; }
    .board-region { min-height: 0; display: flex; flex-direction: column; }
    .board-region :global([data-testid="game-board"]) { flex: 1; min-height: 0; }
    .choices-region { min-height: 0; overflow-y: auto; gap: .5rem; border-top: 0; border-left: 1px solid var(--border); padding: 0 .5rem 1rem 1rem; scrollbar-gutter: stable; }
    .choices-region > :global(*) { flex-shrink: 0; }
    .choices-region > :global(.workspace-decision) { flex: 1; min-height: min-content; }
    .secondary-panels { flex-shrink: 1; min-height: 2rem; overflow-y: auto; overscroll-behavior: contain; }
    [data-testid="current-decision"] { max-height: 3rem; overflow-y: auto; overscroll-behavior: contain; }
    .secondary-content { max-height: 32dvh; overflow-y: auto; overscroll-behavior: contain; }
    .table-jump { position: absolute; width: 1px; height: 1px; overflow: hidden; }
    .table-jump:focus { position: static; width: auto; height: auto; }
  }
</style>
