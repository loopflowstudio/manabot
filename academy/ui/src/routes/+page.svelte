<script lang="ts">
  import { currentGame } from '$lib/current-game.svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import GameSetup from '$lib/components/GameSetup.svelte';
  import { startConfiguredWatch } from '$lib/new-game';
  import { onDestroy } from 'svelte';
  onDestroy(() => watchStart.cancel());
  import { watchStart } from '$lib/watch-start.svelte';
  import { onMount, untrack } from 'svelte';

  import Table from '$lib/table/Table.svelte';
  import { PaymentPreviewController, type PreviewState } from '$lib/payment-preview';
  import PlaySessionStrip from '$lib/components/PlaySessionStrip.svelte';
  import { recordFetch } from '$lib/records';
  import type { PlaySession } from '$lib/play-session';
  import { checkpointIdentity, playerIdentity } from '$lib/player-identity';
  import Feedback from '$lib/components/Feedback.svelte';
  import WorkStatus from '$lib/components/WorkStatus.svelte';
  import StopsPanel from '$lib/components/StopsPanel.svelte';
  import SharedTablePanel from '$lib/components/SharedTablePanel.svelte';
  import { policyChoice, gameStore, type GameStore } from '$lib/game.svelte';
  import { presentationPlayer } from '$lib/presentation.svelte';
  import { sanitizeDeckSelection, DECK_CHOICES } from '$lib/decks';
  import { builtinOpponentName, shortName } from '$lib/opponent-name';
  import { loadLastMatchup, matchupSeat, saveLastMatchup, type LastMatchup } from '$lib/last-matchup';
  import { gameResult } from '$lib/table/result';
  import Strip from '$lib/table/Strip.svelte';
  import { BEAT_PACES, loadBeatPace, saveBeatPace, type BeatPace } from '$lib/table/beats';
  import { logMarks } from '$lib/table/replay';
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
    fetchPaymentPreview,
  } from '$lib/socket.svelte';
  import type { ExperienceFrame, InteractionOffer, StopSide } from '$lib/types';

  let demoName = $state<string | null>(null);
  let learnDemo = $state(false);
  type Editor = {kind:'closed'} | {kind:'new' | 'next'; setup:GameStore; opponentOnly:boolean; pendingAttempt?:string | null};
  let editor = $state<Editor>({kind:'closed'});
  let mounted = $state(false);
  let awaitingRecovery = $state(false);
  let setupStatus = $state('');

  let stopsPending = $state<number | null>(null);

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
      if (opponent && /^[a-f0-9]{64}$/.test(opponent)) gameStore.setOpponentChoice(policyChoice(opponent));
    }
    learnDemo = demoName === 'learn';
    if (demoName) gameStore.setOpponentChoice(learnDemo ? 'random' : 'passive');
    try { awaitingRecovery = !gameStore.resumeToken && sessionStorage.getItem('etude.gui.resume') !== null; } catch { /* In-memory setup remains usable. */ }
    mounted = true;
    connect(params.get('launch'));

    return () => {
      disconnect();
    };
  });

  const tableFrame = $derived(gameStore.restoredDecision?.frame ?? gameStore.protocolFrame);

  // The automatic payer's plan for the offer the table asks about: a
  // read-only query bound to this exact decision. The controller drops an
  // answer for any other frame, and the table asks for nothing while it
  // cannot act, so a commit or a lost connection clears the plan.
  let paymentState = $state<PreviewState | null>(null);
  const payments = new PaymentPreviewController(fetchPaymentPreview, state => { paymentState = state; });
  function previewPayment(offer: InteractionOffer | null): void {
    if (offer && tableFrame && canSubmitLive) void payments.select(tableFrame, offer.id, offer.label);
    else payments.clear();
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
    // A setup address, or one naming decks or an opponent, opens setup; `/` alone is home.
    const wantsNew = ['setup', 'hero', 'villain', 'opponent'].some(key => page.url.searchParams.has(key));
    untrack(() => {
      if (wantsNew) openEditor('new');
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
  const inputReady = $derived(!gameStore.commandPending && gameStore.work?.state !== 'running' && stopsPending === null && gameStore.connection === 'connected');
  // Study retains its separate retry capability; live play is one authority gate.
  const canSubmitLive = $derived(gameActive && pilotCanAct && !participantStudying
    && !gameStore.branchProjection && !gameStore.newGamePending && inputReady);
  const canPassTurn = $derived(canSubmitLive && tableFrame?.action_space === 'PRIORITY' && (tableFrame?.offers.length ?? 0) > 0);
  const canStudyRetry = $derived(participantStudying && !gameStore.branchAttemptId && gameStore.hasCapability('explore_study') && inputReady);
  // A Study branch shows its own projection with no live offers.
  const boardFrame = $derived<ExperienceFrame | null>(tableFrame && gameStore.branchProjection
    ? { ...tableFrame, projection: gameStore.branchProjection, offers: [], prompt: null } : tableFrame);

  $effect(() => {
    if (stopsPending !== null && (gameStore.updateSeq !== stopsPending || gameStore.connection !== 'connected')) stopsPending = null;
  });

  // The opponent is named with how it decides ("Heuristic · scripted",
  // "Search 64 · search"): the server's label once a table exists, the
  // built-in name before. The identity keeps the digest for its description.
  const tableIdentity = $derived(gameStore.table ?? gameStore.interruptedIdentity);
  const opponentIdentity = $derived(tableIdentity?.opponent
    ? checkpointIdentity(tableIdentity.opponent)
    : playerIdentity({ name: tableIdentity?.opponent_label ?? builtinOpponentName(gameStore.opponentChoice) ?? 'Opponent' }));
  const opponentName = $derived(tableIdentity?.opponent_label ?? opponentIdentity.compact);

  // ———————————————— Home: play the last matchup again

  let lastMatchup = $state<LastMatchup | null>(null);
  onMount(() => { lastMatchup = loadLastMatchup(); });
  // Each game this tab plays becomes the next "Play again", as the session echoed its setup.
  $effect(() => {
    const config = gameStore.playSession?.config;
    if (!config || demoName || !gameStore.observation || !tableIdentity) return;
    const decks = gameStore.deckNames ? { hero: gameStore.deckNames.hero, villain: gameStore.deckNames.villain } : null;
    untrack(() => saveLastMatchup({ config, opponent: opponentName, decks }));
  });
  const deckLabel = (key: unknown): string => DECK_CHOICES.find(deck => deck.key === key)?.label ?? 'Imported deck';
  const homeMatchup = $derived(lastMatchup ?? {
    config: gameStore.setupConfig(),
    opponent: builtinOpponentName(gameStore.opponentChoice) ?? 'Registered policy',
    decks: { hero: deckLabel(gameStore.decks.hero), villain: deckLabel(gameStore.decks.villain) },
  });
  const atHome = $derived(mounted && !awaitingRecovery && !demoName && !page.url.searchParams.has('launch')
    && !gameStore.observation && editor.kind === 'closed');

  function playAgain(): void {
    if (gameStore.newGamePending) return;
    gameStore.watchMode = false;
    gameStore.applySetupDraft(homeMatchup.config);
    gameStore.setError(null);
    sendNewGame(gameStore.newGameConfig());
  }

  function syncStopsToServer(): void {
    if (!gameStore.observation && editor.kind !== 'closed') {
      editor.setup.applyServerStops({ ...gameStore.stops, my: [...gameStore.stops.my], opponent: [...gameStore.stops.opponent] });
    }
    if (gameStore.observation && gameStore.hasCapability('configure_match')) {
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

  function handleActionSelect(action: InteractionOffer): void {
    if (participantStudying) {
      if (canStudyRetry) sendRetryDecision(action.id);
      return;
    }
    if (!canSubmitLive) return;
    // The move list names it when the server narrates the move.
    sendAction(action.id);
  }

  const onTable = $derived(gameStore.observation !== null && editor.kind === 'closed' && boardFrame !== null);

  // Who is deciding while you wait: the opponent's search or policy, or the table itself.
  const heroSeat = $derived(gameStore.observation?.agent.player_index ?? null);
  const opponentChoosing = $derived(gameActive && heroSeat !== null && (
    (gameStore.work?.state === 'running' && (gameStore.work.kind === 'choose' || gameStore.work.kind === 'search')
      && gameStore.work.chooser !== null && gameStore.work.chooser !== heroSeat)
    || (tableFrame?.prompt != null && tableFrame.prompt.actor !== heroSeat)));
  const thinking = $derived(!gameActive ? null
    : opponentChoosing ? `${shortName(opponentName)} is ${gameStore.work?.kind === 'search' ? 'searching' : 'choosing'}…`
      : gameStore.commandPending || gameStore.work?.state === 'running' ? 'The table is working…' : null);

  const ending = $derived(gameStore.gameOver && boardFrame
    ? gameResult(boardFrame, gameStore.actionLog, { you: 'You', opponent: shortName(opponentName) }, gameStore.winner) : null);
  // The game so far as a strip of decisions, folded unless opened. Live play
  // shows no past position yet: positions open in Review once the game ends.
  let stripOpen = $state(false);
  const liveMarks = $derived(logMarks(gameStore.actionLog));
  const canContinue = $derived(!demoName && gameStore.playSession !== null && !!gameStore.table?.attempt_id && !gameStore.newGamePending && !gameStore.resumeFailed);

  // How the opponent's moves are read out, remembered per browser.
  let beatPace = $state<BeatPace>(loadBeatPace());
  const BEAT_PACE_LABELS: Record<BeatPace, string> = { off: 'Off', quick: 'Quick', slow: 'Slow' };
</script>

{#snippet notices()}
  {#if gameStore.errorMessage}
    <section role="alert" aria-atomic="true" class="notice error">
      {#if gameStore.resumeFailed}<strong>Interrupted.</strong>{' '}{/if}{gameStore.errorMessage}
      {#if gameStore.resumeFailed && gameStore.recordingId}
        <a class="underline" href={`/replay?trace=${encodeURIComponent(gameStore.recordingId)}`}>Open saved recording</a>
      {/if}
    </section>
  {/if}
  {#if gameStore.searchRetry}
    <section role="alert" class="notice">
      <p>{gameStore.submissionAccepted ? 'Your choice was accepted. ' : ''}{gameStore.searchRetry.message}</p>
      <button class="btn btn-secondary btn-sm mt-1" disabled={gameStore.commandPending || !pilotCanAct || gameStore.connection !== 'connected'} onclick={sendRetrySearch}>{(gameStore.searchRetry.code.startsWith('search_') || gameStore.searchRetry.code.startsWith('mirror_')) ? 'Retry search' : 'Retry bot decision'}</button>
    </section>
  {/if}
  {#if !onTable}<WorkStatus work={gameStore.work} connected={gameStore.connection === 'connected'} inline />{/if}
  {#if gameStore.tableUnavailable}<p role="status" class="type-caption">The previous table is no longer live. Its recording is retained; start the next game explicitly.</p>{/if}
  {#if gameStore.storageWarning}<p role="status" class="type-caption">{gameStore.storageWarning}</p>{/if}
  {#if participantStudying}<p class="type-caption text-ink-2">Study position. {gameStore.branchProjection ? 'Alternate action played; return to the recorded decision to try another.' : 'Choosing an offer tries it in a separate branch.'}</p>{/if}
{/snippet}

{#snippet result()}
  <div data-testid="game-result-dialog" class="result">
    <p class="type-rubric text-ink-3">Game over</p>
    <h2 data-testid="game-result" class="type-display text-display">{ending?.headline}</h2>
    {#if ending}
      <p class="text-ink-2" data-testid="game-result-how">{ending.how}</p>
      <p class="type-caption text-ink-2">{ending.life}</p>
      <p class="type-caption text-ink-3" data-testid="game-result-summary">{ending.summary}</p>
    {/if}
    <div class="result-actions">
      {#if demoName}
        <button data-testid="game-result-action" class="btn btn-primary" onclick={startNewGame}>Play again</button>
      {:else}
        <button data-testid="game-result-action" class="btn btn-primary" disabled={!canContinue} onclick={() => sendContinuation('rematch')}>Rematch</button>
        <button data-testid="game-result-swap" class="btn btn-secondary" disabled={!canContinue} onclick={() => sendContinuation('swap_sides')}>Swap sides</button>
        <button data-testid="game-result-change" class="btn btn-secondary" disabled={!canContinue} onclick={() => openEditor('next', true)}>Change opponent</button>
      {/if}
    </div>
    <p class="type-caption result-links">
      {#if gameStore.recordingId}<a class="underline" data-testid="game-result-recording" href={`/replay?trace=${encodeURIComponent(gameStore.recordingId)}&review=1`}>Review this game</a> · {/if}
      <a class="underline" href="/?setup=1">New game</a>
    </p>
    {#if gameStore.newGamePending}<p role="status" class="type-caption text-ink-2">Preparing the next game…</p>{/if}
  </div>
{/snippet}

{#snippet liveStrip()}
  <Strip marks={liveMarks} selected={null} open={stripOpen} ontoggle={() => (stripOpen = !stripOpen)}
    note={gameStore.gameOver ? null : 'Positions open in Review after the game'} />
{/snippet}

{#snippet menu()}
  <a class="type-caption text-ink-2 hover:text-ink" href="/games">Games</a>
  <details class="table-menu">
    <summary class="btn btn-ghost btn-sm" data-testid="table-menu">Menu</summary>
    <div class="menu-panel" role="region" aria-label="Session, stops and tools">
      <a class="btn btn-secondary btn-sm" href="/?setup=1">New game</a>
      <fieldset class="beat-pace" data-testid="beat-pace">
        <legend class="type-caption text-ink-2">Opponent's moves</legend>
        {#each BEAT_PACES as pace (pace)}
          <label class="type-caption"><input type="radio" name="beat-pace" value={pace} checked={beatPace === pace}
            onchange={() => { beatPace = pace; saveBeatPace(pace); }} />{BEAT_PACE_LABELS[pace]}</label>
        {/each}
      </fieldset>
      {#if !demoName}<PlaySessionStrip onEditNext={() => openEditor('next')} onChangeOpponent={() => openEditor('next', true)} />{/if}
      <p data-testid="automation-status" class="type-caption text-ink-2">Priority-stop automation: {gameStore.stops.auto_pass ? `on · your turn ${gameStore.stops.my.length} / opponent ${gameStore.stops.opponent.length} · stack ${gameStore.stops.stop_on_stack ? 'on' : 'off'}` : 'off · every priority choice'}{gameStore.fastForwarding ? ' · yielding this turn' : ''}{stopsPending !== null ? ' · updating…' : ''}</p>
      <fieldset disabled={!gameStore.hasCapability('configure_match') || participantStudying || !inputReady}>
        <legend class="sr-only">Pilot stop controls</legend>
        <StopsPanel stops={gameStore.stops} onToggleStop={handleToggleStop} onStopOnStackChange={handleStopOnStackChange}
          onAutoPassChange={handleAutoPassChange} onReset={handleResetStops} />
      </fieldset>
      {#if gameStore.table}
        <details><summary class="type-rubric cursor-pointer">Table and Study tools</summary>
          <SharedTablePanel table={gameStore.table} restored={gameStore.restoredDecision} branchAttemptId={gameStore.branchAttemptId}
            announcement={gameStore.tableAnnouncement} onTransferPilot={sendTransferPilot} onRestoreDecision={sendRestoreDecision}
            onReturnFromBranch={sendReturnFromBranch} onReturnToLive={sendReturnToLive} />
        </details>
      {/if}
      {#if gameStore.table?.attempt_id && gameStore.hasCapability('configure_match')}
        {#key gameStore.table.attempt_id}<Feedback attemptId={gameStore.table.attempt_id} />{/key}
      {/if}
    </div>
  </details>
{/snippet}

<main
  class="play-page mx-auto w-full max-w-[1600px] px-4 py-4"
  data-update-seq={gameStore.updateSeq}
  data-frame-revision={gameStore.protocolFrame?.revision ?? -1}
  data-table-revision={gameStore.table?.table_revision ?? -1}
>
  <h1 class="sr-only">{onTable ? 'Current game' : 'New game'}</h1>

  {#if onTable && boardFrame}
    <Table
      frame={boardFrame}
      live={participantStudying ? canStudyRetry : canSubmitLive}
      status={gameStore.decisionStatus}
      you={{ name: 'You', deck: gameStore.deckNames?.hero ?? null }}
      opponent={{ name: opponentName, deck: gameStore.deckNames?.villain ?? null, description: opponentIdentity.description }}
      stops={gameStore.stops}
      connection={gameStore.connection}
      connectionLabel={gameStore.resumeFailed ? 'Interrupted' : gameStore.connection}
      log={gameStore.actionLog}
      {canPassTurn}
      presentation={presentationPlayer}
      {thinking}
      {opponentChoosing}
      workStarted={gameStore.work?.state === 'running' && gameStore.connection === 'connected' ? gameStore.work.started_at : null}
      interrupted={gameStore.resumeFailed}
      payment={paymentState}
      onpreview={previewPayment}
      onact={handleActionSelect}
      onpassturn={handlePassTurn}
      {notices}
      bottom={liveStrip}
      result={gameStore.gameOver ? result : undefined}
      {menu}
      {beatPace}
    />
  {:else if atHome}
    {@render notices()}
    <section class="home" aria-label="Play" data-testid="home">
      <p class="type-rubric text-ink-3">{lastMatchup ? 'Your last matchup' : 'Ready to play'}</p>
      <p class="home-matchup type-title" data-testid="home-matchup">
        <span>You</span>{#if homeMatchup.decks}<span class="text-ink-3">{' · '}{homeMatchup.decks.hero}</span>{/if}<span class="text-ink-3">{', '}{matchupSeat(homeMatchup.config)}</span>
        <span class="text-ink-3">{' vs '}</span><span>{homeMatchup.opponent}</span>{#if homeMatchup.decks}<span class="text-ink-3">{' · '}{homeMatchup.decks.villain}</span>{/if}
      </p>
      <button class="btn btn-primary home-play" data-testid="play-again" disabled={gameStore.newGamePending || gameStore.connection !== 'connected'}
        onclick={playAgain}>{gameStore.newGamePending ? 'Preparing the game…' : lastMatchup ? 'Play again' : 'Play'}</button>
      <p class="type-caption text-ink-2">
        <a class="underline" data-testid="home-setup" href="/?setup=1">Set up a different game</a> ·
        <a class="underline" href="/games">Games</a>
        <span data-testid="connection-badge" data-connection-state={gameStore.connection} role="status" aria-live="polite" aria-atomic="true"
          aria-label={`Connection status: ${gameStore.connection}`} class="type-label uppercase text-ink-3"> · {gameStore.connection}</span>
      </p>
    </section>
  {:else}
    {@render notices()}
    <WorkStatus work={watchStart.work} connected={!watchStart.checking} />
    {#if watchStart.checking}<p role="status">Checking the same Watch setup; loading status is unavailable.</p>{/if}
    <div class="play-sheet overflow-hidden rounded-sm border border-line bg-panel shadow-sheet">
      <div class="sheet-content px-5 pb-4 pt-3">
        <div data-testid="game-header" class="flex flex-wrap items-end justify-between gap-x-8 gap-y-4 border-b border-line pb-4">
          <span class="type-annotation text-ink-2">{gameStore.observation ? 'A game is in progress' : 'No game in progress'}</span>
          <div class="flex min-w-0 max-w-full flex-wrap items-end gap-x-5 gap-y-3">
            {#if learnDemo}<p class="type-caption">Learn demo · UR Lessons vs Random GW Allies. Automated checked setup; Start game repeats it.</p>
            {:else if demoName}<p class="type-caption" data-testid="demo-origin">Precision demo · custom stress decks · automated checked setup. Start game repeats setup.</p>{/if}
            {#if demoName}<button class="btn btn-primary" disabled={gameStore.newGamePending} onclick={startNewGame}>{gameStore.newGamePending ? 'Preparing game…' : 'Start game'}</button>{/if}
            <span data-testid="connection-badge" data-connection-state={gameStore.connection} role="status" aria-live="polite" aria-atomic="true"
              aria-label={gameStore.resumeFailed ? 'Session status: interrupted' : `Connection status: ${gameStore.connection}`}
              class="type-label uppercase text-ink-3">{gameStore.resumeFailed ? 'Interrupted' : gameStore.connection}</span>
          </div>
        </div>
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
            <button class="btn btn-secondary mt-3" disabled={gameStore.newGamePending} onclick={cancelEditor}>{gameStore.observation ? 'Back to the game' : 'Cancel'}</button>
          </section>
        {/if}
        {#if !gameStore.observation}
          <div class="mx-auto max-w-xl py-12 text-center">
            <p class="type-annotation mb-1 text-ink-2">The table is set.</p>
            <p role="status" class="type-caption mb-5 text-ink-2">{gameStore.newGamePending ? 'Preparing the game. This may take a moment…' : 'Start a game to begin.'}</p>
            <div class="mx-auto mt-10 max-w-sm border-t border-line pt-4 text-left">
              <fieldset disabled={!gameStore.hasCapability('configure_match')}>
                <legend class="sr-only">Pilot stop controls</legend>
                <StopsPanel stops={gameStore.stops} onToggleStop={handleToggleStop} onStopOnStackChange={handleStopOnStackChange}
                  onAutoPassChange={handleAutoPassChange} onReset={handleResetStops} />
              </fieldset>
            </div>
          </div>
        {/if}
      </div>
    </div>
  {/if}
</main>

<style>
  .notice { padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg-field); font-size: 13px; }
  .notice.error { border-color: color-mix(in srgb, var(--error) 50%, transparent); background: var(--error-wash); color: var(--error-text); }
  .result { display: grid; gap: 6px; justify-items: start; }
  .result h2 { line-height: 1.1; }
  .result-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
  .result-links { color: var(--text-secondary); }
  .home { max-width: 640px; margin: 12vh auto 0; display: grid; gap: 14px; justify-items: start;
    padding: 28px 30px; border: 1px solid var(--border); border-radius: 12px; background: var(--bg-surface); box-shadow: var(--shadow-sheet, none); }
  .home-matchup { line-height: 1.35; }
  .home-play { font-size: 16px; padding: 10px 22px; }
  .table-menu { position: relative; }
  .table-menu > summary { list-style: none; }
  .table-menu > summary::-webkit-details-marker { display: none; }
  .menu-panel {
    position: absolute; right: 0; top: calc(100% + 8px); z-index: 320; width: 360px; max-height: calc(100dvh - 64px); overflow-y: auto;
    display: grid; gap: 12px; justify-items: start; padding: 14px; border: 1px solid var(--border-strong); border-radius: 10px;
    background: var(--bg-surface); box-shadow: var(--shadow-raised);
  }
  .menu-panel > :global(*) { width: 100%; }
  .menu-panel > a { width: auto; }
  .beat-pace { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; border: 0; padding: 0; margin: 0; }
  .beat-pace legend { float: left; margin-right: 4px; padding: 0; }
  .beat-pace label { display: inline-flex; align-items: center; gap: 4px; cursor: pointer; }
</style>
