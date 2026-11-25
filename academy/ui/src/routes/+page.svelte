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

  import Board from '$lib/board/Board.svelte';
  import { PaymentPreviewController, type PreviewState } from '$lib/payment-preview';
  import PlaySessionStrip from '$lib/components/PlaySessionStrip.svelte';
  import { playerToken, recordFetch } from '$lib/records';
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
  import { scoreName } from '$lib/analysis';
  import { loadLastMatchup, matchupSeat, saveLastMatchup, type LastMatchup } from '$lib/last-matchup';
  import { gameResult } from '$lib/board/result';
  import { numbersLine, type GameNumbers } from '$lib/numbers';
  import { loadAutoTap, saveAutoTap, type FullControlMode } from '$lib/board/auto-tap';
  import { analysedEvidence, liveAnalysedMarks, logMarks } from '$lib/board/replay';
  import { spentOnce } from '$lib/board/phases';
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
    sendShowAnalysis,
    sendSwapSides,
    sendTransferPilot,
    fetchPaymentPreview,
    fetchCombatPreview,
  } from '$lib/socket.svelte';
  import type { ExperienceFrame, InteractionOffer, StopSide } from '$lib/types';

  let demoName = $state<string | null>(null);
  // Review's "Play on from here": a game whose moves before `event` are another game's.
  let playedFrom = $state<{trace_id: string; event: number} | null>(null);
  let learnDemo = $state(false);
  type Editor = {kind:'closed'} | {kind:'new' | 'next'; setup:GameStore; opponentOnly:boolean; pendingAttempt?:string | null};
  let editor = $state<Editor>({kind:'closed'});
  let mounted = $state(false);
  let awaitingRecovery = $state(false);
  let setupStatus = $state('');

  let stopsPending = $state<number | null>(null);
  let autoTap = $state(loadAutoTap());
  let fullControl = $state<FullControlMode>('off');

  function toggleAutoTap(): void {
    if (fullControl !== 'off') {
      fullControl = 'off';
      autoTap = true;
      saveAutoTap(true);
      return;
    }
    autoTap = !autoTap;
    saveAutoTap(autoTap);
  }

  function handleAutoTap(action: InteractionOffer, note: string): void {
    if (participantStudying || !canSubmitLive) return;
    gameStore.appendAutoNote(note);
    sendAction(action.id);
  }

  onMount(() => {
    const params = new URLSearchParams(window.location.search);
    demoName = params.get('demo');
    const from = params.get('played_from'), at = Number(params.get('event'));
    playedFrom = from && Number.isInteger(at) && at >= 0 ? {trace_id: from, event: at} : null;
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

  const decisionFrame = $derived(gameStore.restoredDecision?.frame ?? gameStore.protocolFrame);

  // The automatic payer's plan for the offer the board asks about: a
  // read-only query bound to this exact decision. The controller drops an
  // answer for any other frame, and the board asks for nothing while it
  // cannot act, so a commit or a lost connection clears the plan.
  let paymentState = $state<PreviewState | null>(null);
  const payments = new PaymentPreviewController(fetchPaymentPreview, state => { paymentState = state; });
  function previewPayment(offer: InteractionOffer | null): void {
    if (offer && decisionFrame && canSubmitLive) void payments.select(decisionFrame, offer.id, offer.label);
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

  function playOn(): void {
    if (!playedFrom || gameStore.newGamePending) return;
    sendNewGame({played_from:playedFrom, player_token:playerToken()});
    // One start per link: a reload resumes the game rather than replaying again.
    playedFrom = null;
    void goto('/', {replaceState: true, keepFocus: true});
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
  const canPassTurn = $derived(canSubmitLive && decisionFrame?.action_space === 'PRIORITY' && (decisionFrame?.offers.length ?? 0) > 0);
  const canStudyRetry = $derived(participantStudying && !gameStore.branchAttemptId && gameStore.hasCapability('explore_study') && inputReady);
  // A Study branch shows its own projection with no live offers.
  const boardFrame = $derived<ExperienceFrame | null>(decisionFrame && gameStore.branchProjection
    ? { ...decisionFrame, projection: gameStore.branchProjection, offers: [], prompt: null } : decisionFrame);
  // The boards the opponent's moves were made on, for the frame they came with.
  const stepBoards = $derived(!gameStore.branchProjection && boardFrame && gameStore.stepBoards?.frameHash === boardFrame.frame_hash
    ? gameStore.stepBoards.boards : []);

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
  const deckLabel = (key: unknown): string => {
    const found = DECK_CHOICES.find(deck => deck.key === key)?.label;
    if (found) return found;
    if (typeof key === 'string' && key && key !== 'imported' && key !== 'custom') return key;
    return 'Imported deck';
  };
  const homeMatchup = $derived(lastMatchup ?? {
    config: gameStore.setupConfig(),
    opponent: builtinOpponentName(gameStore.opponentChoice) ?? 'Registered policy',
    decks: { hero: deckLabel(gameStore.decks.hero), villain: deckLabel(gameStore.decks.villain) },
  });
  const atHome = $derived(mounted && !awaitingRecovery && !demoName && !page.url.searchParams.has('launch')
    && !gameStore.observation && editor.kind === 'closed');

  function startHomeTable(): void {
    if (gameStore.newGamePending) return;
    if (gameStore.seedError) { gameStore.setError(gameStore.seedError); return; }
    if (gameStore.watchMode) {
      gameStore.newGamePending = true;
      gameStore.setError(null);
      void startConfiguredWatch(gameStore).then(id => goto(`/replay?trace=${encodeURIComponent(id)}`))
        .catch(error => gameStore.setError(error.message)).finally(() => gameStore.newGamePending = false);
      return;
    }
    gameStore.setError(null);
    sendNewGame(gameStore.newGameConfig());
  }

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

  function handleCycleStop(side: StopSide, step: string): void {
    gameStore.cycleStop(side, step);
    syncStopsToServer();
  }

  // A stop set to hold once is spent when the game arrives at it for you: the
  // client clears it and the move list says so (automation, not your move).
  // Acting again within the step the game stopped at spends nothing.
  let onceChecked = -1;
  let arrivedAt = '';
  $effect(() => {
    const frame = gameStore.protocolFrame;
    if (!frame || frame.revision === onceChecked) return;
    onceChecked = frame.revision;
    if (gameStore.watchMode || frame.status === 'game_over' || frame.prompt?.actor !== frame.projection.agent.player_index) return;
    const turn = frame.projection.turn;
    const at = `${frame.match_id}:${turn.turn_number}:${turn.step}`;
    if (at === arrivedAt) return;
    arrivedAt = at;
    untrack(() => {
      const yourTurn = turn.active_player_id === frame.projection.agent.id;
      const spent = spentOnce(turn, yourTurn, gameStore.stops, gameStore.onceStops);
      if (!spent) return;
      gameStore.spendOnce(spent.side, spent.step);
      syncStopsToServer();
    });
  });

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

  const inGame = $derived(gameStore.observation !== null && editor.kind === 'closed' && boardFrame !== null);

  // Who is deciding while you wait: the opponent's search or policy, or the board itself.
  const heroSeat = $derived(gameStore.observation?.agent.player_index ?? null);
  const opponentChoosing = $derived(gameActive && heroSeat !== null && (
    (gameStore.work?.state === 'running' && (gameStore.work.kind === 'choose' || gameStore.work.kind === 'search')
      && gameStore.work.chooser !== null && gameStore.work.chooser !== heroSeat)
    || (decisionFrame?.prompt != null && decisionFrame.prompt.actor !== heroSeat)));
  const thinking = $derived(!gameActive ? null
    : opponentChoosing ? `${shortName(opponentName)} is ${gameStore.work?.kind === 'search' ? 'searching' : 'choosing'}`
      : gameStore.commandPending || gameStore.work?.state === 'running' ? 'The board is working' : null);

  const ending = $derived(gameStore.gameOver && boardFrame
    ? gameResult(boardFrame, { you: 'You', opponent: shortName(opponentName) }, gameStore.winner) : null);
  // The game's numbers, read once it is over: the server re-executes it.
  let numbers = $state<{ id: string; value: GameNumbers | null } | null>(null);
  $effect(() => {
    const id = gameStore.gameOver ? gameStore.recordingId : null;
    if (!id || numbers?.id === id) return;
    numbers = { id, value: null };
    void recordFetch(`/api/traces/${encodeURIComponent(id)}/numbers`)
      .then(async response => (response.ok ? (await response.json() as { numbers: GameNumbers | null }).numbers : null))
      .then(value => { if (numbers?.id === id) numbers = { id, value }; })
      .catch(() => { /* The result card stands without its numbers. */ });
  });
  // The game so far as a strip of decisions, folded unless opened. Live play
  // shows no past position yet: positions open in Review once the game ends.
  let stripOpen = $state(false);
  // A policy's reading of your decision, shown beside its offers while you
  // show analysis: its probability and estimate on each, nothing selected.
  // A reading for another revision (a decision already made) is not drawn.
  const liveAnalysis = $derived(gameStore.liveAnalysis);
  // With analysis read this game, the strip carries the policy's line from
  // your seat, on Review's fixed 0–100 axis, and opens whenever it is shown.
  const liveScale = $derived.by(() => {
    const name = scoreName(liveAnalysis.points.find(point => point.semantic !== null)?.semantic ?? null);
    if (!name || !liveAnalysis.policy) return null;
    return `${shortName(liveAnalysis.policy.name)}'s ${name.toLowerCase()} (%) from your seat · ${liveAnalysis.points.length} ${liveAnalysis.points.length === 1 ? 'decision' : 'decisions'} read`;
  });
  const analysisOffered = $derived(inGame && !demoName && !gameStore.gameOver && pilotCanAct);
  const liveMarks = $derived(liveScale ? liveAnalysedMarks(gameStore.actionLog, liveAnalysis.points) : logMarks(gameStore.actionLog));
  $effect(() => { if (liveAnalysis.shown) stripOpen = true; });
  const liveEvidence = $derived(liveAnalysis.shown && liveAnalysis.reading && decisionFrame && !participantStudying
    && liveAnalysis.reading.revision === decisionFrame.revision && decisionFrame.offers.length
    ? analysedEvidence(decisionFrame, liveAnalysis.reading.decision, null, liveAnalysis.policy?.name ?? 'The policy') : null);
  const canShowAnalysis = $derived(!demoName && !gameStore.gameOver && pilotCanAct && gameStore.connection === 'connected' && !liveAnalysis.pending);
  const canContinue = $derived(!demoName && gameStore.playSession !== null && !!gameStore.table?.attempt_id && !gameStore.newGamePending && !gameStore.resumeFailed);
  const canSwapSides = $derived(
    gameActive && !demoName && gameStore.hasCapability('configure_match')
    && !participantStudying && !gameStore.branchProjection && !gameStore.searchRetry
    && !gameStore.newGamePending && !gameStore.swapPending && inputReady
  );
  // Close the Settings sheet automatically once a mid-game swap commits so the
  // person immediately sees their new seat; a refusal keeps the sheet open with its error.
  let lastSeatKey = '';
  $effect(() => {
    const matchId = gameStore.protocolFrame?.match_id;
    const seat = heroSeat;
    if (!matchId || seat === null) { lastSeatKey = ''; return; }
    const key = `${matchId}:${seat}`;
    if (lastSeatKey && lastSeatKey.startsWith(`${matchId}:`) && lastSeatKey !== key) {
      document.querySelector<HTMLDetailsElement>('details.game-menu')?.removeAttribute('open');
    }
    lastSeatKey = key;
  });

</script>

{#snippet notices()}
  {#if playedFrom}
    <section class="notice" data-testid="played-from-origin">
      <p>Play on from a recorded game: its moves are replayed to your decision, then you choose.{gameActive ? ' This stops the current game.' : ''}</p>
      <button class="btn btn-primary btn-sm mt-1" data-testid="start-played-from" disabled={gameStore.newGamePending || gameStore.connection !== 'connected'} onclick={playOn}>{gameStore.newGamePending ? 'Preparing game…' : 'Play on'}</button>
    </section>
  {/if}
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
  {#if !inGame}<WorkStatus work={gameStore.work} connected={gameStore.connection === 'connected'} inline />{/if}
  {#if gameStore.tableUnavailable}<p role="status" class="type-caption">The previous game is no longer live. Its recording is retained; start the next game explicitly.</p>{/if}
  {#if gameStore.storageWarning}<p role="status" class="type-caption">{gameStore.storageWarning}</p>{/if}
  {#if participantStudying}<p class="type-caption text-ink-2">Study position. {gameStore.branchProjection ? 'Alternate action played; return to the recorded decision to try another.' : 'Choosing an offer tries it in a separate branch.'}</p>{/if}
{/snippet}

{#snippet result()}
  <div data-testid="game-result-dialog" class="result">
    <p class="type-label text-ink-3">Game over</p>
    <h2 data-testid="game-result" class="type-display text-display">{ending?.headline}</h2>
    {#if ending}
      <p class="text-ink-2" data-testid="game-result-how">{ending.how}</p>
      <p class="type-caption text-ink-2">{ending.life}</p>
      {#if numbers?.value && heroSeat !== null}<p class="type-caption text-ink-3" data-testid="game-result-summary">{numbersLine(numbers.value, heroSeat, { you: 'You', opponent: shortName(opponentName) })}</p>{/if}
    {/if}
    <div class="result-actions">
      {#if demoName}
        <button data-testid="game-result-action" class="btn btn-primary" onclick={startNewGame}>Play again</button>
      {:else}
        {@const heroPool = typeof gameStore.playSession?.draft.hero_deck_pool === 'string' ? gameStore.playSession.draft.hero_deck_pool : null}
        {@const villainPool = typeof gameStore.playSession?.draft.villain_deck_pool === 'string' ? gameStore.playSession.draft.villain_deck_pool : null}
        {#if heroPool || villainPool}
          <button data-testid="game-result-deal-new" class="btn btn-primary" disabled={!canContinue} onclick={() => sendContinuation('rematch', { hero_deck: heroPool ?? gameStore.playSession?.draft.hero_deck, villain_deck: villainPool ?? gameStore.playSession?.draft.villain_deck, seed: null })}>Deal new decks</button>
        {/if}
        <button data-testid="game-result-action" class={(heroPool || villainPool) ? 'btn btn-secondary' : 'btn btn-primary'} disabled={!canContinue} onclick={() => sendContinuation('rematch')}>Rematch</button>
        <button data-testid="game-result-swap" class="btn btn-secondary" disabled={!canContinue} onclick={() => sendContinuation('swap_sides')}>Swap play/draw</button>
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

{#snippet playSettings()}
  <!-- The stop bar under the board sets each step's stop; this holds what it has no cell for: automation on or off, the stack stop and the reset. -->
  <p data-testid="automation-status" class="type-caption text-ink-2">Priority-stop automation: {gameStore.stops.auto_pass ? `on · your turn ${gameStore.stops.my.length} / opponent ${gameStore.stops.opponent.length} · stack ${gameStore.stops.stop_on_stack ? 'on' : 'off'}` : 'off · every priority choice'}{gameStore.fastForwarding ? ' · yielding this turn' : ''}{stopsPending !== null ? ' · updating…' : ''}</p>
  <fieldset disabled={!gameStore.hasCapability('configure_match') || participantStudying || !inputReady}>
    <legend class="sr-only">Pilot stop controls</legend>
    <StopsPanel stops={gameStore.stops} onToggleStop={handleToggleStop} onStopOnStackChange={handleStopOnStackChange}
      onAutoPassChange={handleAutoPassChange} onReset={handleResetStops} />
  </fieldset>
{/snippet}

{#snippet analysisSettings()}
  <div class="live-analysis" data-testid="live-analysis">
    <label class="setting"><span>Show analysis<small>What {liveAnalysis.policy ? shortName(liveAnalysis.policy.name) : 'a policy'} would choose here, and its estimate of the game. A choice you make while it is shown is marked "analysis shown" in Review.</small></span>
      <!-- The sheet closes on the change: it covers the choices the analysis is about. -->
      <input type="checkbox" role="switch" data-testid="show-analysis" checked={liveAnalysis.shown} disabled={!canShowAnalysis}
        onchange={event => { sendShowAnalysis(event.currentTarget.checked); event.currentTarget.closest('details')?.removeAttribute('open'); }} /></label>
    {#if liveAnalysis.policies && liveAnalysis.policies.length > 1}
      <label class="setting"><span>Read by</span>
        <select aria-label="Analysing policy" value={liveAnalysis.policy?.sha256} disabled={!canShowAnalysis}
          onchange={event => sendShowAnalysis(true, event.currentTarget.value)}>
          {#each liveAnalysis.policies as policy (policy.sha256)}<option value={policy.sha256}>{policy.name}</option>{/each}
        </select></label>
    {:else if liveAnalysis.policy}<p class="type-caption text-ink-3">Read by {liveAnalysis.policy.name}</p>{/if}
    {#if liveAnalysis.pending}<p class="type-caption text-ink-3" role="status">{liveAnalysis.shown ? 'Turning off…' : 'Loading the policy…'}</p>{/if}
    {#if liveAnalysis.unavailable}<p class="type-caption text-ink-3" data-testid="live-analysis-unavailable">{liveAnalysis.unavailable} Install a policy provider (managym.bots) to analyse.</p>{/if}
    {#if liveAnalysis.stopped}<p class="type-caption text-ink-3" role="status">{liveAnalysis.stopped}</p>{/if}
  </div>
{/snippet}

{#snippet gameSettings()}
  {#if !demoName}
    <section aria-label="This game" class="game-group">
      <p class="type-label text-ink-2">This game</p>
      <div class="swap-control">
        <button class="btn btn-secondary btn-sm" data-testid="swap-sides" disabled={!canSwapSides} onclick={sendSwapSides}>{gameStore.swapPending ? 'Swapping sides…' : 'Swap sides'}</button>
        <p class="type-caption text-ink-3">Take the opponent's seat and hand; the bot plays yours from here.</p>
        {#if gameStore.swapError}
          <p role="alert" class="type-caption swap-error" data-testid="swap-sides-error">{gameStore.swapError}</p>
        {/if}
      </div>
    </section>
  {/if}
  <section aria-label="Next game" class="game-group">
    <p class="type-label text-ink-2">Next game</p>
    {#if !demoName}<PlaySessionStrip onEditNext={() => openEditor('next')} onChangeOpponent={() => openEditor('next', true)} />{/if}
    <a class="btn btn-secondary btn-sm" href="/?setup=1">New game</a>
  </section>
  {#if gameStore.table}
    <details><summary class="type-label cursor-pointer">Sharing and Study tools</summary>
      <SharedTablePanel table={gameStore.table} restored={gameStore.restoredDecision} branchAttemptId={gameStore.branchAttemptId}
        announcement={gameStore.tableAnnouncement} onTransferPilot={sendTransferPilot} onRestoreDecision={sendRestoreDecision}
        onReturnFromBranch={sendReturnFromBranch} onReturnToLive={sendReturnToLive} />
    </details>
  {/if}
  {#if gameStore.table?.attempt_id && gameStore.hasCapability('configure_match')}
    {#key gameStore.table.attempt_id}<Feedback attemptId={gameStore.table.attempt_id} />{/key}
  {/if}
{/snippet}

<main
  class="play-page mx-auto w-full max-w-[1600px] px-4 py-4"
  data-update-seq={gameStore.updateSeq}
  data-frame-revision={gameStore.protocolFrame?.revision ?? -1}
  data-table-revision={gameStore.table?.table_revision ?? -1}
>
  <h1 class="sr-only">{inGame ? 'Current game' : 'New game'}</h1>

  {#if inGame && boardFrame}
    <Board
      frame={boardFrame}
      live={participantStudying ? canStudyRetry : canSubmitLive}
      status={gameStore.decisionStatus}
      you={{ name: 'You', deck: gameStore.deckNames?.hero ?? null }}
      opponent={{ name: opponentName, deck: gameStore.deckNames?.villain ?? null, description: opponentIdentity.description }}
      stops={gameStore.stops}
      onceStops={gameStore.onceStops}
      oncyclestop={handleCycleStop}
      connection={gameStore.connection}
      connectionLabel={gameStore.resumeFailed ? 'Interrupted' : gameStore.connection}
      log={gameStore.actionLog}
      {canPassTurn}
      presentation={presentationPlayer}
      {thinking}
      {opponentChoosing}
      evidence={liveEvidence}
      workStarted={gameStore.work?.state === 'running' && gameStore.connection === 'connected' ? gameStore.work.started_at : null}
      interrupted={gameStore.resumeFailed}
      payment={paymentState}
      onpreview={previewPayment}
      combatPreview={canSubmitLive && !participantStudying ? fetchCombatPreview : undefined}
      onact={handleActionSelect}
      onpassturn={handlePassTurn}
      {notices}
      strip={{ marks: liveMarks, selected: null, open: stripOpen, ontoggle: () => (stripOpen = !stripOpen),
        note: gameStore.gameOver || liveScale ? null : 'Review the game once it ends to step back through these', scale: liveScale }}
      result={gameStore.gameOver ? result : undefined}
      menu={{ play: playSettings, analysis: analysisOffered ? analysisSettings : undefined, game: gameSettings }}
      chips={liveAnalysis.shown ? [{ key: 'analysis', label: 'Analysis on', clear: () => sendShowAnalysis(false) }] : []}
      {stepBoards}
      {autoTap}
      {fullControl}
      ontoggleautotap={toggleAutoTap}
      onfullcontrol={mode => { fullControl = mode; }}
      onautotap={handleAutoTap}
    />
  {:else if atHome}
    {@render notices()}
    <section class="home-table" aria-label="Play" data-testid="home">
      {#if lastMatchup}
        <div class="last-strip">
          <div>
            <p class="type-label text-ink-3">Your last matchup</p>
            <p class="home-matchup type-title" data-testid="home-matchup">
              <span>You</span>{#if homeMatchup.decks}<span class="text-ink-3">{' · '}{homeMatchup.decks.hero}</span>{/if}<span class="text-ink-3">{', '}{matchupSeat(homeMatchup.config)}</span>
              <span class="text-ink-3">{' vs '}</span><span>{homeMatchup.opponent}</span>{#if homeMatchup.decks}<span class="text-ink-3">{' · '}{homeMatchup.decks.villain}</span>{/if}
            </p>
          </div>
          <button class="btn btn-primary" data-testid="play-again" disabled={gameStore.newGamePending || gameStore.connection !== 'connected'}
            onclick={playAgain}>{gameStore.newGamePending ? 'Preparing the game…' : 'Play again'}</button>
        </div>
      {:else}
        <div class="flex flex-wrap items-baseline justify-between gap-3">
          <div>
            <p class="type-label text-ink-3">Academy · Seat Table</p>
            <h2 class="type-display text-display">Choose seats and start</h2>
          </div>
          <button class="btn btn-secondary btn-sm" data-testid="play-again" disabled={gameStore.newGamePending || gameStore.connection !== 'connected'}
            onclick={startHomeTable}>{gameStore.newGamePending ? 'Preparing the game…' : 'Quick Play'}</button>
        </div>
      {/if}
      <GameSetup setup={gameStore} onStart={startHomeTable} disabled={gameStore.newGamePending || gameStore.connection !== 'connected'}
        applySmartDefaults={!lastMatchup} />
      <p class="type-caption text-ink-2">
        <a class="underline" data-testid="home-setup" href="/?setup=1">Full setup sheet</a> ·
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
          <span class="type-body text-ink-2">{gameStore.observation ? 'A game is in progress' : 'No game in progress'}</span>
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
            <p class="type-body mb-1 text-ink-2">The board is set.</p>
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
  .home-table { max-width: 960px; margin: 16px auto 0; display: grid; gap: 10px;
    padding: 16px 20px; border: 1px solid var(--border); border-radius: 12px; background: var(--bg-surface); box-shadow: var(--shadow-sheet, none); }
  .last-strip { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
  .home-matchup { line-height: 1.35; }
  .live-analysis { display: grid; gap: 6px; }
  .live-analysis select { border: 1px solid var(--border); border-radius: 4px; background: var(--bg-field); padding: 2px 4px; color: var(--text); max-width: 190px; }
  .game-group { display: grid; gap: 6px; justify-items: start; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
  .swap-control { display: grid; gap: 4px; justify-items: start; }
  .swap-error { color: var(--error-text); }
</style>
