<script lang="ts">
  /**
   * Play: the board of the game in progress, or the one setup screen.
   *
   * The address chooses what opens: `?game=<id>` takes that game up,
   * `?launch=<id>` starts a configured launch once, `?watch=1` fills in Watch
   * setup, and `/` alone returns to the unfinished game this browser
   * remembers. Once a game is on the board the address names it, so a reload
   * returns to it.
   */
  import { goto, replaceState } from '$app/navigation';
  import { page } from '$app/state';
  import { onDestroy, onMount, untrack } from 'svelte';

  import ArtCredits from '$lib/components/ArtCredits.svelte';
  import GameSetup from '$lib/components/GameSetup.svelte';
  import { startConfiguredWatch } from '$lib/new-game';
  import { watchStart } from '$lib/watch-start.svelte';
  import Board from '$lib/board/Board.svelte';
  import { PaymentPreviewController, type PreviewState } from '$lib/payment-preview';
  import { checkpointIdentity, playerIdentity } from '$lib/player-identity';
  import WorkStatus from '$lib/components/WorkStatus.svelte';
  import StopsPanel from '$lib/components/StopsPanel.svelte';
  import { policyChoice, gameStore, type PlayerChoice } from '$lib/game.svelte';
  import { presentationPlayer } from '$lib/presentation.svelte';
  import { sanitizeDeckSelection, DECK_CHOICES } from '$lib/decks';
  import { builtinOpponentName, shortName } from '$lib/opponent-name';
  import { scoreName } from '$lib/analysis';
  import { loadLastMatchup, matchupSeat, saveLastMatchup, type LastMatchup } from '$lib/last-matchup';
  import { gameResult } from '$lib/board/result';
  import { numbersLine, type GameNumbers } from '$lib/numbers';
  import { loadAutoTap, saveAutoTap, type FullControlMode } from '$lib/board/auto-tap';
  import { loadWalkPace, saveWalkPace, type WalkPace } from '$lib/board/beats';
  import { trackMarks } from '$lib/board/history';
  import { analysedEvidence, withLiveAnalysisPoints } from '$lib/board/replay';
  import { nextHeldOnce, onceHeldAt, stopStepOf, type OncePlace } from '$lib/board/phases';
  import { parseSearchConfigs } from '$lib/search';
  import {
    connect,
    disconnect,
    leaveGame,
    resumeGame,
    sendAction,
    sendNewGame,
    sendRematch,
    sendPassTurn,
    sendRetrySearch,
    sendSetStops,
    sendShowAnalysis,
    storedGame,
    fetchPaymentPreview,
    fetchCombatPreview,
  } from '$lib/socket.svelte';
  import type { GameSummary, InteractionOffer, StopSide } from '$lib/types';

  onDestroy(() => watchStart.cancel());

  let demoName = $state<string | null>(null);
  // Review's "Play on from here": a game whose moves before `event` are another game's.
  let playedFrom = $state<{trace_id: string; event: number} | null>(null);
  let learnDemo = $state(false);
  /** The game `?game=` asked for: a finished one opens in Replay instead. */
  let askedGame: string | null = null;
  /** A configured launch was sent and has not started or been refused yet. */
  let launching = $state(false);
  let setupNotice = $state('');
  let routed: string | null = null;

  /** Fill Watch setup from a launcher's address; no game starts until Start. */
  async function prefillWatch(params: URLSearchParams): Promise<void> {
    const launch = params.get('launch');
    if (launch) {
      try {
        const response = await fetch(`/api/launch/${encodeURIComponent(launch)}`);
        if (response.ok) gameStore.applySetupDraft(await response.json() as Record<string, unknown>);
        else setupNotice = 'This launch is unknown; run the launch command again or choose a matchup here.';
      } catch { setupNotice = 'The launched matchup could not be read; choose one here.'; }
    }
    gameStore.watchMode = true;
    gameStore.watchPlayerChoice = (params.get('seat1') ?? gameStore.watchPlayerChoice) as PlayerChoice;
    gameStore.setOpponentChoice((params.get('seat2') ?? gameStore.opponentChoice) as PlayerChoice);
    const search = params.get('search');
    if (search !== null) {
      try { [gameStore.watchSearchConfig, gameStore.searchConfig] = parseSearchConfigs(JSON.parse(search)); }
      catch { setupNotice = 'The launched search settings could not be read.'; }
    }
  }

  /** Open what the address asks for. An explicit address wins over the remembered game, which stays in History. */
  function route(params: URLSearchParams): void {
    demoName = params.get('demo');
    learnDemo = demoName === 'learn';
    const from = params.get('played_from'), at = Number(params.get('event'));
    playedFrom = from && Number.isInteger(at) && at >= 0 ? {trace_id: from, event: at} : null;
    askedGame = params.get('game');
    launching = false;
    setupNotice = '';
    const stored = storedGame();
    const decks = sanitizeDeckSelection({ hero: params.get('hero'), villain: params.get('villain') });
    const opponent = params.get('opponent');
    const launch = params.get('launch');
    const watch = params.get('watch') === '1';
    const explicit = demoName !== null || playedFrom !== null || watch || launch !== null || decks !== null || opponent !== null;
    if (askedGame) { resumeGame(askedGame); return; }
    if (!explicit) {
      if (stored) resumeGame(stored);
      else connect();
      return;
    }
    if (gameStore.gameId) leaveGame({ forget: false });
    if (demoName) gameStore.setOpponentChoice(learnDemo ? 'random' : 'passive');
    if (decks) { gameStore.setHeroDeck(decks.hero); gameStore.setVillainDeck(decks.villain); }
    if (opponent && /^[a-f0-9]{64}$/.test(opponent)) gameStore.setOpponentChoice(policyChoice(opponent));
    // Watch setup starts its game over HTTP: it opens no Play connection.
    if (watch) { void prefillWatch(params); return; }
    if (launch) { launching = true; sendNewGame({ launch }); return; }
    connect();
  }

  onMount(() => disconnect);
  $effect(() => {
    const search = page.url.search;
    if (search === routed) return;
    routed = search;
    untrack(() => route(new URLSearchParams(search)));
  });

  // The address names the game on the board, so a reload returns to it.
  $effect(() => {
    const game = gameStore.gameId;
    if (!game || !gameStore.observation || demoName) return;
    launching = false;
    untrack(() => {
      const params = new URLSearchParams(window.location.search);
      if (params.get('game') !== game || [...params.keys()].length > 1) replaceState(`/?game=${encodeURIComponent(game)}`, {});
    });
  });
  // A launch that was refused starts nothing: ordinary setup, at the ordinary address.
  $effect(() => {
    if (!launching || gameStore.newGamePending || gameStore.gameId) return;
    launching = false;
    untrack(() => replaceState('/', {}));
  });

  /**
   * A game the server would not take up. A finished one the address asked for
   * opens in Replay; an unknown one is dropped; any other keeps its reason and
   * a way to its Replay, and stays unfinished in History.
   */
  let refused = $state<{ game: string; message: string } | null>(null);
  $effect(() => {
    const refusal = gameStore.resumeRefused;
    if (!refusal) { refused = null; return; }
    untrack(() => {
      replaceState('/', {});
      void fetch(`/api/traces?q=${encodeURIComponent(refusal.game)}&limit=20`)
        .then(async response => (response.ok ? await response.json() as GameSummary[] : []))
        .catch(() => [] as GameSummary[])
        .then(rows => {
          if (gameStore.resumeRefused !== refusal) return;
          const row = rows.find(candidate => candidate.id === refusal.game);
          if (row?.status === 'completed' && askedGame === refusal.game) void goto(`/replay?trace=${encodeURIComponent(refusal.game)}`, { replaceState: true });
          else if (row && row.status !== 'completed') refused = refusal;
          else gameStore.resumeRefused = null;
        });
    });
  });

  /** New game: setup in place, with the matchup just played selected. The game left stays unfinished in History. */
  function openSetup(): void {
    if (gameStore.setup && !demoName) {
      gameStore.watchMode = false;
      gameStore.applySetupDraft(gameStore.setup);
    }
    leaveGame();
    replaceState(demoName ? `/?demo=${encodeURIComponent(demoName)}` : '/', {});
  }

  let stopsPending = $state<number | null>(null);
  let autoTap = $state(loadAutoTap());
  let fullControl = $state<FullControlMode>('off');
  let walkPace = $state<WalkPace>(loadWalkPace());

  function setWalkPace(pace: WalkPace): void {
    walkPace = pace;
    saveWalkPace(pace);
  }

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
    if (!canSubmitLive) return;
    gameStore.appendAutoNote(note, presentationPlayer.events.at(-1)?.seq ?? -1);
    sendAction(action.id);
  }

  const decisionFrame = $derived(gameStore.protocolFrame);

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

  function playOn(): void {
    if (!playedFrom || gameStore.newGamePending) return;
    // One start per link: the address then names the game it started.
    sendNewGame({ played_from: playedFrom });
    playedFrom = null;
  }

  function startGame(): void {
    if (gameStore.newGamePending) return;
    if (demoName) { sendNewGame({ demo: demoName }); return; }
    if (gameStore.seedError) { gameStore.setError(gameStore.seedError); return; }
    gameStore.setError(null);
    if (gameStore.watchMode) {
      gameStore.newGamePending = true;
      void startConfiguredWatch(gameStore).then(id => goto(`/replay?trace=${encodeURIComponent(id)}`))
        .catch(error => gameStore.setError(error.message)).finally(() => gameStore.newGamePending = false);
      return;
    }
    sendNewGame(gameStore.newGameConfig());
  }

  const gameActive = $derived(gameStore.observation !== null && !gameStore.gameOver && !gameStore.stopped);
  const inputReady = $derived(!gameStore.commandPending && gameStore.work?.state !== 'running' && stopsPending === null && gameStore.connection === 'connected');
  const canSubmitLive = $derived(gameActive && !gameStore.newGamePending && inputReady);
  const canPassTurn = $derived(canSubmitLive && decisionFrame?.action_space === 'PRIORITY' && (decisionFrame?.offers.length ?? 0) > 0);
  // Boards kept this visit (frames and per-line boards, capped at 512) for the game shown.
  const stepBoards = $derived(decisionFrame && gameStore.keptBoards?.matchId === decisionFrame.match_id
    ? gameStore.keptBoards.boards : []);

  $effect(() => {
    if (stopsPending !== null && (gameStore.updateSeq !== stopsPending || gameStore.connection !== 'connected')) stopsPending = null;
  });

  // The opponent is named with how it decides ("Heuristic · scripted",
  // "Search 64 · search"): the server's label once a game is on the board, the
  // built-in name before. The identity keeps the digest for its description.
  const opponentIdentity = $derived(gameStore.opponent
    ? checkpointIdentity(gameStore.opponent)
    : playerIdentity({ name: gameStore.opponentLabel ?? builtinOpponentName(gameStore.opponentChoice) ?? 'Opponent' }));
  const opponentName = $derived(gameStore.opponentLabel ?? opponentIdentity.compact);

  // ———————————————— Setup: play the last matchup again

  let lastMatchup = $state<LastMatchup | null>(null);
  onMount(() => { lastMatchup = loadLastMatchup(); });
  // Each game this tab plays becomes the next "Play again", from the setup the server started it with.
  $effect(() => {
    const config = gameStore.setup;
    if (!config || demoName || !gameStore.observation) return;
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
  function playAgain(): void {
    if (gameStore.newGamePending) return;
    gameStore.watchMode = false;
    gameStore.applySetupDraft(homeMatchup.config);
    gameStore.setError(null);
    sendNewGame(gameStore.newGameConfig());
  }

  function syncStopsToServer(): void {
    if (gameStore.observation) {
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

  // A stop set to hold once holds the game for you and is spent when you leave
  // its window (`nextHeldOnce`): the client clears it and the move list says
  // so (automation, not your move).
  let onceChecked = '';
  let heldOnce: OncePlace | null = null;
  $effect(() => {
    const frame = gameStore.protocolFrame;
    if (!frame) return;
    const frameKey = `${frame.match_id}:${frame.revision}`;
    if (frameKey === onceChecked) return;
    onceChecked = frameKey;
    if (gameStore.watchMode) return;
    const turn = frame.projection.turn;
    const side: StopSide = turn.active_player_id === frame.projection.agent.id ? 'my' : 'opponent';
    untrack(() => {
      const yours = frame.status !== 'game_over' && frame.prompt?.actor === frame.projection.agent.player_index;
      const arrived = yours ? onceHeldAt(turn, side === 'my', gameStore.stops, gameStore.onceStops, frame.action_space) : null;
      const next = nextHeldOnce(
        heldOnce,
        { match: frame.match_id, turn: turn.turn_number, side, step: stopStepOf(turn.step) },
        arrived,
      );
      heldOnce = next.held;
      if (next.spend && gameStore.onceStops[next.spend.side].includes(next.spend.step)) {
        gameStore.spendOnce(next.spend.side, next.spend.step, presentationPlayer.events.at(-1)?.seq ?? -1);
        syncStopsToServer();
      }
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
    // Client-local yield note anchored to the current point on the track.
    gameStore.addHistoryNote(
      'you',
      gameStore.observation?.turn.turn_number ?? null,
      presentationPlayer.events.at(-1)?.seq ?? -1,
      'Pass turn',
    );
  }

  function handleActionSelect(action: InteractionOffer): void {
    if (!canSubmitLive) return;
    // The move list names it when the server narrates the move.
    sendAction(action.id);
  }

  const inGame = $derived(gameStore.observation !== null && decisionFrame !== null);

  // Who is deciding while you wait: the opponent's search or policy, or the board itself.
  const heroSeat = $derived(gameStore.observation?.agent.player_index ?? null);
  const opponentChoosing = $derived(gameActive && heroSeat !== null && (
    (gameStore.work?.state === 'running' && (gameStore.work.kind === 'choose' || gameStore.work.kind === 'search')
      && gameStore.work.chooser !== null && gameStore.work.chooser !== heroSeat)
    || (decisionFrame?.prompt != null && decisionFrame.prompt.actor !== heroSeat)));
  const thinking = $derived(!gameActive ? null
    : opponentChoosing ? `${shortName(opponentName)} is ${gameStore.work?.kind === 'search' ? 'searching' : 'choosing'}`
      : gameStore.commandPending || gameStore.work?.state === 'running' ? 'The board is working' : null);

  const ending = $derived(gameStore.gameOver && decisionFrame
    ? gameResult(decisionFrame, { you: 'You', opponent: shortName(opponentName) }, gameStore.winner) : null);
  // The game's numbers, read once it is over: the server re-executes it.
  let numbers = $state<{ id: string; value: GameNumbers | null } | null>(null);
  $effect(() => {
    const id = gameStore.gameOver ? gameStore.gameId : null;
    if (!id || numbers?.id === id) return;
    numbers = { id, value: null };
    void fetch(`/api/traces/${encodeURIComponent(id)}/numbers`)
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
  const analysisOffered = $derived(inGame && !demoName && !gameStore.gameOver);
  const baseMarks = $derived(trackMarks(
    presentationPlayer.events,
    index => presentationPlayer.labelsAt(index),
    { youSeat: decisionFrame?.projection.agent.player_index ?? 0, you: 'You', opponent: shortName(opponentName) },
  ));
  const liveMarks = $derived(liveScale ? withLiveAnalysisPoints(baseMarks, liveAnalysis.points) : baseMarks);
  $effect(() => { if (liveAnalysis.shown) stripOpen = true; });
  const liveEvidence = $derived(liveAnalysis.shown && liveAnalysis.reading && decisionFrame
    && liveAnalysis.reading.revision === decisionFrame.revision && decisionFrame.offers.length
    ? analysedEvidence(decisionFrame, liveAnalysis.reading.decision, null, liveAnalysis.policy?.name ?? 'The policy') : null);
  const canShowAnalysis = $derived(!demoName && !gameStore.gameOver && gameStore.connection === 'connected' && !liveAnalysis.pending);
  const canContinue = $derived(!demoName && gameStore.setup !== null && !gameStore.newGamePending && !gameStore.stopped);
  const poolOf = (side: 'hero' | 'villain'): string | null => {
    const pool = gameStore.setup?.[`${side}_deck_pool`];
    return typeof pool === 'string' ? pool : null;
  };

</script>

{#snippet notices()}
  {#if playedFrom}
    <section class="notice" data-testid="played-from-origin">
      <p>Play on from a recorded game: its moves are replayed to your decision, then you choose.</p>
      <button class="btn btn-primary btn-sm mt-1" data-testid="start-played-from" disabled={gameStore.newGamePending} onclick={playOn}>{gameStore.newGamePending ? 'Preparing game…' : 'Play on'}</button>
    </section>
  {/if}
  {#if refused}
    <section role="alert" aria-atomic="true" class="notice error" data-testid="resume-refused">
      {refused.message} The game is still in History.
      <a class="underline" href={`/replay?trace=${encodeURIComponent(refused.game)}`}>Open its Replay</a>
    </section>
  {/if}
  {#if gameStore.errorMessage}
    <section role="alert" aria-atomic="true" class="notice error">
      {#if gameStore.stopped}<strong>Stopped.</strong>{' '}{/if}{gameStore.errorMessage}
      {#if gameStore.stopped && gameStore.gameId}
        <a class="underline" href={`/replay?trace=${encodeURIComponent(gameStore.gameId)}`}>Open its Replay</a>
      {/if}
    </section>
  {/if}
  {#if gameStore.searchRetry}
    <section role="alert" class="notice">
      <p>{gameStore.submissionAccepted ? 'Your choice was accepted. ' : ''}{gameStore.searchRetry.message}</p>
      <button class="btn btn-secondary btn-sm mt-1" disabled={gameStore.commandPending || gameStore.connection !== 'connected'} onclick={sendRetrySearch}>{(gameStore.searchRetry.code.startsWith('search_') || gameStore.searchRetry.code.startsWith('mirror_')) ? 'Retry search' : 'Retry bot decision'}</button>
    </section>
  {/if}
  {#if !inGame}<WorkStatus work={gameStore.work} connected={gameStore.connection === 'connected'} inline />{/if}
  {#if setupNotice && !inGame}<p role="status" class="type-caption">{setupNotice}</p>{/if}
  {#if gameStore.storageWarning}<p role="status" class="type-caption">{gameStore.storageWarning}</p>{/if}
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
        <button data-testid="game-result-action" class="btn btn-primary" onclick={startGame}>Play again</button>
      {:else}
        {@const heroPool = poolOf('hero')}
        {@const villainPool = poolOf('villain')}
        {#if heroPool || villainPool}
          <button data-testid="game-result-deal-new" class="btn btn-primary" disabled={!canContinue} onclick={() => sendRematch('rematch', { hero_deck: heroPool ?? gameStore.setup?.hero_deck, villain_deck: villainPool ?? gameStore.setup?.villain_deck, seed: null })}>Deal new decks</button>
        {/if}
        <button data-testid="game-result-action" class={(heroPool || villainPool) ? 'btn btn-secondary' : 'btn btn-primary'} disabled={!canContinue} onclick={() => sendRematch('rematch')}>Rematch</button>
        <button data-testid="game-result-swap" class="btn btn-secondary" disabled={!canContinue} onclick={() => sendRematch('swap_sides')}>Swap sides</button>
      {/if}
    </div>
    <p class="type-caption result-links">
      {#if gameStore.gameId}<a class="underline" data-testid="game-result-recording" href={`/replay?trace=${encodeURIComponent(gameStore.gameId)}&review=1`}>Review this game</a> · {/if}
      <button class="underline" data-testid="game-result-new" onclick={openSetup}>New game</button>
    </p>
    {#if gameStore.newGamePending}<p role="status" class="type-caption text-ink-2">Preparing the next game…</p>{/if}
  </div>
{/snippet}

{#snippet playSettings()}
  <div class="pace-setting" role="group" aria-label="Move pace" data-testid="menu-walk-pace">
    <span class="type-caption text-ink-2">Move pace</span>
    <div class="pace-options">
      <button type="button" class="btn btn-sm" class:btn-primary={walkPace === 'study'} class:btn-secondary={walkPace !== 'study'} aria-pressed={walkPace === 'study'} onclick={() => setWalkPace('study')}>Study (1.4 s)</button>
      <button type="button" class="btn btn-sm" class:btn-primary={walkPace === 'quick'} class:btn-secondary={walkPace !== 'quick'} aria-pressed={walkPace === 'quick'} onclick={() => setWalkPace('quick')}>Quick (0.7 s)</button>
      <button type="button" class="btn btn-sm" class:btn-primary={walkPace === 'off'} class:btn-secondary={walkPace !== 'off'} aria-pressed={walkPace === 'off'} onclick={() => setWalkPace('off')}>Off</button>
    </div>
  </div>
  <!-- The stop bar under the board sets each step's stop; this holds what it has no cell for: automation on or off, the stack stop and the reset. -->
  <p data-testid="automation-status" class="type-caption text-ink-2">Priority-stop automation: {gameStore.stops.auto_pass ? `on · your turn ${gameStore.stops.my.length} / opponent ${gameStore.stops.opponent.length} · stack ${gameStore.stops.stop_on_stack ? 'on' : 'off'}` : 'off · every priority choice'}{gameStore.fastForwarding ? ' · yielding this turn' : ''}{stopsPending !== null ? ' · updating…' : ''}</p>
  <fieldset disabled={!inputReady}>
    <legend class="sr-only">Stop controls</legend>
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
  <button class="btn btn-secondary btn-sm" data-testid="new-game" onclick={openSetup}>New game</button>
{/snippet}

<main
  class="play-page mx-auto w-full max-w-[1600px] px-4 py-4"
  data-update-seq={gameStore.updateSeq}
  data-frame-revision={gameStore.protocolFrame?.revision ?? -1}
  data-game={gameStore.gameId ?? ''}
>
  <h1 class="sr-only">{inGame ? 'Game' : 'Play'}</h1>

  {#if inGame && decisionFrame}
    <Board
      frame={decisionFrame}
      interaction={canSubmitLive ? 'play' : 'read-only'}
      status={gameStore.decisionStatus}
      you={{ name: 'You', deck: gameStore.deckNames?.hero ?? null }}
      opponent={{ name: opponentName, deck: gameStore.deckNames?.villain ?? null, description: opponentIdentity.description }}
      stops={gameStore.stops}
      onceStops={gameStore.onceStops}
      oncyclestop={handleCycleStop}
      connection={gameStore.connection}
      connectionLabel={gameStore.stopped ? 'Stopped' : gameStore.connection}
      notes={gameStore.historyNotes}
      {canPassTurn}
      presentation={presentationPlayer}
      {thinking}
      {opponentChoosing}
      evidence={liveEvidence}
      workStarted={gameStore.work?.state === 'running' && gameStore.connection === 'connected' ? gameStore.work.started_at : null}
      interrupted={gameStore.stopped}
      payment={paymentState}
      onpreview={previewPayment}
      combatPreview={gameActive ? fetchCombatPreview : undefined}
      onact={handleActionSelect}
      pending={gameStore.commandPending}
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
      {walkPace}
      onwalkpace={setWalkPace}
    />
  {:else}
    {@render notices()}
    <WorkStatus work={watchStart.work} connected={!watchStart.checking} />
    {#if watchStart.checking}<p role="status">Checking the same Watch setup; loading status is unavailable.</p>{/if}
    <section class="home-table" aria-label="Play" data-testid="home">
      {#if demoName}
        <div class="last-strip" data-testid="game-header">
          {#if learnDemo}<p class="type-caption">Learn demo · UR Lessons vs Random GW Allies. Automated checked setup; Start game repeats it.</p>
          {:else}<p class="type-caption" data-testid="demo-origin">Precision demo · custom stress decks · automated checked setup. Start game repeats setup.</p>{/if}
          <button class="btn btn-primary" disabled={gameStore.newGamePending} onclick={startGame}>{gameStore.newGamePending ? 'Preparing game…' : 'Start game'}</button>
        </div>
      {:else}
        <div class="last-strip">
          <div>
            <p class="type-label text-ink-3">{lastMatchup ? 'Your last matchup' : 'The default matchup'}</p>
            <p class="home-matchup type-title" data-testid="home-matchup">
              <span>You</span>{#if homeMatchup.decks}<span class="text-ink-3">{' · '}{homeMatchup.decks.hero}</span>{/if}<span class="text-ink-3">{', '}{matchupSeat(homeMatchup.config)}</span>
              <span class="text-ink-3">{' vs '}</span><span>{homeMatchup.opponent}</span>{#if homeMatchup.decks}<span class="text-ink-3">{' · '}{homeMatchup.decks.villain}</span>{/if}
            </p>
          </div>
          <button class="btn btn-primary" data-testid="play-again" disabled={gameStore.newGamePending}
            onclick={playAgain}>{gameStore.newGamePending ? 'Preparing the game…' : lastMatchup ? 'Play again' : 'Play'}</button>
        </div>
        <GameSetup setup={gameStore} onStart={startGame} disabled={gameStore.newGamePending} applySmartDefaults={!lastMatchup} />
        <details>
          <summary class="type-caption cursor-pointer text-ink-2">Priority stops</summary>
          <StopsPanel stops={gameStore.stops} onToggleStop={handleToggleStop} onStopOnStackChange={handleStopOnStackChange}
            onAutoPassChange={handleAutoPassChange} onReset={handleResetStops} />
        </details>
      {/if}
      <p class="type-caption text-ink-2" role="status">{gameStore.newGamePending ? 'Preparing the game. This may take a moment…' : ''}</p>
      <p class="type-caption text-ink-2">
        <span data-testid="connection-badge" data-connection-state={gameStore.connection} role="status" aria-live="polite" aria-atomic="true"
          aria-label={`Connection status: ${gameStore.connection}`} class="type-label uppercase text-ink-3">{gameStore.connection}</span>
      </p>
      <ArtCredits />
    </section>
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
  .pace-setting { display: grid; gap: 6px; }
  .pace-options { display: flex; flex-wrap: wrap; gap: 6px; }
  .live-analysis { display: grid; gap: 6px; }
  .live-analysis select { border: 1px solid var(--border); border-radius: 4px; background: var(--bg-field); padding: 2px 4px; color: var(--text); max-width: 190px; }
</style>
