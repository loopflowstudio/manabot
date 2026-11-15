<script lang="ts">
  import { goto, replaceState } from '$app/navigation';
  import { page } from '$app/state';
  import { tick, untrack } from 'svelte';
  import { offerFocus } from '$lib/offer-map';
  import { recordFetch } from '$lib/records';
  import WatchSetup from '$lib/components/WatchSetup.svelte';
  import Feedback from '$lib/components/Feedback.svelte';
  import GameBoard from '$lib/components/GameBoard.svelte';
  import PresentationHistory from '$lib/components/PresentationHistory.svelte';
  import DecisionSurface from '$lib/components/DecisionSurface.svelte';
  import DecisionAdvice from '$lib/components/DecisionAdvice.svelte';
  import DecisionContext from '$lib/components/DecisionContext.svelte';
  import StudyPanel from '$lib/components/StudyPanel.svelte';
  import Timeline from '$lib/components/Timeline.svelte';
  import { createReplayStore } from '$lib/replay.svelte';
  import { orientBoard } from '$lib/recording';
  import type { RecordedPosition } from '$lib/recording';
  import { createStudyStore } from '$lib/study.svelte';
  import { createPresentationPlayer } from '$lib/presentation.svelte';
  import { mergePresentationLabels, presentationLabelsFromFrame, presentationLabelsFromObservation } from '$lib/presentation';
  import { playerIdentity, recordedIdentity } from '$lib/player-identity';
  import type { RestoredReplayDecision } from '$lib/replay-index';
  import type { InteractionOffer, TraceSummary } from '$lib/types';
  import type { StudyPlanKind, StudySource, StudyRetryResponse, StudyRevealResponse, StudyPlanPreviewResponse } from '$lib/study-runtime';

  const replayStore = createReplayStore();
  const studyStore = createStudyStore();
  const presentationPlayer = createPresentationPlayer();
  let summaries = $state<TraceSummary[]>([]);
  let listError = $state('');
  let studyButton = $state<HTMLButtonElement | null>(null);
  let originatingDecisionButton: HTMLButtonElement | null = null;
  let focusedPlan = $state<StudyPlanKind | null>(null);
  let notes = $state<Record<string, string>>({});
  let historyBusy = $state(false);
  let loaded = '';
  let lastHistoryKey = '';
  let scoreStart = $state(0);
  const current = $derived(replayStore.current);
  const index = $derived(replayStore.index);
  const inStudy = $derived(studyStore.phase !== 'score');
  const informationBoard = $derived(studyStore.preview?.projection ?? studyStore.branchObservation ?? (inStudy ? studyStore.restored?.frame.projection : null) ?? current?.observation ?? null);
  const historyBoard = $derived(inStudy ? informationBoard : current?.decision?.frame.projection ?? current?.observation ?? null);
  const board = $derived(informationBoard ? orientBoard(informationBoard, replayStore.nearSeat) : null);
  const focusedIds = $derived.by(() => {
    const plan = inStudy ? studyStore.plans.find(plan => plan.kind === (focusedPlan ?? studyStore.selectedPlan)) : null;
    return new Set(plan ? offerFocus(plan.offer) : []);
  });
  const markers = $derived(index?.positions.filter((position, i, positions) => i === 0 || position.turn !== positions[i - 1].turn || position.phase !== positions[i - 1].phase || position.step !== positions[i - 1].step) ?? []);
  const score = $derived(index?.positions.slice(scoreStart, scoreStart + 60) ?? []);

  $effect(() => {
    const id = page.url.searchParams.get('trace');
    if (!id || id === loaded) return;
    loaded = id;
    untrack(() => { void open(id); });
  });
  $effect(() => { page.url.searchParams.has('watch'); untrack(() => { void loadList(); }); });

  async function loadList(): Promise<void> {
    try {
      const response = await recordFetch('/api/traces');
      if (!response.ok) throw new Error('Games could not be loaded.');
      summaries = await response.json(); listError = '';
    } catch (reason) { listError = reason instanceof Error ? reason.message : String(reason); }
  }
  async function open(id: string): Promise<void> {
    studyStore.resetSelection(); lastHistoryKey = '';
    let saved: {key?: string; near?: number; reveal?: boolean} = {};
    try { saved = JSON.parse(sessionStorage.getItem(`etude.replay.${id}`) ?? '{}'); } catch { /* Invalid preferences select no private data. */ }
    const params = page.url.searchParams;
    replayStore.nearSeat = Number(params.get('near') ?? saved.near ?? 0) === 1 ? 1 : 0;
    replayStore.reveal = (params.get('view') ?? (saved.reveal ? 'spectator' : 'acting')) === 'spectator';
    try { notes = JSON.parse(localStorage.getItem(`etude.score.notes.${id}`) ?? '{}'); } catch { notes = {}; }
    await replayStore.open(id, params.get('position') ?? saved.key);
    scoreStart = Math.floor(replayStore.selected / 60) * 60;
  }
  function selectRecording(id: string): void { void goto(`/replay?trace=${encodeURIComponent(id)}`); }
  async function request(path: string, init: RequestInit = {}): Promise<Response> {
    const token = sessionStorage.getItem(`etude.watch.${index?.id}`);
    const headers = new Headers(init.headers);
    if (token) headers.set('x-etude-participant-token', token);
    return recordFetch(path, {...init, headers});
  }

  function loadHistory(animate: boolean): void {
    if (!current || !index) return;
    const batches = current.history.map((event, i) => ({events: [event], labels: current.history_labels[i]}));
    presentationPlayer.loadPrefix(batches, `replay:${index.id}:viewer:${current.viewer}`, false);
    if (animate && current.kind === 'consequences') {
      presentationPlayer.playRecordedFrom(current.transition_start);
    }
    presentationPlayer.playing = false;
  }
  $effect(() => {
    current;
    if (!current || inStudy) return;
    untrack(() => { loadHistory(current.key !== lastHistoryKey); lastHistoryKey = current.key; });
  });
  $effect(() => {
    if (!current || !index || inStudy) return;
    const saved = {key: current.key, near: replayStore.nearSeat, reveal: replayStore.reveal};
    untrack(() => {
      sessionStorage.setItem(`etude.replay.${index.id}`, JSON.stringify(saved));
      const url = new URL(location.href);
      url.searchParams.set('trace', index.id); url.searchParams.set('position', saved.key);
      url.searchParams.set('near', String(saved.near)); url.searchParams.set('view', saved.reveal ? 'spectator' : 'acting');
      replaceState(url, {});
    });
  });
  // The transport owns both semantic beats and position advance. Stage timers
  // are disabled for recorded playback; Study keeps its isolated narration.
  $effect(() => {
    if (!replayStore.playing || replayStore.busy || replayStore.producing || inStudy) return;
    const event = presentationPlayer.currentEvent;
    const duration = event ? Math.max(80, event.suggested_ms / replayStore.speed) : 1000 / replayStore.speed;
    const timer = setTimeout(() => {
      if (event) presentationPlayer.advance();
      else void replayStore.next(true);
    }, duration);
    return () => clearTimeout(timer);
  });
  $effect(() => () => replayStore.cancelReads());

  function seek(selected: number): void {
    if (inStudy) return;
    void replayStore.seek(selected);
    scoreStart = Math.floor(selected / 60) * 60;
  }
  async function earlierHistory(): Promise<void> {
    if (!current?.history_before || historyBusy || !index) return;
    historyBusy = true;
    const key = current.key;
    try {
      const earlier = await replayStore.api<RecordedPosition>(`/api/traces/${index.id}/positions/${encodeURIComponent(key)}?before=${current.history_before}`);
      if (current?.key !== key || inStudy) return;
      replayStore.current = {...current, history: [...earlier.history, ...current.history], history_labels: [...earlier.history_labels, ...current.history_labels], history_before: earlier.history_before};
    } catch (reason) { replayStore.error = reason instanceof Error ? reason.message : String(reason); }
    finally { historyBusy = false; }
  }
  function saveNote(key: string, value: string): void {
    notes[key] = value;
    if (index) localStorage.setItem(`etude.score.notes.${index.id}`, JSON.stringify(notes));
  }
  function identity(seat: number) {
    const recorded = index?.players.find(player => player.seat === seat);
    return recorded ? recordedIdentity(recorded) : playerIdentity({name: `Seat ${seat + 1}`});
  }
  async function responseError(response: Response, fallback: string): Promise<string> {
    try {
      const payload = (await response.json()) as { detail?: string };
      const messages: Record<string, string> = {
        study_branch_unavailable: 'Retry is unavailable because this recording no longer has its retained rules root.',
        study_command_not_structured: 'That historical offer is not yet available through the exact Retry authority.',
        study_command_identity_mismatch: 'The Retry offer no longer matches this recorded decision.',
        study_attempt_active: 'Return from the current Study attempt before starting another.',
        study_evidence_unavailable: 'Study evidence is unavailable for this recording.',
        study_evidence_not_revealed: 'Reveal the plans before previewing them.',
        study_attempt_not_found: 'This Study attempt expired. Return to the score and open the decision again.',
      };
      return payload.detail ? (messages[payload.detail] ?? payload.detail.replaceAll('_', ' ')) : fallback;
    } catch {
      return fallback;
    }
  }

  async function openStudyDecision(button: HTMLButtonElement): Promise<void> {
    const decision = replayStore.current?.decision;
    const traceId = replayStore.index?.id;
    if (!decision || !traceId) return;
    replayStore.pause(); replayStore.cancelReads();
    studyStore.setBusy(true); studyStore.setError(null);
    try {
      const selected = await replayStore.api<RecordedPosition & {study_source: StudySource}>(`/api/traces/${traceId}/positions/${encodeURIComponent(decision.address)}?study=true`);
      const restored = await replayStore.api<RestoredReplayDecision>(`/api/traces/${traceId}/decisions/${encodeURIComponent(decision.address)}`);
      if (replayStore.current?.key !== decision.address) throw new Error('The selected choice changed while opening Study.');
      studyStore.restore(selected.study_source, decision, restored);
      focusedPlan = null;
      replayStore.current = {...replayStore.current, observation: decision.frame.projection, revealed: false};
      originatingDecisionButton = button;
      replayStore.reveal = false;
      loadHistory(false);
    } catch (reason) { studyStore.setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { studyStore.setBusy(false); }
  }

  async function retryStudyOffer(offer: InteractionOffer): Promise<void> {
    const traceId = replayStore.index?.id;
    const restored = studyStore.restored;
    if (!traceId || !restored || !restored.frame.prompt) return;
    studyStore.setBusy(true);
    studyStore.setError(null);
    try {
      const command = {
        command_id: `study.${crypto.randomUUID()}`,
        match_id: restored.frame.match_id,
        expected_revision: restored.revision,
        prompt_id: restored.frame.prompt.id,
        offer_id: offer.id,
        answers: [],
      };
      const response = await request(
        `/api/traces/${traceId}/decisions/${encodeURIComponent(restored.address)}/retry`,
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ command }),
        },
      );
      if (!response.ok) throw new Error(await responseError(response, 'Retry failed.'));
      const payload = (await response.json()) as StudyRetryResponse;
      studyStore.acceptRetry(payload);
      presentationPlayer.loadEvents(
        payload.retry.presentation,
        mergePresentationLabels(
          presentationLabelsFromFrame(restored.frame),
          presentationLabelsFromObservation(payload.retry.projection),
        ),
        `replay:${traceId}:study:${payload.attempt_id}:retry`,
      );
    } catch (error) {
      studyStore.setError(error instanceof Error ? error.message : 'Retry failed.');
    } finally {
      studyStore.setBusy(false);
    }
  }

  async function revealStudy(): Promise<void> {
    if (!studyStore.attemptId) return;
    studyStore.setBusy(true);
    studyStore.setError(null);
    try {
      const response = await request(`/api/study-attempts/${studyStore.attemptId}/reveal`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error(await responseError(response, 'Reveal failed.'));
      await studyStore.reveal((await response.json()) as StudyRevealResponse);
      await previewStudyPlan('played');
    } catch (error) {
      studyStore.setError(error instanceof Error ? error.message : 'Reveal failed.');
    } finally {
      studyStore.setBusy(false);
    }
  }

  async function previewStudyPlan(plan: StudyPlanKind): Promise<void> {
    if (!studyStore.attemptId) return;
    studyStore.setBusy(true);
    studyStore.setError(null);
    try {
      const response = await request(`/api/study-attempts/${studyStore.attemptId}/preview`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ plan }),
      });
      if (!response.ok) throw new Error(await responseError(response, 'Plan preview failed.'));
      const payload = (await response.json()) as StudyPlanPreviewResponse;
      studyStore.acceptPreview(payload);
      const restored = studyStore.restored;
      presentationPlayer.loadEvents(
        payload.presentation,
        restored
          ? mergePresentationLabels(
              presentationLabelsFromFrame(restored.frame),
              presentationLabelsFromObservation(payload.projection),
            )
          : presentationLabelsFromObservation(payload.projection),
        `replay:${replayStore.index?.id}:study:${studyStore.attemptId}:${plan}`,
      );
    } catch (error) {
      studyStore.setError(error instanceof Error ? error.message : 'Plan preview failed.');
    } finally {
      studyStore.setBusy(false);
    }
  }

  async function returnToScore(): Promise<void> {
    studyStore.setBusy(true);
    const attemptId = studyStore.attemptId;
    let restoreFocus = false;
    try {
      let returned = studyStore.restored;
      if (attemptId) {
        const response = await request(`/api/study-attempts/${attemptId}/return`, {
          method: 'POST',
        });
        if (!response.ok) throw new Error(await responseError(response, 'Return failed.'));
        returned = (await response.json()) as RestoredReplayDecision;
        const expected = studyStore.restored;
        if (!expected || JSON.stringify(returned) !== JSON.stringify(expected)) {
          throw new Error('Returned decision differs from canonical replay.');
        }
      }
      if (!returned) throw new Error('The canonical Study decision is unavailable.');
      studyStore.returnToScore(returned);

      loadHistory(false);
      restoreFocus = true;
    } catch (error) {
      studyStore.setError(error instanceof Error ? error.message : 'Return failed.');
    } finally {
      studyStore.setBusy(false);
      if (restoreFocus) {
        await tick();
        (originatingDecisionButton?.isConnected ? originatingDecisionButton : studyButton)?.focus();
      }
    }
  }

</script>

<main class="replay-page mx-auto w-full max-w-[1400px] px-4 py-3" class:replay-active={index !== null}>
  <header class="mb-2 flex flex-wrap items-center gap-3">
    <h1 class="type-title text-display">Replay</h1>
    <a class="underline type-caption" href={page.url.searchParams.get('from')?.startsWith('/games?') ? page.url.searchParams.get('from')! : '/games'}>Back to games</a>
    {#if summaries.length}
      <label>Recording <select data-testid="trace-select" class="max-w-64 rounded border border-line bg-field p-2" value={index?.id ?? ''} disabled={inStudy || studyStore.busy || replayStore.producing} onchange={event => selectRecording(event.currentTarget.value)}>
        <option value="" disabled>Select a recording…</option>
        {#each summaries as summary}<option value={summary.id}>{summary.timestamp ?? summary.id}</option>{/each}
      </select></label>
    {/if}
    <button class="btn btn-secondary" disabled={inStudy || studyStore.busy || replayStore.producing} onclick={() => { if (index) void replayStore.open(index.id, current?.key); else void loadList(); }}>Refresh saved head</button>
    {#if listError}<p role="alert">{listError}</p>{/if}
    {#if replayStore.error}<p role="alert">{replayStore.error}</p>{/if}
    {#if studyStore.errorMessage}<p role="alert">{studyStore.errorMessage}</p>{/if}
  </header>
  {#if !index}  {#if page.url.searchParams.has('watch')}
    <details open={!page.url.searchParams.has('trace')} class="mb-4 rounded border border-line bg-panel p-4">
      <summary class="type-label cursor-pointer">{index ? 'Watch another matchup' : 'Choose a bot matchup'}</summary>
      <WatchSetup disabled={inStudy || studyStore.busy || replayStore.producing} onOpen={selectRecording} />
    </details>
  {/if}
{/if}
  {#if index}
    <div class="recording-shell rounded border border-line bg-panel p-4 shadow-sheet">
      <header class="mb-2 flex flex-wrap items-baseline gap-x-3 border-b border-line pb-2">
        <h2 class="type-title">The Score{index.source_kind ? ` · ${index.source_kind === 'training' ? 'Training game' : 'Evaluation game'}` : ''}</h2>
        <p role="status">{replayStore.producing ? 'Producing a decision…' : replayStore.busy ? 'Reading saved recording…' : index.status === 'finished' ? `Finished · ${index.winner === null ? 'Draw' : `Seat ${index.winner + 1} wins`}` : index.status === 'waiting' ? index.watch ? 'Waiting at the live edge' : 'Live game · saved recording only' : 'Interrupted · saved recording only'}</p>
      </header>
      <div class="recording-workspace grid items-start gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
        <aside class="recording-rail" aria-label="Score">
          <label class="type-caption">Turn and phase
            <select class="mt-1 w-full rounded border border-line bg-field p-2" aria-label="Jump to turn and phase" disabled={inStudy || studyStore.busy || replayStore.producing} value={current?.key ?? ''} onchange={event => seek(index!.positions.findIndex(position => position.key === event.currentTarget.value))}>
              <option value="" disabled>Choose a passage</option>
              {#each markers as marker}<option value={marker.key}>Turn {marker.turn ?? '?'} · {marker.phase ?? 'Unknown'} · {marker.step ?? ''}</option>{/each}
            </select>
          </label>
          <div class="my-2 flex gap-2">
            <button class="btn btn-secondary btn-sm" disabled={scoreStart === 0} onclick={() => scoreStart = Math.max(0, scoreStart - 60)}>Earlier score</button>
            <button class="btn btn-secondary btn-sm" disabled={scoreStart + 60 >= index.positions.length} onclick={() => scoreStart += 60}>Later score</button>
          </div>
          <ol data-testid="study-score" class="max-h-[25vh] space-y-1 overflow-y-auto" aria-label="Score lines">
            {#each score as position, offset}
              <li><button data-testid={position.kind === 'choice' ? 'study-decision' : undefined} class="w-full rounded border border-line px-2 py-2 text-left type-caption" aria-current={current?.key === position.key ? 'true' : undefined} disabled={inStudy || studyStore.busy || replayStore.producing} onclick={() => seek(scoreStart + offset)}>
                {scoreStart + offset + 1}. {position.kind === 'state' ? index.status === 'finished' ? 'Result' : 'Latest state' : position.kind === 'choice' ? 'Choice' : position.automatic ? 'Automation consequences' : 'Consequences'} · Turn {position.turn ?? '?'}
                {#if position.actor !== null}<span class="block">Chooser: seat {position.actor + 1}</span>{/if}
              </button></li>
            {/each}
          </ol>
          {#if current}
            <label class="mt-3 block type-caption">Marginalia
              <textarea class="mt-1 w-full rounded border border-line bg-field p-2" aria-label="Note for this position" value={notes[current.note_key] ?? ''} oninput={event => saveNote(current.note_key, event.currentTarget.value)}></textarea>
            </label>
          {/if}
          {#if current}
            {#if current.history_before && !inStudy}<button class="btn btn-secondary btn-sm" disabled={historyBusy} onclick={() => void earlierHistory()}>Load earlier recorded history</button>{/if}
            <PresentationHistory player={presentationPlayer} observation={historyBoard} externallyDriven={!inStudy} />
            {#if !current.consequences_recorded}<p class="type-caption">Consequences not recorded.</p>{/if}
            {#if inStudy && studyStore.restored}
              {#if studyStore.branchObservation}<DecisionContext observation={studyStore.branchObservation} actor={studyStore.branchObservation.decision_context?.actor ?? null} />{/if}
              <StudyPanel phase={studyStore.phase} restored={studyStore.restored} plans={studyStore.plans} selectedPlan={studyStore.selectedPlan} busy={studyStore.busy} errorMessage={studyStore.errorMessage}
                onRetry={offer => void retryStudyOffer(offer)} onReveal={() => void revealStudy()} onReturn={() => void returnToScore()} onPreview={plan => void previewStudyPlan(plan)} onFocusPlan={plan => focusedPlan = plan} />
            {:else if current.decision}
              <DecisionSurface frame={current.decision.frame} readOnly chosenOffer={current.decision.offer_id} />
              <DecisionAdvice mode="watch" frame={current.decision.frame} offers={current.decision.frame.offers} recorded={current.outputs} chosenOffer={current.decision.offer_id} scenarios={[]} selectedScenarioId="" evidence={null} deltas={null} status="ok" reason={null} advisorId="" computeId="" onSelectScenario={() => {}} />
              <button class="btn btn-secondary mt-3" disabled={!index.can_study || studyStore.busy || replayStore.busy} onclick={event => void openStudyDecision(event.currentTarget)} bind:this={studyButton}>Try another line</button>
              {#if !index.can_study}<p class="type-caption">Study requires retained rules state and this game's participant credential.</p>{/if}
            {:else}<p class="type-caption mt-3">{current.kind === 'choice' ? 'Legal offers and Study not recorded.' : 'No decision at this position. Select a choice to inspect its offers and evidence.'}</p>{/if}
          {/if}

        </aside>
        <section class="recording-viewer min-w-0" aria-label="Recording viewer">
          <Timeline selected={replayStore.selected} total={index.positions.length} playing={replayStore.playing} speed={replayStore.speed} disabled={inStudy || studyStore.busy || replayStore.producing} busy={replayStore.busy} canProduce={index.can_produce}
            onPrevious={() => seek(Math.max(0, replayStore.selected - 1))} onNext={() => { replayStore.pause(); void replayStore.next(); }} onTogglePlaying={() => replayStore.playing = !replayStore.playing} onScrub={seek} onSpeedChange={speed => replayStore.speed = speed} />
          {#if current}
            <div class="my-3 flex flex-wrap items-center gap-3 border-b border-line pb-3">
              <p class="type-caption" data-testid="replay-perspective">{current.kind === 'choice' && current.decision ? `Chooser: seat ${current.decision.viewer + 1} · ${current.revealed ? 'Both hands · spectator view' : `Acting view · seat ${current.viewer + 1}`}` : 'Recorded seat-0 view · private post-action views unavailable'}</p>
              <button class="btn btn-secondary btn-sm" onclick={() => replayStore.nearSeat = 1 - replayStore.nearSeat}>Swap seats</button>
              <span class="type-caption">Near: seat {replayStore.nearSeat + 1}</span>
              <label class="type-caption"><input type="checkbox" checked={current.revealed} disabled={inStudy || studyStore.busy || replayStore.busy || !current.can_reveal} onchange={event => { replayStore.reveal = event.currentTarget.checked; void replayStore.seek(replayStore.selected); }} /> Show both hands</label>
              {#if current.kind === 'choice' && !current.can_reveal}<span class="type-caption">Spectator view not recorded.</span>{/if}
            </div>
            {#if current.description}<p class="type-annotation mb-2">{current.description}</p>{/if}
            {#if board}<GameBoard workspace observation={board} {focusedIds} heroIdentity={identity(replayStore.nearSeat)} villainIdentity={identity(1 - replayStore.nearSeat)} winner={board.game_over ? index.winner : undefined} {presentationPlayer} />{/if}

          {/if}
        </section>
      </div>
    </div>
  {:else if replayStore.busy}<p role="status">Reading saved recording…</p>
  {:else if !page.url.searchParams.has('watch')}<p>Select a recording from Games to open its score.</p>{/if}
</main>
<div class="mx-auto w-full max-w-[1400px] px-4">
  {#if index}  {#if page.url.searchParams.has('watch') || index?.watch}
    <details open={!page.url.searchParams.has('trace')} class="mb-4 rounded border border-line bg-panel p-4">
      <summary class="type-label cursor-pointer">{index ? 'Watch another matchup' : 'Choose a bot matchup'}</summary>
      <WatchSetup disabled={inStudy || studyStore.busy || replayStore.producing} onOpen={selectRecording} />
    </details>
  {/if}
{/if}
  {#if index?.can_feedback}<Feedback attemptId={index.id} />{/if}

</div>

<style>
  .recording-rail { min-width: 0; }
  @media (min-width: 1024px) and (min-height: 650px) {
    .replay-active { height: calc(100dvh - 64px); display: flex; flex-direction: column; }
    .replay-active > :global(*) { flex-shrink: 0; }
    .recording-shell { flex: 1; min-height: 0; display: flex; flex-direction: column; }
    .recording-workspace { flex: 1; min-height: 0; align-items: stretch; }
    .recording-rail { min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding-right: .5rem; scrollbar-gutter: stable; }
    .recording-viewer { min-height: 0; display: flex; flex-direction: column; }
    .recording-viewer > :global(*) { flex-shrink: 0; }
    .recording-viewer :global([data-testid="game-board"]) { flex: 1; min-height: 0; }
  }
</style>
