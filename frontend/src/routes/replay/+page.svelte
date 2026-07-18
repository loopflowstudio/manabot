<script lang="ts">
  import { onMount, tick, untrack } from 'svelte';

  import GameBoard from '$lib/components/GameBoard.svelte';
  import DecisionAdvice from '$lib/components/DecisionAdvice.svelte';
  import StudyPanel from '$lib/components/StudyPanel.svelte';
  import Timeline from '$lib/components/Timeline.svelte';
  import { fetchAdviceMeta, postAdvice, type AdviceMeta, type AdviceResponse } from '$lib/advice';
  import type {
    AddressedReplayDecision,
    CanonicalReplayProjectionResponseV1,
    RestoredReplayDecision,
  } from '$lib/replay-index';
  import { createReplayStore } from '$lib/replay.svelte';
  import { replayLogEntries } from '$lib/replay';
  import {
    mergePresentationLabels,
    presentationLabelsFromFrame,
    presentationLabelsFromObservation,
  } from '$lib/presentation';
  import { createPresentationPlayer } from '$lib/presentation.svelte';
  import { createStudyStore } from '$lib/study.svelte';
  import type {
    StudyPlanKind,
    StudyPlanPreviewResponse,
    StudyRetryResponse,
    StudyRevealResponse,
  } from '$lib/study-runtime';
  import type {
    GameLogEntry,
    InteractionOffer,
    Trace,
    TraceSummary,
  } from '$lib/types';

  const replayStore = createReplayStore();
  const studyStore = createStudyStore();
  const presentationPlayer = createPresentationPlayer();
  let originatingDecisionButton: HTMLButtonElement | null = null;
  let activeDecisionAddress = $state<string | null>(null);
  let focusedPlan = $state<StudyPlanKind | null>(null);

  onMount(() => {
    void loadTraces();
    void loadAdviceMeta();
  });

  // The shared advice surface remains at the Score while an opened canonical
  // decision switches the board to the session-bound Retry/Reveal flow.
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

  $effect(() => {
    if (!replayStore.playing) {
      return;
    }
    const timer = setInterval(() => {
      // A replay frame owns all of its ordered semantic beats. Do not replace
      // them with the next authoritative frame mid-sequence.
      if (!presentationPlayer.currentEvent) {
        replayStore.tick();
      }
    }, Math.max(1000 / replayStore.speed, 100));
    return () => clearInterval(timer);
  });

  const currentFrame = $derived(replayStore.frames[replayStore.currentFrameIndex] ?? null);
  const logEntries = $derived(replayStore.trace ? replayLogEntries(replayStore.trace) : []);
  const activeLogEntryId = $derived(
    replayStore.currentFrameIndex > 0
      ? logEntries[replayStore.currentFrameIndex - 1]?.id ?? null
      : null,
  );
  const boardObservation = $derived(
    studyStore.preview?.projection
      ?? studyStore.branchObservation
      ?? studyStore.restored?.frame.projection
      ?? currentFrame?.observation
      ?? null,
  );
  const focusedIds = $derived.by(() => {
    const kind = focusedPlan ?? studyStore.selectedPlan;
    const plan = studyStore.plans.find((candidate) => candidate.kind === kind);
    return new Set(plan?.offer.focus ?? []);
  });

  $effect(() => {
    if (
      !currentFrame
      || studyStore.phase !== 'score'
      || (activeDecisionAddress !== null && studyStore.restored !== null)
    ) {
      untrack(() => presentationPlayer.recover());
      return;
    }
    untrack(() =>
      presentationPlayer.recover(
        currentFrame.presentation,
        currentFrame.presentationLabels,
      ),
    );
  });

  async function loadTraces(): Promise<void> {
    replayStore.setLoadingList(true);
    replayStore.setError(null);

    try {
      const response = await fetch('/api/traces');
      if (!response.ok) {
        throw new Error(`Failed to load traces (${response.status})`);
      }
      const payload = (await response.json()) as TraceSummary[];
      replayStore.setSummaries(payload);
    } catch (error) {
      replayStore.setError(error instanceof Error ? error.message : 'Failed to load traces.');
    } finally {
      replayStore.setLoadingList(false);
    }
  }

  async function loadTrace(traceId: string): Promise<void> {
    replayStore.setLoadingTrace(true);
    replayStore.setError(null);

    try {
      const [traceResponse, decisionsResponse] = await Promise.all([
        fetch(`/api/traces/${traceId}`),
        fetch(`/api/traces/${traceId}/decisions`),
      ]);
      if (!traceResponse.ok) {
        throw new Error(`Failed to load trace ${traceId} (${traceResponse.status})`);
      }
      const payload = (await traceResponse.json()) as Trace;
      replayStore.loadTrace(payload);
      if (decisionsResponse.ok) {
        studyStore.loadProjection(
          (await decisionsResponse.json()) as CanonicalReplayProjectionResponseV1,
        );
      } else {
        studyStore.clearProjection();
      }
      activeDecisionAddress = null;
      focusedPlan = null;
      notes = loadNotes(traceId);
      editingNote = null;
    } catch (error) {
      replayStore.setError(error instanceof Error ? error.message : 'Failed to load trace.');
    } finally {
      replayStore.setLoadingTrace(false);
    }
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

  async function openStudyDecision(
    decision: AddressedReplayDecision,
    button: HTMLButtonElement,
  ): Promise<void> {
    const traceId = replayStore.trace?.id;
    if (!traceId) return;
    studyStore.setBusy(true);
    studyStore.setError(null);
    try {
      const response = await fetch(
        `/api/traces/${traceId}/decisions/${encodeURIComponent(decision.address)}`,
      );
      if (!response.ok) throw new Error(await responseError(response, 'Failed to restore decision.'));
      const restored = (await response.json()) as RestoredReplayDecision;
      studyStore.restore(decision, restored);
      if (replayStore.playing) replayStore.togglePlaying();
      replayStore.setFrame(decision.revision);
      activeDecisionAddress = decision.address;
      originatingDecisionButton = button;
      focusedPlan = null;
      presentationPlayer.recover();
    } catch (error) {
      studyStore.setError(error instanceof Error ? error.message : 'Failed to restore decision.');
    } finally {
      studyStore.setBusy(false);
    }
  }

  async function retryStudyOffer(offer: InteractionOffer): Promise<void> {
    const traceId = replayStore.trace?.id;
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
      const response = await fetch(
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
      presentationPlayer.recover(
        payload.retry.presentation,
        mergePresentationLabels(
          presentationLabelsFromFrame(restored.frame),
          presentationLabelsFromObservation(payload.retry.projection),
        ),
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
      const response = await fetch(`/api/study-attempts/${studyStore.attemptId}/reveal`, {
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
      const response = await fetch(`/api/study-attempts/${studyStore.attemptId}/preview`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ plan }),
      });
      if (!response.ok) throw new Error(await responseError(response, 'Plan preview failed.'));
      const payload = (await response.json()) as StudyPlanPreviewResponse;
      studyStore.acceptPreview(payload);
      const restored = studyStore.restored;
      presentationPlayer.recover(
        payload.presentation,
        restored
          ? mergePresentationLabels(
              presentationLabelsFromFrame(restored.frame),
              presentationLabelsFromObservation(payload.projection),
            )
          : presentationLabelsFromObservation(payload.projection),
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
        const response = await fetch(`/api/study-attempts/${attemptId}/return`, {
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
      focusedPlan = null;
      presentationPlayer.recover();
      restoreFocus = true;
    } catch (error) {
      studyStore.setError(error instanceof Error ? error.message : 'Return failed.');
    } finally {
      studyStore.setBusy(false);
      if (restoreFocus) {
        await tick();
        originatingDecisionButton?.focus();
      }
    }
  }

  function resumeReplay(action: () => void): void {
    studyStore.resetSelection();
    activeDecisionAddress = null;
    focusedPlan = null;
    action();
  }

  function winnerLabel(winner: number | null): string {
    if (winner === null) {
      return 'Draw';
    }
    return winner === 0 ? 'Hero' : 'Opponent';
  }

  // ——— The score: register marks, turn rubrics ———
  // A presentation-level heuristic: the engine does not yet annotate log
  // lines with registers, so the score infers them from game vocabulary.
  function register(entry: GameLogEntry): string {
    const text = entry.text.replace(/^(Hero|Villain|State): /, '');
    if (/^(Attack|Block|Declare)|damage/i.test(text)) return 'R';
    if (/^Play |^Auto-passed|priority window|^Pass turn/i.test(text)) return 'W';
    if (/dies|destroy|sacrific|discard|exile/i.test(text)) return 'B';
    if (/^Cast /.test(text)) return 'U';
    return 'N';
  }

  function lineText(entry: GameLogEntry): string {
    return entry.text.replace(/^(Hero|Villain|State): /, '');
  }

  function actorLabel(actor: GameLogEntry['actor']): string {
    return actor === 'hero' ? 'Hero' : actor === 'villain' ? 'Villain' : 'State';
  }

  // Line i of the score corresponds to frame i+1; the frame carries its turn.
  function turnOf(index: number): number | null {
    return replayStore.frames[index + 1]?.observation.turn.turn_number ?? null;
  }

  // ——— Marginalia: bronze notes, kept per trace ———
  let notes = $state<Record<string, string>>({});
  let editingNote = $state<string | null>(null);
  let draft = $state('');

  function notesKey(traceId: string): string {
    return `etude.score.notes.${traceId}`;
  }
  function loadNotes(traceId: string): Record<string, string> {
    try {
      return JSON.parse(localStorage.getItem(notesKey(traceId)) ?? '{}') ?? {};
    } catch {
      return {};
    }
  }
  function saveNotes(): void {
    const traceId = replayStore.trace?.id;
    if (!traceId) return;
    try {
      localStorage.setItem(notesKey(traceId), JSON.stringify(notes));
    } catch {
      // Storage full or unavailable: the note simply won't persist.
    }
  }
  function beginNote(entryId: string): void {
    editingNote = entryId;
    draft = notes[entryId] ?? '';
  }
  function commitNote(): void {
    if (editingNote === null) return;
    const text = draft.trim();
    if (text) {
      notes[editingNote] = text;
    } else {
      delete notes[editingNote];
    }
    saveNotes();
    editingNote = null;
  }
</script>

<main class="mx-auto w-full max-w-[1400px] px-4 py-6">
  <h1 class="sr-only">Replay</h1>

  {#if replayStore.errorMessage}
    <section class="mb-4 rounded border border-mountain/50 bg-mountain/20 px-4 py-3 text-mountain-ink">
      {replayStore.errorMessage}
    </section>
  {/if}

  {#if studyStore.phase === 'score' && studyStore.errorMessage}
    <section class="mb-4 rounded border border-mountain/50 bg-mountain/20 px-4 py-3 text-mountain-ink">
      {studyStore.errorMessage}
    </section>
  {/if}

  <!-- The score is the spine; the board is the current page of the book. -->
  <div class="overflow-hidden rounded-sm border border-line bg-panel shadow-sheet">
    <div class="px-10 pb-10 pt-6 max-md:px-5 max-md:pb-6">
      <header class="flex flex-wrap items-baseline justify-between gap-3 border-b border-line pb-4">
        <div class="flex items-baseline gap-3">
          <h2 class="type-title text-display">The Score</h2>
          {#if replayStore.trace}
            <span class="type-annotation text-ink-2">
              {winnerLabel(replayStore.trace.winner)} wins · {studyStore.projection?.decisions.length ?? logEntries.length} player decisions
            </span>
          {/if}
        </div>
        <div class="flex flex-wrap items-center gap-2">
          {#if replayStore.summaries.length > 0}
            <label class="type-label flex items-center gap-2 text-ink-2">
              Trace
              <select
                data-testid="trace-select"
                class="type-caption max-w-64 rounded border border-line bg-field px-2 py-1.5"
                value={replayStore.trace?.id ?? ''}
                onchange={(event) => {
                  const id = (event.currentTarget as HTMLSelectElement).value;
                  if (id) void loadTrace(id);
                }}
              >
                <option value="" disabled>Select a trace…</option>
                {#each replayStore.summaries as summary}
                  <option value={summary.id}>
                    {summary.timestamp ?? summary.id} · {winnerLabel(summary.winner)} · {summary.num_events} events
                  </option>
                {/each}
              </select>
            </label>
          {/if}
          <button
            class="btn btn-secondary btn-sm"
            onclick={() => void loadTraces()}
          >
            Refresh
          </button>
        </div>
      </header>

      {#if replayStore.loadingList && replayStore.summaries.length === 0}
        <p class="type-caption py-10 text-center text-ink-2">Loading traces…</p>
      {:else if replayStore.summaries.length === 0}
        <p class="type-annotation py-10 text-center text-ink-3">
          No traces yet. Play a game first.
        </p>
      {:else if replayStore.loadingTrace}
        <p class="type-caption py-10 text-center text-ink-2">Loading replay…</p>
      {:else if currentFrame && replayStore.trace}
        <div class="grid grid-cols-1 lg:grid-cols-[380px_minmax(0,1fr)]">
          <!-- The score column -->
          <section aria-label="Score" class="min-w-0 lg:border-r lg:border-line lg:pr-7">
            <p class="type-caption pb-1 pt-3 text-ink-2">
              Open any canonical player decision; Policy and Search stay sealed until Retry.
            </p>
            <!-- svelte-ignore a11y_no_noninteractive_tabindex (axe requires keyboard access to scrollable regions) -->
            <div
              class="max-h-[70vh] overflow-y-auto pr-1"
              role="region"
              aria-label="Score lines"
              tabindex="0"
            >
              {#if studyStore.projection && studyStore.projection.decisions.length > 0}
                <ol class="m-0 list-none p-0" data-testid="study-score">
                  {#each studyStore.projection.decisions as decision, index}
                    {@const turn = decision.frame.projection.turn.turn_number}
                    {@const previousTurn = index > 0 ? studyStore.projection.decisions[index - 1].frame.projection.turn.turn_number : null}
                    {#if turn !== previousTurn}
                      <li aria-hidden="true" class="rubric type-title pt-3 text-display">
                        Turn {turn}
                      </li>
                    {/if}
                    <li class="score-line" data-register="N">
                      <button
                        type="button"
                        data-testid="study-decision"
                        data-decision-address={decision.address}
                        class={`line-btn min-h-11 ${decision.address === activeDecisionAddress ? 'active' : ''}`}
                        aria-current={decision.address === activeDecisionAddress ? 'true' : undefined}
                        onclick={(event) => void openStudyDecision(decision, event.currentTarget)}
                        disabled={studyStore.busy || studyStore.phase !== 'score'}
                      >
                        <span class="who">#{index + 1}</span>
                        <span class="what">
                          {decision.frame.prompt?.title ?? 'Decision'}
                          <span class="type-caption block text-ink-2">{decision.frame.prompt?.kind ?? 'choice'} · revision {decision.revision}</span>
                        </span>
                      </button>
                    </li>
                  {/each}
                </ol>
              {:else}
                <p class="type-caption py-4 text-ink-3">This legacy recording has no canonical Study Score.</p>
              {/if}

              <details class="mt-4 border-t border-line pt-3">
                <summary class="type-label cursor-pointer text-ink-2">Frame-by-frame replay log</summary>
                <ol class="m-0 mt-2 list-none p-0">
                  {#each logEntries as entry, index}
                    {@const turn = turnOf(index)}
                    {@const previousTurn = index > 0 ? turnOf(index - 1) : null}
                    {#if turn !== null && turn !== previousTurn}
                      <li aria-hidden="true" class="rubric type-title pt-3 text-display">
                        Turn {turn}
                      </li>
                    {/if}
                    <li data-testid="log-entry" class="score-line" data-register={register(entry)}>
                      <button
                        type="button"
                        class={`line-btn ${entry.id === activeLogEntryId ? 'active' : ''}`}
                        aria-current={entry.id === activeLogEntryId ? 'true' : undefined}
                        onclick={() => resumeReplay(() => replayStore.setFrame(index + 1))}
                      >
                        <span class="who">{actorLabel(entry.actor)}</span>
                        <span class="what">{lineText(entry)}</span>
                      </button>
                      {#if editingNote === entry.id}
                        <div class="note-edit">
                          <textarea
                            aria-label={`Note for line ${index + 1}`}
                            placeholder="marginalia…"
                            bind:value={draft}
                            onkeydown={(event) => {
                              if (event.key === 'Escape') editingNote = null;
                              else if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) commitNote();
                            }}
                          ></textarea>
                          <div class="flex gap-1.5 pt-1">
                            <button type="button" class="btn btn-primary btn-sm" onclick={commitNote}>Save</button>
                            <button type="button" class="btn btn-ghost btn-sm" onclick={() => (editingNote = null)}>Cancel</button>
                          </div>
                        </div>
                      {:else if notes[entry.id]}
                        <button
                          type="button"
                          class="note-view"
                          aria-label={`Edit note on line ${index + 1}`}
                          onclick={() => beginNote(entry.id)}
                        >
                          <span aria-hidden="true">❧</span>
                          {notes[entry.id]}
                        </button>
                      {:else}
                        <button
                          type="button"
                          class="note-add"
                          aria-label={`Add a note to line ${index + 1}`}
                          onclick={() => beginNote(entry.id)}
                        >
                          + note
                        </button>
                      {/if}
                    </li>
                  {/each}
                </ol>
              </details>
            </div>
          </section>

          <!-- The plate: the board at the chosen moment -->
          <div class="min-w-0 pt-3 max-lg:mt-4 max-lg:border-t max-lg:border-line lg:pl-7">
            {#if studyStore.phase === 'score'}
              <Timeline
                currentFrame={replayStore.currentFrameIndex}
                totalFrames={replayStore.frames.length}
                playing={replayStore.playing}
                speed={replayStore.speed}
                actionDescription={currentFrame.actionDescription}
                actor={currentFrame.actor}
                onPrevious={() => resumeReplay(() => replayStore.previousFrame())}
                onNext={() => resumeReplay(() => replayStore.nextFrame())}
                onTogglePlaying={() => resumeReplay(() => replayStore.togglePlaying())}
                onScrub={(index) => resumeReplay(() => replayStore.setFrame(index))}
                onSpeedChange={(speed) => replayStore.setSpeed(speed)}
              />
            {:else if studyStore.restored}
              <p class="type-caption border-b border-line pb-3 text-ink-2">
                Canonical decision {studyStore.restored.ordinal + 1} · event cursor {studyStore.restored.presentation_cursor}
              </p>
            {/if}

            <div class="mt-3 border-t border-line">
              {#if boardObservation}
                <GameBoard
                  observation={boardObservation}
                  {focusedIds}
                  winner={boardObservation.game_over ? replayStore.trace.winner : undefined}
                  {presentationPlayer}
                />
              {/if}
            </div>

            {#if studyStore.phase === 'score'}
              <div class="mt-3 border-t border-line pt-4">
                <DecisionAdvice
                  mode="study"
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
            {:else if studyStore.restored}
              <StudyPanel
                phase={studyStore.phase}
                restored={studyStore.restored}
                plans={studyStore.plans}
                selectedPlan={studyStore.selectedPlan}
                busy={studyStore.busy}
                errorMessage={studyStore.errorMessage}
                onRetry={(offer) => void retryStudyOffer(offer)}
                onReveal={() => void revealStudy()}
                onReturn={() => void returnToScore()}
                onPreview={(plan) => void previewStudyPlan(plan)}
                onFocusPlan={(plan) => (focusedPlan = plan)}
              />
            {/if}
          </div>
        </div>
      {:else}
        <p class="type-annotation py-10 text-center text-ink-3">
          Select a trace to open its score.
        </p>
      {/if}
    </div>
  </div>
</main>

<style>
  .rubric::after {
    content: '';
    display: block;
    height: 1px;
    margin-top: 4px;
    background: linear-gradient(to right, var(--border), transparent 82%);
  }

  .score-line {
    padding: 2px 0;
  }
  .line-btn {
    display: grid;
    grid-template-columns: 48px minmax(0, 1fr);
    column-gap: 10px;
    width: 100%;
    text-align: left;
    border: none;
    border-left: 3px solid var(--reg, transparent);
    border-radius: 4px;
    background: color-mix(in srgb, var(--reg, transparent) 10%, transparent);
    padding: 5px 8px 6px 9px;
    font: inherit;
    color: var(--text);
    cursor: pointer;
  }
  .score-line[data-register='R'] { --reg: var(--error); }
  .score-line[data-register='W'] { --reg: var(--warning); }
  .score-line[data-register='U'] { --reg: var(--info); }
  .score-line[data-register='B'] { --reg: var(--neutral); }
  .score-line[data-register='G'] { --reg: var(--success); }
  .line-btn:hover {
    background: color-mix(in srgb, var(--reg, var(--border)) 18%, transparent);
  }
  .line-btn.active {
    background: var(--bg-field);
    box-shadow: inset 0 0 0 1px var(--accent);
  }
  /* Scoped implementations of the voices, value-exact: .who is
     type-rubric, .what is body. */
  .who {
    padding-top: 2px;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 600;
    line-height: 16px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-secondary);
  }
  .what {
    font-size: 14px;
    line-height: 20px;
  }

  .note-view {
    display: block;
    width: 100%;
    border: none;
    background: none;
    padding: 1px 8px 3px 70px;
    text-align: left;
    font-family: var(--font-serif);
    font-style: italic;
    font-size: 13.5px;
    line-height: 20px;
    color: var(--accent-text);
    cursor: pointer;
    overflow-wrap: break-word;
  }
  .note-add {
    display: block;
    border: 1px dashed transparent;
    border-radius: 4px;
    background: none;
    margin-left: 67px;
    padding: 0 5px;
    font-family: var(--font-serif);
    font-style: italic;
    font-size: 12px;
    line-height: 16px;
    color: var(--text-tertiary);
    cursor: pointer;
    opacity: 0.3;
  }
  .score-line:hover .note-add,
  .note-add:focus-visible,
  .score-line:focus-within .note-add {
    opacity: 1;
    border-color: var(--border);
  }
  .note-edit {
    padding: 3px 8px 3px 70px;
  }
  .note-edit textarea {
    width: 100%;
    min-height: 52px;
    resize: vertical;
    background: var(--bg-field);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text);
    font-family: var(--font-serif);
    font-style: italic;
    font-size: 13.5px;
    line-height: 20px;
    padding: 4px 8px;
  }
</style>
