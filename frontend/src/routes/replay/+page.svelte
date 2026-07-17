<script lang="ts">
  import { onMount, untrack } from 'svelte';

  import GameBoard from '$lib/components/GameBoard.svelte';
  import Timeline from '$lib/components/Timeline.svelte';
  import { createReplayStore } from '$lib/replay.svelte';
  import { replayLogEntries } from '$lib/replay';
  import { createPresentationPlayer } from '$lib/presentation.svelte';
  import type { GameLogEntry, Trace, TraceSummary } from '$lib/types';

  const replayStore = createReplayStore();
  const presentationPlayer = createPresentationPlayer();

  onMount(() => {
    void loadTraces();
  });

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

  $effect(() => {
    if (!currentFrame) {
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
      const response = await fetch(`/api/traces/${traceId}`);
      if (!response.ok) {
        throw new Error(`Failed to load trace ${traceId} (${response.status})`);
      }
      const payload = (await response.json()) as Trace;
      replayStore.loadTrace(payload);
      notes = loadNotes(traceId);
      editingNote = null;
    } catch (error) {
      replayStore.setError(error instanceof Error ? error.message : 'Failed to load trace.');
    } finally {
      replayStore.setLoadingTrace(false);
    }
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

  <!-- The score is the spine; the board is the current page of the book. -->
  <div class="overflow-hidden rounded-sm border border-line bg-panel shadow-sheet">
    <div class="px-10 pb-10 pt-6 max-md:px-5 max-md:pb-6">
      <header class="flex flex-wrap items-baseline justify-between gap-3 border-b border-line pb-4">
        <div class="flex items-baseline gap-3">
          <h2 class="type-title text-display">The Score</h2>
          {#if replayStore.trace}
            <span class="type-annotation text-ink-2">
              {winnerLabel(replayStore.trace.winner)} wins · {logEntries.length} decisions
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
              Click a line to turn the board to that moment
            </p>
            <!-- svelte-ignore a11y_no_noninteractive_tabindex (axe requires keyboard access to scrollable regions) -->
            <div
              class="max-h-[70vh] overflow-y-auto pr-1"
              role="region"
              aria-label="Score lines"
              tabindex="0"
            >
              <ol class="m-0 list-none p-0">
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
                      onclick={() => replayStore.setFrame(index + 1)}
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
            </div>
          </section>

          <!-- The plate: the board at the chosen moment -->
          <div class="min-w-0 pt-3 max-lg:mt-4 max-lg:border-t max-lg:border-line lg:pl-7">
            <Timeline
              currentFrame={replayStore.currentFrameIndex}
              totalFrames={replayStore.frames.length}
              playing={replayStore.playing}
              speed={replayStore.speed}
              actionDescription={currentFrame.actionDescription}
              actor={currentFrame.actor}
              onPrevious={() => replayStore.previousFrame()}
              onNext={() => replayStore.nextFrame()}
              onTogglePlaying={() => replayStore.togglePlaying()}
              onScrub={(index) => replayStore.setFrame(index)}
              onSpeedChange={(speed) => replayStore.setSpeed(speed)}
            />

            <div class="mt-3 border-t border-line">
              <GameBoard
                observation={currentFrame.observation}
                focusedIds={new Set()}
                winner={currentFrame.observation.game_over ? replayStore.trace.winner : undefined}
                {presentationPlayer}
              />
            </div>
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
