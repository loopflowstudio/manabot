<script lang="ts">
  import { onDestroy, onMount } from 'svelte';

  import GameBoard from '$lib/components/GameBoard.svelte';
  import GameLog from '$lib/components/GameLog.svelte';
  import Timeline from '$lib/components/Timeline.svelte';
  import { createReplayStore } from '$lib/replay.svelte';
  import { replayLogEntries } from '$lib/replay';
  import type { Trace, TraceSummary } from '$lib/types';

  const replayStore = createReplayStore();
  let playbackTimer: ReturnType<typeof setInterval> | null = null;

  onMount(() => {
    void loadTraces();
  });

  onDestroy(() => {
    if (playbackTimer !== null) {
      clearInterval(playbackTimer);
    }
  });

  $: {
    if (playbackTimer !== null) {
      clearInterval(playbackTimer);
      playbackTimer = null;
    }

    if (replayStore.playing) {
      playbackTimer = setInterval(() => {
        replayStore.tick();
      }, Math.max(1000 / replayStore.speed, 100));
    }
  }

  $: currentFrame = replayStore.frames[replayStore.currentFrameIndex] ?? null;
  $: logEntries = replayStore.trace ? replayLogEntries(replayStore.trace) : [];
  $: activeLogEntryId =
    replayStore.currentFrameIndex > 0
      ? logEntries[replayStore.currentFrameIndex - 1]?.id ?? null
      : null;

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
</script>

<main class="mx-auto w-full max-w-[1600px] p-4">
  <div class="grid grid-cols-1 gap-4 xl:grid-cols-[320px_minmax(0,1fr)_320px]">
    <section class="rounded border border-slate-700 bg-slate-800 p-4">
      <div class="mb-3 flex items-center justify-between gap-3">
        <h1 class="text-lg font-bold">Replay</h1>
        <button class="rounded border border-slate-600 bg-slate-900 px-3 py-2 text-sm hover:border-blue-400" on:click={() => void loadTraces()}>
          Refresh
        </button>
      </div>

      {#if replayStore.loadingList}
        <p class="text-sm text-slate-400">Loading traces…</p>
      {:else if replayStore.summaries.length === 0}
        <p class="text-sm text-slate-400">No traces yet. Play a game first.</p>
      {:else}
        <div class="space-y-2">
          {#each replayStore.summaries as summary}
            <button
              class={`w-full rounded border px-3 py-3 text-left text-sm ${replayStore.trace?.id === summary.id ? 'border-blue-400 bg-slate-900' : 'border-slate-700 bg-slate-900/60 hover:border-slate-500'}`}
              on:click={() => void loadTrace(summary.id)}
            >
              <div class="font-medium text-slate-100">{summary.id}</div>
              <div class="mt-1 text-xs text-slate-400">{summary.timestamp ?? 'Unknown time'}</div>
              <div class="mt-1 text-xs text-slate-400">
                Winner: {winnerLabel(summary.winner)} · Events: {summary.num_events}
              </div>
            </button>
          {/each}
        </div>
      {/if}
    </section>

    <div class="space-y-4">
      {#if replayStore.errorMessage}
        <section class="rounded border border-rose-500/50 bg-rose-900/20 px-4 py-3 text-sm text-rose-200">
          {replayStore.errorMessage}
        </section>
      {/if}

      {#if replayStore.loadingTrace}
        <section class="rounded border border-slate-700 bg-slate-800 p-10 text-center text-slate-300">
          Loading replay…
        </section>
      {:else if currentFrame && replayStore.trace}
        <div class="space-y-4">
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

          <GameBoard
            observation={currentFrame.observation}
            focusedIds={new Set()}
            winner={currentFrame.observation.game_over ? replayStore.trace.winner : undefined}
          />
        </div>
      {:else}
        <section class="rounded border border-slate-700 bg-slate-800 p-10 text-center text-slate-300">
          Select a trace to inspect it.
        </section>
      {/if}
    </div>

    <GameLog entries={logEntries} activeEntryId={activeLogEntryId} />
  </div>
</main>
