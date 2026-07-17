<script lang="ts">
  interface Props {
    currentFrame?: number;
    totalFrames?: number;
    playing?: boolean;
    speed?: number;
    actionDescription?: string | null;
    actor?: 'hero' | 'villain' | null;
    onPrevious?: () => void;
    onNext?: () => void;
    onTogglePlaying?: () => void;
    onScrub?: (index: number) => void;
    onSpeedChange?: (speed: number) => void;
  }

  let {
    currentFrame = 0,
    totalFrames = 0,
    playing = false,
    speed = 1,
    actionDescription = null,
    actor = null,
    onPrevious = undefined,
    onNext = undefined,
    onTogglePlaying = undefined,
    onScrub = undefined,
    onSpeedChange = undefined,
  }: Props = $props();
</script>

<section class="min-w-0">
  <div class="flex flex-wrap items-center gap-2">
    <button
      class="btn btn-secondary btn-sm w-7 px-0"
      aria-label="Previous frame"
      onclick={() => onPrevious?.()}
    >
      ‹
    </button>
    <button
      class="btn btn-secondary btn-sm"
      onclick={() => onTogglePlaying?.()}
    >
      {playing ? 'Pause' : 'Play'}
    </button>
    <button
      class="btn btn-secondary btn-sm w-7 px-0"
      aria-label="Next frame"
      onclick={() => onNext?.()}
    >
      ›
    </button>

    <input
      class="rail min-w-24 flex-1"
      type="range"
      aria-label="Replay position"
      aria-valuetext={`Frame ${totalFrames === 0 ? 0 : currentFrame + 1} of ${totalFrames}`}
      min="0"
      max={Math.max(totalFrames - 1, 0)}
      value={currentFrame}
      oninput={(event) => onScrub?.(Number((event.currentTarget as HTMLInputElement).value))}
    />

    <span class="type-rubric whitespace-nowrap tabular-nums text-ink-2">
      Frame {totalFrames === 0 ? 0 : currentFrame + 1} / {totalFrames}
    </span>

    <label class="type-label flex items-center gap-1.5 text-ink-2">
      Speed
      <select
        class="type-caption min-h-0 rounded border border-line bg-field px-2 py-0.5"
        value={speed}
        onchange={(event) => onSpeedChange?.(Number((event.currentTarget as HTMLSelectElement).value))}
      >
        <option value={1}>1x</option>
        <option value={2}>2x</option>
        <option value={4}>4x</option>
      </select>
    </label>
  </div>

  <div class="mt-2 text-ink-2">
    {#if actionDescription}
      <span class="type-rubric">
        {actor === 'villain' ? 'Villain' : actor === 'hero' ? 'Hero' : 'State'}
      </span>
      <span class="type-annotation text-ink"> {actionDescription}</span>
    {:else}
      <span class="type-annotation">Initial game state</span>
    {/if}
  </div>
</section>

<style>
  input.rail {
    appearance: none;
    -webkit-appearance: none;
    height: 4px;
    border-radius: 2px;
    background: var(--border);
    cursor: pointer;
  }
  input.rail::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 13px;
    height: 13px;
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid var(--ivory);
    box-shadow: 0 1px 3px rgb(0 0 0 / 0.3);
  }
  input.rail::-moz-range-thumb {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid var(--ivory);
  }
</style>
