<script lang="ts">
  export let currentFrame = 0;
  export let totalFrames = 0;
  export let playing = false;
  export let speed = 1;
  export let actionDescription: string | null = null;
  export let actor: 'hero' | 'villain' | null = null;
  export let onPrevious: (() => void) | undefined = undefined;
  export let onNext: (() => void) | undefined = undefined;
  export let onTogglePlaying: (() => void) | undefined = undefined;
  export let onScrub: ((index: number) => void) | undefined = undefined;
  export let onSpeedChange: ((speed: number) => void) | undefined = undefined;
</script>

<section class="rounded border border-slate-700 bg-slate-800 p-4">
  <div class="mb-3 flex flex-wrap items-center gap-2">
    <button class="rounded border border-slate-600 bg-slate-900 px-3 py-2 text-sm hover:border-blue-400" on:click={() => onPrevious?.()}>
      Previous
    </button>
    <button class="rounded border border-slate-600 bg-slate-900 px-3 py-2 text-sm hover:border-blue-400" on:click={() => onTogglePlaying?.()}>
      {playing ? 'Pause' : 'Play'}
    </button>
    <button class="rounded border border-slate-600 bg-slate-900 px-3 py-2 text-sm hover:border-blue-400" on:click={() => onNext?.()}>
      Next
    </button>

    <label class="ml-auto flex items-center gap-2 text-sm text-slate-300">
      Speed
      <select
        class="rounded border border-slate-600 bg-slate-900 px-2 py-1"
        value={speed}
        on:change={(event) => onSpeedChange?.(Number((event.currentTarget as HTMLSelectElement).value))}
      >
        <option value={1}>1x</option>
        <option value={2}>2x</option>
        <option value={4}>4x</option>
      </select>
    </label>
  </div>

  <input
    class="w-full"
    type="range"
    min="0"
    max={Math.max(totalFrames - 1, 0)}
    value={currentFrame}
    on:input={(event) => onScrub?.(Number((event.currentTarget as HTMLInputElement).value))}
  />

  <div class="mt-3 flex flex-wrap items-center justify-between gap-3 text-sm text-slate-300">
    <span>Frame {totalFrames === 0 ? 0 : currentFrame + 1} / {totalFrames}</span>
    <span class="text-right">
      {#if actionDescription}
        <span class={actor === 'villain' ? 'text-amber-300' : actor === 'hero' ? 'text-emerald-300' : 'text-slate-300'}>
          {actor === 'villain' ? 'Villain' : actor === 'hero' ? 'Hero' : 'State'}
        </span>
        : {actionDescription}
      {:else}
        Initial game state
      {/if}
    </span>
  </div>
</section>
