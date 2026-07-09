<script lang="ts">
  import type { OpponentChoice } from '$lib/game.svelte';

  export let value: OpponentChoice = 'search-64';
  export let checkpointPath = '';
  export let checkpointDeterministic = false;
  export let disabled = false;
  export let onChange: ((value: OpponentChoice) => void) | undefined = undefined;
  export let onCheckpointPathChange: ((value: string) => void) | undefined = undefined;
  export let onCheckpointDeterministicChange: ((value: boolean) => void) | undefined =
    undefined;
</script>

<div class="flex flex-wrap items-center gap-3">
  <label class="flex items-center gap-2 text-sm text-slate-300">
    Opponent
    <select
      class="rounded border border-slate-600 bg-slate-900 px-3 py-2 text-sm"
      value={value}
      disabled={disabled}
      on:change={(event) =>
        onChange?.((event.currentTarget as HTMLSelectElement).value as OpponentChoice)}
    >
      <option value="search-16">Search 16 (fast)</option>
      <option value="search-64">Search 64 (default)</option>
      <option value="search-256">Search 256 (strong)</option>
      <option value="checkpoint">Policy checkpoint (.pt)</option>
      <option value="random">Random</option>
      <option value="passive">Passive</option>
    </select>
  </label>

  {#if value === 'checkpoint'}
    <label class="flex items-center gap-2 text-sm text-slate-300">
      Path
      <input
        type="text"
        class="w-72 rounded border border-slate-600 bg-slate-900 px-3 py-2 text-sm"
        placeholder="/abs/path/to/step_65536.pt"
        value={checkpointPath}
        disabled={disabled}
        on:input={(event) =>
          onCheckpointPathChange?.((event.currentTarget as HTMLInputElement).value)}
      />
    </label>
    <label class="flex items-center gap-2 text-sm text-slate-300">
      <input
        type="checkbox"
        class="rounded border-slate-600 bg-slate-900"
        checked={checkpointDeterministic}
        disabled={disabled}
        on:change={(event) =>
          onCheckpointDeterministicChange?.(
            (event.currentTarget as HTMLInputElement).checked,
          )}
      />
      Argmax
    </label>
  {/if}
</div>
