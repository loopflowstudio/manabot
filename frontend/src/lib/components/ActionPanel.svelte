<script lang="ts">
  import type { ActionOption } from '$lib/types';

  export let actions: ActionOption[] = [];
  export let selectedTargetId: number | null = null;
  export let highlightedActionIndexes = new Set<number>();
  export let disabled = false;
  export let onHoverAction: ((action: ActionOption | null) => void) | undefined = undefined;
  export let onSelectAction: ((action: ActionOption) => void) | undefined = undefined;
  export let onClearSelection: (() => void) | undefined = undefined;
</script>

<aside class="rounded border border-slate-700 bg-slate-800 p-4">
  <div class="mb-3 flex items-center justify-between gap-3">
    <h2 class="text-sm font-semibold uppercase tracking-wide text-slate-300">Actions</h2>
    {#if selectedTargetId !== null}
      <button class="text-xs text-slate-400 underline hover:text-slate-200" on:click={() => onClearSelection?.()}>
        Show all
      </button>
    {/if}
  </div>

  {#if selectedTargetId !== null}
    <p class="mb-3 text-xs text-slate-400">Filtered to actions for selected board target.</p>
  {/if}

  <div class="space-y-2">
    {#if actions.length === 0}
      <p class="text-sm text-slate-400">No actions available.</p>
    {:else}
      {#each actions as action}
        <button
          class={`w-full rounded border px-3 py-2 text-left text-sm transition ${highlightedActionIndexes.has(action.index) ? 'border-amber-300 bg-slate-800' : 'border-slate-600 bg-slate-900 hover:border-blue-400 hover:bg-slate-800'} ${disabled ? 'cursor-not-allowed opacity-60' : ''}`}
          on:mouseenter={() => onHoverAction?.(action)}
          on:mouseleave={() => onHoverAction?.(null)}
          on:click={() => onSelectAction?.(action)}
          disabled={disabled}
        >
          <div class="font-medium">{action.description}</div>
          <div class="mt-1 text-xs text-slate-400">{action.type}</div>
        </button>
      {/each}
    {/if}
  </div>
</aside>
