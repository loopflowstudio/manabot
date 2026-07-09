<script lang="ts">
  import type { ActionOption } from '$lib/types';

  interface Props {
    actions?: ActionOption[];
    selectedTargetId?: number | null;
    highlightedActionIndexes?: Set<number>;
    disabled?: boolean;
    fastForwarding?: boolean;
    canPassTurn?: boolean;
    onHoverAction?: (action: ActionOption | null) => void;
    onSelectAction?: (action: ActionOption) => void;
    onClearSelection?: () => void;
    onPassTurn?: () => void;
  }

  let {
    actions = [],
    selectedTargetId = null,
    highlightedActionIndexes = new Set<number>(),
    disabled = false,
    fastForwarding = false,
    canPassTurn = false,
    onHoverAction = undefined,
    onSelectAction = undefined,
    onClearSelection = undefined,
    onPassTurn = undefined,
  }: Props = $props();
</script>

<aside class="rounded border border-slate-700 bg-slate-800 p-4">
  <div class="mb-3 flex items-center justify-between gap-3">
    <div class="flex items-center gap-2">
      <h2 class="text-sm font-semibold uppercase tracking-wide text-slate-300">Actions</h2>
      {#if disabled}
        <span class="rounded bg-slate-700 px-2 py-0.5 text-xs font-semibold text-slate-300">Game over</span>
      {:else if fastForwarding}
        <span data-testid="auto-passing" class="animate-pulse rounded bg-sky-600/30 px-2 py-0.5 text-xs font-semibold text-sky-300">Auto-passing…</span>
      {:else if actions.length > 0}
        <span class="rounded bg-emerald-600/30 px-2 py-0.5 text-xs font-semibold text-emerald-300">Your move</span>
      {/if}
    </div>
    <div class="flex items-center gap-3">
      {#if selectedTargetId !== null}
        <button class="text-xs text-slate-400 underline hover:text-slate-200" onclick={() => onClearSelection?.()}>
          Show all
        </button>
      {/if}
      <button
        data-testid="pass-turn"
        class="rounded border border-slate-600 bg-slate-900 px-2 py-1 text-xs font-semibold text-slate-300 transition hover:border-sky-400 hover:text-sky-300 disabled:cursor-not-allowed disabled:opacity-50"
        disabled={!canPassTurn || fastForwarding}
        onclick={() => onPassTurn?.()}
        title="Auto-pass every priority window until the turn ends"
      >
        Pass Turn (F6)
      </button>
    </div>
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
          data-testid="action-option"
          class={`w-full rounded border px-3 py-2 text-left text-sm transition ${highlightedActionIndexes.has(action.index) ? 'border-amber-300 bg-slate-800' : 'border-slate-600 bg-slate-900 hover:border-blue-400 hover:bg-slate-800'} ${disabled ? 'cursor-not-allowed opacity-60' : ''}`}
          onmouseenter={() => onHoverAction?.(action)}
          onmouseleave={() => onHoverAction?.(null)}
          onclick={() => onSelectAction?.(action)}
          {disabled}
        >
          <div class="font-medium">{action.description}</div>
          <div class="mt-1 text-xs text-slate-400">{action.type}</div>
        </button>
      {/each}
    {/if}
  </div>
</aside>
