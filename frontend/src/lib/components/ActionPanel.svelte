<script lang="ts">
  import { tick } from 'svelte';

  import { DECISION_PROMPTS } from '$lib/prompt-instructions';
  import type { ActionOption } from '$lib/types';

  interface Props {
    actions?: ActionOption[];
    actionSpaceKind?: string;
    selectedTargetId?: number | null;
    highlightedActionIndexes?: Set<number>;
    disabled?: boolean;
    fastForwarding?: boolean;
    canPassTurn?: boolean;
    focusKey?: string;
    onHoverAction?: (action: ActionOption | null) => void;
    onSelectAction?: (action: ActionOption) => void;
    onClearSelection?: () => void;
    onPassTurn?: () => void;
  }

  let {
    actions = [],
    actionSpaceKind = '',
    selectedTargetId = null,
    highlightedActionIndexes = new Set<number>(),
    disabled = false,
    fastForwarding = false,
    canPassTurn = false,
    focusKey = '',
    onHoverAction = undefined,
    onSelectAction = undefined,
    onClearSelection = undefined,
    onPassTurn = undefined,
  }: Props = $props();

  const decisionPrompt = $derived(
    actions.length > 0 && !disabled ? DECISION_PROMPTS[actionSpaceKind] ?? null : null,
  );
  let actionList: HTMLDivElement | null = $state(null);
  let lastFocusKey = '';

  $effect(() => {
    const nextFocusKey = focusKey;
    const shouldFocus = nextFocusKey !== '' && nextFocusKey !== lastFocusKey && actions.length > 0 && !disabled;
    if (!shouldFocus) {
      return;
    }

    lastFocusKey = nextFocusKey;
    void tick().then(() => {
      actionList?.querySelector<HTMLButtonElement>('[data-testid="action-option"]')?.focus();
    });
  });
</script>

<aside
  data-testid="action-panel"
  data-action-space-kind={actionSpaceKind}
  aria-labelledby="action-panel-heading"
  aria-describedby={decisionPrompt ? 'decision-prompt' : undefined}
  class="rounded border border-slate-700 bg-slate-800 p-4"
>
  <div class="mb-3 flex items-center justify-between gap-3">
    <div class="flex items-center gap-2">
      <h2 id="action-panel-heading" class="text-sm font-semibold uppercase tracking-wide text-slate-300">Actions</h2>
      {#if disabled}
        <span role="status" aria-live="polite" aria-atomic="true" class="rounded bg-slate-700 px-2 py-0.5 text-xs font-semibold text-slate-300">Game over</span>
      {:else if fastForwarding}
        <span role="status" aria-live="polite" aria-atomic="true" data-testid="auto-passing" class="animate-pulse rounded bg-sky-600/30 px-2 py-0.5 text-xs font-semibold text-sky-300">Auto-passing…</span>
      {:else if actions.length > 0}
        <span role="status" aria-live="polite" aria-atomic="true" class="rounded bg-emerald-600/30 px-2 py-0.5 text-xs font-semibold text-emerald-300">Your move</span>
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

  {#if decisionPrompt}
    <p
      id="decision-prompt"
      data-testid="decision-prompt"
      data-kind={actionSpaceKind}
      class="mb-3 rounded border border-violet-500/40 bg-violet-900/20 px-3 py-2 text-sm text-violet-200"
    >
      {decisionPrompt}
    </p>
  {/if}

  {#if selectedTargetId !== null}
    <p class="mb-3 text-xs text-slate-400">Filtered to actions for selected board target.</p>
  {/if}

  <div
    bind:this={actionList}
    class="space-y-2"
    role="group"
    aria-labelledby="action-panel-heading"
    aria-describedby={decisionPrompt ? 'decision-prompt' : undefined}
  >
    {#if actions.length === 0}
      <p class="text-sm text-slate-400">No actions available.</p>
    {:else}
      {#each actions as action}
        <button
          data-testid="action-option"
          data-offer-id={action.index}
          data-action-type={action.type}
          data-action-description={action.description}
          aria-label={action.description}
          aria-describedby={decisionPrompt ? 'decision-prompt' : undefined}
          class={`w-full rounded border px-3 py-2 text-left text-sm transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-300 ${highlightedActionIndexes.has(action.index) ? 'border-amber-300 bg-slate-800' : 'border-slate-600 bg-slate-900 hover:border-blue-400 hover:bg-slate-800'} ${disabled ? 'cursor-not-allowed opacity-60' : ''}`}
          onmouseenter={() => onHoverAction?.(action)}
          onmouseleave={() => onHoverAction?.(null)}
          onfocus={() => onHoverAction?.(action)}
          onblur={() => onHoverAction?.(null)}
          onclick={() => onSelectAction?.(action)}
          {disabled}
        >
          <div class="font-medium">{action.description}</div>
          <div aria-hidden="true" class="mt-1 text-xs text-slate-400">{action.type}</div>
        </button>
      {/each}
    {/if}
  </div>
</aside>
