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
  class="min-w-0"
>
  <div class="mb-2 flex items-baseline justify-between gap-3">
    <div class="flex items-baseline gap-2.5">
      <h2 id="action-panel-heading" class="type-title text-display">Actions</h2>
      {#if disabled}
        <span role="status" aria-live="polite" aria-atomic="true" class="type-label rounded-full bg-swamp/20 px-2 py-1 text-ink-2">Game over</span>
      {:else if fastForwarding}
        <span role="status" aria-live="polite" aria-atomic="true" data-testid="auto-passing" class="type-label animate-pulse rounded-full bg-island/20 px-2 py-1 text-ink">Auto-passing…</span>
      {:else if actions.length > 0}
        <span role="status" aria-live="polite" aria-atomic="true" class="type-label whitespace-nowrap uppercase text-mountain-ink">Your move</span>
      {/if}
    </div>
    <div class="flex items-center gap-3">
      {#if selectedTargetId !== null}
        <button class="type-label inline-block py-1 text-ink-2 underline underline-offset-2 hover:text-ink" onclick={() => onClearSelection?.()}>
          Show all
        </button>
      {/if}
      <button
        data-testid="pass-turn"
        class="btn btn-secondary btn-sm"
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
      class="type-annotation mb-3 border-l-2 border-action py-0.5 pl-3 text-ink"
    >
      {decisionPrompt}
    </p>
  {/if}

  {#if selectedTargetId !== null}
    <p class="type-caption mb-3 text-ink-2">Filtered to actions for selected board target.</p>
  {/if}

  <div
    bind:this={actionList}
    class="space-y-2"
    role="group"
    aria-labelledby="action-panel-heading"
    aria-describedby={decisionPrompt ? 'decision-prompt' : undefined}
  >
    {#if actions.length === 0}
      <p class="type-caption text-ink-3">No actions available.</p>
    {:else}
      {#each actions as action}
        <button
          data-testid="action-option"
          data-offer-id={action.index}
          data-action-type={action.type}
          data-action-description={action.description}
          aria-label={action.description}
          aria-describedby={decisionPrompt ? 'decision-prompt' : undefined}
          class={`w-full rounded border px-3 py-2 text-left transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-action ${highlightedActionIndexes.has(action.index) ? 'border-plains-ink bg-panel' : 'border-line-strong bg-field hover:border-action hover:bg-panel'} ${disabled ? 'cursor-not-allowed opacity-60' : ''}`}
          onmouseenter={() => onHoverAction?.(action)}
          onmouseleave={() => onHoverAction?.(null)}
          onfocus={() => onHoverAction?.(action)}
          onblur={() => onHoverAction?.(null)}
          onclick={() => onSelectAction?.(action)}
          {disabled}
        >
          <div class="font-medium">{action.description}</div>
        </button>
      {/each}
    {/if}
  </div>
</aside>
