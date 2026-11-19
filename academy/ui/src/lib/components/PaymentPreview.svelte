<script lang="ts">
  import { visibleCardLabel } from '$lib/decision-labels';
  import { manaText, pooledManaText } from '$lib/mana';
  import type { PreviewState } from '$lib/payment-preview';
  import type { Observation } from '$lib/types';

  interface Props {
    state: PreviewState | null;
    observation: Observation;
    chosenSourceFirst: boolean;
    onClear: () => void;
    onRetry: () => void;
  }
  let { state, observation, chosenSourceFirst, onClear, onRetry }: Props = $props();
  const preview = $derived(state?.preview);
  let region: HTMLElement | null = null;
  // Clear and Retry remove their own button; keep keyboard focus in the panel.
  function keepFocus(action: () => void): void {
    region?.focus();
    action();
  }
</script>

<!-- Keep this scroll region stable as the pointer moves between offers. -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard access to the scrolling payment explanation) -->
<section bind:this={region} aria-label="Payment preview" class="h-40 shrink-0 overflow-y-auto border-t border-line py-1.5 type-caption" tabindex="0" data-testid="payment-preview">
  <div class="flex items-baseline justify-between gap-2">
    <h3 class="type-rubric text-display">Payment preview</h3>
    {#if state}<button class="underline text-ink-2" aria-label="Clear preview" onclick={() => keepFocus(onClear)}>Clear</button>{/if}
  </div>
  {#if !state}
    <p class="text-ink-2">Focus or point to a payment choice.</p>
  {:else if state.loading}
    <p role="status">Reading automatic payment…</p>
  {:else if state.error}
    <p role="status">Couldn’t load this payment preview.</p>
    <button class="btn btn-ghost btn-sm" onclick={() => keepFocus(onRetry)}>Retry</button>
  {:else if preview}
    <p>{state.label}{#if preview.cost}{` · Cost ${preview.cost}`}{/if}</p>
    {#if preview.state === 'exact'}
      {#if preview.contributions.length}
        <p class="font-semibold">Already paid</p>
        <ul aria-label="Already paid" class="list-disc pl-5">
          {#each preview.contributions as paid}<li>{paid}</li>{/each}
        </ul>
      {/if}
      {#if preview.sources.length}
        <p class="font-semibold">Would tap</p>
        <ol aria-label="Would tap" class="list-decimal pl-5">
          {#each preview.sources as source, index}
            <li>
              {#if chosenSourceFirst && index === 0}Your choice · {/if}
              {source.source.kind === 'object' ? visibleCardLabel(observation, source.source.id.entity) ?? source.label : source.label} → {manaText(source.output.amounts)}
              {#each source.output.classes as part}<span> · {pooledManaText(part)}</span>{/each}
            </li>
          {/each}
        </ol>
      {:else}
        <p>No automatic source taps needed.</p>
      {/if}
      {#if preview.pool_after_spending}
        <p>Left over · {preview.pool_after_spending.amounts.some(Boolean) ? manaText(preview.pool_after_spending.amounts) : 'nothing floating'}</p>
        {#each preview.pool_after_spending.classes as part}<p>{pooledManaText(part)}</p>{/each}
      {:else}
        <p>Choose which mana to spend when prompted.</p>
      {/if}
      {#if preview.pool_after_production}
        <details open={!preview.pool_after_spending}>
          <summary>Pool after production · {manaText(preview.pool_after_production.amounts)}</summary>
          {#each preview.pool_after_production.classes as part}<p>{pooledManaText(part)}</p>{/each}
        </details>
      {/if}
    {:else}
      <p role="status">{preview.explanation}</p>
    {/if}
  {/if}
</section>
