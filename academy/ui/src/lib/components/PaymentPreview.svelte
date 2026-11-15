<script lang="ts">
  import { visibleCardLabel } from '$lib/decision-labels';
  import type { Observation } from '$lib/types';
  import { manaText, pooledManaText } from '$lib/mana';
  import type { PreviewState } from '$lib/payment-preview';
  interface Props {
    state: PreviewState | null;
    observation: Observation;
    onClose: () => void;
    onRetry: () => void;
  }
  let { state, observation, onClose, onRetry }: Props = $props();
  const scope = $derived(state?.preview?.scope);
  const activationFee = $derived(scope === 'activation_fee');
</script>
<!-- A fixed height: changing the proposal on hover must never move the offers
     above it, or the pointer would land on a different offer. -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard access to the scrolling payment explanation) -->
<section aria-label="Automatic payment preview" class="h-28 shrink-0 overflow-y-auto border-t border-line py-2 type-caption" tabindex="0" data-testid="payment-preview">
  <p class="type-rubric text-plains-ink">{activationFee ? 'Activation fee proposal' : scope === 'spell_cost' ? 'Spell cost proposal' : 'Payment proposal'}</p>
  {#if !state}
    <p class="text-ink-2">Point to or focus a choice to see what the automatic payer would tap.</p>
  {:else}
    <p>{state.label}</p>
    {#if state.loading}
      <p role="status">Reading automatic payment…</p>
    {:else if state.error}
      <p role="status">Payment preview unavailable. Legal actions remain available.</p>
      <button class="btn btn-ghost btn-sm" onclick={onRetry}>Retry preview</button>
    {:else if state.preview}
      {@const preview = state.preview}
      <p role="status">{preview.explanation}</p>
      {#if preview.cost}
        <p>{activationFee ? 'Activation fee' : 'Mana cost'}: {preview.cost}</p>
      {/if}
      {#if preview.state === 'exact'}
        {#if preview.sources.length}
          <ol class="list-decimal pl-5">
            {#each preview.sources as source}
              <li>
                {source.source.kind === 'object' ? visibleCardLabel(observation, source.source.id.entity) ?? source.label : source.label}: {manaText(source.output.amounts)}
                {#each source.output.classes as part}
                  <span> · {pooledManaText(part)}</span>
                {/each}
              </li>
            {/each}
          </ol>
        {:else}
          <p>No automatic source taps needed.</p>
        {/if}
        {#if preview.pool_after_production}
          <p>{activationFee ? 'Pool before fee spending' : 'Pool after production'}: {manaText(preview.pool_after_production.amounts)}</p>
        {/if}
        {#if preview.pool_after_spending}
          <p>{activationFee ? 'Pool after fee, before ability output or effects' : scope === 'spell_cost' ? 'Pool after cost, before spell effects' : 'Pool after spending'}: {manaText(preview.pool_after_spending.amounts)}</p>
        {/if}
      {/if}
      {#if preview.continuation.ability_output}
        <p>Choose the ability's output after activation when prompted.</p>
      {/if}
      {#if preview.continuation.outer_payment}
        <p>The outer payment still follows.</p>
      {/if}
    {/if}
  {/if}
  <p class="text-ink-2">Preview spends nothing. Mana activations commit; closing this proposal does not untap or refund them.</p>
  {#if state}<button class="btn btn-ghost btn-sm" onclick={onClose}>Close preview</button>{/if}
</section>
