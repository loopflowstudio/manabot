<script lang="ts">
  import { stageLabel, subjectLabel } from '$lib/decision-labels';
  import type { Observation } from '$lib/types';
  let { observation, actor = null }: { observation: Observation; actor?: number | null } = $props();
  const context = $derived(observation.decision_context);
</script>

<section aria-label="Decision context" class="space-y-3">
  <h2 class="type-title text-display">Who decides · {context ? `Seat ${context.actor + 1}` : actor !== null ? `Seat ${actor + 1}` : observation.game_over ? 'Game over' : 'Decision details unavailable'}</h2>
  <div class="border-l-2 border-action pl-3" data-testid="decision-context">
    <h3 class="font-medium">What is happening</h3>
    {#if context}
      <p class="type-caption capitalize">{stageLabel(context.stage)}</p>
      <p class="type-caption">{context.instruction}</p>
      {#if context.source}<p>Source: {subjectLabel(observation, context.source)}</p>{/if}
      {#if context.chosen_mode !== null}<p>Chosen mode: {context.chosen_mode + 1}</p>{/if}
      {#each context.targets as target}<p>Selected target {target.clause + 1}: {subjectLabel(observation, target.object)}</p>{/each}
      {#if context.reserved_sacrifice}<p>Reserved, unpaid sacrifice: {subjectLabel(observation, context.reserved_sacrifice)}</p>{/if}
      {#if context.mana_cost}<p>Mana to pay: {context.mana_cost} · available pools are on the board</p>{/if}
    {:else}
      <p class="type-caption text-ink-2">Details not recorded.</p>
    {/if}
  </div>
  {#if context?.recent_facts.length}
    <details><summary class="type-rubric">What happened · recent actions</summary>
      <ol class="type-caption list-decimal pl-5">{#each context.recent_facts as fact}<li>{fact}</li>{/each}</ol>
    </details>
  {/if}
</section>
