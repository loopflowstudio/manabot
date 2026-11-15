<script lang="ts">
  import { visibleCardLabel } from '$lib/decision-labels';
  import type { Observation } from '$lib/types';
  let { observation }: { observation: Observation } = $props();
  function subject(id: number): string {
    if (id === observation.agent.id) return `Seat ${observation.agent.player_index + 1} payment`;
    if (id === observation.opponent.id) return `Seat ${observation.opponent.player_index + 1} payment`;
    return visibleCardLabel(observation, id) ?? 'Visible object';
  }
</script>
{#if observation.unavailable_actions === undefined}
  <p class="type-caption text-ink-2">Unavailable-action explanations not recorded.</p>
{:else if observation.unavailable_actions.length}
  <details class="border-b border-line py-2">
    <summary class="type-label cursor-pointer">Why unavailable?</summary>
    <ul class="type-caption space-y-2 py-2">{#each observation.unavailable_actions as refusal}<li>{subject(refusal.object)}: {refusal.message}</li>{/each}</ul>
  </details>
{/if}
