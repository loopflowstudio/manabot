<script lang="ts">
  import { visibleCardLabel } from '$lib/decision-labels';
  import type { Observation } from '$lib/types';
  let { observation }: { observation: Observation } = $props();
  function subject(id: number): string {
    if (id === observation.agent.id) return `Seat ${observation.agent.player_index + 1} payment`;
    if (id === observation.opponent.id) return `Seat ${observation.opponent.player_index + 1} payment`;
    return visibleCardLabel(observation, id) ?? 'Visible object';
  }
  // Copies of one card in one zone with the same native refusal read as one
  // line; a copy with a different reason keeps its own exact label.
  const groups = $derived.by(() => {
    const byKey = new Map<string, {label: string; base: string; message: string; count: number}>();
    for (const refusal of observation.unavailable_actions ?? []) {
      const label = subject(refusal.object);
      const base = label.replace(/ · copy \d+ of \d+$/, '');
      const key = `${base}\u0000${refusal.message}`;
      const group = byKey.get(key);
      if (group) group.count += 1;
      else byKey.set(key, {label, base, message: refusal.message, count: 1});
    }
    return [...byKey.values()].map(group => ({
      subject: group.count > 1 ? `${group.base} · ${group.count} copies` : group.label,
      message: group.message,
    }));
  });
</script>
{#if observation.unavailable_actions === undefined}
  <p class="type-caption text-ink-2">Unavailable-action explanations not recorded.</p>
{:else if observation.unavailable_actions.length}
  <details class="border-b border-line py-2">
    <summary class="type-label cursor-pointer">Why unavailable?</summary>
    <ul class="type-caption space-y-2 py-2">{#each groups as group}<li>{group.subject}: {group.message}</li>{/each}</ul>
  </details>
{/if}
