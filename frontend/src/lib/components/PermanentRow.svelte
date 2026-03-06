<script lang="ts">
  import type { PermanentState } from '$lib/types';

  import Card from './Card.svelte';

  export let label: string;
  export let permanents: PermanentState[] = [];
  export let focusedIds = new Set<number>();
  export let clickableTargets: Map<number, number[]> | undefined = undefined;
  export let onSelectTarget: ((objectId: number) => void) | undefined = undefined;
  export let onHoverTarget: ((objectId: number | null) => void) | undefined = undefined;
  export let onPreviewCard:
    | ((card: { name: string | null; power: number | null; toughness: number | null } | null) => void)
    | undefined = undefined;
</script>

<div class="min-h-24">
  <div class="mb-2 text-xs uppercase tracking-wide text-slate-500">{label}</div>
  <div class="flex flex-wrap gap-3">
    {#if permanents.length === 0}
      <div class="rounded border border-dashed border-slate-700 px-3 py-5 text-xs text-slate-500">No permanents</div>
    {/if}

    {#each permanents as permanent}
      <Card
        name={permanent.name ?? 'Unknown Permanent'}
        power={permanent.power}
        toughness={permanent.toughness}
        focused={focusedIds.has(permanent.id)}
        clickable={clickableTargets?.has(permanent.id) ?? false}
        tapped={permanent.tapped}
        dimmed={permanent.summoning_sick}
        damage={permanent.damage}
        onSelect={() => onSelectTarget?.(permanent.id)}
        onHoverStart={() => {
          onHoverTarget?.(clickableTargets?.has(permanent.id) ? permanent.id : null);
          onPreviewCard?.({
            name: permanent.name,
            power: permanent.power,
            toughness: permanent.toughness,
          });
        }}
        onHoverEnd={() => {
          onHoverTarget?.(null);
          onPreviewCard?.(null);
        }}
      />
    {/each}
  </div>
</div>
