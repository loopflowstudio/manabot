<script lang="ts">
  import type { PermanentState } from '$lib/types';

  import Card from './Card.svelte';

  // A staff of permanents on the sheet: just the cards on their line —
  // the zone rubric lives in the sheet's margin column.
  interface Props {
    permanents?: PermanentState[];
    focusedIds?: Set<number>;
    clickableTargets?: Map<number, number[]>;
    onSelectTarget?: (objectId: number) => void;
    onHoverTarget?: (objectId: number | null) => void;
    onPreviewCard?: (
      card: { name: string | null; power: number | null; toughness: number | null } | null,
    ) => void;
  }

  let {
    permanents = [],
    focusedIds = new Set<number>(),
    clickableTargets = undefined,
    onSelectTarget = undefined,
    onHoverTarget = undefined,
    onPreviewCard = undefined,
  }: Props = $props();
</script>

<div class="flex flex-wrap items-end gap-2">
  {#if permanents.length === 0}
    <div class="type-caption px-1 py-1 text-ink-3">No permanents</div>
  {/if}

  {#each permanents as permanent}
    <Card
      name={permanent.name ?? 'Unknown Permanent'}
      power={permanent.power === 0 && permanent.toughness === 0 ? null : permanent.power}
      toughness={permanent.power === 0 && permanent.toughness === 0 ? null : permanent.toughness}
      focused={focusedIds.has(permanent.id)}
      clickable={clickableTargets?.has(permanent.id) ?? false}
      tapped={permanent.tapped}
      dimmed={permanent.summoning_sick}
      counters={permanent.plus1_counters}
      damage={permanent.damage}
      onSelect={() => onSelectTarget?.(permanent.id)}
      onHoverStart={() => {
        onHoverTarget?.(clickableTargets?.has(permanent.id) ? permanent.id : null);
        onPreviewCard?.({
          name: permanent.name,
          power: permanent.power === 0 && permanent.toughness === 0 ? null : permanent.power,
          toughness:
            permanent.power === 0 && permanent.toughness === 0 ? null : permanent.toughness,
        });
      }}
      onHoverEnd={() => {
        onHoverTarget?.(null);
        onPreviewCard?.(null);
      }}
    />
  {/each}
</div>
