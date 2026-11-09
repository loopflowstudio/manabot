<script lang="ts">
  import { copyLabel } from '$lib/decision-labels';
  import { permanentNotes } from '$lib/permanent-notes';
  import type { CardPreview, PermanentState } from '$lib/types';

  import Card from './Card.svelte';

  // A staff of permanents on the sheet: just the cards on their line —
  // the zone rubric lives in the sheet's margin column.
  interface Props {
    permanents?: PermanentState[];
    allPermanents?: PermanentState[];
    /** Every visible permanent at the table, for attachments across seats. */
    tablePermanents?: PermanentState[];
    cardWidth?: number;
    focusedIds?: Set<number>;
    clickableTargets?: Map<number, number[]>;
    onSelectTarget?: (objectId: number) => void;
    onHoverTarget?: (objectId: number | null) => void;
    onPreviewCard?: (card: CardPreview | null) => void;
  }

  let {
    permanents = [],
    allPermanents = permanents,
    tablePermanents = allPermanents,
    cardWidth = undefined,
    focusedIds = new Set<number>(),
    clickableTargets = undefined,
    onSelectTarget = undefined,
    onHoverTarget = undefined,
    onPreviewCard = undefined,
  }: Props = $props();
</script>

<div class="flex flex-wrap items-end gap-1.5" style:--card-width={cardWidth === undefined ? undefined : `${cardWidth}px`}>
  {#if permanents.length === 0}
    <div class="type-caption px-1 py-1 text-ink-3">No permanents</div>
  {/if}

  {#each permanents as permanent}
    <Card
      name={permanent.name ?? 'Unknown Permanent'}
      copyDescription={copyLabel(allPermanents, permanent.id)}
      power={permanent.power === 0 && permanent.toughness === 0 ? null : permanent.power}
      toughness={permanent.power === 0 && permanent.toughness === 0 ? null : permanent.toughness}
      focused={focusedIds.has(permanent.id)}
      clickable={onSelectTarget !== undefined}
      tapped={permanent.tapped}
      dimmed={permanent.summoning_sick}
      counters={permanent.plus1_counters}
      temporarySubtypes={permanent.temp_subtypes ?? []}
      notes={permanentNotes(permanent, tablePermanents)}
      damage={permanent.damage}
      onSelect={() => onSelectTarget?.(permanent.id)}
      onHoverStart={() => {
        onHoverTarget?.(clickableTargets?.has(permanent.id) ? permanent.id : null);
        onPreviewCard?.({
          name: permanent.name,
          text_box: permanent.text_box,
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
