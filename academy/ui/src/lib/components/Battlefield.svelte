<script lang="ts">
  import { groupBattlefield, type BattlefieldLane } from '$lib/battlefield-layout';
  import type { CardPreview, PermanentState } from '$lib/types';
  import BattlefieldZone from './BattlefieldZone.svelte';

  interface Props {
    permanents: PermanentState[];
    /** Every visible permanent at the table, for attachments across seats. */
    tablePermanents: PermanentState[];
    focusedIds?: Set<number>;
    clickableTargets?: Map<number, number[]>;
    onSelectTarget?: (id: number) => void;
    onHoverTarget?: (id: number | null) => void;
    onPreviewCard?: (card: CardPreview | null) => void;
  }
  let { permanents, tablePermanents, focusedIds = new Set<number>(), clickableTargets,
    onSelectTarget, onHoverTarget, onPreviewCard }: Props = $props();
  const groups = $derived(groupBattlefield(permanents));
  const lanes: { lane: BattlefieldLane; label: string }[] = [
    { lane: 'creatures', label: 'Creatures' },
    { lane: 'lands', label: 'Lands' },
    { lane: 'other', label: 'Other permanents' },
    { lane: 'tokens', label: 'Artifact tokens' },
  ];
</script>

<div class="battlefield-grid" data-testid="battlefield">
  {#each lanes.filter(({ lane }) => groups[lane].length > 0) as { lane, label }}
    <BattlefieldZone {lane} {label} permanents={groups[lane]} allPermanents={permanents} {tablePermanents}
      {focusedIds} {clickableTargets} {onSelectTarget} {onHoverTarget} {onPreviewCard} />
  {/each}
</div>
<div class="empty-lanes flex flex-wrap gap-x-3 type-caption text-ink-2">
  {#each lanes.filter(({ lane }) => groups[lane].length === 0) as { lane, label }}
    <span role="group" aria-label={label} data-battlefield-lane={lane}>{label} · 0</span>
  {/each}
</div>

<style>
  .battlefield-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.5rem;
    min-width: 0;
    flex: 1;
    grid-auto-rows: minmax(126px, 1fr);
  }
  .battlefield-grid:empty, .empty-lanes:empty { display: none; }
</style>
