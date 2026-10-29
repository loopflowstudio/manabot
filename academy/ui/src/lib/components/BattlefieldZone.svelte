<script lang="ts">
  import { onMount } from 'svelte';
  import { battlefieldPageLayout } from '$lib/battlefield-layout';
  import type { CardPreview, PermanentState } from '$lib/types';
  import PermanentRow from './PermanentRow.svelte';

  interface Props {
    label: string;
    lane: string;
    permanents: PermanentState[];
    allPermanents: PermanentState[];
    tablePermanents?: PermanentState[];
    focusedIds?: Set<number>;
    clickableTargets?: Map<number, number[]>;
    onSelectTarget?: (id: number) => void;
    onHoverTarget?: (id: number | null) => void;
    onPreviewCard?: (card: CardPreview | null) => void;
  }
  let { label, lane, permanents, allPermanents, tablePermanents = allPermanents, focusedIds = new Set<number>(),
    clickableTargets, onSelectTarget, onHoverTarget, onPreviewCard }: Props = $props();
  let box: HTMLDivElement;
  let width = $state(120);
  let height = $state(100);
  let page = $state(0);
  const layout = $derived(battlefieldPageLayout(width, height, permanents.length));
  const pages = $derived(Math.max(1, Math.ceil(permanents.length / layout.capacity)));
  const visible = $derived(permanents.slice(page * layout.capacity, (page + 1) * layout.capacity));
  $effect(() => { if (page >= pages) page = pages - 1; });
  $effect(() => {
    const index = permanents.findIndex(permanent => focusedIds.has(permanent.id));
    if (index >= 0) page = Math.floor(index / layout.capacity);
  });
  onMount(() => {
    const observer = new ResizeObserver(([entry]) => {
      width = entry.contentRect.width;
      height = entry.contentRect.height;
    });
    observer.observe(box);
    return () => observer.disconnect();
  });
</script>

<div role="group" aria-label={label} data-battlefield-lane={lane} class="lane min-h-0 min-w-0 rounded border border-line p-2">
  <h3 class="type-rubric truncate text-ink-2" title={label}>{label} · {permanents.length}</h3>
  <div bind:this={box} class="min-h-0 overflow-hidden p-1">
    {#if permanents.length}
      <PermanentRow permanents={visible} {allPermanents} {tablePermanents} cardWidth={layout.cardWidth}
        {focusedIds} {clickableTargets} {onSelectTarget} {onHoverTarget} {onPreviewCard} />
    {:else}<p class="type-caption text-ink-3">Empty</p>{/if}
  </div>
  <div class="flex min-h-6 items-center justify-between gap-1 type-caption">
    {#if pages > 1}
      <button aria-label={`Previous ${label.toLowerCase()}`} disabled={page === 0} onclick={() => page -= 1} class="min-h-6 min-w-6 px-1 underline disabled:opacity-30">←</button>
      <span role="status" class="text-ink-2">{page + 1}/{pages}</span>
      <button aria-label={`Next ${label.toLowerCase()}`} disabled={page + 1 >= pages} onclick={() => page += 1} class="min-h-6 min-w-6 px-1 underline disabled:opacity-30">→</button>
    {/if}
  </div>
</div>

<style>
  .lane { display: grid; grid-template-rows: auto minmax(0, 1fr) 1.5rem; }
</style>
