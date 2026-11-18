<script lang="ts">
  /**
   * One battlefield row. It measures the space it is given and sizes its
   * cards with `fitRow`, so a crowded row shrinks, piles and wraps instead of
   * spilling under its neighbours.
   */
  import TableCard from './TableCard.svelte';
  import { fitRow, pileCounts, type PermanentItem, type Pile, type RowLimits } from './view';

  interface Props {
    piles: Pile[];
    label: string;
    limits: RowLimits;
    live: boolean;
    asked: Set<number>;
    /** Permanents that acted or were acted on since your last move. */
    touched: Set<number>;
    toward: 'up' | 'down';
    align?: 'start' | 'end';
    names: Map<number, string>;
    onactivate: (item: PermanentItem, element: HTMLElement) => void;
    onpreview: (item: PermanentItem | null, element: HTMLElement | null) => void;
  }

  let { piles, label, limits, live, asked, touched, toward, align = 'start', names, onactivate, onpreview }: Props = $props();

  let width = $state(0);
  let height = $state(0);
  // A pixel of slack keeps rounding from wrapping a row that just fits.
  const fit = $derived(fitRow(pileCounts(piles), width - 2, height, limits));
  const cardHeight = $derived(Math.round(fit.card * 7 / 5));

  function blocked(item: PermanentItem): string | null {
    const attacker = item.permanent.blocking;
    return attacker === undefined ? null : names.get(attacker) ?? null;
  }
</script>

<div
  class="field"
  class:end={align === 'end'}
  class:overflow={fit.overflow}
  role="group"
  aria-label={label}
  bind:clientWidth={width}
  bind:clientHeight={height}
  data-testid="field-row"
  data-card-width={fit.card}
  data-lines={fit.lines}
  style:--lean={fit.lines > 1 ? '8%' : '20%'}
>
  {#if width > 0}
    {#each piles as pile (pile.key)}
      {@const members = pile.items.flatMap(item => [item, ...item.attached.map(attached => ({ ...attached, host: item }))])}
      {@const face = pile.tapped ? cardHeight : fit.card}
      <div class="pile" style:width={`${face + (members.length - 1) * fit.fan}px`} style:height={`${cardHeight}px`}>
        {#each members as item, index (item.permanent.id)}
          <div class="member" style:left={`${index * fit.fan}px`} style:z-index={members.length - index}>
            <TableCard
              name={item.permanent.name ?? 'Unknown permanent'}
              width={fit.card}
              permanent={item.permanent}
              offers={item.offers}
              {live}
              asked={asked.has(item.permanent.id)}
              touched={touched.has(item.permanent.id)}
              side={toward === 'up' ? 'you' : 'opponent'}
              {toward}
              count={index === 0 && pile.items.length > 1 ? pile.items.length : null}
              blocks={blocked(item)}
              onactivate={element => onactivate(item, element)}
              onpreview={element => onpreview(element ? item : null, element)}
            />
          </div>
        {/each}
      </div>
    {/each}
  {/if}
</div>

<style>
  .field {
    position: relative; min-height: 0; min-width: 0; height: 100%;
    display: flex; flex-wrap: wrap; align-content: center; align-items: center; gap: 8px 8px;
  }
  .field.end { align-content: center; }
  .field.overflow { flex-wrap: nowrap; overflow-x: auto; }
  .pile { position: relative; flex: none; }
  .member { position: absolute; top: 0; }
  .member:hover, .member:focus-within { z-index: 50 !important; }
</style>
