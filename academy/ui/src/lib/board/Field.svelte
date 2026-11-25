<script lang="ts">
  /**
   * One battlefield row. It measures the space it is given and sizes its
   * cards with `fitRow`, so a crowded row shrinks, piles and wraps instead of
   * spilling under its neighbours. A tapped card lies on its side in the
   * width its pile reserves for it, centred on the upright cards' line.
   */
  import BoardCard from './BoardCard.svelte';
  import type { PlannedSource } from './payment';
  import { fitRow, footprint, pileShapes, pileWidth, unpilePlanned, type PermanentItem, type Pile, type RowLimits } from './view';

  /** Pixels a lean may pass the row's edge: the rows' padding, short of the other half's cards. */
  const LEAN_SLACK = 6;

  interface Props {
    piles: Pile[];
    label: string;
    limits: RowLimits;
    live: boolean;
    asked: Set<number>;
    /** Permanents that acted or were acted on since your last move. */
    touched: Set<number>;
    /** Objects the opponent's move being read out is about. */
    beat?: Set<number>;
    /** Damage dealt to each permanent since your last move. */
    damage: Map<number, number>;
    /** Sources the automatic payer would tap, by permanent id. */
    planned: Map<number, PlannedSource> | null;
    /** Copies to take out of their piles: the pending payment's own plan. */
    unpiled: ReadonlyMap<number, unknown> | null;
    toward: 'up' | 'down';
    align?: 'start' | 'end';
    names: Map<number, string>;
    /** The person may act now: their permanents nothing acts on step back (lands never do). */
    acting?: boolean;
    /** Permanents something on the stack targets. */
    targeted?: Set<number>;
    /** A combat draft's place for each creature it concerns, and what pressing it does. */
    draft?: { places: Map<number, 'can' | 'in' | 'armed'>; notes: Map<number, string> } | null;
    sources?: ReadonlySet<number>;
    drop?: ReadonlySet<number>;
    onactivate: (item: PermanentItem, element: HTMLElement) => void;
    onpreview: (item: PermanentItem | null, element: HTMLElement | null) => void;
  }

  let { sources, drop, piles, label, limits, live, asked, touched, beat = new Set(), damage, planned, unpiled, toward, align = 'start', names, acting = false, targeted = new Set(), draft = null, onactivate, onpreview }: Props = $props();

  const shown = $derived(unpiled === null ? piles : unpilePlanned(piles, unpiled));

  let width = $state(0);
  let height = $state(0);
  // A pixel of slack keeps rounding from wrapping a row that just fits.
  const shapes = $derived(pileShapes(shown));
  const fit = $derived(fitRow(shapes, width - 2, height, limits));
  const cardHeight = $derived(footprint(fit.card, true));
  // An attacker steps toward the player it attacks: a fifth of its height
  // (less when lines are stacked), but never further than the row's own free
  // space and padding, so it stops short of the defending row and its P/T
  // stays in sight however full that row is.
  const lean = $derived(Math.max(0, Math.min(
    Math.round(cardHeight * (fit.lines > 1 ? 0.08 : 0.2)),
    Math.floor((height - fit.lines * cardHeight - (fit.lines - 1) * 8) / 2) + LEAN_SLACK,
  )));

  /** Only the person's own permanents step back, and never a land: lands are tapped for mana, not chosen. */
  function idle(item: PermanentItem): boolean {
    if (!acting || toward !== 'up' || item.permanent.types?.is_land) return false;
    return draft ? !draft.places.has(item.permanent.id) : item.offers.length === 0;
  }

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
  style:--lean={`${lean}px`}
  style:gap={`${limits.gap}px`}
>
  {#if width > 0}
    {#each shown as pile, at (pile.key)}
      {@const members = pile.items.flatMap(item => [item, ...item.attached.map(attached => ({ ...attached, host: item }))])}
      <div class="pile" style:width={`${pileWidth(shapes[at], fit.card, fit.fan)}px`} style:height={`${cardHeight}px`}>
        {#each members as item, index (item.permanent.id)}
          <!-- A leaning attacker paints over its neighbours and the seam. -->
          <div class="member" style:left={`${index * fit.fan}px`} style:top={item.permanent.tapped ? `${Math.round((cardHeight - fit.card) / 2)}px` : '0px'}
            style:z-index={item.permanent.attacking ? members.length + 20 : members.length - index}>
            <BoardCard
              name={item.permanent.name ?? 'Unknown permanent'}
              width={fit.card}
              shape="field"
              permanent={item.permanent}
              offers={draft ? [] : item.offers}
              {live}
              draft={draft?.places.get(item.permanent.id) ?? null}
              note={draft?.notes.get(item.permanent.id) ?? null}
              connectSource={sources?.has(item.permanent.id)} drop={drop?.has(item.permanent.id)}
              asked={asked.has(item.permanent.id)}
              touched={touched.has(item.permanent.id)}
              beat={beat.has(item.permanent.id)}
              dealt={damage.get(item.permanent.id) ?? 0}
              planned={planned?.get(item.permanent.id) ?? null}
              paid={item.paid}
              side={toward === 'up' ? 'you' : 'opponent'}
              {toward}
              count={index === 0 && pile.items.length > 1 ? pile.items.length : null}
              piled={index > 0 && !('host' in item)}
              dim={idle(item) ? 'idle' : null}
              targeted={targeted.has(item.permanent.id)}
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
    display: flex; flex-wrap: wrap; align-content: center; align-items: center;
  }
  .field.end { align-content: center; }
  .field.overflow { flex-wrap: nowrap; overflow-x: auto; }
  .pile { position: relative; flex: none; }
  .member { position: absolute; top: 0; }
  .member:hover, .member:focus-within { z-index: 50 !important; }
</style>
