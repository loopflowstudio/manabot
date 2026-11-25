<script lang="ts">
  /**
   * The game's decisions as one line under the board, drawn while open; its
   * toggle sits in the stop bar's row (`Board`), so a folded strip costs the
   * board no height. Where a chooser's value was recorded it is a point on that
   * chooser's line; turns are ruled and numbered. With an analysis the strip is
   * a chart: taller, a 0–100% axis with gridlines, and hovering names the
   * nearest point (its turn, where it was, its score). In a recording the open strip is the
   * recording's one navigable line (`onjump`): a slider over it, so a click,
   * a drag or the arrow keys move to a position. Live play has no past
   * positions to show yet, so its marks only count.
   */
  import { labelledTurns, type StripMark } from './replay';

  interface Props {
    marks: StripMark[];
    selected: number | null;
    onjump?: (at: number) => void;
    /** Why marks cannot be opened, when they cannot. */
    note?: string | null;
    /** With an analysis: the caption, and values are 0–1 scores on a fixed 0–100 axis. */
    scale?: string | null;
  }

  let { marks, selected, onjump, note = null, scale = null }: Props = $props();

  const HEIGHT = $derived(scale !== null ? 80 : 44);
  const decisions = $derived(marks.filter(mark => mark.decision));
  const values = $derived(decisions.map(mark => mark.value).filter((value): value is number => value !== null));
  // An analysis's scores sit on a fixed 0–100 axis. Recorded values keep their
  // own scale: probabilities on [0, 1], signed values on [-1, 1], else their range.
  const range = $derived.by(() => {
    if (scale !== null) return { low: 0, high: 1 };
    if (!values.length) return null;
    const low = Math.min(...values), high = Math.max(...values);
    if (low >= 0 && high <= 1) return { low: 0, high: 1 };
    if (low >= -1 && high <= 1) return { low: -1, high: 1 };
    return { low, high: high === low ? low + 1 : high };
  });
  const span = $derived(Math.max(1, marks.length - 1));
  const x = (at: number): number => (at / span) * 1000;
  const y = (value: number): number => range ? 4 + (1 - (value - range.low) / (range.high - range.low)) * (HEIGHT - 8) : HEIGHT / 2;
  const line = (side: 'you' | 'opponent'): string => decisions
    .filter(mark => mark.side === side && mark.value !== null).map(mark => `${x(mark.at)},${y(mark.value!)}`).join(' ');

  const turn = $derived(selected !== null ? marks[selected]?.turn ?? null : null);

  // Hover: the valued decision nearest the pointer, named with its score.
  const valued = $derived(decisions.filter(mark => mark.value !== null));
  let hover = $state<{ mark: StripMark; left: number } | null>(null);
  function track(event: PointerEvent): void {
    const box = (event.currentTarget as HTMLElement).getBoundingClientRect();
    if (!valued.length || box.width === 0) { hover = null; return; }
    const at = ((event.clientX - box.left) / box.width) * span;
    const mark = valued.reduce((best, item) => Math.abs(item.at - at) < Math.abs(best.at - at) ? item : best);
    hover = { mark, left: (mark.at / span) * 100 };
  }
  const readout = (mark: StripMark): string => scale !== null
    ? `${mark.label} · ${Math.round(mark.value! * 100)}%`
    : `${mark.label} · ${mark.value!.toFixed(2)}`;
</script>

<section class="strip" aria-label="Decisions" data-testid="decision-strip" data-open="true">
  {#if values.length || note || scale !== null}
    <p class="caption type-caption text-ink-3" data-testid="strip-caption">{#if scale !== null}{scale}{:else if values.length}Value recorded at {values.length} of {decisions.length} decisions{/if}{#if note}<span class="note">{note}</span>{/if}</p>
  {/if}
  <div class="plot" class:chart={scale !== null}>
  {#if scale !== null}
    <div class="axis type-caption text-ink-3" aria-hidden="true" style:height={`${HEIGHT}px`}>
      {#each [1, 0.5, 0] as tick (tick)}<span style:top={`${y(tick)}px`}>{tick * 100}%</span>{/each}
    </div>
  {/if}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="lane" class:jumps={!!onjump} onpointermove={track} onpointerleave={() => (hover = null)}>
  <svg class="line" style:height={`${HEIGHT}px`} viewBox={`0 0 1000 ${HEIGHT}`} preserveAspectRatio="none" aria-hidden="true">
    {#if range && range.low < 0}<line class="zero" x1="0" x2="1000" y1={y(0)} y2={y(0)} />{/if}
    {#if scale !== null}
      {#each [0.25, 0.75] as grid (grid)}<line class="grid" x1="0" x2="1000" y1={y(grid)} y2={y(grid)} />{/each}
      <line class="zero" data-testid="strip-midline" x1="0" x2="1000" y1={y(0.5)} y2={y(0.5)} />
    {/if}
    {#each marks as mark (mark.key)}
      {#if mark.turnStart}<line class="turn" x1={x(mark.at)} x2={x(mark.at)} y1="0" y2={HEIGHT} />{/if}
    {/each}
    <polyline class="value" points={line('you')} data-testid="strip-value" data-points={decisions.filter(mark => mark.side === 'you' && mark.value !== null).length} />
    {#if scale !== null}
      {#each valued as mark (mark.key)}<circle class="point" class:theirs={mark.side === 'opponent'} cx={x(mark.at)} cy={y(mark.value!)} r="1.75" />{/each}
    {/if}
    {#each decisions as mark (mark.key)}
      {#if mark.differs && mark.value !== null}<circle class="differs" cx={x(mark.at)} cy={y(mark.value)} r="2.5"><title>{mark.label} · chose otherwise</title></circle>{/if}
    {/each}
    <polyline class="value theirs" points={line('opponent')} />
    {#if selected !== null && marks[selected]}<line class="here" x1={x(selected)} x2={x(selected)} y1="0" y2={HEIGHT} />{/if}
    {#if hover}<line class="hover" x1={x(hover.mark.at)} x2={x(hover.mark.at)} y1="0" y2={HEIGHT} />{/if}
  </svg>
  {#if hover}
    <span class="dot" style:left={`${hover.left}%`} style:top={`${y(hover.mark.value!)}px`}></span>
    <p class="readout type-caption" data-testid="strip-readout" class:flip={hover.left > 60} style:left={`${hover.left}%`}>{readout(hover.mark)}</p>
  {/if}
  {#if onjump && selected !== null}
    <input type="range" class="seek" aria-label="Replay position" min="0" max={Math.max(0, marks.length - 1)} value={selected}
      aria-valuetext={`Position ${selected + 1} of ${marks.length}${turn !== null ? `, turn ${turn}` : ''}`}
      oninput={event => onjump(Number(event.currentTarget.value))} />
  {/if}
  </div>
  </div>
  <div class="turns" class:chart={scale !== null} aria-hidden="true">
    {#each labelledTurns(marks, 0.02) as mark (mark.key)}<span style:left={`${(mark.at / span) * 100}%`}>{mark.turn ?? ''}</span>{/each}
  </div>
</section>

<style>
  .strip { padding: 8px 12px 4px; border-top: 1px solid color-mix(in srgb, var(--border) 60%, transparent); }
  .caption { display: flex; gap: 12px; margin: 0 0 4px; }
  .note { margin-left: auto; }
  .lane { position: relative; border-radius: var(--radius-sm); }
  .lane:focus-within { outline: 2px solid var(--accent); outline-offset: 1px; }
  .plot { display: flex; gap: 6px; }
  .plot > .lane { flex: 1; min-width: 0; }
  .axis { position: relative; width: 34px; flex: none; }
  .axis span { position: absolute; right: 0; transform: translateY(-50%); font-variant-numeric: tabular-nums; line-height: 1; }
  .line { display: block; width: 100%; background: var(--bg-field); border-radius: var(--radius-sm); }
  .grid { stroke: var(--border); stroke-width: 1; vector-effect: non-scaling-stroke; opacity: .7; }
  .hover { stroke: var(--border-strong); stroke-width: 1; vector-effect: non-scaling-stroke; }
  .dot { position: absolute; width: 7px; height: 7px; margin: -3.5px 0 0 -3.5px; border-radius: 50%; background: var(--info-text, var(--accent)); pointer-events: none; }
  .readout { position: absolute; bottom: calc(100% + 4px); transform: translateX(-6px); margin: 0; padding: 2px 8px; white-space: nowrap; pointer-events: none;
    background: var(--bg-surface, var(--bg-field)); color: var(--text); border: 1px solid var(--border-strong); border-radius: var(--radius-sm); z-index: 2; }
  .readout.flip { transform: translateX(calc(-100% + 6px)); }
  .turns.chart { margin-left: 40px; }
  /* The slider is the line's input: invisible, the full lane, a hairline thumb so
     a click lands on the position drawn under it. */
  .seek { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: pointer; }
  .seek::-webkit-slider-thumb { appearance: none; width: 1px; height: 80px; }
  .seek::-moz-range-thumb { width: 1px; height: 80px; border: 0; }
  .turn { stroke: var(--border); stroke-width: 1; vector-effect: non-scaling-stroke; }
  .zero { stroke: var(--border-strong); stroke-dasharray: 3 3; vector-effect: non-scaling-stroke; }
  .value { fill: none; stroke: var(--info-text, var(--accent)); stroke-width: 1.5; vector-effect: non-scaling-stroke; }
  .value.theirs { stroke: var(--text-tertiary); }
  /* Every read decision is a dot, so one reading alone still shows. */
  .point { fill: var(--info-text, var(--accent)); }
  .point.theirs { fill: var(--text-tertiary); }
  /* Where you and the policy chose differently: a neutral fact, not a fault. */
  .differs { fill: var(--text-secondary); vector-effect: non-scaling-stroke; }
  .here { stroke: var(--accent); stroke-width: 2; vector-effect: non-scaling-stroke; }
  .turns { position: relative; height: 16px; }
  .turns span { position: absolute; transform: translateX(2px); font: 400 12px/16px var(--font-sans); color: var(--text-tertiary); }
</style>
