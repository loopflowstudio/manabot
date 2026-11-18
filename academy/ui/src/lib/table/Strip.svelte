<script lang="ts">
  /**
   * The game's decisions as one line under the board, folded to a single row
   * until opened. Each decision is a tick, coloured by whose it was; where the
   * chooser's value was recorded it is a point on that chooser's line instead.
   * Turns are ruled and numbered. In a recording a click jumps to the nearest
   * position (`onjump`); the keyboard path is the transport and the move list.
   * Live play has no past positions to show yet, so its marks only count.
   */
  import { labelledTurns, type StripMark } from './replay';

  interface Props {
    marks: StripMark[];
    selected: number | null;
    open: boolean;
    ontoggle: () => void;
    onjump?: (at: number) => void;
    /** Why marks cannot be opened, when they cannot. */
    note?: string | null;
  }

  let { marks, selected, open, ontoggle, onjump, note = null }: Props = $props();

  const HEIGHT = 44;
  const decisions = $derived(marks.filter(mark => mark.decision));
  const values = $derived(decisions.map(mark => mark.value).filter((value): value is number => value !== null));
  // Values keep their own scale: probabilities on [0, 1], signed values on [-1, 1], else their range.
  const range = $derived.by(() => {
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

  function jump(event: MouseEvent): void {
    if (!onjump || !marks.length) return;
    const box = (event.currentTarget as SVGElement).getBoundingClientRect();
    onjump(Math.round(((event.clientX - box.left) / box.width) * span));
  }
</script>

<section class="strip" class:open aria-label="Decisions" data-testid="decision-strip" data-open={open}>
  <button type="button" class="head type-rubric" aria-expanded={open} onclick={ontoggle} data-testid="strip-toggle">
    <span class="chevron" aria-hidden="true">{open ? '▾' : '▸'}</span>
    Decisions · {decisions.length}{#if values.length}<span class="text-ink-3"> · value recorded at {values.length}</span>{/if}
    {#if note}<span class="note type-caption text-ink-3">{note}</span>{/if}
  </button>
  {#if open}
    <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions (the transport and the move list are the keyboard path to every position) -->
    <svg class="line" class:jumps={!!onjump} viewBox={`0 0 1000 ${HEIGHT}`} preserveAspectRatio="none" onclick={jump} aria-hidden="true">
      {#if range && range.low < 0}<line class="zero" x1="0" x2="1000" y1={y(0)} y2={y(0)} />{/if}
      {#each marks as mark (mark.key)}
        {#if mark.turnStart}<line class="turn" x1={x(mark.at)} x2={x(mark.at)} y1="0" y2={HEIGHT} />{/if}
        {#if mark.decision && mark.value === null}
          <line class="tick" class:theirs={mark.side === 'opponent'} x1={x(mark.at)} x2={x(mark.at)} y1={mark.side === 'opponent' ? 6 : HEIGHT / 2} y2={mark.side === 'opponent' ? HEIGHT / 2 : HEIGHT - 6}><title>{mark.label}</title></line>
        {/if}
      {/each}
      <polyline class="value" points={line('you')} />
      <polyline class="value theirs" points={line('opponent')} />
      {#if selected !== null && marks[selected]}<line class="here" x1={x(selected)} x2={x(selected)} y1="0" y2={HEIGHT} />{/if}
    </svg>
    <div class="turns" aria-hidden="true">
      {#each labelledTurns(marks, 0.02) as mark (mark.key)}<span style:left={`${(mark.at / span) * 100}%`}>{mark.turn ?? ''}</span>{/each}
    </div>
  {/if}
</section>

<style>
  .strip { padding: 0 12px 4px; border-top: 1px solid color-mix(in srgb, var(--border) 60%, transparent); }
  .head { display: flex; align-items: baseline; gap: 6px; width: 100%; padding: 4px 0; color: var(--text-secondary); text-align: left; }
  .head:hover { color: var(--text); }
  .chevron { width: 10px; }
  .note { margin-left: auto; text-transform: none; letter-spacing: 0; }
  .line { display: block; width: 100%; height: 44px; background: var(--bg-field); border-radius: 4px; }
  .line.jumps { cursor: pointer; }
  .turn { stroke: var(--border); stroke-width: 1; vector-effect: non-scaling-stroke; }
  .zero { stroke: var(--border-strong); stroke-dasharray: 3 3; vector-effect: non-scaling-stroke; }
  .tick { stroke: var(--info-text, var(--accent)); stroke-width: 2; vector-effect: non-scaling-stroke; }
  .tick.theirs { stroke: var(--text-tertiary); }
  .value { fill: none; stroke: var(--info-text, var(--accent)); stroke-width: 1.5; vector-effect: non-scaling-stroke; }
  .value.theirs { stroke: var(--text-tertiary); }
  .here { stroke: var(--accent); stroke-width: 2; vector-effect: non-scaling-stroke; }
  .turns { position: relative; height: 12px; }
  .turns span { position: absolute; transform: translateX(2px); font: 600 9px/12px var(--font-mono); color: var(--text-tertiary); }
</style>
