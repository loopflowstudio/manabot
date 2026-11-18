<script lang="ts">
  /**
   * The turn's steps along the bottom of the board, and the stops set on
   * them, laid out as MTGO does: one cell per step, your turn's stop above
   * the step's name and the opponent's below, a rule between phases, the
   * current step marked. Pressing a stop cycles it off → every time → once
   * (`nextStopState`); the Menu's checkboxes edit the same `StopsConfig` and
   * stay the full-size control. The stops are one toolbar: Tab reaches it
   * once, arrow keys move between them. Without `oncycle` (a recording) the
   * bar is read-only.
   */
  import type { Snippet } from 'svelte';
  import type { StopSide } from '$lib/types';
  import { nextStopState, STOP_STATE_WORDS, type PhaseTrack, type StopState } from './phases';

  interface Props {
    track: PhaseTrack;
    oncycle?: (side: StopSide, step: string) => void;
    /** Under the legend, in the rail's column: the decision strip's toggle. */
    aside?: Snippet;
  }

  let { track, oncycle, aside }: Props = $props();

  const sides: StopSide[] = ['my', 'opponent'];
  const SIDE_WORDS: Record<StopSide, string> = { my: 'your turn', opponent: "the opponent's turn" };
  const editable = $derived(oncycle !== undefined && !track.every);

  // Roving focus: one stop is in the Tab order; arrows move it.
  let focusAt = $state({ side: 0, step: 0 });
  let toolbar = $state<HTMLElement | null>(null);
  function onkeydown(event: KeyboardEvent): void {
    const moves: Record<string, [number, number]> = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0], Home: [0, -99], End: [0, 99] };
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    event.stopPropagation();
    const side = Math.max(0, Math.min(sides.length - 1, focusAt.side + move[0]));
    const step = Math.max(0, Math.min(track.marks.length - 1, focusAt.step + move[1]));
    focusAt = { side, step };
    toolbar?.querySelector<HTMLElement>(`[data-side="${sides[side]}"][data-step="${track.marks[step].key}"]`)?.focus();
  }

  function words(side: StopSide, label: string, state: StopState): string {
    if (track.every) return `Stop at ${label} on ${SIDE_WORDS[side]}: priority stops are off, every step stops`;
    return `Stop at ${label} on ${SIDE_WORDS[side]}: ${STOP_STATE_WORDS[state]}${editable ? `. Press for ${STOP_STATE_WORDS[nextStopState(state)]}` : ''}`;
  }
</script>

<div class="stopbar" data-testid="stop-bar">
  <!-- svelte-ignore a11y_interactive_supports_focus (focus is roving: one stop is tabbable) -->
  <div class="steps" role={editable ? 'toolbar' : 'list'} aria-label={editable ? 'Stops: arrow keys move, Space changes' : 'Steps of the turn'}
    bind:this={toolbar} onkeydown={editable ? onkeydown : undefined}>
    <span class="key" aria-hidden="true">
      <span class:active={track.active === 'my'}>▲ Your turn</span>
      <span class:active={track.active === 'opponent'}>▼ Their turn</span>
    </span>
    {#each track.marks as mark, step (mark.key)}
      <div class="cell" class:rule={mark.phaseStart} class:current={mark.current} role={editable ? 'group' : 'listitem'}
        aria-label={mark.label} aria-current={mark.current ? 'step' : undefined}>
        {#each sides as side, row (side)}
          {@const state = track.every ? 'every' : mark.state[side]}
          {#if editable}
            <button type="button" class={`stop ${state}`} class:below={row === 1} data-testid={`bar-stop-${side}-${mark.key}`} data-state={state}
              data-side={side} data-step={mark.key} aria-label={words(side, mark.label, state)} title={words(side, mark.label, state)}
              tabindex={focusAt.side === row && focusAt.step === step ? 0 : -1}
              onfocus={() => (focusAt = { side: row, step })} onclick={() => oncycle?.(side, mark.key)}><i aria-hidden="true"></i></button>
          {:else}
            <span class={`stop ${state}`} class:below={row === 1} data-testid={`bar-stop-${side}-${mark.key}`} data-state={state} title={words(side, mark.label, state)}>
              <i aria-hidden="true"></i><span class="sr-only">{state === 'off' ? '' : `, stops on ${SIDE_WORDS[side]}`}</span>
            </span>
          {/if}
          {#if row === 0}<span class="name" aria-hidden={editable}>{mark.short}</span>{/if}
        {/each}
      </div>
    {/each}
  </div>
  <div class="side">
    <p class="legend">
      {#if track.every}
        Stops off: every step stops
      {:else}
        <span><i class="stop off" aria-hidden="true"><i></i></i>Off</span>
        <span><i class="stop every" aria-hidden="true"><i></i></i>Every time</span>
        <span><i class="stop once" aria-hidden="true"><i></i></i>Once</span>
      {/if}
    </p>
    {@render aside?.()}
  </div>
</div>

<style>
  /* 64 px tall. A cell is three rows: your stop, the step's name, the opponent's
     stop. Each stop's button fills its row to the bar's edge, a large target,
     and draws its 12 px marker against the name (1 px away), so at least 8 px
     stays between a marker and the bar's top and bottom edges. */
  .stopbar { grid-column: 1 / -1; display: grid; grid-template-columns: subgrid; height: 64px; background: var(--bg-surface); border-top: 1px solid var(--border); }
  .steps { display: grid; grid-template-columns: var(--plate) repeat(9, minmax(0, 1fr)); min-width: 0; }
  .key { display: grid; align-content: center; gap: 4px; padding-left: 14px; font: 14px/20px var(--font-sans); color: var(--text-tertiary); white-space: nowrap; }
  .key .active { color: var(--text); }
  .cell { display: grid; grid-template-rows: minmax(0, 1fr) 20px minmax(0, 1fr); row-gap: 1px; min-width: 0; border-left: 1px solid transparent; }
  .cell.rule { border-left-color: var(--border); }
  .cell.current { background: var(--bg-muted); }
  .name { font: 14px/20px var(--font-sans); color: var(--text-secondary); text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .current .name { color: var(--text); font-weight: 600; }

  /* A stop is a small square: hollow when off, filled every time, a dot inside once. */
  .stop { display: grid; justify-items: center; align-items: end; min-height: 0; padding: 0; border: 0; background: none; border-radius: var(--radius-sm); }
  .stop.below { align-items: start; }
  button.stop { cursor: pointer; }
  .stop i { position: relative; width: 12px; height: 12px; border-radius: 0; box-shadow: inset 0 0 0 1.5px var(--text-tertiary); }
  .stop.every i { background: var(--text); box-shadow: inset 0 0 0 1.5px var(--text); }
  .stop.once i { box-shadow: inset 0 0 0 1.5px var(--text); }
  .stop.once i::after { content: ''; position: absolute; inset: 3px; border-radius: 0; background: var(--text); }
  button.stop:hover i { box-shadow: inset 0 0 0 2px var(--text); }
  button.stop:hover { background: var(--bg-field); }
  button.stop:focus-visible { outline: 2px solid var(--text); outline-offset: -2px; }

  /* The rail's column: the legend, and under it whatever the board puts there
     (the decision strip's toggle), each one 20 px line. */
  .side { display: grid; align-content: center; justify-items: center; gap: 4px; min-width: 0; padding: 0 8px; border-left: 1px solid var(--border); }
  .legend { display: flex; gap: 12px; align-items: center; justify-content: center; margin: 0; font: 14px/20px var(--font-sans); color: var(--text-secondary); text-align: center; white-space: nowrap; }
  .legend > span { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
  .legend .stop { display: inline-grid; align-items: center; }
</style>
