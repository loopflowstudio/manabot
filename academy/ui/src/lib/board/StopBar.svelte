<script lang="ts">
  /**
   * The turn's steps along the bottom of the board, and the stops set on
   * them, laid out as MTGO does: one cell per step, your turn's stop above
   * the step's name and the opponent's below, a rule between phases, the
   * current step marked. Pressing a stop cycles it off → every time → once
   * (`nextStopState`); the Game popover's checkboxes edit the same `StopsConfig` and
   * stay the full-size control. The stops are one toolbar: Tab reaches it
   * once, arrow keys move between them. Without `oncycle` (a recording) the
   * bar is only the step track: where in the turn this is, with no stops,
   * since stops are a setting of live play.
   */
  import type { Snippet } from 'svelte';
  import { ACTIVE_COMBAT_STOP_STEPS } from '$lib/stops';
  import type { StopSide } from '$lib/types';
  import { nextStopState, STOP_STATE_WORDS, type PhaseMark, type PhaseTrack, type StopState } from './phases';

  interface Props {
    track: PhaseTrack;
    oncycle?: (side: StopSide, step: string) => void;
    /** Under the legend, in the rail's column: the decision strip's toggle. */
    aside?: Snippet;
    /** The turn shown, named at the head of the bar and coloured by whose it is. */
    turn?: number;
  }

  let { track, oncycle, aside, turn }: Props = $props();

  const sides: StopSide[] = ['my', 'opponent'];
  const SIDE_WORDS: Record<StopSide, string> = { my: 'your turn', opponent: "the opponent's turn" };
  const playing = $derived(oncycle !== undefined);
  const editable = $derived(playing && !track.every);

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

  function words(side: StopSide, mark: PhaseMark, state: StopState): string {
    if (track.every) return `Stop at ${mark.label} on ${SIDE_WORDS[side]}: full control is on, every priority choice pauses`;
    const activeCombat = ACTIVE_COMBAT_STOP_STEPS.has(mark.key) ? ' (active combat with a non-mana play)' : '';
    return `Stop at ${mark.label} on ${SIDE_WORDS[side]}${activeCombat}: ${STOP_STATE_WORDS[state]}${editable ? `. Press for ${STOP_STATE_WORDS[nextStopState(state)]}` : ''}`;
  }
</script>

<div class="stopbar" data-testid="stop-bar" data-turn-side={track.active}>
  <!-- svelte-ignore a11y_interactive_supports_focus (focus is roving: one stop is tabbable) -->
  <div class="steps" role={editable ? 'toolbar' : 'list'} aria-label={editable ? 'Stops: arrow keys move, Space changes' : 'Steps of the turn'}
    bind:this={toolbar} onkeydown={editable ? onkeydown : undefined}>
    <!-- Whose turn it is, in that side's colour; the current step's cell carries the same colour on that side's edge. -->
    <div class="key">
      {#if turn !== undefined}<p class="turn" data-testid="turn-number">Turn {turn} · {track.active === 'my' ? 'yours' : 'theirs'}</p>{/if}
      <span aria-hidden="true" class:active={track.active === 'my'}>▲ Your turn</span>
      <span aria-hidden="true" class:active={track.active === 'opponent'}>▼ Their turn</span>
    </div>
    {#each track.marks as mark, step (mark.key)}
      <div class="cell" class:rule={mark.phaseStart} class:current={mark.current} role={editable ? 'group' : 'listitem'}
        aria-label={mark.label} aria-current={mark.current ? 'step' : undefined}>
        {#each sides as side, row (side)}
          {@const state = track.every ? 'every' : mark.state[side]}
          {@const tip = words(side, mark, state)}
          {#if editable}
            <button type="button" class={`stop ${state}`} class:below={row === 1} data-testid={`bar-stop-${side}-${mark.key}`} data-state={state}
              data-side={side} data-step={mark.key} aria-label={tip} title={tip}
              tabindex={focusAt.side === row && focusAt.step === step ? 0 : -1}
              onfocus={() => (focusAt = { side: row, step })} onclick={() => oncycle?.(side, mark.key)}><i aria-hidden="true"></i></button>
          {:else if playing}
            <span class={`stop ${state}`} class:below={row === 1} data-testid={`bar-stop-${side}-${mark.key}`} data-state={state} title={tip}>
              <i aria-hidden="true"></i><span class="sr-only">{state === 'off' ? '' : `, stops on ${SIDE_WORDS[side]}`}</span>
            </span>
          {/if}
          {#if !playing}<span aria-hidden="true"></span>{/if}
          {#if row === 0}<span class="name" aria-hidden={editable}>{mark.short}</span>{/if}
        {/each}
      </div>
    {/each}
  </div>
  <div class="side">
    {#if playing}
    <p class="legend">
      {#if track.every}
        Full control: every priority choice pauses
      {:else}
        <span><i class="stop off" aria-hidden="true"><i></i></i>Off</span>
        <span><i class="stop every" aria-hidden="true"><i></i></i>Every time</span>
        <span><i class="stop once" aria-hidden="true"><i></i></i>Once</span>
      {/if}
    </p>
    {/if}
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
  .key .active { color: var(--side-text); font-weight: 600; }
  .key .turn { margin: 0; font-weight: 600; color: var(--text); }
  .key:has(.turn) { gap: 0; font-size: 12px; line-height: 16px; }
  .key:has(.turn) .turn { font-size: 14px; line-height: 20px; }
  /* Arena's pairing: your turn is blue, the opponent's orange. The whole bar takes a faint wash of the side's colour;
     the current step's solid block mixes toward --text so --bg ink stays above 4.5:1 in both light and dark themes. */
  .stopbar[data-turn-side='my'] { --side: var(--vivid-u); }
  .stopbar[data-turn-side='opponent'] { --side: var(--vivid-r); }
  .stopbar[data-turn-side] {
    --side-text: color-mix(in srgb, var(--side) 72%, var(--text));
    --side-block: color-mix(in srgb, var(--side) 78%, var(--text));
    background: color-mix(in srgb, var(--side) 8%, var(--bg-surface));
  }
  .cell { display: grid; grid-template-rows: minmax(0, 1fr) 20px minmax(0, 1fr); row-gap: 1px; min-width: 0; border-left: 1px solid transparent; }
  .cell.rule { border-left-color: var(--border); }
  /* The current step is a solid block of the side's colour; its name and stops are drawn in the page ground's colour on it. */
  .cell.current { background: var(--side-block, var(--bg-muted)); }
  .name { font: 14px/20px var(--font-sans); color: var(--text-secondary); text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .current .name { color: var(--text); font-weight: 600; }
  .stopbar[data-turn-side] .current .name { color: var(--bg); }
  .stopbar[data-turn-side] .current .stop i { box-shadow: inset 0 0 0 1.5px var(--bg); }
  .stopbar[data-turn-side] .current .stop.every i { background: var(--bg); }
  .stopbar[data-turn-side] .current .stop.once i::after { background: var(--bg); }
  .stopbar[data-turn-side] .current button.stop:hover { background: color-mix(in srgb, var(--bg) 22%, transparent); }
  .stopbar[data-turn-side] .current button.stop:focus-visible { outline-color: var(--bg); }

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
