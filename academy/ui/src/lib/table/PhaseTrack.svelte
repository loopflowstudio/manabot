<script lang="ts">
  /**
   * The turn's steps in the top bar, and the stops set on them: two rows of
   * markers under the steps, yours and the opponent's, each filled where the
   * game stops on that player's turn. Pressing a marker toggles that stop
   * (the same `StopsConfig` the Menu edits, whose checkboxes stay the
   * full-size control). Both rows show at once so the opponent's stops can
   * be set on your own turn; the row of the turn being played is in full
   * ink. The markers are one toolbar: Tab reaches it once, arrow keys move
   * between markers. Without `ontoggle` (a recording) they are read-only.
   */
  import type { StopSide } from '$lib/types';
  import type { PhaseTrack } from './phases';

  interface Props {
    track: PhaseTrack;
    turn: number;
    yourTurn: boolean;
    ontoggle?: (side: StopSide, step: string) => void;
  }

  let { track, turn, yourTurn, ontoggle }: Props = $props();

  const sides: StopSide[] = ['my', 'opponent'];
  const SIDE_WORDS: Record<StopSide, string> = { my: 'your turn', opponent: "the opponent's turn" };
  const editable = $derived(ontoggle !== undefined && !track.every);

  // Roving focus: one marker is in the Tab order; arrows move it.
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

  function words(side: StopSide, label: string, set: boolean): string {
    return `Stop at ${label} on ${SIDE_WORDS[side]}: ${track.every ? 'priority stops are off, every step stops' : set ? 'on' : 'off'}`;
  }
</script>

<div class="track" data-testid="phase-track">
  <span class="type-rubric turn" data-testid="turn-number">Turn {turn} · {yourTurn ? 'yours' : 'theirs'}</span>
  <span class="step sides" aria-hidden="true"><span class="label type-rubric">&nbsp;</span><span class="side" class:active={track.active === 'my'}>You</span><span class="side" class:active={track.active === 'opponent'}>Them</span></span>
  <!-- svelte-ignore a11y_interactive_supports_focus (focus is roving: one marker is tabbable) -->
  <div class="steps" role={editable ? 'toolbar' : 'list'} aria-label={editable ? 'Stops: arrow keys move, Space toggles' : `Turn ${turn} steps`} bind:this={toolbar} onkeydown={editable ? onkeydown : undefined}>
    {#each track.marks as mark, step (mark.key)}
      <div class="step" class:current={mark.current} role={editable ? 'group' : 'listitem'} aria-label={mark.label} aria-current={mark.current ? 'step' : undefined}>
        <span class="label type-rubric" aria-hidden={editable}>{mark.short}</span>
        {#each sides as side, row (side)}
          {@const set = track.every || mark.set[side]}
          {#if editable}
            <button type="button" class="marker" class:set class:active={track.active === side} data-testid={`track-stop-${side}-${mark.key}`}
              data-side={side} data-step={mark.key} aria-pressed={mark.set[side]} aria-label={words(side, mark.label, mark.set[side])}
              title={words(side, mark.label, mark.set[side])} tabindex={focusAt.side === row && focusAt.step === step ? 0 : -1}
              onfocus={() => (focusAt = { side: row, step })} onclick={() => ontoggle?.(side, mark.key)}><i aria-hidden="true"></i></button>
          {:else}
            <span class="marker" class:set class:active={track.active === side} title={words(side, mark.label, set)}>
              <i aria-hidden="true"></i><span class="sr-only">{set ? `, stops on ${SIDE_WORDS[side]}` : ''}</span>
            </span>
          {/if}
        {/each}
      </div>
    {/each}
  </div>
</div>

<style>
  .track { display: flex; align-items: center; gap: 10px; margin: 0 auto; flex-shrink: 0; height: 100%; }
  .turn { color: var(--text); white-space: nowrap; }
  .steps { display: flex; gap: 1px; height: 100%; }
  .step { display: grid; grid-auto-rows: min-content; justify-items: stretch; align-content: center; min-width: 0; }
  .label { padding: 1px 6px; border-radius: 10px; color: var(--text-tertiary); white-space: nowrap; text-align: center; }
  .current .label { background: var(--accent); color: var(--ivory); }

  /* A marker is a small square: hollow when the step passes, filled where it stops. */
  .marker { display: grid; place-items: center; height: 10px; min-height: 0; padding: 0; border: 0; background: none; cursor: pointer; border-radius: 3px; }
  span.marker { cursor: default; }
  .marker i { width: 7px; height: 7px; border-radius: 1.5px; box-shadow: inset 0 0 0 1px var(--text-tertiary); }
  .marker.set i { background: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
  button.marker:hover i { box-shadow: inset 0 0 0 1.5px var(--text); }
  button.marker:focus-visible { outline: 2px solid var(--text); outline-offset: -1px; }

  /* The turn's own row in full ink, the other quieter. */
  .marker:not(.active) i { opacity: 0.55; }
  .sides { margin-right: 2px; }
  .sides .label { visibility: hidden; }
  .side { height: 10px; font: 11px/10px var(--font-sans); color: var(--text-tertiary); text-align: right; }
  .side.active { color: var(--text); }
</style>
