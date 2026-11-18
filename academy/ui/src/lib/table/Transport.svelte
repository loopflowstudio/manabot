<script lang="ts">
  /**
   * A recording's transport, in place of the hand's live input: step by
   * position or by decision, play at a speed, and scrub a line ticked by turn
   * with each decision marked. At a Watch's live edge, Next asks the bots for
   * the next decision (Continue). Keys are the page's: ←/→ step, Shift+←/→
   * jump decisions, Space plays, Home/End.
   */
  import { labelledTurns, type StripMark } from './replay';

  interface Props {
    selected: number;
    total: number;
    marks: StripMark[];
    playing: boolean;
    speed: number;
    disabled?: boolean;
    busy?: boolean;
    producing?: boolean;
    canProduce?: boolean;
    retrySearch?: boolean;
    retryBot?: boolean;
    onPrevious: () => void;
    onNext: () => void;
    onDecision: (direction: -1 | 1) => void;
    onTogglePlaying: () => void;
    onScrub: (index: number) => void;
    onSpeedChange: (speed: number) => void;
  }

  let {
    selected, total, marks, playing, speed, disabled = false, busy = false, producing = false, canProduce = false,
    retrySearch = false, retryBot = false, onPrevious, onNext, onDecision, onTogglePlaying, onScrub, onSpeedChange,
  }: Props = $props();

  const edge = $derived(selected + 1 === total);
  const span = $derived(Math.max(1, total - 1));
  const at = (index: number): string => `${(index / span) * 100}%`;
  const turn = $derived(marks[selected]?.turn ?? null);
  const numbered = $derived(new Set(labelledTurns(marks, 0.045).map(mark => mark.at)));
  const hasDecisionBefore = $derived(marks.some(mark => mark.decision && mark.at < selected));
  const hasDecisionAfter = $derived(marks.some(mark => mark.decision && mark.at > selected));
</script>

<section class="transport" aria-label="Replay transport" data-testid="replay-transport">
  <div class="buttons">
    <button type="button" class="btn btn-ghost btn-sm" disabled={disabled || !hasDecisionBefore} onclick={() => onDecision(-1)}
      aria-label="Previous decision" title="Previous decision (Shift+←)">⇤</button>
    <button type="button" class="btn btn-secondary btn-sm" disabled={disabled || selected === 0} onclick={onPrevious} title="Previous position (←)">Previous</button>
    <button type="button" class="btn btn-secondary btn-sm play" disabled={disabled || (producing && edge && !playing) || (!playing && edge && (!canProduce || retrySearch || retryBot))}
      onclick={onTogglePlaying} title="Play or pause (Space)">{playing ? 'Pause' : 'Play'}</button>
    <button type="button" class="btn btn-primary btn-sm" disabled={disabled || (producing && edge) || busy || (edge && !canProduce)} onclick={onNext} title="Next position (→)">
      {edge && canProduce ? retrySearch ? 'Retry search' : retryBot ? 'Retry bot decision' : 'Continue' : 'Next'}
    </button>
    <button type="button" class="btn btn-ghost btn-sm" disabled={disabled || !hasDecisionAfter} onclick={() => onDecision(1)}
      aria-label="Next decision" title="Next decision (Shift+→)">⇥</button>
  </div>
  <div class="scrub">
    <input type="range" aria-label="Replay position" aria-valuetext={`Position ${selected + 1} of ${total}${turn !== null ? `, turn ${turn}` : ''}`}
      min="0" max={Math.max(0, total - 1)} value={selected} {disabled} oninput={event => onScrub(Number(event.currentTarget.value))} />
    <div class="ticks" aria-hidden="true">
      {#each marks as mark (mark.key)}
        {#if mark.turnStart}<i class="turn" style:left={at(mark.at)}>{#if numbered.has(mark.at)}<b>{mark.turn ?? ''}</b>{/if}</i>{/if}
        {#if mark.decision}<i class="decision" class:theirs={mark.side === 'opponent'} style:left={at(mark.at)}></i>{/if}
      {/each}
    </div>
  </div>
  <span class="type-caption counter" data-testid="position-counter">Position {selected + 1} / {total}</span>
  <label class="type-caption speed">Speed
    <select value={speed} onchange={event => onSpeedChange(Number(event.currentTarget.value))}>{#each [1, 2, 4, 16] as value (value)}<option {value}>{value}×</option>{/each}</select>
  </label>
</section>

<style>
  .transport { display: flex; align-items: center; gap: 12px; padding: 6px 12px; min-width: 0; }
  .buttons { display: flex; gap: 4px; flex: none; }
  .play { min-width: 56px; }
  .scrub { position: relative; flex: 1 1 auto; min-width: 120px; padding-bottom: 12px; }
  .scrub input { width: 100%; display: block; accent-color: var(--accent); }
  .ticks { position: absolute; left: 8px; right: 8px; bottom: 0; height: 12px; pointer-events: none; }
  .ticks i { position: absolute; top: 0; }
  .turn { width: 1px; height: 12px; background: var(--border-strong); }
  .turn b { position: absolute; left: 3px; top: -1px; font: 600 9px/12px var(--font-mono); color: var(--text-tertiary); }
  .decision { width: 3px; height: 5px; margin-left: -1px; top: 1px !important; border-radius: 1px; background: var(--info-text, var(--accent)); }
  .decision.theirs { background: var(--text-tertiary); }
  .counter { flex: none; font-variant-numeric: tabular-nums; color: var(--text-secondary); }
  .speed { flex: none; display: flex; align-items: center; gap: 4px; color: var(--text-secondary); }
  .speed select { border: 1px solid var(--border); border-radius: 4px; background: var(--bg-field); padding: 1px 4px; color: var(--text); }
</style>
