<script lang="ts">
  let { selected, total, playing, speed, disabled = false, busy = false, producing = false, canProduce = false, retrySearch = false, retryBot = false, onPrevious, onNext, onTogglePlaying, onScrub, onSpeedChange }: {
    selected: number; total: number; playing: boolean; speed: number;
    disabled?: boolean; busy?: boolean; producing?: boolean; canProduce?: boolean; retrySearch?: boolean; retryBot?: boolean;
    onPrevious: () => void; onNext: () => void; onTogglePlaying: () => void;
    onScrub: (index: number) => void; onSpeedChange: (speed: number) => void;
  } = $props();
  const edge = $derived(selected + 1 === total);
</script>
<section aria-label="Replay transport" class="flex flex-wrap items-center gap-2">
  <button class="btn btn-secondary btn-sm" disabled={disabled || producing || selected === 0} onclick={onPrevious}>Previous</button>
  <button class="btn btn-secondary btn-sm" disabled={disabled || (producing && !playing) || (!playing && edge && !canProduce)} onclick={onTogglePlaying}>{playing ? 'Pause' : 'Play'}</button>
  <button class="btn btn-primary btn-sm" disabled={disabled || producing || busy || (edge && !canProduce)} onclick={onNext}>{edge && canProduce ? retrySearch ? 'Retry search' : retryBot ? 'Retry bot decision' : 'Continue' : 'Next'}</button>
  <input class="min-w-24 flex-1" type="range" aria-label="Replay position" aria-valuetext={`Position ${selected + 1} of ${total}`} min="0" max={Math.max(0, total - 1)} value={selected} disabled={disabled || producing} oninput={event => onScrub(Number(event.currentTarget.value))} />
  <span class="type-caption tabular-nums" data-testid="position-counter">Position {selected + 1} / {total}</span>
  <label class="type-caption">Speed <select class="rounded border border-line bg-field p-1" value={speed} onchange={event => onSpeedChange(Number(event.currentTarget.value))}>{#each [1, 2, 4, 16] as value}<option value={value}>{value}×</option>{/each}</select></label>
</section>
