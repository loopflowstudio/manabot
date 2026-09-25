<script lang="ts">
  import type { OpponentChoice } from '$lib/game.svelte';

  interface Props {
    value?: OpponentChoice;
    trainedLabel?: string;
    trainedAvailable?: boolean;
    disabled?: boolean;
    onChange?: (value: OpponentChoice) => void;

  }

  let {
    value = 'search-64',
    trainedLabel = 'Trained opponent unavailable',
    trainedAvailable = false,
    disabled = false,
    onChange = undefined,

  }: Props = $props();
</script>

<div class="flex flex-wrap items-end gap-3">
  <label class="flex flex-col gap-1.5">
    <span class="type-label text-ink-2">Opponent</span>
    <select
      data-testid="opponent-select"
      class="rounded border border-line bg-field px-3 py-2"
      {value}
      {disabled}
      onchange={(event) =>
        onChange?.((event.currentTarget as HTMLSelectElement).value as OpponentChoice)}
    >
      <option value="search-16">Search 16 (fast)</option>
      <option value="search-64">Search 64 (default)</option>
      <option value="search-256">Search 256 (strong)</option>
      <option value="checkpoint" disabled={!trainedAvailable}>{trainedLabel}</option>
      <option value="random">Random</option>
      <option value="passive">Passive</option>
    </select>
  </label>

</div>
