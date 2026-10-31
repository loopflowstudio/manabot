<script lang="ts">
  import { checkpointChoice, type OpponentChoice } from '$lib/game.svelte';
  import type { Decklists, TrainedOpponent } from '$lib/types';

  interface Props {
    value?: OpponentChoice;
    // Trained opponents from GET /api/opponents; unavailable entries stay
    // listed, disabled, with the server's reason.
    trained?: TrainedOpponent[];
    catalogError?: string | null;
    // Closed decklists keep the Search options listed, disabled, with why.
    decklists?: Decklists;
    disabled?: boolean;
    onChange?: (value: OpponentChoice) => void;
  }

  let {
    value = 'search-64',
    trained = [],
    catalogError = null,
    decklists = 'open',
    disabled = false,
    onChange = undefined,
  }: Props = $props();

  const searchOptions: { value: OpponentChoice; label: string }[] = [
    { value: 'search-16', label: 'Search 16 (fast)' },
    { value: 'search-64', label: 'Search 64 (default)' },
    { value: 'search-256', label: 'Search 256 (strong)' },
  ];
  const searchUnavailable = $derived(decklists === 'closed');

  function optionLabel(opponent: TrainedOpponent): string {
    const parts = [opponent.name];
    if (opponent.world) parts.push(opponent.world);
    if (opponent.step !== null) parts.push(`step ${opponent.step.toLocaleString('en-US')}`);
    parts.push(opponent.sha256.slice(0, 10));
    const label = parts.join(' · ');
    return opponent.available ? label : `${label} — ${opponent.reason ?? 'unavailable'}`;
  }

  const selected = $derived(
    trained.find((opponent) => checkpointChoice(opponent.sha256) === value) ?? null,
  );
  const unlisted = $derived(
    value.startsWith('checkpoint:') && selected === null ? value : null,
  );
  const details = $derived(
    selected
      ? [selected.producer, selected.decks.length ? selected.decks.join(' vs ') : null,
          selected.model, selected.parameters !== null
          ? `${selected.parameters.toLocaleString('en-US')} parameters`
          : null]
          .filter(Boolean)
          .join(' · ')
      : '',
  );
</script>

<div class="flex min-w-0 max-w-full flex-wrap items-end gap-3">
  <label class="flex min-w-0 max-w-full flex-col gap-1.5">
    <span class="type-label text-ink-2">Opponent</span>
    <select
      data-testid="opponent-select"
      class="w-full min-w-0 max-w-[28rem] rounded border border-line bg-field px-3 py-2"
      {value}
      {disabled}
      aria-describedby="opponent-details"
      onchange={(event) =>
        onChange?.((event.currentTarget as HTMLSelectElement).value as OpponentChoice)}
    >
      {#each searchOptions as option (option.value)}
        <option value={option.value} disabled={searchUnavailable}
          >{option.label}{searchUnavailable ? ' — needs open decklists' : ''}</option>
      {/each}
      <option value="random">Random</option>
      <option value="passive">Passive</option>
      {#if trained.length || unlisted}
        <optgroup label="Trained checkpoints">
          {#each trained as opponent (opponent.sha256)}
            <option
              data-testid="trained-opponent"
              value={checkpointChoice(opponent.sha256)}
              disabled={!opponent.available}
            >{optionLabel(opponent)}</option>
          {/each}
          {#if unlisted}
            <option value={unlisted}>Trained · {unlisted.slice('checkpoint:'.length, 'checkpoint:'.length + 10)}</option>
          {/if}
        </optgroup>
      {/if}
    </select>
  </label>
  <p id="opponent-details" data-testid="opponent-details" class="type-caption min-w-0 break-words text-ink-2">
    {#if details}{details}{:else if catalogError}{catalogError}{:else if !trained.length}No trained checkpoints found.{/if}
  </p>
</div>
