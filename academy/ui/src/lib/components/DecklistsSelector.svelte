<script lang="ts">
  import { isSearchChoice, type PlayerChoice } from '$lib/game.svelte';
  import type { Decklists } from '$lib/types';

  interface Props {
    value?: Decklists;
    // The bots that will play. A search bot keeps Closed listed, disabled, with why.
    players?: PlayerChoice[];
    disabled?: boolean;
    onChange?: (value: Decklists) => void;
  }

  let {
    value = 'open',
    players = ['search-64'],
    disabled = false,
    onChange = undefined,
  }: Props = $props();

  const closedUnavailable = $derived(players.some(isSearchChoice));
</script>

<label class="type-caption flex min-w-0 max-w-full items-center gap-2 text-ink-2" data-testid="decklists-choice">
  Decklists
  <select
    class="min-w-0 flex-1 rounded border border-line bg-panel px-2 py-1"
    {value}
    {disabled}
    onchange={(event) => onChange?.((event.currentTarget as HTMLSelectElement).value as Decklists)}
  >
    <option value="open">Open — both main decks and sideboards are known</option>
    <option value="closed" disabled={closedUnavailable}
      >Closed — neither player sees the other's lists{closedUnavailable
        ? ' — search opponents need open decklists'
        : ''}</option>
  </select>
</label>
