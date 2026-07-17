<script lang="ts">
  import { DECK_CHOICES, type DeckChoice } from '$lib/decks';

  import DeckIdentity from './DeckIdentity.svelte';

  interface Props {
    hero?: DeckChoice;
    villain?: DeckChoice;
    disabled?: boolean;
    onHeroChange?: (value: DeckChoice) => void;
    onVillainChange?: (value: DeckChoice) => void;
  }

  let {
    hero = 'ur_lessons',
    villain = 'gw_allies',
    disabled = false,
    onHeroChange = undefined,
    onVillainChange = undefined,
  }: Props = $props();

  function labelFor(key: DeckChoice): string {
    return DECK_CHOICES.find((deck) => deck.key === key)?.label ?? key;
  }
</script>

<div class="flex flex-wrap items-end gap-3">
  <label class="flex flex-col gap-1.5">
    <span class="type-label text-ink-2">Your deck</span>
    <span class="flex items-center gap-1.5 rounded border border-line bg-field py-2 pl-3 pr-1">
      <DeckIdentity name={labelFor(hero)} symbolsOnly />
      <select
        data-testid="deck-select-hero"
        class="min-h-0 border-0 bg-transparent pr-2"
        value={hero}
        {disabled}
        onchange={(event) =>
          onHeroChange?.((event.currentTarget as HTMLSelectElement).value as DeckChoice)}
      >
        {#each DECK_CHOICES as deck (deck.key)}
          <option value={deck.key}>{deck.label}</option>
        {/each}
      </select>
    </span>
  </label>

  <label class="flex flex-col gap-1.5">
    <span class="type-label text-ink-2">Opponent deck</span>
    <span class="flex items-center gap-1.5 rounded border border-line bg-field py-2 pl-3 pr-1">
      <DeckIdentity name={labelFor(villain)} symbolsOnly />
      <select
        data-testid="deck-select-villain"
        class="min-h-0 border-0 bg-transparent pr-2"
        value={villain}
        {disabled}
        onchange={(event) =>
          onVillainChange?.((event.currentTarget as HTMLSelectElement).value as DeckChoice)}
      >
        {#each DECK_CHOICES as deck (deck.key)}
          <option value={deck.key}>{deck.label}</option>
        {/each}
      </select>
    </span>
  </label>
</div>
