<script lang="ts">
  import { DECK_CHOICES, type DeckChoice } from '$lib/decks';

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
</script>

<div class="flex flex-wrap items-center gap-3">
  <label class="flex items-center gap-2 text-sm text-slate-300">
    Your deck
    <select
      data-testid="deck-select-hero"
      class="rounded border border-slate-600 bg-slate-900 px-3 py-2 text-sm"
      value={hero}
      {disabled}
      onchange={(event) =>
        onHeroChange?.((event.currentTarget as HTMLSelectElement).value as DeckChoice)}
    >
      {#each DECK_CHOICES as deck (deck.key)}
        <option value={deck.key}>{deck.label}</option>
      {/each}
    </select>
  </label>

  <label class="flex items-center gap-2 text-sm text-slate-300">
    Opponent deck
    <select
      data-testid="deck-select-villain"
      class="rounded border border-slate-600 bg-slate-900 px-3 py-2 text-sm"
      value={villain}
      {disabled}
      onchange={(event) =>
        onVillainChange?.((event.currentTarget as HTMLSelectElement).value as DeckChoice)}
    >
      {#each DECK_CHOICES as deck (deck.key)}
        <option value={deck.key}>{deck.label}</option>
      {/each}
    </select>
  </label>
</div>
