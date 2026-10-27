<script lang="ts">
  import { CURATED_PACK, ELEMENTAL_WORLD_PACK } from '$lib/curated-pack';
  import { DECK_CHOICES, type DeckChoice } from '$lib/decks';
  import CardImage from './CardImage.svelte';
  import DeckIdentity from './DeckIdentity.svelte';

  interface Props {
    hero: DeckChoice;
    villain: DeckChoice;
    disabled?: boolean;
    heroLabel?: string;
    villainLabel?: string;
    onHeroChange: (value: DeckChoice) => void;
    onVillainChange: (value: DeckChoice) => void;
  }

  let { hero, villain, disabled = false, heroLabel = 'Your deck', villainLabel = 'Opponent deck', onHeroChange, onVillainChange }: Props = $props();
  const roster = ELEMENTAL_WORLD_PACK.roster!;
  const otherDecks = DECK_CHOICES.filter((row) => !row.key.startsWith('elemental_'));
  const allDecks = [...roster, CURATED_PACK.matchup.hero, CURATED_PACK.matchup.villain];
  let inspected = $state<DeckChoice | null>(null);
  const inspectedKey = $derived(inspected ?? hero);
  const choice = $derived(DECK_CHOICES.find((row) => row.key === inspectedKey)!);
  const deck = $derived(allDecks.find((row) => row.deck_id === inspectedKey));
  const cards = $derived(Object.entries(deck?.cards ?? {}).sort(([a], [b]) => a.localeCompare(b)));
  const lessons = $derived(Object.entries(deck?.sideboard ?? {}).sort(([a], [b]) => a.localeCompare(b)));
  const fullArt = import.meta.glob('/src/lib/card-art/full/*.jpg', { eager: true, query: '?url', import: 'default' }) as Record<string, string>;
  let preview = $state<string | null>(null);
  let previewDialog: HTMLDialogElement;

  function imageFor(name: string): string | undefined {
    const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    return fullArt[`/src/lib/card-art/full/${slug}.jpg`];
  }

  function inspectCard(name: string): void {
    preview = name;
    previewDialog.showModal();
  }

  function labelFor(key: DeckChoice): string {
    return DECK_CHOICES.find((row) => row.key === key)?.label ?? key;
  }
  const lessonCount = $derived(lessons.reduce((sum, [, count]) => sum + count, 0));

  function title(name: string): string {
    return name.replace(/^Elemental /, '');
  }
</script>

<section aria-label="Choose decks" class="my-6 w-full min-w-0 border-b border-line pb-6" data-testid="deck-browser">
  <div class="mb-5 flex flex-wrap items-baseline justify-between gap-2">
    <div>
      <p class="type-rubric text-ink-2">The Elemental collection</p>
      <h2 class="type-title text-display">Ten decks. Choose your matchup.</h2>
    </div>
    <p class="type-caption text-ink-2">40 cards each · individual Lesson pools · mirrors welcome</p>
  </div>
  <div class="mb-5 grid gap-3 border-y border-line py-3 sm:grid-cols-2" aria-live="polite">
    <div data-testid="selected-deck-hero" data-deck-id={hero}><p class="type-label text-ink-2">{heroLabel}</p><p class="mt-1"><DeckIdentity name={title(labelFor(hero))} /></p></div>
    <div data-testid="selected-deck-villain" data-deck-id={villain}><p class="type-label text-ink-2">{villainLabel}</p><p class="mt-1"><DeckIdentity name={title(labelFor(villain))} /></p></div>
  </div>
  <div class="grid gap-6 lg:grid-cols-[minmax(15rem,1fr)_minmax(0,2fr)]">
    <nav aria-label="Browse Elemental decks" class="grid content-start grid-cols-1 gap-1 sm:grid-cols-2 lg:grid-cols-1">
      {#each roster as row, index (row.deck_id)}
        <button
          class={`flex min-h-14 items-center gap-3 border-l-2 px-3 py-3 text-left ${inspectedKey === row.deck_id ? 'border-action bg-panel-muted' : 'border-transparent hover:bg-panel-muted'}`}
          aria-pressed={inspectedKey === row.deck_id}
          onclick={() => inspected = row.deck_id as DeckChoice}
          data-testid={`browse-${row.deck_id}`}
        >
          <span class="type-caption w-5 shrink-0 text-ink-3" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
          <span class="flex-1 text-sm"><DeckIdentity name={title(row.display_name)} /></span>
          <span class="type-label text-action">{[hero === row.deck_id ? (heroLabel === 'Your deck' ? 'You' : heroLabel) : '', villain === row.deck_id ? (villainLabel === 'Opponent deck' ? 'Opponent' : villainLabel) : ''].filter(Boolean).join(' / ')}</span>
        </button>
      {/each}
      <details class="mt-3 border-t border-line pt-3 sm:col-span-2 lg:col-span-1">
        <summary class="type-caption cursor-pointer text-ink-2">Other decks</summary>
        {#each otherDecks as row (row.key)}
          <button class="mt-1 block w-full rounded p-3 text-left text-sm aria-pressed:bg-panel-muted" aria-pressed={inspectedKey === row.key} onclick={() => inspected = row.key} data-testid={`browse-${row.key}`}><DeckIdentity name={row.label} /></button>
        {/each}
      </details>
    </nav>

    <div class="min-w-0 border-t border-line pt-4 lg:border-l lg:border-t-0 lg:pl-6 lg:pt-0" data-testid="deck-detail">
      <div class="mb-4 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h3 class="type-title text-display"><DeckIdentity name={title(choice.label)} /></h3>
          <p class="type-caption mt-1 text-ink-2">{#if deck}{deck.card_count} main deck cards · {lessonCount} Lessons{:else}Practice deck{/if}</p>
        </div>
        <div class="flex flex-wrap gap-2">
          <button class="btn border border-line bg-field aria-pressed:border-action aria-pressed:text-action" {disabled} aria-pressed={hero === inspectedKey} onclick={() => onHeroChange(inspectedKey)} data-testid="assign-deck-hero">{heroLabel === 'Your deck' ? 'Use for me' : `Use for ${heroLabel.toLowerCase()}`}</button>
          <button class="btn border border-line bg-field aria-pressed:border-action aria-pressed:text-action" {disabled} aria-pressed={villain === inspectedKey} onclick={() => onVillainChange(inspectedKey)} data-testid="assign-deck-villain">{villainLabel === 'Opponent deck' ? 'Use for opponent' : `Use for ${villainLabel.toLowerCase()}`}</button>
        </div>
      </div>
      <div class="grid gap-x-5 gap-y-1 sm:grid-cols-2" aria-label="Main deck">
        {#each cards as [name, count] (name)}
          <button class="flex items-center gap-3 border-b border-line/50 py-2 text-left hover:bg-panel-muted" onclick={() => inspectCard(name)} aria-label={`Inspect ${name}`} data-testid="deck-card">
            <div class="h-16 w-24 shrink-0 overflow-hidden rounded border border-line" aria-hidden="true">
              <CardImage {name} className="h-full w-full" />
            </div>
            <span class="type-caption w-4 shrink-0 text-right text-ink-2">{count}</span>
            <span class="text-sm">{name}</span>
          </button>
        {/each}
      </div>
      {#if deck}<div class="mt-5 border-t border-line pt-3">
        <h4 class="type-rubric text-ink-2">Lesson pool</h4>
        <ul class="mt-2 grid gap-x-5 gap-y-1 sm:grid-cols-2">
          {#each lessons as [name, count] (name)}
            <li><button class="flex w-full items-center gap-3 py-2 text-left hover:bg-panel-muted" onclick={() => inspectCard(name)} aria-label={`Inspect ${name}`}><div class="h-12 w-20 shrink-0 overflow-hidden rounded border border-line" aria-hidden="true"><CardImage {name} className="h-full w-full" /></div><span class="text-sm"><span class="mr-2 text-ink-2">{count}</span>{name}</span></button></li>
          {/each}
        </ul>
      </div>{:else}<p class="type-caption mt-4 text-ink-2">A blue and red practice deck. Select an Elemental deck above to browse its complete list and Lesson pool.</p>{/if}
    </div>
  </div>
</section>

<dialog bind:this={previewDialog} class="m-auto max-h-[95dvh] max-w-[95vw] rounded-lg border border-line bg-panel p-4 text-ink shadow-sheet backdrop:bg-black/70" aria-label={preview ? `Card preview: ${preview}` : 'Card preview'}>
  <div class="mb-3 flex items-center justify-between gap-4"><h3 class="type-title">{preview}</h3><button class="btn border border-line bg-field" onclick={() => previewDialog.close()}>Close preview</button></div>
  {#if preview}
    {#if imageFor(preview)}
      <img src={imageFor(preview)} alt={preview} class="mx-auto max-h-[75dvh] max-w-full rounded-lg" data-testid="full-card-preview" />
    {:else}
      <CardImage name={preview} size="normal" className="mx-auto h-72 w-52 rounded-lg" />
      <p class="type-caption mt-3">Printed card image unavailable.</p>
    {/if}
  {/if}
</dialog>
