<script lang="ts">
  import type { BoardActionTarget, CardState, PlayerState } from '$lib/types';

  import Card from './Card.svelte';
  import CardBack from './CardBack.svelte';

  export let label: string;
  export let player: PlayerState;
  export let opponent = false;
  export let focusedIds = new Set<number>();
  export let clickableTargets: Map<number, BoardActionTarget> | undefined = undefined;
  export let onSelectTarget: ((objectId: number) => void) | undefined = undefined;
  export let onHoverTarget: ((objectId: number | null) => void) | undefined = undefined;
  export let onPreviewCard:
    | ((card: { name: string | null; power: number | null; toughness: number | null } | null) => void)
    | undefined = undefined;

  $: hiddenHandCount = player.hand_hidden_count ?? player.zone_counts.HAND ?? player.hand.length;

  function isClickable(cardId: number): boolean {
    return clickableTargets?.has(cardId) ?? false;
  }

  function preview(card: CardState): void {
    onPreviewCard?.({
      name: card.name,
      power: card.types.is_creature ? card.power : null,
      toughness: card.types.is_creature ? card.toughness : null,
    });
  }
</script>

<article class={`rounded border p-3 ${focusedIds.has(player.id) ? 'border-blue-400' : 'border-slate-700'}`}>
  <div class="mb-3 flex items-center justify-between gap-4">
    <div>
      <h2 class="text-sm uppercase tracking-wide text-slate-400">{label}</h2>
      <div class="text-sm text-slate-300">Library: {player.library_count}</div>
    </div>
    <div class={`rounded bg-slate-900 px-3 py-1 text-2xl font-bold ${focusedIds.has(player.id) ? 'ring-1 ring-blue-400' : ''}`}>
      {player.life}
    </div>
  </div>

  <div class="space-y-4">
    <div>
      <h3 class="mb-2 text-xs uppercase tracking-wide text-slate-400">Hand ({hiddenHandCount})</h3>
      <div class="flex flex-wrap gap-2">
        {#if opponent}
          {#each Array(hiddenHandCount) as _, index}
            <div aria-label={`Hidden card ${index + 1}`}>
              <CardBack />
            </div>
          {/each}
        {:else if player.hand.length > 0}
          {#each player.hand as card}
            <Card
              name={card.name}
              power={card.types.is_creature ? card.power : null}
              toughness={card.types.is_creature ? card.toughness : null}
              focused={focusedIds.has(card.id)}
              clickable={isClickable(card.id)}
              onSelect={() => onSelectTarget?.(card.id)}
              onHoverStart={() => {
                onHoverTarget?.(isClickable(card.id) ? card.id : null);
                preview(card);
              }}
              onHoverEnd={() => {
                onHoverTarget?.(null);
                onPreviewCard?.(null);
              }}
            />
          {/each}
        {:else}
          <div class="rounded border border-dashed border-slate-700 px-3 py-5 text-xs text-slate-500">Empty hand</div>
        {/if}
      </div>
    </div>

    <div>
      <h3 class="mb-2 text-xs uppercase tracking-wide text-slate-400">Graveyard ({player.graveyard.length})</h3>
      <div class="flex flex-wrap gap-2">
        {#if player.graveyard.length === 0}
          <div class="rounded border border-dashed border-slate-700 px-3 py-2 text-xs text-slate-500">Empty graveyard</div>
        {/if}
        {#each player.graveyard as card}
          <Card
            name={card.name}
            power={card.types.is_creature ? card.power : null}
            toughness={card.types.is_creature ? card.toughness : null}
            focused={focusedIds.has(card.id)}
            onHoverStart={() => preview(card)}
            onHoverEnd={() => onPreviewCard?.(null)}
          />
        {/each}
      </div>
    </div>

    <div>
      <h3 class="mb-2 text-xs uppercase tracking-wide text-slate-400">Exile ({player.exile.length})</h3>
      <div class="flex flex-wrap gap-2">
        {#if player.exile.length === 0}
          <div class="rounded border border-dashed border-slate-700 px-3 py-2 text-xs text-slate-500">Empty exile</div>
        {/if}
        {#each player.exile as card}
          <Card
            name={card.name}
            power={card.types.is_creature ? card.power : null}
            toughness={card.types.is_creature ? card.toughness : null}
            focused={focusedIds.has(card.id)}
            onHoverStart={() => preview(card)}
            onHoverEnd={() => onPreviewCard?.(null)}
          />
        {/each}
      </div>
    </div>
  </div>
</article>
