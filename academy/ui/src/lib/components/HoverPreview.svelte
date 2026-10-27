<script lang="ts">
  import type { CardPreview } from '$lib/types';
  import Card from './Card.svelte';

  interface Props {
    card?: CardPreview | null;
  }

  let { card = null }: Props = $props();
</script>

{#if card?.name}
  <div class="pointer-events-none fixed bottom-4 right-4 z-20 hidden xl:block">
    <div class="rounded-xl border border-line bg-panel/90 p-3 shadow-raised backdrop-blur">
      <Card name={card.name} power={card.power} toughness={card.toughness} size="normal" />
      {#if card.text_box}
        <p class="mt-3 max-w-72 whitespace-pre-line text-sm text-ink" data-testid="card-rules-text">{card.text_box}</p>
      {/if}
    </div>
  </div>
{/if}
