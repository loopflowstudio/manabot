<script lang="ts">
  import Card from './Card.svelte';

  export let name: string | null = null;
  export let power: number | null = null;
  export let toughness: number | null = null;
</script>

{#if name}
  <div class="pointer-events-none fixed bottom-4 right-4 z-20 hidden xl:block">
    <div class="rounded-xl border border-slate-700 bg-slate-950/90 p-3 shadow-2xl backdrop-blur">
      <Card name={name} power={power} toughness={toughness} size="normal" />
    </div>
  </div>
{/if}
