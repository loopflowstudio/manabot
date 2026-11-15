<script lang="ts">
  interface Props { active: boolean; budget: string; }
  let { active, budget }: Props = $props();
  let elapsed = $state(0);
  $effect(() => {
    if (!active) return;
    const started = performance.now();
    elapsed = 0;
    const timer = setInterval(() => elapsed = (performance.now() - started) / 1000, 100);
    return () => clearInterval(timer);
  });
</script>

{#if active}
  <p role="status" class="type-caption py-2 text-ink-2" data-testid="search-thinking">
    Waiting for the bot decision · {budget}<span aria-hidden="true">{` · ${elapsed.toFixed(1)} s`}</span>
  </p>
{/if}
