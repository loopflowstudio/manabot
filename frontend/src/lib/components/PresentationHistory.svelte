<script lang="ts">
  import { presentationBeat } from '$lib/presentation';
  import type { PresentationPlayer } from '$lib/presentation.svelte';

  let { player }: { player: PresentationPlayer } = $props();
  const recent = $derived(
    player.events.slice(-12).map(event => presentationBeat(event, player.labels)),
  );
</script>

<section aria-label="Recent consequences" class="my-3 border-y border-line py-2" data-testid="presentation-history">
  <details open>
    <summary class="type-rubric cursor-pointer text-ink-2">Recent consequences (up to 12 events)</summary>
    <p class="type-caption mt-1 text-ink-2">The board shows the current position. Narration describes events that already happened.</p>
    {#if recent.length > 0}
      <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard access to scrollable history) -->
      <ol class="mt-2 max-h-40 overflow-y-auto pl-5 text-sm text-ink" tabindex="0" aria-label="Recent game events">
        {#each recent as row (row.seq)}
          <li data-presentation-seq={row.seq} class="py-0.5">{row.detail}</li>
        {/each}
      </ol>
    {:else}
      <p class="type-caption mt-1 text-ink-2">No narrated events are available for this position.</p>
    {/if}
  </details>
</section>
