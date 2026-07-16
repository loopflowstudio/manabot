<script lang="ts">
  import type { GameLogEntry } from '$lib/types';

  interface Props {
    entries?: GameLogEntry[];
    activeEntryId?: string | null;
  }

  let { entries = [], activeEntryId = null }: Props = $props();

  function actorClass(actor: GameLogEntry['actor']): string {
    switch (actor) {
      case 'hero':
        return 'text-emerald-300';
      case 'villain':
        return 'text-amber-300';
      default:
        return 'text-slate-300';
    }
  }
</script>

<section class="rounded border border-slate-700 bg-slate-800 p-4">
  <h2 class="mb-3 text-base font-semibold text-accent-text">Game Log</h2>

  <!-- svelte-ignore a11y_no_noninteractive_tabindex (axe requires keyboard access to scrollable regions) -->
  <div
    class="max-h-[32rem] space-y-2 overflow-y-auto pr-1"
    role="region"
    aria-label="Game log entries"
    tabindex="0"
  >
    {#if entries.length === 0}
      <p class="text-sm text-slate-400">Actions will appear here.</p>
    {:else}
      {#each entries as entry}
        <div
          data-testid="log-entry"
          class={`rounded border px-3 py-2 text-sm ${entry.id === activeEntryId ? 'border-blue-400 bg-slate-900' : 'border-slate-700 bg-slate-900/60'}`}
        >
          <div class={`font-medium ${actorClass(entry.actor)}`}>{entry.text}</div>
        </div>
      {/each}
    {/if}
  </div>
</section>
