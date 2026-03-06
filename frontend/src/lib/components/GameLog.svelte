<script lang="ts">
  import type { GameLogEntry } from '$lib/types';

  export let entries: GameLogEntry[] = [];
  export let activeEntryId: string | null = null;

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
  <h2 class="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-300">Game Log</h2>

  <div class="max-h-[32rem] space-y-2 overflow-y-auto pr-1">
    {#if entries.length === 0}
      <p class="text-sm text-slate-400">Actions will appear here.</p>
    {:else}
      {#each entries as entry}
        <div class={`rounded border px-3 py-2 text-sm ${entry.id === activeEntryId ? 'border-blue-400 bg-slate-900' : 'border-slate-700 bg-slate-900/60'}`}>
          <div class={`font-medium ${actorClass(entry.actor)}`}>{entry.text}</div>
        </div>
      {/each}
    {/if}
  </div>
</section>
