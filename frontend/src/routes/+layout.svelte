<script lang="ts">
  import type { Snippet } from 'svelte';

  import '../app.css';

  let { children }: { children?: Snippet } = $props();
</script>

<svelte:head>
  <title>Etude Fantasia</title>
</svelte:head>

<div class="min-h-screen bg-slate-900 text-slate-100">
  <header class="border-b border-slate-800 bg-slate-950/90 backdrop-blur">
    <div class="mx-auto flex w-full max-w-[1600px] items-center justify-between px-4 py-3">
      <div class="leading-none">
        <div data-testid="brand-name" class="font-serif text-2xl font-semibold tracking-[-0.01em] text-accent-text">Etude Fantasia</div>
        <div class="mt-1 font-mono text-[9px] uppercase tracking-[0.18em] text-slate-400">Play · replay · study</div>
      </div>
      <nav class="flex items-center gap-3 text-sm text-slate-300">
        <a class="rounded border border-transparent px-3 py-2 hover:border-slate-700 hover:bg-slate-800 hover:text-accent-text" href="/">Play</a>
        <a class="rounded border border-transparent px-3 py-2 hover:border-slate-700 hover:bg-slate-800 hover:text-accent-text" href="/replay">Replay</a>
      </nav>
    </div>
  </header>

  {@render children?.()}
</div>
