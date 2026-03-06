<script lang="ts">
  import '../app.css';
</script>

<svelte:head>
  <title>manabot gui</title>
</svelte:head>

<div class="min-h-screen bg-slate-900 text-slate-100">
  <header class="border-b border-slate-800 bg-slate-950/90 backdrop-blur">
    <div class="mx-auto flex w-full max-w-[1600px] items-center justify-between px-4 py-3">
      <div>
        <div class="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">manabot</div>
        <div class="text-lg font-bold text-slate-100">GUI</div>
      </div>
      <nav class="flex items-center gap-3 text-sm text-slate-300">
        <a class="rounded px-3 py-2 hover:bg-slate-800 hover:text-white" href="/">Play</a>
        <a class="rounded px-3 py-2 hover:bg-slate-800 hover:text-white" href="/replay">Replay</a>
      </nav>
    </div>
  </header>

  <slot />
</div>
