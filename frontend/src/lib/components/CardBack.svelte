<script lang="ts">
  export let focused = false;
  export let clickable = false;
  export let className = '';
</script>

<div
  class={`aspect-[5/7] w-20 rounded-lg border bg-gradient-to-br from-slate-700 via-slate-800 to-slate-950 p-2 text-center shadow ${focused ? 'border-blue-400 ring-1 ring-blue-400/70' : 'border-slate-600'} ${clickable ? 'cursor-pointer hover:border-amber-300' : ''} ${className}`}
>
  <div class="flex h-full items-center justify-center rounded-md border border-slate-500/60 bg-slate-900/70 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-300">
    Hidden
  </div>
</div>
