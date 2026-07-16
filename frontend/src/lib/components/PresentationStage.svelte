<script lang="ts">
  import { onMount } from 'svelte';

  import type { PresentationPlayer } from '$lib/presentation.svelte';

  interface Props {
    player: PresentationPlayer;
  }

  let { player }: Props = $props();
  const beat = $derived(player.currentBeat);
  const event = $derived(player.currentEvent);

  onMount(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const applyPreference = (): void => player.setReducedMotion(media.matches);
    applyPreference();
    media.addEventListener('change', applyPreference);
    return () => media.removeEventListener('change', applyPreference);
  });

  $effect(() => {
    if (!player.playing || !event) {
      return;
    }
    const timer = window.setTimeout(() => player.advance(), player.effectiveDurationMs);
    return () => window.clearTimeout(timer);
  });
</script>

{#if beat && event}
  <aside
    data-testid="presentation-stage"
    data-presentation-seq={beat.seq}
    data-presentation-kind={event.kind.kind}
    data-reduced-motion={player.reducedMotion}
    class={`pointer-events-none absolute inset-x-4 top-16 z-10 flex justify-center ${player.reducedMotion ? '' : 'transition duration-200 ease-out'}`}
  >
    <div
      class={`pointer-events-auto w-full max-w-xl rounded-xl border px-4 py-3 shadow-2xl backdrop-blur ${
        beat.importance === 'critical'
          ? 'border-rose-300/70 bg-rose-950/90'
          : beat.importance === 'emphasized'
            ? 'border-amber-300/60 bg-slate-950/90'
            : 'border-cyan-300/40 bg-slate-950/90'
      }`}
    >
      <div class="flex items-start justify-between gap-4">
        <div role="status" aria-live="polite" aria-atomic="true">
          <p class="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
            Beat {player.currentIndex + 1} of {player.events.length}
          </p>
          <h2 class="mt-1 text-base font-bold text-white">{beat.heading}</h2>
          <p class="mt-1 text-sm text-slate-200">{beat.detail}</p>
        </div>
        <div class="flex shrink-0 gap-2">
          <button
            type="button"
            class={`rounded border px-2.5 py-1.5 text-xs font-semibold ${
              player.speed > 1
                ? 'border-amber-300 bg-amber-300/20 text-amber-100'
                : 'border-slate-600 bg-slate-900 text-slate-200 hover:border-slate-400'
            }`}
            aria-pressed={player.speed > 1}
            onclick={() => player.setFastForward(player.speed === 1)}
          >
            {player.speed > 1 ? 'Fast 4×' : 'Fast-forward'}
          </button>
          <button
            type="button"
            class="rounded border border-slate-600 bg-slate-900 px-2.5 py-1.5 text-xs font-semibold text-slate-200 hover:border-slate-400"
            onclick={() => player.skipCurrent()}
          >
            Skip beat
          </button>
          <button
            type="button"
            class="rounded border border-slate-600 bg-slate-900 px-2.5 py-1.5 text-xs font-semibold text-slate-200 hover:border-slate-400"
            onclick={() => player.finishSequence()}
          >
            Finish
          </button>
        </div>
      </div>
    </div>
  </aside>
{/if}
