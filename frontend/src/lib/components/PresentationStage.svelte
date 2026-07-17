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
    class={`pointer-events-none absolute inset-x-4 top-16 z-10 flex justify-center ${player.reducedMotion ? '' : 'motion-move'}`}
  >
    <div
      class={`pointer-events-auto w-full max-w-xl rounded-xl border px-4 py-3 shadow-raised backdrop-blur ${
        beat.importance === 'critical'
          ? 'border-mountain/80 bg-panel/95'
          : beat.importance === 'emphasized'
            ? 'border-plains/60 bg-panel/95'
            : 'border-island/40 bg-panel/95'
      }`}
    >
      <div class="flex items-start justify-between gap-4">
        <div role="status" aria-live="polite" aria-atomic="true">
          <p class="type-rubric text-ink-2">
            Beat {player.currentIndex + 1} of {player.events.length}
          </p>
          <h2 class="type-title mt-1 text-ink">{beat.heading}</h2>
          <p class="mt-1 text-ink">{beat.detail}</p>
        </div>
        <div class="flex shrink-0 gap-2">
          <button
            type="button"
            class={`btn btn-secondary btn-sm ${player.speed > 1 ? 'border-plains bg-plains/20' : ''}`}
            aria-pressed={player.speed > 1}
            onclick={() => player.setFastForward(player.speed === 1)}
          >
            {player.speed > 1 ? 'Fast 4×' : 'Fast-forward'}
          </button>
          <button
            type="button"
            class="btn btn-secondary btn-sm"
            onclick={() => player.skipCurrent()}
          >
            Skip beat
          </button>
          <button
            type="button"
            class="btn btn-secondary btn-sm"
            onclick={() => player.finishSequence()}
          >
            Finish
          </button>
        </div>
      </div>
    </div>
  </aside>
{/if}
