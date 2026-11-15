<script lang="ts">
  import type { PresentationPlayer } from '$lib/presentation.svelte';

  interface Props {
    player: PresentationPlayer;
    externallyDriven?: boolean;
  }

  let { player, externallyDriven = false }: Props = $props();
  const beat = $derived(player.currentBeat);
  const event = $derived(player.currentEvent);

  $effect(() => {
    const activePlayer = player;
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const applyPreference = (): void => activePlayer.setReducedMotion(media.matches);
    applyPreference();
    media.addEventListener('change', applyPreference);
    return () => media.removeEventListener('change', applyPreference);
  });

  $effect(() => {
    if (externallyDriven || !player.playing || !event) {
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
    class="border-b border-line pb-3"
  >
    <div
      class={`w-full border-l-2 pl-3 ${
        beat.importance === 'critical'
          ? 'border-mountain'
          : beat.importance === 'emphasized'
            ? 'border-plains'
            : 'border-island'
      }`}
    >
      <div class="flex flex-wrap items-start justify-between gap-4">
        <div role="status" aria-live="polite" aria-atomic="true">
          <p class="type-rubric text-ink-2">
            Narration · {player.remaining} remaining
          </p>
          <h2 class="type-label mt-1 text-ink">{beat.heading}</h2>
          <p class="mt-1 text-ink">{beat.detail}</p>
        </div>
        {#if !externallyDriven}<div class="flex flex-wrap gap-2">
          <button
            type="button"
            class="btn btn-secondary btn-sm"
            aria-pressed={!player.playing}
            onclick={() => player.togglePlayback()}
          >
            {player.playing ? 'Pause narration' : 'Resume narration'}
          </button>
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
        </div>{/if}
      </div>
    </div>
  </aside>
{/if}
