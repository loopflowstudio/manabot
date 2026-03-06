<script lang="ts">
  import type { Observation } from '$lib/types';

  import HoverPreview from './HoverPreview.svelte';
  import PermanentRow from './PermanentRow.svelte';
  import PlayerArea from './PlayerArea.svelte';

  export let observation: Observation;
  export let focusedIds = new Set<number>();
  export let clickableTargets: Map<number, number[]> | undefined = undefined;
  export let onSelectTarget: ((objectId: number) => void) | undefined = undefined;
  export let onHoverTarget: ((objectId: number | null) => void) | undefined = undefined;
  export let winner: number | null | undefined = undefined;
  export let overlayActionLabel: string | null = null;
  export let onOverlayAction: (() => void) | undefined = undefined;

  let previewName: string | null = null;
  let previewPower: number | null = null;
  let previewToughness: number | null = null;
  $: stackCards = [...observation.opponent.stack, ...observation.agent.stack];

  function setPreview(
    card: { name: string | null; power: number | null; toughness: number | null } | null,
  ): void {
    previewName = card?.name ?? null;
    previewPower = card?.power ?? null;
    previewToughness = card?.toughness ?? null;
  }
</script>

<section class="relative space-y-4 rounded border border-slate-700 bg-slate-800 p-4">
  <div class="rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-center text-sm font-semibold text-slate-200">
    Turn {observation.turn.turn_number} · {observation.turn.phase} · {observation.turn.step}
  </div>

  <PlayerArea
    label="Opponent"
    player={observation.opponent}
    opponent={true}
    {focusedIds}
    {clickableTargets}
    {onSelectTarget}
    {onHoverTarget}
    onPreviewCard={setPreview}
  />

  <section class="rounded border border-slate-700 bg-slate-900/60 p-3">
    <h3 class="mb-3 text-xs uppercase tracking-wide text-slate-400">Battlefield</h3>
    <div class="space-y-4">
      <PermanentRow
        label="Opponent"
        permanents={observation.opponent.battlefield}
        {focusedIds}
        {clickableTargets}
        {onSelectTarget}
        {onHoverTarget}
        onPreviewCard={setPreview}
      />
      <PermanentRow
        label="Hero"
        permanents={observation.agent.battlefield}
        {focusedIds}
        {clickableTargets}
        {onSelectTarget}
        {onHoverTarget}
        onPreviewCard={setPreview}
      />
    </div>
  </section>

  <PlayerArea
    label="Hero"
    player={observation.agent}
    {focusedIds}
    {clickableTargets}
    {onSelectTarget}
    {onHoverTarget}
    onPreviewCard={setPreview}
  />

  {#if stackCards.length > 0}
    <section class="rounded border border-indigo-500/40 bg-indigo-900/20 p-3">
      <h3 class="mb-2 text-xs uppercase tracking-wide text-indigo-200">Stack</h3>
      <div class="flex flex-wrap gap-2 text-xs text-slate-100">
        {#each stackCards as card}
          <button
            type="button"
            class={`rounded border px-3 py-2 text-left ${focusedIds.has(card.id) ? 'border-blue-400 bg-slate-800' : 'border-indigo-400/50 bg-slate-900/80'}`}
            on:mouseenter={() => {
              setPreview({
                name: card.name,
                power: card.types.is_creature ? card.power : null,
                toughness: card.types.is_creature ? card.toughness : null,
              });
            }}
            on:mouseleave={() => setPreview(null)}
          >
            {card.name}
          </button>
        {/each}
      </div>
    </section>
  {/if}

  {#if observation.game_over}
    <div class="absolute inset-0 grid place-items-center rounded bg-slate-950/80">
      <div class="rounded border border-slate-600 bg-slate-900 p-6 text-center shadow-xl">
        <h2 class="mb-2 text-2xl font-bold">Game Over</h2>
        <p class="mb-4 text-slate-300">
          {#if winner === null}
            Draw
          {:else if winner === 0}
            Hero wins
          {:else}
            Opponent wins
          {/if}
        </p>
        {#if overlayActionLabel && onOverlayAction}
          <button
            class="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-500"
            on:click={() => onOverlayAction?.()}
          >
            {overlayActionLabel}
          </button>
        {/if}
      </div>
    </div>
  {/if}
</section>

<HoverPreview name={previewName} power={previewPower} toughness={previewToughness} />
