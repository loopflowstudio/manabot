<script lang="ts">
  import { tick } from 'svelte';

  import type { Observation } from '$lib/types';
  import type { PresentationPlayer } from '$lib/presentation.svelte';

  import HoverPreview from './HoverPreview.svelte';
  import PermanentRow from './PermanentRow.svelte';
  import PlayerArea from './PlayerArea.svelte';
  import PresentationStage from './PresentationStage.svelte';

  interface Props {
    observation: Observation;
    focusedIds?: Set<number>;
    clickableTargets?: Map<number, number[]>;
    onSelectTarget?: (objectId: number) => void;
    onHoverTarget?: (objectId: number | null) => void;
    winner?: number | null;
    overlayActionLabel?: string | null;
    onOverlayAction?: () => void;
    presentationPlayer?: PresentationPlayer;
  }

  let {
    observation,
    focusedIds = new Set<number>(),
    clickableTargets = undefined,
    onSelectTarget = undefined,
    onHoverTarget = undefined,
    winner = undefined,
    overlayActionLabel = null,
    onOverlayAction = undefined,
    presentationPlayer = undefined,
  }: Props = $props();

  let previewName = $state<string | null>(null);
  let previewPower = $state<number | null>(null);
  let previewToughness = $state<number | null>(null);
  let resultDialog: HTMLDivElement | null = $state(null);
  let resultAction: HTMLButtonElement | null = $state(null);
  const stackCards = $derived([...observation.opponent.stack, ...observation.agent.stack]);

  $effect(() => {
    if (!observation.game_over) {
      return;
    }
    void tick().then(() => (resultAction ?? resultDialog)?.focus());
  });

  function setPreview(
    card: { name: string | null; power: number | null; toughness: number | null } | null,
  ): void {
    previewName = card?.name ?? null;
    previewPower = card?.power ?? null;
    previewToughness = card?.toughness ?? null;
  }

  function keepResultFocus(event: KeyboardEvent): void {
    if (event.key !== 'Tab') {
      return;
    }
    event.preventDefault();
    (resultAction ?? resultDialog)?.focus();
  }
</script>

<section data-testid="game-board" class="relative space-y-4 rounded border border-slate-700 bg-slate-800 p-4">
  {#if presentationPlayer}
    <PresentationStage player={presentationPlayer} />
  {/if}

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
          <div
            role="img"
            aria-label={card.name}
            class={`rounded border px-3 py-2 text-left ${focusedIds.has(card.id) ? 'border-blue-400 bg-slate-800' : 'border-indigo-400/50 bg-slate-900/80'}`}
            onmouseenter={() => {
              setPreview({
                name: card.name,
                power: card.types.is_creature ? card.power : null,
                toughness: card.types.is_creature ? card.toughness : null,
              });
            }}
            onmouseleave={() => setPreview(null)}
          >
            {card.name}
          </div>
        {/each}
      </div>
    </section>
  {/if}

  {#if observation.game_over}
    <div class="absolute inset-0 grid place-items-center rounded bg-slate-950/80">
      <div
        bind:this={resultDialog}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="game-over-heading"
        aria-describedby="game-result"
        tabindex="-1"
        data-testid="game-result-dialog"
        onkeydown={keepResultFocus}
        class="rounded border border-slate-600 bg-slate-900 p-6 text-center shadow-xl"
      >
        <h2 id="game-over-heading" class="mb-2 text-2xl font-bold">Game Over</h2>
        <p id="game-result" data-testid="game-result" class="mb-4 text-slate-300">
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
            bind:this={resultAction}
            data-testid="game-result-action"
            class="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-500"
            onclick={() => onOverlayAction?.()}
          >
            {overlayActionLabel}
          </button>
        {/if}
      </div>
    </div>
  {/if}
</section>

<HoverPreview name={previewName} power={previewPower} toughness={previewToughness} />
