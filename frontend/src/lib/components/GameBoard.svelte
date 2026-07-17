<script lang="ts">
  import { tick } from 'svelte';

  import type { Observation } from '$lib/types';
  import type { PresentationPlayer } from '$lib/presentation.svelte';

  import HoverPreview from './HoverPreview.svelte';
  import PlayerArea from './PlayerArea.svelte';
  import PresentationStage from './PresentationStage.svelte';

  // The score column of the sheet: the opponent's region, the stack — the
  // one thing on the page allowed to burn red — and the hero's region,
  // separated by rules, never boxes.
  interface Props {
    observation: Observation;
    focusedIds?: Set<number>;
    clickableTargets?: Map<number, number[]>;
    onSelectTarget?: (objectId: number) => void;
    onHoverTarget?: (objectId: number | null) => void;
    winner?: number | null;
    deckNames?: { hero: string; villain: string } | null;
    heroLabel?: string;
    villainLabel?: string;
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
    deckNames = null,
    heroLabel = 'Hero',
    villainLabel = 'Opponent',
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

  // Notation, not enumeration: the turn line writes human notation; raw
  // engine identifiers stay in data attributes for Study surfaces.
  function titleWords(value: string): string {
    return value
      .toLowerCase()
      .split('_')
      .filter(Boolean)
      .map((word) => word[0].toUpperCase() + word.slice(1))
      .join(' ');
  }

  const turnLine = $derived.by(() => {
    const phase = titleWords(observation.turn.phase);
    const step = titleWords(observation.turn.step.replace(/_STEP$/, ''));
    const stage = step === phase ? phase : `${phase} · ${step}`;
    return `Turn ${observation.turn.turn_number} · ${stage}`;
  });
</script>

<section data-testid="game-board" class="relative min-w-0">
  {#if presentationPlayer}
    <PresentationStage player={presentationPlayer} />
  {/if}

  <!-- The turn, set as a tempo marking. -->
  <div
    data-phase={observation.turn.phase}
    data-step={observation.turn.step}
    class="tempo flex items-center gap-4 pt-1"
  >
    <span class="type-rubric text-ink-2">
      {turnLine}
    </span>
  </div>

  <PlayerArea
    label={villainLabel}
    player={observation.opponent}
    deckName={deckNames?.villain ?? null}
    opponent={true}
    {focusedIds}
    {clickableTargets}
    {onSelectTarget}
    {onHoverTarget}
    onPreviewCard={setPreview}
  />

  {#if stackCards.length > 0}
    <section class="border-t border-line" aria-label="The stack">
      <div class="stack-staff">
        <div class="type-rubric pt-2 text-right text-mountain-ink">
          The Stack
        </div>
        <div class="flex flex-wrap items-end gap-2 py-2">
          {#each stackCards as card}
            <div
              role="img"
              aria-label={card.name}
              class={`relative ${focusedIds.has(card.id) ? 'stack-hit' : ''}`}
              onmouseenter={() => {
                setPreview({
                  name: card.name,
                  power: card.types.is_creature ? card.power : null,
                  toughness: card.types.is_creature ? card.toughness : null,
                });
              }}
              onmouseleave={() => setPreview(null)}
            >
              <span
                class="type-annotation inline-block rounded border border-mountain/50 bg-mountain/10 px-3 py-2 text-ink"
              >
                <b class="not-italic font-semibold text-mountain-ink">{card.name}</b>
              </span>
            </div>
          {/each}
        </div>
      </div>
    </section>
  {/if}

  <div class="border-t border-line">
    <PlayerArea
      label={heroLabel}
      player={observation.agent}
      deckName={deckNames?.hero ?? null}
      {focusedIds}
      {clickableTargets}
      {onSelectTarget}
      {onHoverTarget}
      onPreviewCard={setPreview}
    />
  </div>

  {#if observation.game_over}
    <!-- The scrim layer: content is never dimmed by its own opacity. -->
    <div class="absolute inset-0 z-[190] grid place-items-center rounded bg-[color-mix(in_srgb,var(--text)_45%,transparent)]">
      <div
        bind:this={resultDialog}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="game-over-heading"
        aria-describedby="game-result"
        tabindex="-1"
        data-testid="game-result-dialog"
        onkeydown={keepResultFocus}
        class="z-[200] rounded-lg border border-line bg-panel p-6 text-center shadow-raised"
      >
        <h2 id="game-over-heading" class="type-display mb-2 text-display">Game Over</h2>
        <p id="game-result" data-testid="game-result" class="mb-4 text-ink-2">
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
            class="btn btn-primary"
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

<style>
  .tempo::before,
  .tempo::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }
  .tempo span {
    white-space: nowrap;
  }

  .stack-staff {
    display: grid;
    grid-template-columns: 88px minmax(0, 1fr);
    column-gap: 18px;
    align-items: end;
    padding: 6px 0;
  }
  @media (max-width: 640px) {
    .stack-staff {
      grid-template-columns: 1fr;
    }
    .stack-staff > :first-child {
      padding-top: 6px;
      text-align: left;
    }
  }
  .stack-hit {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 6px;
  }
</style>
