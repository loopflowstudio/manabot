<script lang="ts">
  import { visibleCardLabel, copyLabel } from '$lib/decision-labels';
  import { tick } from 'svelte';

  import type { CardPreview, Observation } from '$lib/types';
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

  let inspected = $state<{ id: number; card: CardPreview } | null>(null);
  let preview = $state<CardPreview | null>(null);
  let resultDialog: HTMLDivElement | null = $state(null);
  let resultAction: HTMLButtonElement | null = $state(null);
  const stackCards = $derived([...observation.opponent.stack, ...observation.agent.stack]);

  $effect(() => {
    if (!observation.game_over) {
      return;
    }
    void tick().then(() => (resultAction ?? resultDialog)?.focus());
  });

  $effect(() => {
    observation;
    inspected = null;
  });

  function inspectTarget(id: number): void {
    const players = [observation.agent, observation.opponent];
    const card = players.flatMap(player => [...player.hand, ...player.graveyard, ...player.exile, ...player.stack, ...player.battlefield, ...(player.inspected ?? [])]).find(card => card.id === id);
    inspected = card ? { id, card: { name: card.name, text_box: card.text_box, power: card.power, toughness: card.toughness } } : null;
    onSelectTarget?.(id);
  }

  function setPreview(card: CardPreview | null): void {
    preview = card;
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
    const owner = observation.turn.active_player_id === observation.agent.id ? heroLabel : villainLabel;
    const ownership = owner === 'You' ? 'Your turn' : `${owner}'s turn`;
    return `Turn ${observation.turn.turn_number} · ${ownership} · ${stage}`;
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
        definitionNames={observation.definition_names ?? {}}
    label={villainLabel}
    player={observation.opponent}
    deckName={deckNames?.villain ?? null}
    opponent={true}
    {focusedIds}
    {clickableTargets}
    onSelectTarget={inspectTarget}
    {onHoverTarget}
    onPreviewCard={setPreview}
  />

  {#if observation.stack_objects !== undefined}
    <section aria-label="Ordered stack" data-testid="ordered-stack" class="border-t border-line py-3">
      <h3 class="type-title">Stack · top first</h3>
      {#if observation.stack_objects.length === 0}<p class="type-caption text-ink-2">Empty stack</p>{/if}
      <ol class="type-caption space-y-2">
        {#each observation.stack_objects as entry, index}
          <li>{index === 0 ? 'Top' : index + 1} · {entry.label} · {entry.kind.replaceAll('_', ' ')} · Seat {entry.controller + 1}
            {#if entry.source_permanent !== undefined}
              {#if [...observation.agent.battlefield, ...observation.opponent.battlefield].some(permanent => permanent.id === entry.source_permanent)}
                <button class="underline" onclick={() => inspectTarget(entry.source_permanent!)}>Inspect source</button>
              {:else}<span>Source no longer on the battlefield</span>{/if}
            {/if}
            {#if entry.chosen_mode !== undefined} · Mode {entry.chosen_mode + 1}{/if}
          </li>
        {/each}
      </ol>
    </section>
  {/if}

  {#if stackCards.length > 0}
    <section class="border-t border-line" aria-label="The stack">
      <div class="stack-staff">
        <div class="type-rubric pt-2 text-right text-mountain-ink">
          {observation.stack_objects === undefined ? 'Stack cards · order not recorded' : 'Inspect stack cards'}
        </div>
        <div class="flex flex-wrap items-end gap-2 py-2">
          {#each stackCards as card}
            <button
              type="button"
              onclick={() => inspectTarget(card.id)}
              aria-label={card.name}
              class={`relative ${focusedIds.has(card.id) ? 'stack-hit' : ''}`}
              onmouseenter={() => {
                setPreview({
                  name: card.name,
                  text_box: card.text_box,
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
            </button>
          {/each}
        </div>
      </div>
    </section>
  {/if}

  <div class="border-t border-line">
    <PlayerArea
        definitionNames={observation.definition_names ?? {}}
      label={heroLabel}
      player={observation.agent}
      deckName={deckNames?.hero ?? null}
      {focusedIds}
      {clickableTargets}
      onSelectTarget={inspectTarget}
      {onHoverTarget}
      onPreviewCard={setPreview}
    />
  </div>

  {#if observation.agent.inspected?.length}
    <section aria-label="Private inspection" class="border-t border-line py-3">
      <h3 class="type-title">Private inspection · not library order</h3>
      {#each observation.agent.inspected as card}
        <button class="btn btn-secondary m-1" onclick={() => inspectTarget(card.id)}>{card.name}{#if copyLabel(observation.agent.inspected ?? [], card.id)} · {copyLabel(observation.agent.inspected ?? [], card.id)}{/if}</button>
      {/each}
    </section>
  {/if}
  {#if inspected}
    <section aria-label="Inspected card" class="border-y border-line py-3">
      <h3 class="font-medium">{visibleCardLabel(observation, inspected.id) ?? inspected.card.name}</h3>
      <p class="type-caption whitespace-pre-line">{inspected.card.text_box}</p>
      <p class="type-caption text-ink-2">Legal offers determine what this card can do now.</p>
      <button class="btn btn-secondary mt-2" onclick={() => inspected = null}>Close inspection</button>
    </section>
  {/if}

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

<HoverPreview card={preview} />

<style>
  .tempo::before,
  .tempo::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }
  .tempo span {
    min-width: 0;
    overflow-wrap: anywhere;
    text-align: center;
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
