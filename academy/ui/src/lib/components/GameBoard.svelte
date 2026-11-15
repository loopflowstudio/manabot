<script lang="ts">
  import { turnStageLabel, visibleCardLabel } from '$lib/decision-labels';
  import { tick, setContext, untrack } from 'svelte';

  import type { CardPreview, CardState, Observation, PermanentState } from '$lib/types';
  import type { PresentationPlayer } from '$lib/presentation.svelte';

  import HoverPreview from './HoverPreview.svelte';
  import { playerIdentity, type PlayerIdentity } from '$lib/player-identity';
  import PlayerArea from './PlayerArea.svelte';
  import Schedule from './Schedule.svelte';
  import { PRESENTATION_CONTEXT, presentationSubjects } from '$lib/presentation';

  // The score column of the sheet: the opponent's region, the stack — the
  // one thing on the page allowed to burn red — and the hero's region,
  // separated by rules, never boxes.
  interface Props {
    observation: Observation;
    focusedIds?: Set<number>;
    combatMarkers?: Map<number, string>;
    clickableTargets?: Map<number, number[]>;
    onSelectTarget?: (objectId: number) => void;
    onInspect?: () => void;
    onHoverTarget?: (objectId: number | null) => void;
    winner?: number | null;
    deckNames?: { hero: string; villain: string } | null;
    heroIdentity?: PlayerIdentity;
    villainIdentity?: PlayerIdentity;
    overlayActionLabel?: string | null;
    onOverlayAction?: () => void;
    presentationPlayer?: PresentationPlayer;
    workspace?: boolean;
    playMode?: boolean;
    combatPlanning?: boolean;
    canSelectForPlay?: boolean;
    selectedId?: number | null;
    gestureActive?: boolean;
    dragLabel?: string | null;
    onHandPointerDown?: (event: PointerEvent) => void;
    onHandPointerMove?: (event: PointerEvent) => void;
    onHandPointerUp?: (event: PointerEvent) => void;
    onHandPointerCancel?: () => void;
    onHandClickCapture?: (event: MouseEvent) => void;
  }

  let {
    observation,
    focusedIds = new Set<number>(), combatMarkers = new Map<number, string>(),
    clickableTargets = undefined,
    onSelectTarget = undefined,
    onInspect = undefined,
    onHoverTarget = undefined,
    winner = undefined,
    deckNames = null,
    heroIdentity = playerIdentity({ name: 'Hero' }),
    villainIdentity = playerIdentity({ name: 'Opponent' }),
    overlayActionLabel = null,
    onOverlayAction = undefined,
    presentationPlayer = undefined,
    workspace = false,
    playMode = false,
    combatPlanning = false,
    canSelectForPlay = true,
    selectedId = null,
    gestureActive = false,
    dragLabel = null,
    onHandPointerDown,
    onHandPointerMove,
    onHandPointerUp,
    onHandPointerCancel,
    onHandClickCapture,
  }: Props = $props();

  setContext(PRESENTATION_CONTEXT, () => presentationPlayer?.currentEvent ?? null);

  // Keep the permitted object itself: reference identity certifies seating-only
  // updates, while a fetched historical snapshot must close inspection.
  let inspected = $state.raw<CardState | PermanentState | null>(null);
  const visibleCards = $derived([observation.agent, observation.opponent].flatMap(player => [
    ...player.hand, ...player.graveyard, ...player.exile, ...player.stack,
    ...player.battlefield, ...(player.inspected ?? []),
  ]));
  let inspectionButton: HTMLElement | null = null;
  let inspectionNotice = $state('');
  let boardRegion: HTMLElement | null = $state(null);
  let preview = $state<CardPreview | null>(null);
  let resultDialog: HTMLDivElement | null = $state(null);
  let resultAction: HTMLButtonElement | null = $state(null);
  const tablePermanents = $derived([...observation.agent.battlefield, ...observation.opponent.battlefield]);
  const narratedSubjects = $derived(presentationPlayer?.currentEvent ? presentationSubjects(presentationPlayer.currentEvent) : []);
  const narratedStacks = $derived(new Set(narratedSubjects.flatMap(subject => subject.kind === 'stack' ? [subject.id] : [])));
  const stackCards = $derived([...observation.opponent.stack, ...observation.agent.stack]);
  const interactiveResult = $derived(!!overlayActionLabel && !!onOverlayAction);

  $effect(() => {
    if (!observation.game_over || !interactiveResult) {
      return;
    }
    void tick().then(() => (resultAction ?? resultDialog)?.focus());
  });

  $effect(() => {
    const visible = visibleCards;
    untrack(() => {
      // The legacy board cannot certify incarnation across fetched snapshots.
      // A seating-only update retains the same permitted object reference.
      if (!inspected || visible.includes(inspected)) return;
      inspected = null;
      inspectionNotice = 'Historical card details are unavailable at this position.';
      void tick().then(() => (inspectionButton?.isConnected ? inspectionButton : boardRegion)?.focus({preventScroll: true}));
    });
  });

  function selectTarget(id: number): void {
    if (!playMode || !canSelectForPlay) inspectTarget(id);
    onSelectTarget?.(id);
  }

  function inspectTarget(id: number): void {
    onInspect?.();
    inspectionButton = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    inspected = visibleCards.find(card => card.id === id) ?? null;
    inspectionNotice = inspected ? '' : 'Historical card details are unavailable at this position.';
  }

  function setPreview(card: CardPreview | null): void {
    preview = card;
  }

  function keepResultFocus(event: KeyboardEvent): void {
    if (!interactiveResult || event.key !== 'Tab') {
      return;
    }
    event.preventDefault();
    (resultAction ?? resultDialog)?.focus();
  }

  const turnLine = $derived.by(() => {
    const stage = turnStageLabel(observation.turn.phase, observation.turn.step);
    const owner = observation.turn.active_player_id === observation.agent.id ? heroIdentity.text : villainIdentity.text;
    const ownership = owner === 'You' ? 'Your turn' : `${owner}'s turn`;
    return `Turn ${observation.turn.turn_number} · ${ownership} · ${stage}`;
  });
</script>

<!-- svelte-ignore a11y_no_static_element_interactions (delegates pointer gestures from descendant card buttons; explicit actions serve keyboard input) -->
<section bind:this={boardRegion} tabindex="-1" aria-label="Game board" data-testid="game-board" class="relative min-w-0" class:workspace-board={workspace}
  onclickcapture={onHandClickCapture} onpointerdown={onHandPointerDown} onpointermove={onHandPointerMove} onpointerup={onHandPointerUp}
  onpointercancel={onHandPointerCancel} onlostpointercapture={onHandPointerCancel}>
  {#if playMode}
    <div class="play-controls" class:combat-controls={combatPlanning}>
    <div class="play-drop-area type-caption border-b border-line py-1" class:dragging={gestureActive} data-testid="play-drop-area" aria-live="polite">
      {combatPlanning ? 'Plan combat · select a creature, then its destination.' : gestureActive && dragLabel ? `Release to ${dragLabel}` : 'Play area · select a card, then choose its labeled action; drag a single hand play here.'}
      {#if selectedId !== null && selectedId !== observation.agent.id && selectedId !== observation.opponent.id}<button class="btn btn-ghost btn-sm" onclick={() => inspectTarget(selectedId!)}>Inspect card</button>{/if}
    </div>
    <div class="flex flex-wrap gap-2" role="group" aria-label="Player targets">
      {#each [{ player: observation.agent, label: heroIdentity.text }, { player: observation.opponent, label: villainIdentity.text }] as seat}
        {#if clickableTargets?.has(seat.player.id)}<button class="btn btn-secondary btn-sm" onclick={() => selectTarget(seat.player.id)}>Target {seat.label}</button>{/if}
      {/each}
    </div>
    </div>
  {/if}

  <!-- The turn, set as a tempo marking. -->
  <div
    data-narrated={presentationPlayer?.currentEvent?.kind.kind === 'turn_started' && presentationPlayer.currentEvent.kind.turn_number === observation.turn.turn_number || undefined}
    data-phase={observation.turn.phase}
    data-step={observation.turn.step}
    class="tempo flex items-center gap-4 pt-1"
  >
    <span class="type-rubric text-ink-2">
      {turnLine}
    </span>
  </div>

  <details class="schedule-details type-caption"><summary>Turn and combat schedule</summary>
  <Schedule schedule={observation.schedule} heroSeat={observation.agent.player_index}
    heroLabel={heroIdentity.text} villainLabel={villainIdentity.text} />
  </details>

  <PlayerArea
        definitionNames={observation.definition_names ?? {}}
    identity={villainIdentity}
    player={observation.opponent}
    {tablePermanents}
    deckName={deckNames?.villain ?? null}
    opponent={true}
    {focusedIds} {combatMarkers}
    {clickableTargets}
    onSelectTarget={selectTarget}
    {onHoverTarget}
    onPreviewCard={setPreview}
  />

  {#if observation.stack_objects !== undefined}
    <section aria-label="Ordered stack" data-testid="ordered-stack" class="ordered-stack border-t border-line py-1">
      <h3 class="type-title">Stack · top first</h3>
      {#if observation.stack_objects.length === 0}<p class="type-caption text-ink-2">Empty stack</p>{/if}
      <ol class="type-caption space-y-2">
        {#each observation.stack_objects as entry, index}
          <li>{index === 0 ? 'Top' : index + 1} · {entry.label} · {entry.kind.replaceAll('_', ' ')} · Seat {entry.controller + 1}
            {#if entry.source_permanent !== undefined}
              {#if tablePermanents.some(permanent => permanent.id === entry.source_permanent)}
                <button class="underline" onclick={() => { inspectTarget(entry.source_permanent!); onSelectTarget?.(entry.source_permanent!); }}>Inspect source</button>
              {:else}<span>Source no longer on the battlefield</span>{/if}
            {/if}
            {#if entry.chosen_mode !== undefined} · Mode {entry.chosen_mode + 1}{/if}
            {#if entry.x !== undefined} · X = {entry.x}{/if}
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
              data-object-id={card.id}
              data-narrated-seq={narratedStacks.has(card.id) ? presentationPlayer?.currentEvent?.seq : undefined}
              onclick={() => selectTarget(card.id)}
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
                <b class="not-italic font-semibold text-mountain-ink">{card.name}</b>{#if card.x !== undefined}
                  <span> · X = {card.x}</span>{/if}
              </span>
            </button>
          {/each}
        </div>
      </div>
    </section>
  {/if}

  <div class="hero-area border-t border-line">
    <PlayerArea
        definitionNames={observation.definition_names ?? {}}
      identity={heroIdentity}
      player={observation.agent}
      {tablePermanents}
      deckName={deckNames?.hero ?? null}
      compactHand={combatPlanning && workspace}
      {focusedIds} {combatMarkers}
      {clickableTargets}
      onSelectTarget={selectTarget}
      {onHoverTarget}
      onPreviewCard={setPreview}
    />
  </div>

  {#if observation.agent.inspected?.length}
    <section aria-label="Private inspection" class="border-t border-line py-3">
      <h3 class="type-title">Private inspection · not library order</h3>
      {#each observation.agent.inspected as card}
        <button class="btn btn-secondary m-1" onclick={() => selectTarget(card.id)}>{visibleCardLabel(observation, card.id) ?? card.name}</button>
      {/each}
    </section>
  {/if}
  {#if inspectionNotice}<p role="status" class="type-caption">{inspectionNotice}</p>{/if}
  {#if inspected}
    <section aria-label="Inspected card" class="border-y border-line py-3">
      <h3 class="font-medium">{visibleCardLabel(observation, inspected.id) ?? inspected.name}</h3>
      <p class="type-caption whitespace-pre-line">{inspected.text_box}</p>
      {#each observation.unavailable_actions?.filter(item => item.object === inspected?.id) ?? [] as refusal}
        <p class="type-caption mt-2" data-testid="unavailable-action">{refusal.message}</p>
      {/each}
      {#if observation.unavailable_actions === undefined}<p class="type-caption text-ink-2">Unavailable-action explanations not recorded.</p>{/if}
      <p class="type-caption text-ink-2">Legal offers determine what this card can do now.</p>
      <button class="btn btn-secondary mt-2" onclick={() => inspected = null}>Close inspection</button>
    </section>
  {/if}

  {#if observation.game_over}
    <!-- The scrim layer: content is never dimmed by its own opacity. -->
    <div class={interactiveResult ? "absolute inset-0 z-[190] grid place-items-center rounded bg-[color-mix(in_srgb,var(--text)_45%,transparent)]" : "shrink-0 border-t border-line pt-1"}>
      <div
        bind:this={resultDialog}
        role={interactiveResult ? 'alertdialog' : 'status'}
        aria-modal={interactiveResult ? 'true' : undefined}
        aria-labelledby="game-over-heading"
        aria-describedby="game-result"
        tabindex="-1"
        data-testid="game-result-dialog"
        onkeydown={keepResultFocus}
        class={interactiveResult ? "z-[200] rounded-lg border border-line bg-panel p-6 text-center shadow-raised" : "flex flex-wrap items-baseline gap-3"}
      >
        <h2 id="game-over-heading" class={interactiveResult ? "type-display mb-2 text-display" : "type-title text-display"}>Game Over</h2>
        <p id="game-result" data-testid="game-result" class={interactiveResult ? "mb-4 text-ink-2" : "type-caption text-ink-2"}>
          {#if winner === undefined}
            Result not recorded
          {:else if winner === null}
            Draw
          {:else}
            {@const winnerName = winner === observation.agent.player_index ? heroIdentity.text : villainIdentity.text}
            {winnerName === 'You' ? 'You win' : `${winnerName} wins`}{!interactiveResult ? ` · Seat ${winner! + 1}` : ''}
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
  @media (min-width: 1024px) and (min-height: 650px) {
    [data-testid="game-board"] { display: flex; flex-direction: column; }
    [data-testid="game-board"] > :global(.player-area) { flex: 1; }
    [data-testid="game-board"] > :global(.player-area.empty-battlefield) { flex: 0 0 auto; }
    .hero-area { flex: 0 0 auto; display: flex; flex-direction: column; min-height: 0; }
    .workspace-board .hero-area { flex: 1 1 150px; }
    .hero-area > :global(.player-area) { flex: 1; }
    .schedule-details, .ordered-stack { max-height: 90px; overflow-y: auto; flex-shrink: 0; }
    [aria-label="Inspected card"], [aria-label="Private inspection"], [aria-label="The stack"] { max-height: 130px; overflow-y: auto; flex-shrink: 0; }
  }

  .play-controls { display: contents; }
  .combat-controls { display: flex; align-items: center; flex-wrap: wrap; gap: .5rem; flex-shrink: 0; }
  .combat-controls .play-drop-area { border: 0; }
  .play-drop-area { flex-shrink: 0; min-height: 32px; }
  .play-drop-area.dragging { outline: 2px dashed var(--accent); background: var(--bg-field); }
  .tempo[data-narrated] { background: var(--bg-field); }

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
  [data-narrated-seq] { outline: 2px solid var(--info); outline-offset: 2px; }

  .stack-hit {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 6px;
  }
</style>
