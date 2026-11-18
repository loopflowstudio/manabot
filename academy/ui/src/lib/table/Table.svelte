<script lang="ts">
  /**
   * The table: one screen for a game. The board is the centre; cards with a
   * legal offer are outlined and clicking one commits its single offer or
   * opens its several at the card. The rail repeats every offer by number.
   *
   * The table owns only presentation state (open menus, the preview, where
   * cards were drawn a moment ago, and the revision of the person's last move,
   * which scopes "since your last move"). It commits nothing itself: `onact`
   * receives the exact offer a person chose, and the caller binds it to the
   * frame's revision as one Command. What happened is read from the
   * authority's presentation events; motion only shows it.
   *
   * A recorded game draws the same table read-only (`live` false): the caller
   * supplies where the recap starts (`since`), the move list (`moves`, which
   * may jump), the recorded policy (`evidence`, shown on each offer's card and
   * row) and a bottom edge (`bottom`: the transport and the decision strip).
   */
  import { onMount, setContext, untrack, type Snippet } from 'svelte';
  import type { PreviewState } from '$lib/payment-preview';
  import type { PresentationPlayer } from '$lib/presentation.svelte';
  import type { CardState, ConnectionState, ExperienceFrame, GameLogEntry, InteractionOffer, StopsConfig } from '$lib/types';
  import Field from './Field.svelte';
  import Inspect from './Inspect.svelte';
  import Lines from './Lines.svelte';
  import { prefersReducedMotion, Travel } from './motion';
  import { moveList, type MoveTurn } from './moves';
  import { EVIDENCE, type OfferEvidence } from './replay';
  import { paymentPlan, previewable } from './payment';
  import { primaryLabel, promptLine } from './prompt';
  import { nameBook, recentView, tableLines, turnOwners, visibleNames } from './recent';
  import Plate from './Plate.svelte';
  import Rail from './Rail.svelte';
  import TableCard from './TableCard.svelte';
  import { phaseTrack } from './phases';
  import { CARD_RATIO, FRAME_RATIO, handStrip, pileCounts, splitHalf, tableView, type HalfSplit, type PermanentItem, type PlacedOffer, type RowLimits, type StackItem } from './view';

  interface Props {
    frame: ExperienceFrame;
    live: boolean;
    status: string;
    you: { name: string; deck: string | null };
    opponent: { name: string; deck: string | null; description?: string };
    stops: StopsConfig;
    /** The live connection; a recording has none. */
    connection?: ConnectionState | null;
    connectionLabel?: string;
    log: GameLogEntry[];
    canPassTurn: boolean;
    /** The live presentation track: what happened, as the authority reports it. */
    presentation: PresentationPlayer;
    /** Set while the opponent is choosing, or the table is working, in words. */
    thinking: string | null;
    /** The opponent is the one choosing now. */
    opponentChoosing: boolean;
    /** When the work the table waits on started, epoch seconds; null when nothing runs. */
    workStarted: number | null;
    /** The live session is gone; offers are the last recorded decision's. */
    interrupted: boolean;
    /** The automatic payer's answer for the offer last asked about, if any. */
    payment?: PreviewState | null;
    /** Ask for the payment preview of this offer, or of none. Asking sends no action. */
    onpreview?: (offer: InteractionOffer | null) => void;
    onact: (offer: InteractionOffer) => void;
    onpassturn?: () => void;
    /** Where "since your last move" starts, when the caller knows (a recording); else the person's last commit. */
    since?: number | null;
    /** The move list, when the caller keeps it (a recording); else read from `log`. */
    moves?: MoveTurn[];
    /** Jump to the position a move names; moves are buttons when set. */
    onjump?: (id: string) => void;
    /** The move the shown position belongs to. */
    currentMove?: string | null;
    /** The recorded policy at a recorded choice. */
    evidence?: OfferEvidence | null;
    /** The recap's heading, when it is not the person's last move. */
    recapTitle?: string;
    /** Under the board: a recording's transport and decision strip, or live play's strip. */
    bottom?: Snippet;
    notices?: Snippet;
    result?: Snippet;
    menu?: Snippet;
  }

  let {
    frame, live, status, you, opponent, stops, connection = null, connectionLabel = '', log, canPassTurn,
    presentation, thinking, opponentChoosing, workStarted, interrupted, payment = null, onpreview = () => {}, onact, onpassturn = () => {},
    since: sinceGiven, moves: movesGiven, onjump, currentMove = null, evidence = null, recapTitle, bottom, notices, result, menu,
  }: Props = $props();

  // Cards read the recorded policy for their own offers.
  setContext(EVIDENCE, () => evidence);

  // The table is the whole screen; the page behind it must not scroll.
  onMount(() => {
    const root = document.documentElement;
    const previous = root.style.overflow;
    root.style.overflow = 'hidden';
    return () => { root.style.overflow = previous; };
  });

  const view = $derived(tableView(frame));
  const projection = $derived(frame.projection);
  const yourTurn = $derived(projection.turn.active_player_id === projection.agent.id);
  const track = $derived(phaseTrack(projection.turn, stops, yourTurn));
  const names = $derived(new Map([...projection.agent.battlefield, ...projection.opponent.battlefield]
    .map(permanent => [permanent.id, permanent.name ?? 'permanent'])));
  const whose = $derived(frame.status === 'game_over' ? 'Game over'
    : !frame.prompt ? 'Waiting'
      : frame.prompt.actor === projection.agent.player_index
        ? you.name === 'You' ? (live ? 'Your decision' : 'Your decision · waiting') : `${you.name}'s decision`
        : `${opponent.name}'s decision`);
  // ———————————————— Payment preview

  // The offer the board previews: the one a person points at or focuses, when
  // the authority can preview it (a cast, a target, a tap), else the pending
  // payment itself. Previewing is a read-only query; it commits nothing.
  let pointed = $state<PlacedOffer | null>(null);
  const payOffer = $derived(view.primary?.offer.verb === 'pay' ? view.primary : null);
  const previewing = $derived(!live ? null : pointed && previewable(pointed.offer) ? pointed : payOffer);
  $effect(() => { onpreview(previewing?.offer ?? null); });
  const answered = $derived(payment && previewing && payment.offer === previewing.offer.id && !payment.loading ? payment.preview : null);
  const plan = $derived(paymentPlan(answered, frame));
  // Piles part around the pending payment's own plan only, and keep that
  // layout while a person points at other lands: a preview never moves a card
  // from under the pointer.
  let payPlan = $state.raw<{ revision: number; sources: ReadonlyMap<number, unknown> } | null>(null);
  $effect(() => {
    if (plan && previewing === payOffer) payPlan = { revision: frame.revision, sources: plan.sources };
  });
  const unpiled = $derived(payPlan && payPlan.revision === frame.revision ? payPlan.sources : null);
  const primaryText = $derived(primaryLabel(frame, view, payOffer && previewing === payOffer ? answered : null));
  const line = $derived(promptLine(frame, view, plan !== null && previewing === payOffer && plan.sources.size > 0));
  const paymentNote = $derived(payOffer && previewing === payOffer
    ? payment?.error ? 'The payment preview is unavailable; Pay still pays automatically.'
      : answered && answered.state !== 'exact' && answered.state !== 'not_applicable' ? answered.explanation : null
    : null);
  function point(offers: PlacedOffer[] | null): void {
    pointed = offers?.length === 1 ? offers[0] : null;
  }

  // ———————————————— What happened since your last move

  // The revision your last move was made at; events after it are "since your last move".
  let since = $state.raw<{ revision: number; frame: ExperienceFrame; logLength: number } | null>(null);
  const players = $derived({ you: you.name, opponent: opponent.name });
  const sinceRevision = $derived(sinceGiven !== undefined ? sinceGiven : since?.revision ?? null);
  const book = $derived(nameBook(since && sinceGiven === undefined ? [since.frame, frame] : [frame], since && sinceGiven === undefined ? log.slice(since.logLength) : []));
  const recent = $derived(recentView(frame, sinceRevision, presentation.events, index => presentation.labelsAt(index), players, book));
  const lines = $derived(tableLines(frame, presentation.events, recent.resolved, sinceRevision));
  const moves = $derived(movesGiven ?? moveList(log, players, turnOwners(frame, presentation.events), visibleNames(frame, presentation.labels)));

  // ———————————————— Motion

  let root = $state<HTMLElement | null>(null);
  const travel = new Travel();
  $effect.pre(() => {
    void frame.revision;
    travel.snapshot(root);
  });
  $effect(() => {
    void frame.revision;
    travel.play(root, prefersReducedMotion());
  });

  // Battlefield frames are landscape so names stay legible as rows shrink;
  // lands and other support permanents are smaller tiles of the same frame.
  const CREATURES: RowLimits = { gap: 8, minCard: 64, maxCard: 172, maxLines: 3, ratio: FRAME_RATIO };
  const SUPPORT: RowLimits = { gap: 6, minCard: 52, maxCard: 128, maxLines: 2, ratio: FRAME_RATIO };
  const ROW_GAP = 6;

  // Each half divides its height between its two rows by what they hold.
  let theirRows = $state({ width: 0, height: 0 });
  let yourRows = $state({ width: 0, height: 0 });
  function split(side: typeof view.you, size: { width: number; height: number }): HalfSplit | null {
    if (size.width <= 0 || size.height <= 0) return null;
    return splitHalf(pileCounts(side.creatures), pileCounts(side.support), size.width - 22, size.height - 16,
      { gap: ROW_GAP, creatures: CREATURES, support: SUPPORT });
  }
  const theirSplit = $derived(split(view.opponent, theirRows));
  const yourSplit = $derived(split(view.you, yourRows));
  const rowsStyle = (half: HalfSplit | null, size: { height: number }, creaturesFirst: boolean): string | undefined => {
    if (!half) return undefined;
    const rest = Math.max(0, size.height - 12 - ROW_GAP - half.creatures);
    return creaturesFirst ? `${half.creatures}px ${rest}px` : `${rest}px ${half.creatures}px`;
  };

  // ———————————————— Menus and preview

  type Menu = { title: string; offers: PlacedOffer[]; x: number; y: number };
  type Zone = { title: string; cards: CardState[]; x: number; y: number };
  /** What is inspected: a name, and the object it names when there is one. */
  type Subject = { name: string; text: string | null; permanent: number | null; card: number | null };
  let menuAt = $state<Menu | null>(null);
  let zoneAt = $state<Zone | null>(null);
  let menuElement = $state<HTMLElement | null>(null);

  // Inspecting: a preview floats beside the card under the pointer or focus,
  // or the rail's docked pane keeps the last card inspected.
  const INSPECT_KEY = 'academy.table.inspect';
  let inspectMode = $state<'float' | 'dock'>(typeof localStorage !== 'undefined' && localStorage.getItem(INSPECT_KEY) === 'dock' ? 'dock' : 'float');
  function toggleInspect(): void {
    inspectMode = inspectMode === 'dock' ? 'float' : 'dock';
    localStorage.setItem(INSPECT_KEY, inspectMode);
  }
  let hovered = $state<{ subject: Subject; x: number; y: number } | null>(null);
  let resultFolded = $state(false);
  $effect(() => { if (!result) resultFolded = false; });
  let docked = $state<Subject | null>(null);

  // A new frame replaces the cards a menu or preview was opened from.
  $effect(() => {
    void frame.revision;
    menuAt = null;
    hovered = null;
    pointed = null;
    untrack(stopDwell);
  });
  $effect(() => {
    if (menuAt) menuElement?.querySelector<HTMLButtonElement>('button')?.focus({ preventScroll: true });
  });

  function act(placed: PlacedOffer): void {
    if (!live) return;
    menuAt = null;
    hush();
    since = { revision: frame.revision, frame, logLength: log.length };
    onact(placed.offer);
  }

  function passTurn(): void {
    if (!canPassTurn) return;
    since = { revision: frame.revision, frame, logLength: log.length };
    onpassturn();
  }

  function activate(title: string, offers: PlacedOffer[], element: HTMLElement): void {
    hush();
    if (!live || !offers.length) return;
    if (offers.length === 1) { act(offers[0]); return; }
    const rect = element.getBoundingClientRect();
    menuAt = { title, offers, x: Math.min(rect.left, window.innerWidth - 300), y: rect.bottom + 6 > window.innerHeight - 160 ? Math.max(8, rect.top - 8 - 36 * offers.length - 30) : rect.bottom + 6 };
  }

  // Inspect opens only when asked: after the pointer rests on a card for a
  // moment, or at once for keyboard focus. After a commit the pointer stays
  // quiet until it moves, so the card that slides under a resting pointer
  // opens nothing; any frame change closes it.
  const DWELL_MS = 380;
  let dwell: ReturnType<typeof setTimeout> | null = null;
  let pointer = { x: -1, y: -1 };
  let quietAt: { x: number; y: number } | null = null;
  function onPointerMove(event: PointerEvent): void {
    pointer = { x: event.clientX, y: event.clientY };
    if (quietAt && Math.hypot(pointer.x - quietAt.x, pointer.y - quietAt.y) > 4) quietAt = null;
  }
  function hush(): void {
    stopDwell();
    hovered = null;
    quietAt = { ...pointer };
  }
  function stopDwell(): void {
    if (dwell !== null) clearTimeout(dwell);
    dwell = null;
  }
  function show(subject: Subject | null, element: HTMLElement | null): void {
    stopDwell();
    if (!subject || !element) { hovered = null; return; }
    docked = subject;
    if (element.matches(':focus-visible')) { place(subject, element); return; }
    // Quiet is read when the dwell ends: the pointer may still be arriving.
    dwell = setTimeout(() => { dwell = null; if (!quietAt && element.isConnected) place(subject, element); }, DWELL_MS);
  }
  function place(subject: Subject, element: HTMLElement): void {
    const rect = element.getBoundingClientRect();
    const flip = rect.right + 290 > window.innerWidth - 290;
    hovered = { subject, x: flip ? rect.left - 282 : rect.right + 12, y: Math.max(48, Math.min(rect.top - 40, window.innerHeight - 440)) };
  }

  // The inspected object as it is in this frame, by id; a name alone when it has left.
  const permanents = $derived(new Map([...projection.agent.battlefield, ...projection.opponent.battlefield].map(permanent => [permanent.id, permanent])));
  const cards = $derived(new Map([projection.agent, projection.opponent]
    .flatMap(player => [...player.hand, ...player.graveyard, ...player.exile, ...player.stack]).map(card => [card.id, card])));
  const shown = $derived(inspectMode === 'dock' ? docked : hovered?.subject ?? null);
  const shownPermanent = $derived(shown?.permanent != null ? permanents.get(shown.permanent) ?? null : null);
  const shownCard = $derived(shown?.card != null ? cards.get(shown.card) ?? null : null);

  function onPermanent(item: PermanentItem, element: HTMLElement): void {
    activate(item.permanent.name ?? 'Permanent', item.offers, element);
  }
  function previewPermanent(item: PermanentItem | null, element: HTMLElement | null): void {
    point(item?.offers ?? null);
    show(item ? { name: item.permanent.name ?? 'Permanent', text: item.permanent.text_box ?? null, permanent: item.permanent.id, card: null } : null, element);
  }
  function previewCard(card: CardState | null, element: HTMLElement | null): void {
    point(card ? view.hand.find(item => item.card.id === card.id)?.offers ?? null : null);
    show(card ? { name: card.name, text: card.text_box ?? null, permanent: null, card: card.id } : null, element);
  }
  function previewName(name: string | null, element: HTMLElement | null): void {
    show(name ? { name, text: textOf(name), permanent: null, card: null } : null, element);
  }
  function onStack(item: StackItem, element: HTMLElement): void {
    activate(item.entry.label, item.offers, element);
  }

  /** Rules text for a name in the log, from any card of that name this viewer can see. */
  function textOf(name: string): string | null {
    const players = [projection.agent, projection.opponent];
    const card = players.flatMap(player => [...player.hand, ...player.graveyard, ...player.exile, ...player.stack]).find(candidate => candidate.name === name);
    const permanent = players.flatMap(player => player.battlefield).find(candidate => candidate.name === name);
    return card?.text_box ?? permanent?.text_box ?? null;
  }
  function openZone(title: string, cards: CardState[], element: HTMLElement): void {
    const rect = element.getBoundingClientRect();
    zoneAt = zoneAt?.title === title ? null : { title, cards, x: rect.right + 8, y: Math.min(rect.top, window.innerHeight - 320) };
  }

  // ———————————————— Hand

  // The hand yields height to the board: an empty hand is a label, and a hand
  // grows only into height the board's rows would not use (`handStrip`).
  const boardCount = $derived(projection.agent.battlefield.length + projection.opponent.battlefield.length);
  let boardHeight = $state(0);
  const handTrack = $derived(view.hand.length === 0 ? '30px' : `${handStrip(boardHeight, [
    { creatures: pileCounts(view.opponent.creatures), support: pileCounts(view.opponent.support), width: theirRows.width },
    { creatures: pileCounts(view.you.creatures), support: pileCounts(view.you.support), width: yourRows.width },
  ], { gap: ROW_GAP, creatures: CREATURES, support: SUPPORT },
  { min: boardCount >= 24 ? 96 : 112, max: Math.max(112, Math.min(236, Math.round(boardHeight * 0.32))), step: 8 })}px`);
  let handWidth = $state(0);
  let handHeight = $state(0);
  const handCard = $derived(Math.max(56, Math.min(150, Math.floor((handHeight - 22) * CARD_RATIO))));
  const handStep = $derived(view.hand.length <= 1 ? handCard + 10
    : Math.min(handCard + 10, Math.max(24, (handWidth - 24 - handCard) / (view.hand.length - 1))));

  // ———————————————— Keyboard

  function typing(target: EventTarget | null): boolean {
    return target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
  }

  /** Space presses a focused control, except a card with nothing to do: then it is the primary. */
  function ownsSpace(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    const card = target.closest<HTMLElement>('[data-testid=table-card]');
    if (card) return card.dataset.offerCount !== '0';
    return target instanceof HTMLButtonElement || target instanceof HTMLAnchorElement || target.tagName === 'SUMMARY';
  }

  function onKeydown(event: KeyboardEvent): void {
    if (event.metaKey || event.ctrlKey || event.altKey || typing(event.target)) return;
    if (event.key === 'Escape') { menuAt = null; zoneAt = null; hovered = null; stopDwell(); return; }
    if (event.key === 'F6') { event.preventDefault(); passTurn(); return; }
    if (/^[1-9]$/.test(event.key)) {
      const placed = (menuAt?.offers ?? view.list).find(candidate => candidate.number === Number(event.key)) ?? null;
      if (placed && live) { event.preventDefault(); act(placed); }
      return;
    }
    if (event.key === ' ' && !ownsSpace(event.target) && view.primary && live) {
      event.preventDefault();
      act(view.primary);
    }
  }
</script>

<svelte:window onkeydown={onKeydown} onpointermove={onPointerMove} />

<div class="table" data-testid="table" data-action-space={frame.action_space} data-live={live} data-thinking={thinking !== null}
  data-turn={projection.turn.turn_number} data-step={projection.turn.step} data-turn-owner={yourTurn ? 'you' : 'opponent'} bind:this={root}>
  <header class="topbar">
    <a href="/games" class="brand" aria-label="Academy · Games">Academy</a>
    <p class="matchup type-caption" data-testid="deck-names">
      <span class="text-ink">{you.name}</span>{#if you.deck}<span class="text-ink-3">{' · '}{you.deck}</span>{/if}
      <span class="text-ink-3">{' vs '}</span>
      <span class="text-ink" title={opponent.description}>{opponent.name}</span>{#if opponent.deck}<span class="text-ink-3">{' · '}{opponent.deck}</span>{/if}
    </p>
    <nav class="track" aria-label={`Turn ${projection.turn.turn_number}, ${yourTurn ? 'yours' : "opponent's"}`}>
      <span class="type-rubric turn" data-testid="turn-number">Turn {projection.turn.turn_number} · {yourTurn ? 'yours' : 'theirs'}</span>
      <ol>
        {#each track as mark (mark.key)}
          <li class="type-rubric" class:current={mark.current} aria-current={mark.current ? 'step' : undefined}
            title={`${mark.label}${mark.stop ? ' · stop' : ''}`}>
            {mark.short}{#if mark.stop}<i class="stop" aria-hidden="true"></i><span class="sr-only">, stop</span>{/if}
          </li>
        {/each}
      </ol>
    </nav>
    <div class="right">
      {#if connection}
        <span class="connection" data-state={connection}>
          <i aria-hidden="true"></i>
          <span data-testid="connection-badge" data-connection-state={connection} role="status" aria-live="polite" aria-atomic="true"
            aria-label={`Connection status: ${connectionLabel}`} class="type-rubric">{connectionLabel}</span>
        </span>
      {/if}
      <button type="button" class="inspect-toggle type-caption" data-testid="inspect-toggle" aria-pressed={inspectMode === 'dock'}
        title="Show inspected cards in a pane on the rail, or beside the card" onclick={toggleInspect}>{inspectMode === 'dock' ? 'Inspect: docked' : 'Inspect: beside card'}</button>
      {#if menu}{@render menu()}{/if}
    </div>
  </header>

  <div class="board" id="table-board" data-testid="game-board" aria-label="Board" bind:clientHeight={boardHeight} style:grid-template-rows={`minmax(0, 1fr) minmax(0, 1fr) ${handTrack}`}>
    <section class="half opponent" aria-label={`${opponent.name}'s battlefield`}>
      <Plate player={projection.opponent} name={opponent.name} deck={opponent.deck} offers={view.opponent.offers} {live} damage={recent.damage.players.get(projection.opponent.player_index) ?? 0}
        active={!yourTurn} choosing={opponentChoosing} side="opponent" onactivate={element => activate(opponent.name, view.opponent.offers, element)} onzone={(zone, cards, element) => openZone(`${opponent.name} · ${zone}`, cards, element)} />
      <div class="rows" bind:clientWidth={theirRows.width} bind:clientHeight={theirRows.height} style:grid-template-rows={rowsStyle(theirSplit, theirRows, false)}>
        <div class="row support"><Field piles={view.opponent.support} label={`${opponent.name}'s lands and other permanents`} limits={SUPPORT} {live} asked={view.asked} touched={recent.touched} damage={recent.damage.objects} planned={plan?.sources ?? null} {unpiled} toward="down" {names} onactivate={onPermanent} onpreview={previewPermanent} /></div>
        <div class="row creatures"><Field piles={view.opponent.creatures} label={`${opponent.name}'s creatures`} limits={CREATURES} {live} asked={view.asked} touched={recent.touched} damage={recent.damage.objects} planned={plan?.sources ?? null} {unpiled} toward="down" {names} onactivate={onPermanent} onpreview={previewPermanent} /></div>
      </div>
    </section>
    <section class="half you" aria-label="Your battlefield">
      <Plate player={projection.agent} name={you.name} deck={you.deck} offers={view.you.offers} {live} damage={recent.damage.players.get(projection.agent.player_index) ?? 0}
        active={yourTurn} side="you" onactivate={element => activate(you.name, view.you.offers, element)} onzone={(zone, cards, element) => openZone(`Your ${zone.toLowerCase()}`, cards, element)} />
      <div class="rows" bind:clientWidth={yourRows.width} bind:clientHeight={yourRows.height} style:grid-template-rows={rowsStyle(yourSplit, yourRows, true)}>
        <div class="row creatures"><Field piles={view.you.creatures} label="Your creatures" limits={CREATURES} {live} asked={view.asked} touched={recent.touched} damage={recent.damage.objects} planned={plan?.sources ?? null} {unpiled} toward="up" {names} onactivate={onPermanent} onpreview={previewPermanent} /></div>
        <div class="row support"><Field piles={view.you.support} label="Your lands and other permanents" limits={SUPPORT} {live} asked={view.asked} touched={recent.touched} damage={recent.damage.objects} planned={plan?.sources ?? null} {unpiled} toward="up" {names} onactivate={onPermanent} onpreview={previewPermanent} /></div>
      </div>
    </section>
    <section class="hand" aria-label={`Your hand, ${view.hand.length} cards`} bind:clientWidth={handWidth} bind:clientHeight={handHeight}>
      <span class="type-rubric hand-label text-ink-3">Hand · {view.hand.length}</span>
      <div class="hand-cards" style:width={`${handCard + Math.max(0, view.hand.length - 1) * handStep}px`}>
        {#each view.hand as item, index (item.card.id)}
          <div class="hand-card" class:playable={live && item.offers.length > 0} style:left={`${index * handStep}px`}>
            <TableCard name={item.card.name} width={handCard} card={item.card} offers={item.offers} {live} asked={view.asked.has(item.card.id)}
              onactivate={element => activate(item.card.name, item.offers, element)}
              onpreview={element => previewCard(element ? item.card : null, element)} />
          </div>
        {/each}
      </div>
    </section>

    {#if result}
      <!-- Over the board, not over the screen: the final position stays readable around it. -->
      <section class="result-card" class:folded={resultFolded} aria-label="Game result" data-testid="result-card">
        {#if resultFolded}
          <button type="button" class="btn btn-secondary btn-sm" onclick={() => (resultFolded = false)}>Show the result</button>
        {:else}
          {@render result()}
          <button type="button" class="fold type-caption" onclick={() => (resultFolded = true)}>Look at the board</button>
        {/if}
      </section>
    {/if}

    {#if view.tray.length && !result}
      <section class="tray" aria-label="Choices from outside the table" data-testid="tray">
        <h2 class="type-rubric text-ink-2">{frame.prompt?.title ?? 'Choose'}</h2>
        <div class="tray-items">
          {#each view.tray as item (item.placed.offer.id)}
            {#if item.name}
              <div class="tray-card">
                <TableCard name={item.name} width={104} offers={[item.placed]} {live}
                  onactivate={() => act(item.placed)}
                  onpreview={element => previewName(element ? item.name : null, element)} />
                <span class="type-caption">{item.placed.offer.label}</span>
              </div>
            {:else}
              <button type="button" class="btn btn-secondary" disabled={!live} onclick={() => act(item.placed)}>
                <kbd>{item.placed.number}</kbd>{item.placed.offer.label}
              </button>
            {/if}
          {/each}
        </div>
      </section>
    {/if}
  </div>

  <div class="rail-slot" id="table-choices">
    <Rail {view} {frame} {live} {line} {primaryText} {status} {whose} {thinking} {workStarted} {interrupted} {canPassTurn} {recent} {moves}
      opponentName={opponent.name} onact={act} onpassturn={passTurn} {paymentNote} onpointoffer={placed => (pointed = placed)}
      onstack={onStack} onpreviewname={previewName} {notices} over={result !== undefined} dock={inspectMode === 'dock' ? dock : undefined}
      {evidence} {recapTitle} {onjump} {currentMove} />
  </div>

  {#if bottom}<div class="bottom">{@render bottom()}</div>{/if}
</div>

<Lines {lines} {root} revision={frame.revision} />

{#if menuAt}
  <div class="menu" role="menu" tabindex="-1" aria-label={`Choices for ${menuAt.title}`} bind:this={menuElement} data-testid="card-menu"
    style:left={`${menuAt.x}px`} style:top={`${menuAt.y}px`}>
    <p class="type-rubric text-ink-2">{menuAt.title}</p>
    {#each menuAt.offers as placed (placed.offer.id)}
      <button type="button" role="menuitem" onclick={() => act(placed)} data-offer-id={placed.offer.id}><kbd>{placed.number}</kbd>{placed.offer.label}</button>
    {/each}
    <button type="button" role="menuitem" class="cancel" onclick={() => (menuAt = null)}><kbd>Esc</kbd>Cancel</button>
  </div>
{/if}

{#if zoneAt}
  <div class="menu zone" role="dialog" aria-label={zoneAt.title} style:left={`${zoneAt.x}px`} style:top={`${zoneAt.y}px`}>
    <p class="type-rubric text-ink-2">{zoneAt.title}</p>
    <ol>
      {#each zoneAt.cards as card (card.id)}
        <li><button type="button" onmouseenter={event => previewCard(card, event.currentTarget)} onmouseleave={() => previewCard(null, null)}
          onfocus={event => previewCard(card, event.currentTarget)} onblur={() => previewCard(null, null)}>{card.name}</button></li>
      {/each}
    </ol>
    <button type="button" class="cancel" onclick={() => (zoneAt = null)}><kbd>Esc</kbd>Close</button>
  </div>
{/if}

{#snippet dock()}
  <!-- svelte-ignore a11y_no_noninteractive_tabindex (a scrolling region must be reachable by keyboard) -->
  <section class="dock" aria-label="Inspect" data-testid="inspect-dock" tabindex="0">
    {#if docked}
      <Inspect name={docked.name} text={shownPermanent?.text_box ?? shownCard?.text_box ?? docked.text} permanent={shownPermanent} card={shownCard}
        blocks={shownPermanent?.blocking !== undefined ? names.get(shownPermanent.blocking) ?? null : null} art={72} />
    {:else}
      <p class="type-annotation text-ink-3">Hover or focus a card to inspect it.</p>
    {/if}
  </section>
{/snippet}

{#if inspectMode === 'float' && hovered}
  <div class="preview" aria-hidden="true" style:left={`${hovered.x}px`} style:top={`${hovered.y}px`} data-testid="card-preview">
    <Inspect name={hovered.subject.name} text={shownPermanent?.text_box ?? shownCard?.text_box ?? hovered.subject.text} permanent={shownPermanent} card={shownCard}
      blocks={shownPermanent?.blocking !== undefined ? names.get(shownPermanent.blocking) ?? null : null} />
  </div>
{/if}

<style>
  .table {
    position: fixed; inset: 0; z-index: 10; display: grid;
    grid-template-columns: minmax(0, 1fr) clamp(250px, 21vw, 300px);
    grid-template-rows: 40px minmax(0, 1fr) auto;
    background: var(--bg); color: var(--text);
  }
  .topbar {
    grid-column: 1 / -1; display: flex; align-items: center; gap: 16px; padding: 0 14px;
    border-bottom: 1px solid var(--border); background: var(--bg-surface); min-width: 0;
  }
  .brand { font-family: var(--font-serif); font-weight: 600; font-size: 19px; color: var(--accent-text); }
  .matchup { min-width: 0; flex: 0 1 auto; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  /* The matchup gives way first: the phase track and the controls keep their width. */
  .track { display: flex; align-items: center; gap: 12px; margin: 0 auto; flex-shrink: 0; }
  .track .turn { color: var(--text); white-space: nowrap; }
  .track ol { display: flex; gap: 2px; }
  .track li { position: relative; padding: 2px 7px; border-radius: 10px; color: var(--text-tertiary); white-space: nowrap; }
  .track li.current { background: var(--accent); color: var(--ivory); }
  .track .stop { position: absolute; left: 50%; bottom: -3px; width: 4px; height: 4px; margin-left: -2px; border-radius: 2px; background: var(--accent); }
  .track li.current .stop { background: var(--ivory); }
  .right { display: flex; align-items: center; gap: 14px; flex-shrink: 0; }
  .connection { display: inline-flex; align-items: center; gap: 6px; color: var(--text-tertiary); }
  .connection i { width: 6px; height: 6px; border-radius: 3px; background: var(--neutral); }
  .connection[data-state='connected'] i { background: var(--success); }
  .connection[data-state='connecting'] i, .connection[data-state='reconnecting'] i { background: var(--warning); }

  .board {
    position: relative; min-height: 0; min-width: 0; display: grid;
    transition: grid-template-rows var(--motion-move) var(--motion-move-ease);
  }
  .half { display: grid; grid-template-columns: 148px minmax(0, 1fr); min-height: 0; border-bottom: 1px solid var(--border); }
  .half.opponent { background: color-mix(in srgb, var(--bg-muted) 55%, var(--bg)); }
  .half :global(.plate) { border-right: 1px solid var(--border); }
  .rows { display: grid; min-height: 0; min-width: 0; padding: 6px 10px; gap: 6px; }
  .opponent .rows { grid-template-rows: minmax(0, 42fr) minmax(0, 58fr); }
  .you .rows { grid-template-rows: minmax(0, 58fr) minmax(0, 42fr); }
  .row { min-height: 0; min-width: 0; }
  .row.support { border-top: 1px dashed transparent; }
  .opponent .row.creatures { border-top: 1px solid color-mix(in srgb, var(--border) 60%, transparent); padding-top: 4px; }
  .you .row.support { border-top: 1px solid color-mix(in srgb, var(--border) 60%, transparent); padding-top: 4px; }

  .hand { position: relative; min-height: 0; display: flex; align-items: flex-end; justify-content: center; padding: 0 12px 8px; overflow: hidden; }
  .hand-label { position: absolute; left: 14px; bottom: 10px; }
  .hand-cards { position: relative; height: 100%; }
  .hand-card { position: absolute; bottom: 0; transition: transform var(--motion-fast) ease-out; }
  .hand-card:hover, .hand-card:focus-within { transform: translateY(-10px); z-index: 30; }

  .tray {
    position: absolute; left: 50%; top: 50%; transform: translate(-50%, -60%); z-index: 40;
    padding: 12px 16px 14px; border: 1px solid var(--border-strong); border-radius: 10px;
    background: var(--bg-surface); box-shadow: var(--shadow-raised); max-width: 90%;
  }
  .tray-items { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 8px; align-items: flex-start; }
  .tray-card { display: grid; justify-items: center; gap: 6px; width: 120px; text-align: center; }

  .result-card {
    position: absolute; left: 50%; top: 44%; transform: translate(-50%, -50%); z-index: 45; width: min(440px, 80%);
    display: grid; gap: 10px; padding: 18px 20px 14px; border: 1px solid var(--border-strong); border-radius: 12px;
    background: color-mix(in srgb, var(--bg-surface) 94%, transparent); box-shadow: var(--shadow-raised);
    animation: appear var(--motion-move) ease-out;
  }
  @keyframes appear { from { opacity: 0; } }
  .result-card.folded { top: auto; bottom: 12px; left: auto; right: 12px; transform: none; width: auto; padding: 6px; }
  .result-card .fold { justify-self: end; color: var(--text-secondary); }
  .result-card .fold:hover { color: var(--text); }

  .rail-slot { min-height: 0; grid-row: 2 / span 2; grid-column: 2; }
  .bottom { grid-row: 3; grid-column: 1; min-width: 0; border-top: 1px solid var(--border); background: var(--bg-surface); }

  .menu {
    position: fixed; z-index: 300; min-width: 220px; max-width: 320px; padding: 8px; display: grid; gap: 2px;
    border: 1px solid var(--border-strong); border-radius: 8px; background: var(--bg-surface); box-shadow: var(--shadow-raised);
  }
  .menu, .preview { animation: open var(--motion-fast) ease-out; transform-origin: top left; }
  @keyframes open { from { opacity: 0; transform: scale(0.97); } }
  .menu p { padding: 0 6px 4px; }
  .menu button { display: flex; align-items: center; gap: 8px; text-align: left; padding: 6px 8px; border-radius: 5px; font-size: 13.5px; color: var(--text); width: 100%; }
  .menu button:hover, .menu button:focus-visible { background: var(--bg-field); outline: 1px solid var(--border-strong); }
  .menu .cancel { color: var(--text-secondary); font-size: 12px; }
  .zone ol { max-height: 260px; overflow-y: auto; }
  kbd { font-family: var(--font-mono); font-size: 10px; line-height: 14px; padding: 0 4px; border-radius: 3px; border: 1px solid currentColor; opacity: 0.7; min-width: 16px; text-align: center; }

  .preview {
    position: fixed; z-index: 250; width: 270px; max-height: calc(100vh - 60px); overflow: hidden; pointer-events: none;
    padding: 10px; border-radius: 8px; border: 1px solid var(--border-strong); background: var(--bg-surface); box-shadow: var(--shadow-raised);
  }
  .dock { max-height: 100%; overflow-y: auto; }
  .inspect-toggle { color: var(--text-secondary); padding: 2px 6px; border-radius: 4px; font-size: 12px; white-space: nowrap; }
  .inspect-toggle:hover { color: var(--text); background: var(--bg-field); }

  @media (prefers-reduced-motion: reduce) {
    .hand-card, .board { transition: none; }
    .menu, .preview, .result-card { animation: none; }
  }
</style>
