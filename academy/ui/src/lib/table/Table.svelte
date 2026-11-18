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
   */
  import { onMount, type Snippet } from 'svelte';
  import type { PresentationPlayer } from '$lib/presentation.svelte';
  import { artUrl, loadArt } from '$lib/card-art';
  import type { CardState, ConnectionState, ExperienceFrame, GameLogEntry, InteractionOffer, StopsConfig } from '$lib/types';
  import Field from './Field.svelte';
  import Lines from './Lines.svelte';
  import { prefersReducedMotion, Travel } from './motion';
  import { moveList } from './moves';
  import { promptLine } from './prompt';
  import { nameBook, recentView, tableLines, turnOwners, visibleNames } from './recent';
  import Plate from './Plate.svelte';
  import Rail from './Rail.svelte';
  import TableCard from './TableCard.svelte';
  import { phaseTrack } from './phases';
  import { CARD_RATIO, pileCounts, splitHalf, tableView, type HalfSplit, type PermanentItem, type PlacedOffer, type RowLimits, type StackItem } from './view';

  interface Props {
    frame: ExperienceFrame;
    live: boolean;
    status: string;
    you: { name: string; deck: string | null };
    opponent: { name: string; deck: string | null; description?: string };
    stops: StopsConfig;
    connection: ConnectionState;
    connectionLabel: string;
    log: GameLogEntry[];
    canPassTurn: boolean;
    /** The live presentation track: what happened, as the authority reports it. */
    presentation: PresentationPlayer;
    /** Set while the opponent is choosing, or the table is working, in words. */
    thinking: string | null;
    /** The opponent is the one choosing now. */
    opponentChoosing: boolean;
    /** The live session is gone; offers are the last recorded decision's. */
    interrupted: boolean;
    onact: (offer: InteractionOffer) => void;
    onpassturn: () => void;
    notices?: Snippet;
    result?: Snippet;
    menu?: Snippet;
  }

  let {
    frame, live, status, you, opponent, stops, connection, connectionLabel, log, canPassTurn,
    presentation, thinking, opponentChoosing, interrupted, onact, onpassturn, notices, result, menu,
  }: Props = $props();

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
      : frame.prompt.actor === projection.agent.player_index ? (live ? 'Your decision' : 'Your decision · waiting') : `${opponent.name}'s decision`);
  const line = $derived(promptLine(frame, view));

  // ———————————————— What happened since your last move

  // The revision your last move was made at; events after it are "since your last move".
  let since = $state.raw<{ revision: number; frame: ExperienceFrame; logLength: number } | null>(null);
  const players = $derived({ you: you.name, opponent: opponent.name });
  const book = $derived(nameBook(since ? [since.frame, frame] : [frame], since ? log.slice(since.logLength) : []));
  const recent = $derived(recentView(frame, since?.revision ?? null, presentation.events, index => presentation.labelsAt(index), players, book));
  const lines = $derived(tableLines(frame, presentation.events, recent.resolved));
  const moves = $derived(moveList(log, players, turnOwners(frame, presentation.events), visibleNames(frame, presentation.labels)));

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

  const CREATURES: RowLimits = { gap: 8, minCard: 40, maxCard: 132, maxLines: 3 };
  const SUPPORT: RowLimits = { gap: 8, minCard: 36, maxCard: 104, maxLines: 2 };
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
  type Preview = { name: string; text: string | null; x: number; y: number; flip: boolean };
  type Zone = { title: string; cards: CardState[]; x: number; y: number };
  let menuAt = $state<Menu | null>(null);
  let zoneAt = $state<Zone | null>(null);
  let preview = $state<Preview | null>(null);
  let menuElement = $state<HTMLElement | null>(null);

  // A new frame replaces the cards a menu or preview was opened from.
  $effect(() => {
    void frame.revision;
    menuAt = null;
    preview = null;
  });
  $effect(() => {
    if (menuAt) menuElement?.querySelector<HTMLButtonElement>('button')?.focus({ preventScroll: true });
  });

  function act(placed: PlacedOffer): void {
    if (!live) return;
    menuAt = null;
    since = { revision: frame.revision, frame, logLength: log.length };
    onact(placed.offer);
  }

  function passTurn(): void {
    if (!canPassTurn) return;
    since = { revision: frame.revision, frame, logLength: log.length };
    onpassturn();
  }

  function activate(title: string, offers: PlacedOffer[], element: HTMLElement): void {
    if (!live || !offers.length) return;
    if (offers.length === 1) { act(offers[0]); return; }
    const rect = element.getBoundingClientRect();
    menuAt = { title, offers, x: Math.min(rect.left, window.innerWidth - 300), y: rect.bottom + 6 > window.innerHeight - 160 ? Math.max(8, rect.top - 8 - 36 * offers.length - 30) : rect.bottom + 6 };
  }

  function show(name: string | null, text: string | null, element: HTMLElement | null): void {
    if (!name || !element) { preview = null; return; }
    const rect = element.getBoundingClientRect();
    const flip = rect.right + 260 > window.innerWidth - 280;
    preview = { name, text, x: flip ? rect.left - 252 : rect.right + 12, y: Math.max(48, Math.min(rect.top - 40, window.innerHeight - 360)), flip };
  }

  const previewFull = $derived(preview ? artUrl(preview.name, 'full') : null);
  let previewLoaded = $state<string | null>(null);
  $effect(() => {
    const url = previewFull;
    if (!url) return;
    let active = true;
    void loadArt(url).then(ok => { if (active && ok) previewLoaded = url; });
    return () => { active = false; };
  });

  function onPermanent(item: PermanentItem, element: HTMLElement): void {
    activate(item.permanent.name ?? 'Permanent', item.offers, element);
  }
  function previewPermanent(item: PermanentItem | null, element: HTMLElement | null): void {
    show(item?.permanent.name ?? null, item?.permanent.text_box ?? null, element);
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

  // The hand yields height to the board: an empty hand is a label, a short
  // hand or a crowded board gets a lower strip, and only a full hand on a
  // sparse board gets the full one.
  const boardCount = $derived(projection.agent.battlefield.length + projection.opponent.battlefield.length);
  const handTrack = $derived(view.hand.length === 0 ? '30px'
    : boardCount >= 24 ? 'clamp(92px, 14vh, 128px)'
      : boardCount >= 14 || view.hand.length <= 2 ? 'clamp(104px, 17vh, 160px)'
        : 'clamp(118px, 21vh, 190px)');
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
    if (event.key === 'Escape') { menuAt = null; zoneAt = null; preview = null; return; }
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

<svelte:window onkeydown={onKeydown} />

<div class="table" data-testid="table" data-action-space={frame.action_space} data-live={live} data-thinking={thinking !== null} bind:this={root}>
  <header class="topbar">
    <a href="/games" class="brand" aria-label="Academy · Games">Academy</a>
    <p class="matchup type-caption" data-testid="deck-names">
      <span class="text-ink">You</span>{#if you.deck}<span class="text-ink-3"> · {you.deck}</span>{/if}
      <span class="text-ink-3"> vs </span>
      <span class="text-ink" title={opponent.description}>{opponent.name}</span>{#if opponent.deck}<span class="text-ink-3"> · {opponent.deck}</span>{/if}
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
      <span class="connection" data-state={connection}>
        <i aria-hidden="true"></i>
        <span data-testid="connection-badge" data-connection-state={connection} role="status" aria-live="polite" aria-atomic="true"
          aria-label={`Connection status: ${connectionLabel}`} class="type-rubric">{connectionLabel}</span>
      </span>
      {#if menu}{@render menu()}{/if}
    </div>
  </header>

  <div class="board" id="table-board" aria-label="Board" style:grid-template-rows={`minmax(0, 1fr) minmax(0, 1fr) ${handTrack}`}>
    <section class="half opponent" aria-label={`${opponent.name}'s battlefield`}>
      <Plate player={projection.opponent} name={opponent.name} deck={opponent.deck} offers={view.opponent.offers} {live}
        active={!yourTurn} choosing={opponentChoosing} side="opponent" onactivate={element => activate(opponent.name, view.opponent.offers, element)} onzone={(zone, cards, element) => openZone(`${opponent.name} · ${zone}`, cards, element)} />
      <div class="rows" bind:clientWidth={theirRows.width} bind:clientHeight={theirRows.height} style:grid-template-rows={rowsStyle(theirSplit, theirRows, false)}>
        <div class="row support"><Field piles={view.opponent.support} label={`${opponent.name}'s lands and other permanents`} limits={SUPPORT} {live} asked={view.asked} touched={recent.touched} toward="down" {names} onactivate={onPermanent} onpreview={previewPermanent} /></div>
        <div class="row creatures"><Field piles={view.opponent.creatures} label={`${opponent.name}'s creatures`} limits={CREATURES} {live} asked={view.asked} touched={recent.touched} toward="down" {names} onactivate={onPermanent} onpreview={previewPermanent} /></div>
      </div>
    </section>
    <section class="half you" aria-label="Your battlefield">
      <Plate player={projection.agent} name={you.name} deck={you.deck} offers={view.you.offers} {live}
        active={yourTurn} side="you" onactivate={element => activate(you.name, view.you.offers, element)} onzone={(zone, cards, element) => openZone(`Your ${zone.toLowerCase()}`, cards, element)} />
      <div class="rows" bind:clientWidth={yourRows.width} bind:clientHeight={yourRows.height} style:grid-template-rows={rowsStyle(yourSplit, yourRows, true)}>
        <div class="row creatures"><Field piles={view.you.creatures} label="Your creatures" limits={CREATURES} {live} asked={view.asked} touched={recent.touched} toward="up" {names} onactivate={onPermanent} onpreview={previewPermanent} /></div>
        <div class="row support"><Field piles={view.you.support} label="Your lands and other permanents" limits={SUPPORT} {live} asked={view.asked} touched={recent.touched} toward="up" {names} onactivate={onPermanent} onpreview={previewPermanent} /></div>
      </div>
    </section>
    <section class="hand" aria-label={`Your hand, ${view.hand.length} cards`} bind:clientWidth={handWidth} bind:clientHeight={handHeight}>
      <span class="type-rubric hand-label text-ink-3">Hand · {view.hand.length}</span>
      <div class="hand-cards" style:width={`${handCard + Math.max(0, view.hand.length - 1) * handStep}px`}>
        {#each view.hand as item, index (item.card.id)}
          <div class="hand-card" class:playable={live && item.offers.length > 0} style:left={`${index * handStep}px`}>
            <TableCard name={item.card.name} width={handCard} card={item.card} offers={item.offers} {live} asked={view.asked.has(item.card.id)}
              onactivate={element => activate(item.card.name, item.offers, element)}
              onpreview={element => show(element ? item.card.name : null, item.card.text_box ?? null, element)} />
          </div>
        {/each}
      </div>
    </section>

    {#if view.tray.length}
      <section class="tray" aria-label="Choices from outside the table" data-testid="tray">
        <h2 class="type-rubric text-ink-2">{frame.prompt?.title ?? 'Choose'}</h2>
        <div class="tray-items">
          {#each view.tray as item (item.placed.offer.id)}
            {#if item.name}
              <div class="tray-card">
                <TableCard name={item.name} width={104} offers={[item.placed]} {live}
                  onactivate={() => act(item.placed)}
                  onpreview={element => show(element ? item.name : null, null, element)} />
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
    <Rail {view} {frame} {live} {line} {status} {whose} {thinking} {interrupted} {canPassTurn} {recent} {moves}
      opponentName={opponent.name} onact={act} onpassturn={passTurn}
      onstack={onStack} onpreviewname={(name, element) => show(name, name ? textOf(name) : null, element)} {notices} {result} />
  </div>
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
        <li><button type="button" onmouseenter={event => show(card.name, card.text_box ?? null, event.currentTarget)} onmouseleave={() => (preview = null)}
          onfocus={event => show(card.name, card.text_box ?? null, event.currentTarget)} onblur={() => (preview = null)}>{card.name}</button></li>
      {/each}
    </ol>
    <button type="button" class="cancel" onclick={() => (zoneAt = null)}><kbd>Esc</kbd>Close</button>
  </div>
{/if}

{#if preview}
  <div class="preview" aria-hidden="true" style:left={`${preview.x}px`} style:top={`${preview.y}px`} data-testid="card-preview">
    {#if previewLoaded === previewFull}
      <img src={previewFull} alt="" />
    {:else}
      <div class="preview-text">
        <p class="type-title">{preview.name}</p>
        {#if preview.text}<p class="type-caption">{preview.text}</p>{/if}
      </div>
    {/if}
  </div>
{/if}

<style>
  .table {
    position: fixed; inset: 0; z-index: 10; display: grid;
    grid-template-columns: minmax(0, 1fr) clamp(250px, 21vw, 300px);
    grid-template-rows: 40px minmax(0, 1fr);
    background: var(--bg); color: var(--text);
  }
  .topbar {
    grid-column: 1 / -1; display: flex; align-items: center; gap: 16px; padding: 0 14px;
    border-bottom: 1px solid var(--border); background: var(--bg-surface); min-width: 0;
  }
  .brand { font-family: var(--font-serif); font-weight: 600; font-size: 19px; color: var(--accent-text); }
  .matchup { min-width: 0; flex: 0 1 auto; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .track { display: flex; align-items: center; gap: 12px; margin: 0 auto; min-width: 0; }
  .track .turn { color: var(--text); white-space: nowrap; }
  .track ol { display: flex; gap: 2px; }
  .track li { position: relative; padding: 2px 7px; border-radius: 10px; color: var(--text-tertiary); white-space: nowrap; }
  .track li.current { background: var(--accent); color: var(--ivory); }
  .track .stop { position: absolute; left: 50%; bottom: -3px; width: 4px; height: 4px; margin-left: -2px; border-radius: 2px; background: var(--accent); }
  .track li.current .stop { background: var(--ivory); }
  .right { display: flex; align-items: center; gap: 14px; }
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

  .rail-slot { min-height: 0; }

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

  .preview { position: fixed; z-index: 250; width: 240px; pointer-events: none; }
  .preview img { width: 240px; border-radius: 12px; box-shadow: var(--shadow-raised); display: block; }
  .preview-text { padding: 12px; border-radius: 10px; border: 1px solid var(--border-strong); background: var(--bg-surface); box-shadow: var(--shadow-raised); display: grid; gap: 6px; white-space: pre-line; }

  @media (prefers-reduced-motion: reduce) {
    .hand-card, .board { transition: none; }
    .menu, .preview { animation: none; }
  }
</style>
