<script lang="ts">
  import { region } from './region';
  /**
   * The rail beside the board: the stack and what happened since your last
   * move, the prompt in one plain line with its buttons, every legal choice by
   * number, and the log as a move list. The list is the guarantee that the
   * board hides nothing: it holds every offer of the frame, in order.
   *
   * One rule divides the rail's height. What you must read to act (notices,
   * the stack with its recap, the decision) keeps its whole height; its long
   * lists scroll inside their own bounds. The two reference lists (every
   * legal choice, then the moves) share what is left, each above a floor, and
   * scroll inside. No block is cut by its neighbour; on a window too short
   * for the floors the rail itself scrolls.
   */
  import type { Snippet } from 'svelte';
  import type { ExperienceFrame } from '$lib/types';
  import type { HistoryLine, HistoryTurn, RecentHistoryItem } from './history';
  import type { Segment } from './tell';
  import { committed, offerChoiceLabel, offerText, type PrimaryLabel } from './prompt';
  import { targetNames, type ResolvedSpell } from './recent';
  import { formatProbability, type OfferEvidence } from './replay';
  import { formatScore, markText, type DecisionMark } from '$lib/analysis';
  import ManaText from './ManaText.svelte';
  import Thumb from './Thumb.svelte';
  import { stackName, stackProvenance, type PlacedOffer, type StackItem, type BoardView } from './view';

  interface Props {
    view: BoardView;
    /** Stack entries for the drawn board (may be an earlier cursor board while `view` stays live). */
    stack?: StackItem[];
    frame: ExperienceFrame;
    live: boolean;
    /** What to do now, in plain words. */
    line: string;
    /** How to answer, apart from what is asked. */
    hint: string | null;
    /** What the primary button does next. */
    primaryText: PrimaryLabel | null;
    status: string;
    whose: string;
    /** "Search 64 is choosing" while someone else decides; the rail adds the seconds. */
    thinking: string | null;
    /** The opponent is the one deciding: an empty choice list waits for them by name. */
    opponentChoosing: boolean;
    /** When the awaited work started, epoch seconds. */
    workStarted: number | null;
    interrupted: boolean;
    canPassTurn: boolean;
    /** Unread history tail derived from `history` (empty whenever a cursor is active). */
    recentFacts: readonly RecentHistoryItem[];
    /** Spells cast since your last move that left the stack, for lingering target arrows. */
    resolvedSpells: readonly ResolvedSpell[];
    history: HistoryTurn[];
    opponentName: string;
    onact: (placed: PlacedOffer) => void;
    onpassturn: () => void;
    stackSources?: Set<number>;
    drop?: number;
    onstack: (item: StackItem, element: HTMLElement) => void;
    onpreviewname: (name: string | null, element: HTMLElement | null) => void;
    /** A row under the pointer or focus, whose payment the board may preview. */
    onpointoffer: (placed: PlacedOffer | null) => void;
    /** Why the board shows no planned taps for the payment, when the preview says. */
    paymentNote: string | null;
    notices?: Snippet;
    /** The game has ended; its result card sits over the board. */
    over?: boolean;
    /** The recorded policy at a recorded choice. */
    evidence?: OfferEvidence | null;
    /** An analysis's mark on each of your moves, by move id. */
    moveMarks?: ReadonlyMap<string, DecisionMark>;
    /** Your moves chosen with a live analysis shown, by move id. */
    shownMoves?: ReadonlySet<string>;
    /** The recap's heading. */
    recapTitle?: string;
    /** Moves jump to their position when set (Review). */
    onjump?: (id: string, boardSeq: number | null) => void;
    /** Read-only (a recording, or a policy's line): no empty stack and no answer buttons beside the list. */
    recording?: boolean;
    currentMove?: string | null;
    /** Select a history line to move the board cursor to it (live play). */
    onselectline?: (line: HistoryLine) => void;
    /** `HistoryLine.id`s matching the active cursor / walk stop. */
    cursorLineIds?: ReadonlySet<string>;
    /** Cast/activation seq marked when a standalone resolution line is selected. */
    cursorSourceSeq?: number | null;
    /** A combat draft's controls, in place of the per-creature buttons. */
    draft?: Snippet;
    /** Why a confirmed combat declaration stopped before all of it was sent. */
    stopped?: string | null;
  }

  let {
    view, stack: stackGiven, frame, live, line, hint, primaryText, status, whose, thinking, opponentChoosing, workStarted, interrupted, canPassTurn, recentFacts, resolvedSpells, history, opponentName,
    stackSources, drop, onact, onpassturn, onstack, onpreviewname, onpointoffer, paymentNote, notices, over = false,
    evidence = null, moveMarks = new Map(), shownMoves = new Set(), recapTitle = 'Since your last move', onjump, recording = false, currentMove = null, onselectline, cursorLineIds, cursorSourceSeq = null, draft, stopped = null,
  }: Props = $props();

  const stack = $derived(stackGiven ?? view.stack);

  /** No choices because the opponent is deciding: the list waits for them by name instead of counting 0. */
  const waitingFor = $derived(frame.offers.length === 0 && opponentChoosing);
  const recordedP = (placed: PlacedOffer): number | undefined => evidence?.probability.get(placed.offer.id);
  const isChosen = (placed: PlacedOffer): boolean => evidence?.chosen === placed.offer.id;
  const estimateOf = (placed: PlacedOffer): number | undefined => evidence?.estimate.get(placed.offer.id);
  const isFirst = (placed: PlacedOffer): boolean => evidence?.first === placed.offer.id;
  /** "you", "its choice", both or neither: who chose an analysed offer. */
  const choosers = (placed: PlacedOffer): string => [isChosen(placed) ? 'you' : '', isFirst(placed) ? 'its choice' : ''].filter(Boolean).join(' · ');

  const context = $derived(frame.projection.decision_context);
  const paid = $derived(committed(frame));
  const detail = $derived(context?.instruction ?? frame.prompt?.instruction ?? '');
  const secondary = $derived(view.buttons.filter(placed => placed !== view.primary));
  const mine = (controller: number): boolean => controller === frame.projection.agent.player_index;
  const resolved = $derived(new Map(resolvedSpells.map(spell => [spell.id, spell])));

  // The recap shows its first lines; the rest open on demand and fold again when the recap changes.
  const RECAP_LINES = 3;
  let recapOpen = $state(false);
  $effect(() => { void recentFacts; recapOpen = false; });
  const facts = $derived(recapOpen ? recentFacts : recentFacts.slice(0, RECAP_LINES));

  // Seconds the opponent or the server has been working, beside "… is choosing".
  let elapsed = $state(0);
  $effect(() => {
    const started = workStarted;
    if (started === null || thinking === null) { elapsed = 0; return; }
    const update = (): void => { elapsed = Math.max(0, Math.floor(Date.now() / 1000 - started)); };
    update();
    const timer = setInterval(update, 1000);
    return () => clearInterval(timer);
  });

  let logList = $state<HTMLElement | null>(null);
  let expanded = $state<Set<string>>(new Set());
  const hasCursor = $derived((cursorLineIds?.size ?? 0) > 0);
  function isLineCurrent(entry: HistoryLine, fallbackId = false): boolean {
    if (hasCursor) return cursorLineIds!.has(entry.id);
    const key = entry.jumpKey ?? (fallbackId ? entry.id : null);
    return currentMove !== null && key === currentMove;
  }
  function toggleExpanded(id: string): void {
    const next = new Set(expanded);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    expanded = next;
  }
  // The list follows the active cursor / walk stop, the shown Review position, or the newest line live.
  $effect(() => {
    void history.length;
    void history.at(-1)?.lines.length;
    void history.at(-1)?.lines.at(-1)?.children.length;
    void cursorLineIds;
    const current = currentMove;
    // Only the list scrolls: scrolling an element into view would also move a rail that is taller than a short window.
    if (!hasCursor && current === null && !onjump) { if (logList) logList.scrollTop = logList.scrollHeight; return; }
    const row = logList?.querySelector<HTMLElement>('[aria-current="true"]');
    if (!row || !logList) return;
    const top = row.offsetTop - logList.offsetTop;
    if (top < logList.scrollTop || top + row.offsetHeight > logList.scrollTop + logList.clientHeight) logList.scrollTop = top - logList.clientHeight / 2;
  });

  function anchorText(placed: PlacedOffer): string {
    const anchor = placed.anchor;
    if (anchor.kind === 'player') return 'player';
    if (anchor.kind === 'tray') return 'tray';
    return '';
  }

  function hover(name: string): { onmouseenter: (event: MouseEvent) => void; onmouseleave: () => void; onfocus: (event: FocusEvent) => void; onblur: () => void } {
    return {
      onmouseenter: event => onpreviewname(name, event.currentTarget as HTMLElement),
      onmouseleave: () => onpreviewname(null, null),
      onfocus: event => onpreviewname(name, event.currentTarget as HTMLElement),
      onblur: () => onpreviewname(null, null),
    };
  }
</script>

{#snippet lineSegments(parts: readonly Segment[], focusable: boolean)}
  {#each parts as segment, segIdx (segIdx)}
    {#if segment.card}
      {#if focusable}
        <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_static_element_interactions (focus shows the card; a button would not wrap inside a line) -->
        <span class="card-name" tabindex="0" {...hover(segment.card)}>{segment.text}</span>
      {:else}
        <span class="card-name" {...hover(segment.card)}>{segment.text}</span>
      {/if}
    {:else}
      {segment.text}
    {/if}
  {/each}
{/snippet}

{#snippet moveWhat(entry: HistoryLine, prevWho: string, focusable: boolean)}
  <span class="what">{#if entry.who !== '' && prevWho !== entry.who}<span class="who">{entry.who}</span>{/if}{@render lineSegments(entry.segments, focusable)}{#if entry.count > 1}<span class="times">&nbsp;×{entry.count}</span>{/if}</span>
{/snippet}

<aside class="rail" class:recording aria-label="Stack, decision and legal choices">
  {#if notices}<div class="notices">{@render notices()}</div>{/if}

  <!-- A recording draws the stack only while something is on it. -->
  {#if !recording || stack.length || recentFacts.length}
  <section use:region={{ across: !onjump }} class="block stack" aria-label="Stack">
    {#if stack.length === 0 && !recording}
      <h2><span class="type-heading text-ink-2">Stack</span></h2>
      <p class="type-caption text-ink-3">Empty</p>
    {:else if stack.length}
      <h2><span class="type-heading text-ink-2">Stack · {stack.length}</span></h2>
      <ol>
        {#each stack as item, index (item.entry.id)}
          {@const hitPlayers = item.entry.combat_damage_to_players?.map(seat => mine(seat) ? 'you' : opponentName).join(' and ')}
          {@const damageHistory = hitPlayers ? `Source dealt combat damage to ${hitPlayers} this turn` : ''}
          {@const provenance = stackProvenance(item.entry)}
          {@const targets = targetNames(item.entry, frame, { you: 'You', opponent: opponentName })}
          <li>
            <button type="button" class="stack-entry" class:legal={live && item.offers.length > 0} class:top={index === 0}
              data-testid="stack-entry" data-stack-entry={item.entry.id} data-connect-source={stackSources?.has(item.entry.id) || undefined} data-drop={drop === item.entry.id || undefined}
              data-stack-card={item.card?.id ?? (item.announcing ? item.entry.source_permanent : item.entry.id)}
              data-announcing={item.announcing}
              data-travel-id={item.card ? `card:${item.card.id}` : `stack:${item.entry.id}`}
              data-travel-source={item.entry.source_permanent !== undefined ? `perm:${item.entry.source_permanent}` : undefined}
              data-travel-side={mine(item.entry.controller) ? 'you' : 'opponent'}
              data-travel-name={item.card?.name ?? item.entry.label.split(' — ')[0]}
              aria-label={`${item.entry.label}${provenance.map(tag => `, ${tag}`).join('')}${targets.length ? `, targeting ${targets.join(' and ')}` : ''}${damageHistory ? `. ${damageHistory}` : ''}, ${mine(item.entry.controller) ? 'yours' : opponentName + "'s"}${item.offers.length ? ` — ${item.offers.map(placed => placed.offer.label).join('; ')}` : ''}`}
              onclick={event => onstack(item, event.currentTarget)}
              {...hover(item.card?.name ?? item.entry.label)}>
              <Thumb name={item.card?.name ?? item.entry.label} />
              <span class="entry-text">
                <span class="entry-title">
                  <span class="truncate entry-name">{stackName(item.entry)}</span>
                  {#each provenance as tag (tag)}<span class="tag type-caption" data-testid="stack-provenance">{tag}</span>{/each}
                </span>
                {#if targets.length}<span class="type-caption targets" data-testid="stack-targets">→ {targets.join(', ')}</span>{/if}
                {#if damageHistory}<span class="type-caption text-ink-3">{damageHistory}</span>{/if}
                <span class="type-caption text-ink-3">{mine(item.entry.controller) ? 'You' : opponentName}{item.announcing ? ` · ${item.announcing === 'spell' ? 'casting' : 'activating'}` : index === 0 ? ' · resolves next' : ''}</span>
              </span>
            </button>
          </li>
        {/each}
      </ol>
    {/if}
    {#if recentFacts.length}
      <div class="recent" data-testid="recent">
        <h3><span class="type-heading text-ink-2">{recapTitle}</span></h3>
        <ol>
          {#each facts as fact (fact.seq)}
            {@const spell = fact.spell === null ? undefined : resolved.get(fact.spell)}
            {@const isFactJump = Boolean(onjump && fact.line.jumpKey)}
            {@const isFactSelectable = Boolean(onselectline && fact.line.boardSeq !== null)}
            <li class="type-caption fact" class:theirs={fact.by === 'opponent'} data-testid="recent-fact">
              {#if spell}
                <span class="resolved" data-stack-card={spell.id} tabindex="-1" {...hover(spell.name)}><Thumb name={spell.name} /></span>
              {/if}
              {#if isFactJump || isFactSelectable}
                <button type="button" class="fact-btn" title={fact.text}
                  onclick={() => (isFactJump ? onjump!(fact.line.jumpKey!, fact.line.boardSeq) : onselectline!(fact.line))}>
                  <span class="fact-text">{@render lineSegments(fact.segments, false)}</span>
                </button>
              {:else}
                <span class="fact-text" title={fact.text}>{@render lineSegments(fact.segments, true)}</span>
              {/if}
            </li>
          {/each}
        </ol>
        {#if recentFacts.length > RECAP_LINES}
          <button type="button" class="more type-caption" data-testid="recent-more" aria-expanded={recapOpen} onclick={() => (recapOpen = !recapOpen)}>
            {recapOpen ? 'Fewer' : `${recentFacts.length - RECAP_LINES} more`}
          </button>
        {/if}
      </div>
    {/if}
  </section>
  {/if}

  <section use:region={{ across: !onjump }} class="block prompt" aria-label="Decision" data-testid="action-panel" data-action-space-kind={frame.action_space}>
    <p class="type-heading whose" class:yours={live}>
      {#if thinking}<span data-testid="work-status">{thinking}{#if elapsed >= 1}<span class="elapsed">{` · ${elapsed} s`}</span>{/if}</span>{:else}{whose}{/if}
    </p>
    {#if !over}
      {#if interrupted}
        <p class="type-body text-ink-2" data-testid="decision-prompt">The live session is unavailable. These choices belong to the last recorded decision and cannot be submitted.</p>
      {:else if frame.prompt}
        <p class="type-prompt text-ink" data-testid="decision-prompt" data-kind={frame.action_space}><ManaText text={line} size={16} /></p>
        {#if paid.length}
          <div class="committed" data-testid="decision-committed">
            <h3 class="type-caption text-ink-3">Committed to this cost</h3>
            <ol>{#each paid as phrase, at (at)}<li class="type-body text-ink">{phrase}</li>{/each}</ol>
          </div>
        {/if}
        {#if hint}<p class="type-caption text-ink-3 hint" data-testid="decision-hint">{hint}</p>{/if}
      {/if}
      {#if evidence}
        <div class="evidence" data-testid="recorded-decision-evidence" aria-label="Recorded at this decision">
          {#each evidence.summary as fact (fact)}<p class="type-caption text-ink-2">{fact}</p>{/each}
          {#if evidence.search || evidence.provenance.length}
            <details class="search-detail" data-testid="search-detail">
              <summary class="type-caption text-ink-3">Search detail</summary>
              {#if evidence.search}
                <table class="type-caption">
                  <thead><tr><th>Offer</th>{#if evidence.search.some(row => row.prior !== null)}<th>Raw policy</th>{/if}<th>Search value</th><th>Updated policy</th></tr></thead>
                  <tbody>
                    {#each evidence.search as row (row.id)}
                      <tr class:chosen={row.id === evidence.chosen}>
                        <td>{row.id === evidence.chosen ? '✓ ' : ''}{row.label}</td>
                        {#if evidence.search.some(candidate => candidate.prior !== null)}<td>{row.prior === null ? '—' : formatProbability(row.prior)}</td>{/if}
                        <td>{row.score === null ? '—' : row.score.toFixed(3)}</td>
                        <td>{row.updated === null ? '—' : formatProbability(row.updated)}</td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              {/if}
              {#each evidence.provenance as line (line)}<p class="type-caption text-ink-3 provenance">{line}</p>{/each}
            </details>
          {/if}
        </div>
      {/if}
      {#if stopped}<p class="type-caption text-ink-2" data-testid="combat-stopped" role="status">{stopped}</p>{/if}
      {#if draft}
        {@render draft()}
      {:else if !recording}
      <!-- A recording's answers are the list below; buttons here would repeat them, disabled. -->
      {#if view.primary}
        <button type="button" class="btn btn-primary primary" class:chosen={isChosen(view.primary)} data-testid="primary-action" data-offer-id={view.primary.offer.id} aria-label={primaryText?.description ?? view.primary.offer.label} title={primaryText?.description ?? view.primary.offer.label}
          disabled={!live} onclick={() => onact(view.primary!)}>
          <span>{#if isChosen(view.primary)}✓ {/if}<ManaText text={primaryText?.text ?? offerText(view.primary.offer)} /></span>{#if recordedP(view.primary) !== undefined}<span class="p">{formatProbability(recordedP(view.primary)!)}</span>{:else if live}<kbd>Space</kbd>{/if}
        </button>
        {#if paymentNote}<p class="type-caption text-ink-3 payment-note" data-testid="payment-note">{paymentNote}</p>{/if}
      {/if}
      {#each secondary as placed (placed.offer.id)}
        <button type="button" class="btn btn-secondary secondary" class:chosen={isChosen(placed)} data-testid="prompt-action" data-offer-id={placed.offer.id}
          disabled={!live} onclick={() => onact(placed)}>
          <kbd>{placed.number}</kbd>{#if isChosen(placed)}✓ {/if}<ManaText text={offerText(placed.offer)} />{#if recordedP(placed) !== undefined}<span class="p">{formatProbability(recordedP(placed)!)}</span>{/if}
        </button>
      {/each}
      {/if}
      {#if canPassTurn}
        <button type="button" class="pass-turn type-caption" data-testid="pass-turn" onclick={onpassturn}
          title="Pass every priority window until the turn ends (Shift+Enter or F6)">Pass turn <kbd>⇧↵</kbd></button>
      {/if}
      <!-- The engine's own words stay one click away, under the answers. -->
      {#if detail && frame.prompt && !interrupted}
        <details class="detail">
          <summary class="type-caption text-ink-3">Rules detail</summary>
          <p class="type-caption text-ink-2">{detail}</p>
        </details>
      {/if}
    {/if}
    <!-- While you may act, or once the game is over, the heading above already says what the status would. -->
    <p class="type-caption text-ink-3 status" class:sr-only={live || thinking !== null || frame.status === 'game_over'} data-testid="current-decision" role="status" aria-live="polite" aria-atomic="true">{thinking ?? status}</p>
  </section>

  <!-- A finished game has no choices to list; the result card says what happened. Read-only, a position without a decision lists none. -->
  {#if (!over && !recording) || frame.offers.length}
  <section use:region={{ across: !onjump }} class="block choices" aria-label="All legal choices">
    <h2 class="heading type-heading text-ink-2">
      <span data-testid="choices-heading">{evidence?.heading ?? (interrupted || (!live && evidence) ? 'Recorded choices' : 'All legal choices')}</span>
      {#if !waitingFor}<span class="count" data-testid="legal-choice-count">{frame.offers.length}<span class="sr-only"> {interrupted ? 'recorded' : 'legal'} choices</span></span>{/if}
    </h2>
    {#if waitingFor}<p class="type-caption text-ink-3" data-testid="choices-waiting">Waiting for {opponentName}</p>{/if}
    <!-- An open combat draft is the way to declare; each creature's own offer stays one click away for inspection. -->
    {#if draft}
      <details class="raw" data-testid="per-creature-choices">
        <summary class="type-caption text-ink-3">Each creature's choice</summary>
        <p class="type-caption text-ink-3">Confirming the draft sends these one creature at a time.</p>
        {@render choiceList()}
      </details>
    {:else}
      {@render choiceList()}
    {/if}
  </section>
  {/if}

  <section use:region={{ across: false }} class="block log" aria-label="History">
    <h2><span class="type-heading text-ink-2">History</span></h2>
    <ol class="turns" data-testid="game-log" bind:this={logList}>
      {#each history as turn, index (`${turn.turn}:${index}`)}
        <li class="turn" data-turn={turn.turn}>
          <!-- A recording numbers the turn in its first row's gutter; the heading stays for screen readers. -->
          <p class="turn-head type-label" class:sr-only={onjump !== undefined}><span class="number">{turn.turn ?? '·'}</span>{#if turn.owner}<span class="owner">{turn.owner}</span>{/if}</p>
          <ol>
            {#each turn.lines as entry, at (entry.id)}
              {@const jumpId = entry.jumpKey ?? entry.id}
              {@const prevWho = at > 0 ? turn.lines[at - 1]?.who : ''}
              {@const isCurrent = isLineCurrent(entry, true)}
              {@const isSource = cursorSourceSeq !== null && entry.seq === cursorSourceSeq}
              {@const childCurrent = entry.children.some(child => isLineCurrent(child))}
              {@const isJump = Boolean(onjump && entry.jumpKey)}
              {@const isSelectable = !entry.passes && Boolean(onselectline && entry.boardSeq !== null)}
              {@const mark = moveMarks.get(entry.id)}
              {#if isJump || isSelectable}
                <!-- A recording's line jumps to its position; a live line selects its cursor board. -->
                <li class="move type-caption jump" class:quiet={entry.passes !== null} class:auto={entry.actor === 'auto'} class:you={entry.actor === 'you'}
                  data-testid={entry.passes ? 'pass-run' : 'log-entry'} data-actor={entry.actor}>
                  <button type="button" aria-current={isCurrent ? 'true' : undefined} data-cursor-source={isSource || undefined}
                    data-move={isJump ? jumpId : undefined} data-history-seq={entry.boardSeq ?? undefined}
                    title={isJump && !entry.passes ? entry.segments.map(segment => segment.text).join('') : undefined}
                    onclick={() => (isJump ? onjump!(jumpId, entry.boardSeq) : onselectline!(entry))}>
                    {#if isJump}
                      <span class="gutter" aria-hidden="true" title={at === 0 && turn.owner ? `Turn ${turn.turn ?? ''}, ${turn.owner}` : undefined}>{at === 0 ? turn.turn ?? '·' : ''}</span>
                      <span class="who" class:repeat={!entry.passes && prevWho === entry.who}>{entry.passes ? '' : entry.who}</span>
                      <span class="what">{#if entry.passes}passed {entry.passes.map(part => `${part.who}${part.count > 1 ? ` ×${part.count}` : ''}`).join(', ')}{:else}{@render lineSegments(entry.segments, false)}{#if entry.count > 1}<span class="times">&nbsp;×{entry.count}</span>{/if}{/if}</span>
                      {#if mark}{@const same = mark.agreed === mark.of}
                        <span class="reading">
                          <span class="mark type-caption" data-testid="move-mark" data-same={same} data-agreed={mark.agreed} data-of={mark.of}>{markText(mark)}</span>
                          <svg class="track" viewBox="0 0 32 8" aria-hidden="true" data-testid="move-track">
                            <line x1="0" x2="32" y1="4" y2="4" />
                            {#if mark.its !== null}<circle class="its" cx={3 + mark.its * 26} cy="4" r="2.5" />{/if}
                            {#if mark.yours !== null}<circle class="yours" cx={3 + mark.yours * 26} cy="4" r="1.75" />{/if}
                          </svg>
                        </span>
                        <span class="sr-only">{mark.of > 1 ? `${mark.agreed} of ${mark.of} choices same as the policy` : same ? 'Same as the policy' : 'Differs from the policy'}{mark.yours !== null ? `, estimate after yours ${formatScore(mark.yours)}` : ''}{mark.its !== null && !same ? `, after its choice ${formatScore(mark.its)}` : ''}</span>
                      {/if}
                      {#if shownMoves.has(entry.id)}<span class="shown type-caption" data-testid="analysis-shown"><span class="sr-only">, </span>analysis shown</span>{/if}
                    {:else}
                      {@render moveWhat(entry, prevWho, false)}
                    {/if}
                  </button>
                </li>
              {:else if entry.passes}
                <li class="move type-caption quiet" data-testid="pass-run">
                  <span class="what">passed{#each entry.passes as part, idx (part.actor)}{idx ? ',' : ''} <span data-testid="log-entry" data-actor={part.actor} class:auto={part.actor === 'auto'}>{part.who}{#if part.count > 1}&nbsp;×{part.count}{/if}</span>{/each}</span>
                </li>
              {:else}
                <li class="move type-caption" class:auto={entry.actor === 'auto'} class:you={entry.actor === 'you'} data-testid="log-entry" data-actor={entry.actor}>
                  {@render moveWhat(entry, prevWho, true)}
                </li>
              {/if}
              {#if entry.children.length > 0}
                {#if turn.busy && !expanded.has(entry.id) && !childCurrent}
                  <li class="consequence-summary type-caption">
                    <button type="button" class="consequence-toggle" data-testid="consequence-toggle" aria-expanded="false" onclick={() => toggleExpanded(entry.id)}>
                      {entry.children.length} {entry.children.length === 1 ? 'consequence' : 'consequences'}
                    </button>
                  </li>
                {:else}
                  <li class="consequences-wrap">
                    {#if turn.busy}
                      <button type="button" class="consequence-toggle type-caption" data-testid="consequence-toggle" aria-expanded="true" onclick={() => toggleExpanded(entry.id)}>
                        Hide {entry.children.length} {entry.children.length === 1 ? 'consequence' : 'consequences'}
                      </button>
                    {/if}
                    <ol class="consequences">
                      {#each entry.children as child (child.id)}
                        {@const isChildJump = Boolean(onjump && child.jumpKey)}
                        {@const isChildInteractive = isChildJump || Boolean(onselectline && child.boardSeq !== null)}
                        <li class="consequence type-caption" class:jump={isChildInteractive} data-testid="history-consequence" data-depth="1">
                          {#if isChildInteractive}
                            <button type="button" aria-current={isLineCurrent(child) ? 'true' : undefined}
                              data-move={isChildJump ? child.jumpKey! : undefined}
                              data-history-seq={child.boardSeq ?? undefined}
                              onclick={() => (isChildJump ? onjump!(child.jumpKey!, child.boardSeq) : onselectline!(child))}>
                              <span class="what">{@render lineSegments(child.segments, false)}</span>
                            </button>
                          {:else}
                            <span class="what">{@render lineSegments(child.segments, true)}</span>
                          {/if}
                        </li>
                      {/each}
                    </ol>
                  </li>
                {/if}
              {/if}
            {/each}
          </ol>
        </li>
      {/each}
    </ol>
  </section>
  {#snippet choiceList()}
    <ol class="list" aria-label="Legal choices">
      {#each view.list as placed (placed.offer.id)}
        {@const choiceText = offerChoiceLabel(placed, view.list, frame.projection)}
        <li>
          <button type="button" class="row" class:chosen={isChosen(placed)} data-testid="action-option" data-offer-id={placed.offer.id}
            data-action-type={placed.offer.action_type} data-action-description={placed.offer.label} data-chosen={isChosen(placed) || undefined}
            aria-label={`${choiceText}${recordedP(placed) !== undefined ? `, ${evidence?.heading ? 'policy' : 'recorded policy'} ${formatProbability(recordedP(placed)!)}` : ''}${estimateOf(placed) !== undefined ? `, estimate after ${formatScore(estimateOf(placed)!)}` : ''}${evidence?.heading ? (choosers(placed) ? `, ${choosers(placed)}` : '') : isChosen(placed) ? ', chosen' : ''}`}
            disabled={!live} onclick={() => onact(placed)}
            onmouseenter={() => onpointoffer(placed)} onmouseleave={() => onpointoffer(null)}
            onfocus={() => onpointoffer(placed)} onblur={() => onpointoffer(null)}>
            <kbd class:hidden-key={placed.number > 9}>{placed.number}</kbd>
            <span class="label" title={recording ? choiceText : undefined}>{#if isChosen(placed)}<span class="check" aria-hidden="true">✓ </span>{/if}<ManaText text={choiceText} />{#if !recording && evidence?.heading && choosers(placed)}<span class="who-chose type-caption" aria-hidden="true" data-testid="offer-choosers">{choosers(placed)}</span>{/if}</span>
            {#if recording && evidence?.heading && choosers(placed)}<span class="who-chose type-caption" aria-hidden="true" data-testid="offer-choosers">{choosers(placed)}</span>{/if}
            {#if estimateOf(placed) !== undefined}<span class="estimate" aria-hidden="true" data-testid="offer-estimate" title="Estimate after this choice, 0–100 from your seat: the next decision's reading, in a world re-dealt from what you knew">{formatScore(estimateOf(placed)!)}</span>{/if}
            {#if recordedP(placed) !== undefined}
              <span class="p" aria-hidden="true" style:--p={recordedP(placed)}>{formatProbability(recordedP(placed)!)}</span>
            {/if}
            {#if anchorText(placed)}<span class="type-caption text-ink-3">{anchorText(placed)}</span>{/if}
          </button>
        </li>
      {/each}
    </ol>
  {/snippet}
</aside>

<style>
  .rail { display: flex; flex-direction: column; min-height: 0; height: 100%; overflow-y: auto; border-left: 1px solid var(--border); background: var(--bg-surface); }
  .block { padding: 10px 14px; border-bottom: 1px solid var(--border); }
  .notices { padding: 8px 14px 0; display: grid; gap: 6px; }
  h2 { margin-bottom: 4px; line-height: 16px; }
  .stack { flex: none; }
  .stack ol { display: grid; gap: 3px; }
  /* Three entries show (a spell, its answer and the answer to that); a deeper stack scrolls. The padding keeps a legal entry's ring inside the scroll box. */
  .stack > ol { max-height: 150px; overflow-y: auto; margin: -2px; padding: 2px; }
  .stack-entry {
    width: 100%; display: flex; align-items: center; gap: 8px; text-align: left; padding: 3px 6px; border-radius: var(--radius-sm);
    border: 1px solid var(--border); background: var(--bg-field); font-size: 14px; line-height: 20px; color: var(--text);
  }
  .stack-entry.top { border-color: var(--border-strong); }
  /* Being cast or activated: on the stack by the rules, its costs still being chosen. */
  .stack-entry[data-announcing] { border-style: dashed; border-color: var(--planned); }
  .stack-entry.legal { box-shadow: 0 0 0 2px var(--legal); }
  .entry-text { display: grid; min-width: 0; }
  .entry-name { font-family: var(--font-serif); font-weight: 600; }
  .targets { color: var(--text); overflow-wrap: anywhere; }
  .entry-title { display: flex; align-items: baseline; gap: 4px; min-width: 0; }
  .tag { flex: none; padding: 0 4px; border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-secondary); }
  .recent { margin-top: 8px; }
  .recent h3 { margin-bottom: 3px; }
  .recent ol { display: grid; gap: 3px; }
  /* Folded, the recap is three lines; opened, it scrolls within a quarter of the window. */
  .recent ol { max-height: 25vh; overflow-y: auto; }
  .fact { display: flex; align-items: center; gap: 6px; color: var(--text-secondary); }
  .fact.theirs { color: var(--text); }
  .fact-btn {
    flex: 1 1 auto; min-width: 0; min-height: 0; padding: 1px 4px; margin: 0 -4px; border-radius: var(--radius-sm);
    text-align: left; color: inherit; background: none; cursor: pointer;
  }
  .fact-btn:hover { background: var(--bg-field); }
  .fact-text { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2; overflow: hidden; }
  .resolved { display: inline-flex; border-radius: var(--radius-sm); outline: 1px dashed color-mix(in srgb, var(--accent) 70%, transparent); outline-offset: 1px; }
  .prompt { flex: none; display: grid; grid-template-columns: minmax(0, 1fr); gap: 6px; }
  .whose { color: var(--text-secondary); }
  .whose.yours { color: var(--text); }
  .detail summary, .raw summary { cursor: pointer; width: max-content; }
  .detail p { margin-top: 2px; max-height: 6em; overflow-y: auto; }
  /* One line at any rail width: the full text is the accessible name and the title. */
  .primary { height: 40px; padding-block: 6px; width: 100%; font-size: 14px; justify-content: center; white-space: nowrap; text-align: center; }
  .primary span { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  .primary kbd { flex: none; opacity: 1; }
  .payment-note { margin-top: -4px; }
  /* Two caption lines are kept so a hint that rewords on hover moves nothing below it. */
  .hint { min-height: 32px; }
  .committed ol { list-style: none; margin: 2px 0 0; padding: 0; }
  .committed li { padding-left: 10px; border-left: 2px solid var(--planned); line-height: 20px; }
  .committed li + li { margin-top: 2px; }
  .primary:disabled { background: var(--bg-muted); color: var(--text-secondary); opacity: 1; }
  .secondary { width: 100%; height: auto; min-height: 32px; padding-block: 5px; justify-content: flex-start; text-align: left; white-space: normal; }
  .pass-turn { justify-self: center; display: inline-flex; gap: 6px; align-items: center; color: var(--text-secondary); padding: 2px 6px; border-radius: var(--radius-sm); }
  .pass-turn:hover { color: var(--text); background: var(--bg-field); }
  .status { min-height: 16px; }
  .choices { flex: 0 1 auto; min-height: 0; max-height: 40%; display: flex; flex-direction: column; }
  .heading { display: flex; justify-content: space-between; }
  .raw { min-height: 0; display: flex; flex-direction: column; gap: 4px; }
  .raw[open] { flex: 1; }
  /* A draft's per-creature answers are not numbered keys. */
  .raw kbd { display: none; }
  .list { flex: 1; min-height: 0; overflow-y: auto; display: grid; grid-template-columns: minmax(0, 1fr); align-content: start; gap: 0; margin: 0 -6px; padding: 2px 6px; }
  .row {
    width: 100%; display: flex; align-items: baseline; gap: 8px; padding: 2px 6px; border-radius: var(--radius-sm); text-align: left;
    border: 1px solid transparent; font-size: 14px; line-height: 20px; color: var(--text); background: none;
  }
  .row:hover:not(:disabled) { border-color: var(--border-strong); background: var(--bg-field); }
  .row:disabled { color: var(--text-secondary); cursor: default; }
  /* A choice's words are the game's face, and its row grows to fit them wrapped. */
  .row .label { flex: 1; min-width: 0; font: 16px/24px var(--font-serif); overflow-wrap: break-word; }
  /* Reviewing, a choice keeps one line like its move, so the bounded list shows several; the whole choice is its title. */
  .recording .row .label { font-size: 14px; line-height: 20px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .recording .row .who-chose { flex: none; margin-left: 0; }
  .hidden-key { visibility: hidden; }
  .log { flex: 1 1 220px; min-height: 120px; display: flex; flex-direction: column; border-bottom: 0; }
  /* Reviewing, the moves are what you read: they take the remaining height, and the choices list keeps a bounded share. */
  .recording .log { flex: 1 1 0; min-height: 200px; }
  .recording .choices { flex: 0 1 auto; min-height: 0; max-height: 26%; }
  .turns { flex: 1; min-height: 0; overflow-y: auto; display: grid; align-content: start; gap: 6px; }
  .turn ol { display: grid; gap: 1px; }
  .turn-head { display: flex; gap: 8px; align-items: baseline; margin-bottom: 2px; color: var(--text-secondary); border-bottom: 1px solid color-mix(in srgb, var(--border) 60%, transparent); }
  .turn-head .number { min-width: 18px; font-variant-numeric: tabular-nums; color: var(--text); }
  .move { padding: 1px 4px; color: var(--text); line-height: 20px; }
  /* What happened is the game's face; who did it and the count stay the interface's. */
  .move .what { display: block; min-width: 0; font: 14px/20px var(--font-serif); overflow-wrap: break-word; }
  .move .who {
    display: inline; margin-right: 5px; font: 500 12px/16px var(--font-sans); color: var(--text-tertiary);
  }
  .move.you .who { color: var(--info-text); }
  .move.auto { color: var(--text-tertiary); }
  .move.quiet { color: var(--text-tertiary); }
  .more { min-height: 0; color: var(--text-secondary); padding: 1px 4px; border-radius: var(--radius-sm); text-decoration: underline dotted; }
  .more:hover { color: var(--text); }
  .elapsed { color: var(--text-tertiary); font-variant-numeric: tabular-nums; }
  .card-name { font-family: var(--font-serif); font-weight: 600; text-decoration: underline dotted color-mix(in srgb, currentColor 45%, transparent); text-underline-offset: 2px; cursor: help; }
  .card-name:hover, .card-name:focus-visible { color: var(--accent-text); }
  .times { color: var(--text-tertiary); font-variant-numeric: tabular-nums; }
  .evidence { display: grid; gap: 1px; padding: 4px 8px; border-left: 2px solid var(--border-strong); min-width: 0; }
  .search-detail summary { cursor: pointer; width: max-content; }
  .search-detail table { width: 100%; margin-top: 4px; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  .search-detail th { text-align: left; font-weight: 600; color: var(--text-tertiary); padding: 0 4px 2px 0; }
  .search-detail td { padding: 1px 4px 1px 0; vertical-align: top; }
  .search-detail tr.chosen td { font-weight: 600; }
  .provenance { overflow-wrap: anywhere; }
  .p { margin-left: auto; flex: none; font: 600 12px/16px var(--font-sans); font-variant-numeric: tabular-nums; color: var(--text-secondary);
    padding: 0 4px; border-radius: var(--radius-sm); background: linear-gradient(90deg, color-mix(in srgb, var(--legal) 28%, transparent) calc(var(--p, 0) * 100%), transparent 0); }
  .row.chosen, .secondary.chosen { border-color: var(--text); }
  .row.chosen .label { font-weight: 600; }
  .row:disabled.chosen { color: var(--text); }
  .primary.chosen { outline: 2px solid var(--text); outline-offset: 2px; }
  .primary .p { color: inherit; background: none; }
  .move.jump { padding: 0; }
  .jump button { width: 100%; min-height: 0; display: block; padding: 2px 5px; border-radius: var(--radius-sm); text-align: left; color: inherit; }
  .jump button:hover { background: var(--bg-field); }
  .jump button[aria-current='true'] { background: color-mix(in srgb, var(--accent) 16%, transparent); box-shadow: inset 2px 0 0 var(--accent); }
  .jump button[data-cursor-source='true'] { background: color-mix(in srgb, var(--accent) 9%, transparent); box-shadow: inset 2px 0 0 var(--border-strong); }
  /* A recording's row also carries its mark, so its words keep one line, as a score's move list does; the whole
     move is the button's title and its accessible name. */
  .recording .turn + .turn { border-top: 1px solid color-mix(in srgb, var(--border) 60%, transparent); padding-top: 2px; }
  .recording .turns { gap: 0; }
  .recording .turn { min-width: 0; }
  .gutter { flex: none; width: 16px; font-variant-numeric: tabular-nums; color: var(--text); }
  .recording .move.jump button { min-width: 0; min-height: 24px; display: flex; align-items: baseline; gap: 6px; padding: 2px 4px; }
  .recording .move.jump .who { flex: none; width: 44px; margin-right: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .recording .move.jump .who.repeat { visibility: hidden; }
  .recording .move.jump .what { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  /* An analysis's reading of your move: neutral ink, never a grade. */
  .reading { flex: none; display: inline-flex; gap: 4px; align-items: center; }
  .mark { font-family: var(--font-sans); color: var(--text-tertiary); }
  .shown { flex: none; font-family: var(--font-sans); color: var(--text-tertiary); }
  .mark[data-same='false'] { color: var(--text-secondary); }
  .track { width: 32px; height: 8px; }
  .track line { stroke: var(--border); stroke-width: 1; }
  .track .its { fill: none; stroke: var(--text-tertiary); stroke-width: 1; }
  .track .yours { fill: var(--text-secondary); }
  /* Who chose it follows the choice on its line, wrapping under it only when the line is full. */
  .who-chose { margin-left: 6px; font: 400 12px/16px var(--font-sans); color: var(--text-tertiary); white-space: nowrap; }
  .estimate { flex: none; font: 400 12px/16px var(--font-sans); font-variant-numeric: tabular-nums; color: var(--text-tertiary); min-width: 2ch; text-align: right; }
  .consequences-wrap, .consequence-summary { padding-left: 6px; }
  .consequences { display: grid; gap: 1px; border-left: 1px solid var(--border); padding-left: 7px; margin: 1px 0 3px; }
  .consequence { color: var(--text-secondary); line-height: 20px; }
  .consequence .what { display: block; font: 14px/20px var(--font-serif); overflow-wrap: break-word; }
  .consequence-toggle { min-height: 0; color: var(--text-tertiary); padding: 1px 3px; text-decoration: underline dotted; }
  .consequence-toggle:hover { color: var(--text); }
</style>
