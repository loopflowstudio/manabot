<script lang="ts">
  import { region } from './region';
  /**
   * The rail beside the board: the stack and what happened since your last
   * move, the prompt in one plain line with its buttons, every legal choice by
   * number, and the log as a move list. The list is the guarantee that the
   * board hides nothing: it holds every offer of the frame, in order.
   */
  import type { Snippet } from 'svelte';
  import type { ExperienceFrame } from '$lib/types';
  import type { HistoryLine, HistoryTurn } from './history';
  import type { Segment } from './tell';
  import { committed, type PrimaryLabel } from './prompt';
  import type { RecentView } from './recent';
  import { formatProbability, type OfferEvidence } from './replay';
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
    recent: RecentView;
    history: HistoryTurn[];
    opponentName: string;
    onact: (placed: PlacedOffer) => void;
    onpassturn: () => void;
    onstack: (item: StackItem, element: HTMLElement) => void;
    onpreviewname: (name: string | null, element: HTMLElement | null) => void;
    /** A row under the pointer or focus, whose payment the board may preview. */
    onpointoffer: (placed: PlacedOffer | null) => void;
    /** Why the board shows no planned taps for the payment, when the preview says. */
    paymentNote: string | null;
    notices?: Snippet;
    /** The game has ended; its result card sits over the board. */
    over?: boolean;
    /** The docked inspect pane, when inspecting docks to the rail. */
    dock?: Snippet;
    /** The recorded policy at a recorded choice. */
    evidence?: OfferEvidence | null;
    /** The recap's heading. */
    recapTitle?: string;
    /** Moves jump to their position when set (Review). */
    onjump?: (id: string, boardSeq: number | null) => void;
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
    view, stack: stackGiven, frame, live, line, hint, primaryText, status, whose, thinking, opponentChoosing, workStarted, interrupted, canPassTurn, recent, history, opponentName,
    onact, onpassturn, onstack, onpreviewname, onpointoffer, paymentNote, notices, over = false, dock,
    evidence = null, recapTitle = 'Since your last move', onjump, currentMove = null, onselectline, cursorLineIds, cursorSourceSeq = null, draft, stopped = null,
  }: Props = $props();

  const stack = $derived(stackGiven ?? view.stack);

  /** No choices because the opponent is deciding: the list waits for them by name instead of counting 0. */
  const waitingFor = $derived(frame.offers.length === 0 && opponentChoosing);
  const recordedP = (placed: PlacedOffer): number | undefined => evidence?.probability.get(placed.offer.id);
  const isChosen = (placed: PlacedOffer): boolean => evidence?.chosen === placed.offer.id;

  const context = $derived(frame.projection.decision_context);
  const paid = $derived(committed(frame));
  const detail = $derived(context?.instruction ?? frame.prompt?.instruction ?? '');
  const secondary = $derived(view.buttons.filter(placed => placed !== view.primary));
  const mine = (controller: number): boolean => controller === frame.projection.agent.player_index;
  const resolved = $derived(new Map(recent.resolved.map(spell => [spell.id, spell])));

  // The recap shows its first lines; the rest open on demand and fold again when the recap changes.
  const RECAP_LINES = 3;
  let recapOpen = $state(false);
  $effect(() => { void recent.facts; recapOpen = false; });
  const facts = $derived(recapOpen ? recent.facts : recent.facts.slice(0, RECAP_LINES));

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

  let logEnd = $state<HTMLElement | null>(null);
  let logList = $state<HTMLElement | null>(null);
  let expanded = $state<Set<string>>(new Set());
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
    const hasCursor = (cursorLineIds?.size ?? 0) > 0;
    const current = currentMove;
    if (!hasCursor && current === null && !onjump) { logEnd?.scrollIntoView({ block: 'end' }); return; }
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

<aside class="rail" aria-label="Stack, decision and legal choices">
  {#if notices}<div class="notices">{@render notices()}</div>{/if}

  {#if dock}<div class="block docked">{@render dock()}</div>{/if}

  <section use:region={{ across: !onjump }} class="block stack" aria-label="Stack">
    <h2><span class="type-heading text-ink-2">Stack{#if stack.length} · {stack.length}{/if}</span></h2>
    {#if stack.length === 0}
      <p class="type-caption text-ink-3">Empty</p>
    {:else}
      <ol>
        {#each stack as item, index (item.entry.id)}
          {@const hitPlayers = item.entry.combat_damage_to_players?.map(seat => mine(seat) ? 'you' : opponentName).join(' and ')}
          {@const damageHistory = hitPlayers ? `Source dealt combat damage to ${hitPlayers} this turn` : ''}
          {@const provenance = stackProvenance(item.entry)}
          <li>
            <button type="button" class="stack-entry" class:legal={live && item.offers.length > 0} class:top={index === 0}
              data-testid="stack-entry" data-stack-card={item.card?.id ?? (item.announcing ? item.entry.source_permanent : undefined)}
              data-announcing={item.announcing}
              data-travel-id={item.card ? `card:${item.card.id}` : `stack:${item.entry.id}`}
              data-travel-source={item.entry.source_permanent !== undefined ? `perm:${item.entry.source_permanent}` : undefined}
              data-travel-side={mine(item.entry.controller) ? 'you' : 'opponent'}
              data-travel-name={item.card?.name ?? item.entry.label.split(' — ')[0]}
              aria-label={`${item.entry.label}${provenance.map(tag => `, ${tag}`).join('')}${damageHistory ? `. ${damageHistory}` : ''}, ${mine(item.entry.controller) ? 'yours' : opponentName + "'s"}${item.offers.length ? ` — ${item.offers.map(placed => placed.offer.label).join('; ')}` : ''}`}
              onclick={event => onstack(item, event.currentTarget)}
              {...hover(item.card?.name ?? item.entry.label)}>
              <Thumb name={item.card?.name ?? item.entry.label} />
              <span class="entry-text">
                <span class="entry-title">
                  <span class="truncate entry-name">{stackName(item.entry)}</span>
                  {#each provenance as tag (tag)}<span class="tag type-caption" data-testid="stack-provenance">{tag}</span>{/each}
                </span>
                {#if damageHistory}<span class="type-caption text-ink-3">{damageHistory}</span>{/if}
                <span class="type-caption text-ink-3">{mine(item.entry.controller) ? 'You' : opponentName}{item.announcing ? ` · ${item.announcing === 'spell' ? 'casting' : 'activating'}` : index === 0 ? ' · resolves next' : ''}</span>
              </span>
            </button>
          </li>
        {/each}
      </ol>
    {/if}
    {#if recent.facts.length}
      <div class="recent" data-testid="recent">
        <h3><span class="type-heading text-ink-2">{recapTitle}</span></h3>
        <ol>
          {#each facts as fact (fact.seq)}
            {@const spell = fact.spell === null ? undefined : resolved.get(fact.spell)}
            <li class="type-caption fact" class:theirs={fact.by === 'opponent'} data-testid="recent-fact">
              {#if spell}
                <span class="resolved" data-stack-card={spell.id} tabindex="-1" {...hover(spell.name)}><Thumb name={spell.name} /></span>
              {/if}
              <span class="fact-text" title={fact.text}>{fact.text}</span>
            </li>
          {/each}
        </ol>
        {#if recent.facts.length > RECAP_LINES}
          <button type="button" class="more type-caption" data-testid="recent-more" aria-expanded={recapOpen} onclick={() => (recapOpen = !recapOpen)}>
            {recapOpen ? 'Fewer' : `${recent.facts.length - RECAP_LINES} more`}
          </button>
        {/if}
      </div>
    {/if}
  </section>

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
      {:else}
      {#if view.primary}
        <button type="button" class="btn btn-primary primary" class:chosen={isChosen(view.primary)} data-testid="primary-action" data-offer-id={view.primary.offer.id} aria-label={primaryText?.description ?? view.primary.offer.label} title={primaryText?.description ?? view.primary.offer.label}
          disabled={!live} onclick={() => onact(view.primary!)}>
          <span>{#if isChosen(view.primary)}✓ {/if}<ManaText text={primaryText?.text ?? view.primary.offer.label} /></span>{#if recordedP(view.primary) !== undefined}<span class="p">{formatProbability(recordedP(view.primary)!)}</span>{:else if live}<kbd>Space</kbd>{/if}
        </button>
        {#if paymentNote}<p class="type-caption text-ink-3 payment-note" data-testid="payment-note">{paymentNote}</p>{/if}
      {/if}
      {#each secondary as placed (placed.offer.id)}
        <button type="button" class="btn btn-secondary secondary" class:chosen={isChosen(placed)} data-testid="prompt-action" data-offer-id={placed.offer.id}
          disabled={!live} onclick={() => onact(placed)}>
          <kbd>{placed.number}</kbd>{#if isChosen(placed)}✓ {/if}<ManaText text={placed.offer.label} />{#if recordedP(placed) !== undefined}<span class="p">{formatProbability(recordedP(placed)!)}</span>{/if}
        </button>
      {/each}
      {/if}
      {#if canPassTurn}
        <button type="button" class="pass-turn type-caption" data-testid="pass-turn" onclick={onpassturn}
          title="Pass every priority window until the turn ends">Pass turn <kbd>F6</kbd></button>
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

  <!-- A finished game has no choices to list; the result card says what happened. -->
  {#if !over || frame.offers.length}
  <section use:region={{ across: !onjump }} class="block choices" aria-label="All legal choices">
    <h2 class="heading type-heading text-ink-2">
      <span>{interrupted || (!live && evidence) ? 'Recorded choices' : 'All legal choices'}</span>
      {#if !waitingFor}<span class="count" data-testid="legal-choice-count">{frame.offers.length}<span class="sr-only"> {interrupted ? 'recorded' : 'legal'} choices</span></span>{/if}
    </h2>
    {#if waitingFor}<p class="type-caption text-ink-3" data-testid="choices-waiting">Waiting for {opponentName}</p>{/if}
    <ol class="list" aria-label="Legal choices">
      {#each view.list as placed (placed.offer.id)}
        <li>
          <button type="button" class="row" class:chosen={isChosen(placed)} data-testid="action-option" data-offer-id={placed.offer.id}
            data-action-type={placed.offer.action_type} data-action-description={placed.offer.label} data-chosen={isChosen(placed) || undefined}
            aria-label={`${placed.offer.label}${recordedP(placed) !== undefined ? `, recorded policy ${formatProbability(recordedP(placed)!)}` : ''}${isChosen(placed) ? ', chosen' : ''}`}
            disabled={!live} onclick={() => onact(placed)}
            onmouseenter={() => onpointoffer(placed)} onmouseleave={() => onpointoffer(null)}
            onfocus={() => onpointoffer(placed)} onblur={() => onpointoffer(null)}>
            <kbd class:hidden-key={placed.number > 9}>{placed.number}</kbd>
            <span class="label">{#if isChosen(placed)}<span class="check" aria-hidden="true">✓ </span>{/if}<ManaText text={placed.offer.label} /></span>
            {#if recordedP(placed) !== undefined}
              <span class="p" aria-hidden="true" style:--p={recordedP(placed)}>{formatProbability(recordedP(placed)!)}</span>
            {/if}
            {#if anchorText(placed)}<span class="type-caption text-ink-3">{anchorText(placed)}</span>{/if}
          </button>
        </li>
      {/each}
    </ol>
  </section>
  {/if}

  <section use:region={{ across: false }} class="block log" aria-label="History">
    <h2><span class="type-heading text-ink-2">History</span></h2>
    <ol class="turns" data-testid="game-log" bind:this={logList}>
      {#each history as turn, index (`${turn.turn}:${index}`)}
        <li class="turn" data-turn={turn.turn}>
          <p class="turn-head type-label"><span class="number">{turn.turn ?? '·'}</span>{#if turn.owner}<span class="owner">{turn.owner}</span>{/if}</p>
          <ol>
            {#each turn.lines as entry, at (entry.id)}
              {@const jumpId = entry.jumpKey ?? entry.id}
              {@const prevWho = at > 0 ? turn.lines[at - 1]?.who : ''}
              {@const hasCursor = (cursorLineIds?.size ?? 0) > 0}
              {@const isCurrent = hasCursor ? cursorLineIds!.has(entry.id) : (currentMove !== null && currentMove === jumpId)}
              {@const isSource = cursorSourceSeq !== null && entry.seq === cursorSourceSeq}
              {@const childCurrent = hasCursor
                ? entry.children.some(child => cursorLineIds!.has(child.id))
                : (currentMove !== null && entry.children.some(child => child.jumpKey === currentMove))}
              {@const isJump = Boolean(onjump && entry.jumpKey)}
              {@const isSelectable = !entry.passes && Boolean(onselectline && entry.boardSeq !== null)}
              {#if isJump || isSelectable}
                <!-- A recording's line jumps to its position; a live line selects its cursor board. -->
                <li class="move type-caption jump" class:quiet={entry.passes !== null} class:auto={entry.actor === 'auto'} class:you={entry.actor === 'you'}
                  data-testid={entry.passes ? 'pass-run' : 'log-entry'} data-actor={entry.actor}>
                  <button type="button" aria-current={isCurrent ? 'true' : undefined} data-cursor-source={isSource || undefined}
                    data-move={isJump ? jumpId : undefined} data-history-seq={entry.boardSeq ?? undefined}
                    onclick={() => (isJump ? onjump!(jumpId, entry.boardSeq) : onselectline!(entry))}>
                    <span class="who" class:repeat={!entry.passes && entry.who !== '' && prevWho === entry.who}>{entry.passes ? '' : entry.who}</span>
                    <span class="what">{#if entry.passes}passed {entry.passes.map(part => `${part.who}${part.count > 1 ? ` ×${part.count}` : ''}`).join(', ')}{:else}{@render lineSegments(entry.segments, false)}{#if entry.count > 1}<span class="times">&nbsp;×{entry.count}</span>{/if}{/if}</span>
                  </button>
                </li>
              {:else if entry.passes}
                <li class="move type-caption quiet" data-testid="pass-run">
                  <span class="who" aria-hidden="true"></span>
                  <span class="what">passed{#each entry.passes as part, idx (part.actor)}{idx ? ',' : ''} <span data-testid="log-entry" data-actor={part.actor} class:auto={part.actor === 'auto'}>{part.who}{#if part.count > 1}&nbsp;×{part.count}{/if}</span>{/each}</span>
                </li>
              {:else}
                <li class="move type-caption" class:auto={entry.actor === 'auto'} class:you={entry.actor === 'you'} data-testid="log-entry" data-actor={entry.actor}>
                  <!-- A score names the player once per run of their moves. -->
                  <span class="who" class:repeat={entry.who !== '' && prevWho === entry.who}>{entry.who}</span>
                  <span class="what">{@render lineSegments(entry.segments, true)}{#if entry.count > 1}<span class="times">&nbsp;×{entry.count}</span>{/if}</span>
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
                        {@const isChildCurrent = hasCursor ? cursorLineIds!.has(child.id) : (currentMove !== null && child.jumpKey === currentMove)}
                        {@const isChildJump = Boolean(onjump && child.jumpKey)}
                        {@const isChildSelectable = Boolean(onselectline && child.boardSeq !== null)}
                        {#if isChildJump || isChildSelectable}
                          <li class="consequence type-caption jump" data-testid="history-consequence" data-depth="1">
                            <button type="button" aria-current={isChildCurrent ? 'true' : undefined}
                              data-move={isChildJump ? child.jumpKey! : undefined}
                              data-history-seq={child.boardSeq ?? undefined}
                              onclick={() => (isChildJump ? onjump!(child.jumpKey!, child.boardSeq) : onselectline!(child))}>
                              <span class="what">{@render lineSegments(child.segments, false)}</span>
                            </button>
                          </li>
                        {:else}
                          <li class="consequence type-caption" data-testid="history-consequence" data-depth="1">
                            <span class="what">{@render lineSegments(child.segments, true)}</span>
                          </li>
                        {/if}
                      {/each}
                    </ol>
                  </li>
                {/if}
              {/if}
            {/each}
          </ol>
        </li>
      {/each}
      <li bind:this={logEnd} aria-hidden="true"></li>
    </ol>
  </section>
</aside>

<style>
  .rail { display: flex; flex-direction: column; min-height: 0; height: 100%; border-left: 1px solid var(--border); background: var(--bg-surface); }
  .block { padding: 10px 14px; border-bottom: 1px solid var(--border); }
  .notices { padding: 8px 14px 0; display: grid; gap: 6px; }
  h2 { margin-bottom: 4px; line-height: 16px; }
  .stack { max-height: 34vh; overflow-y: auto; }
  /* A crowded rail never cuts the top two stack entries (a spell and what it answers): the block keeps room for its
     heading and one entry (88 px) or two (135 px); deeper entries and the recap below scroll. */
  .stack:has(.stack-entry) { min-height: 88px; }
  .stack:has(li + li .stack-entry) { min-height: 135px; }
  .stack ol { display: grid; gap: 3px; }
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
  .entry-title { display: flex; align-items: baseline; gap: 4px; min-width: 0; }
  .tag { flex: none; padding: 0 4px; border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-secondary); }
  .recent { margin-top: 8px; }
  .recent h3 { margin-bottom: 3px; }
  .recent ol { display: grid; gap: 3px; }
  .fact { display: flex; align-items: center; gap: 8px; color: var(--text-secondary); }
  .fact.theirs { color: var(--text); }
  .fact-text { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2; overflow: hidden; }
  .resolved { display: inline-flex; border-radius: var(--radius-sm); outline: 1px dashed color-mix(in srgb, var(--accent) 70%, transparent); outline-offset: 1px; }
  .prompt { display: grid; grid-template-columns: minmax(0, 1fr); gap: 6px; }
  .whose { color: var(--text-secondary); }
  .whose.yours { color: var(--text); }
  .detail summary { cursor: pointer; width: max-content; }
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
  .choices { flex: 1 1 auto; min-height: 110px; display: flex; flex-direction: column; }
  .heading { display: flex; justify-content: space-between; }
  .list { flex: 1; min-height: 0; overflow-y: auto; display: grid; align-content: start; gap: 0; margin: 0 -6px; padding: 2px 6px; }
  .row {
    width: 100%; display: flex; align-items: baseline; gap: 8px; padding: 2px 6px; border-radius: var(--radius-sm); text-align: left;
    border: 1px solid transparent; font-size: 14px; line-height: 20px; color: var(--text); background: none;
  }
  .row:hover:not(:disabled) { border-color: var(--border-strong); background: var(--bg-field); }
  .row:disabled { color: var(--text-secondary); cursor: default; }
  /* A choice's words are the game's face, and its row grows to fit them wrapped. */
  .row .label { flex: 1; min-width: 0; font: 16px/24px var(--font-serif); overflow-wrap: break-word; }
  .hidden-key { visibility: hidden; }
  .log { flex: 0 1 190px; min-height: 92px; display: flex; flex-direction: column; border-bottom: 0; }
  .turns { flex: 1; min-height: 0; overflow-y: auto; mask-image: linear-gradient(transparent, var(--text) 14px); display: grid; align-content: start; gap: 4px; }
  .turn-head { display: flex; gap: 8px; align-items: baseline; color: var(--text-secondary); border-bottom: 1px solid color-mix(in srgb, var(--border) 60%, transparent); }
  .turn-head .number { min-width: 18px; font-variant-numeric: tabular-nums; color: var(--text); }
  .move { display: flex; align-items: baseline; gap: 6px; padding-left: 26px; color: var(--text); line-height: 16px; }
  /* What happened is the game's face; who did it and the count stay the interface's. */
  .move .what { flex: 1; min-width: 0; font: 14px/20px var(--font-serif); }
  .move .who { flex: none; width: 60px; color: var(--text-tertiary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .move .who.repeat { visibility: hidden; }
  .move.you .who { color: var(--info-text); }
  .move.auto { color: var(--text-tertiary); }
  .move.quiet { color: var(--text-tertiary); }
  .more { color: var(--text-secondary); padding: 0 4px; margin-left: 30px; border-radius: var(--radius-sm); text-decoration: underline dotted; }
  .more:hover { color: var(--text); }
  .elapsed { color: var(--text-tertiary); font-variant-numeric: tabular-nums; }
  .docked { flex: none; max-height: 36%; overflow-y: auto; }
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
  .move.jump { padding-left: 0; }
  .move.jump button { width: 100%; display: flex; align-items: baseline; gap: 6px; padding: 0 4px 0 26px; border-radius: var(--radius-sm); text-align: left; color: inherit; }
  .move.jump button:hover { background: var(--bg-field); }
  .move.jump button[aria-current='true'] { background: color-mix(in srgb, var(--accent) 16%, transparent); box-shadow: inset 2px 0 0 var(--accent); }
  .move.jump button[data-cursor-source='true'] { background: color-mix(in srgb, var(--accent) 9%, transparent); box-shadow: inset 2px 0 0 var(--border-strong); }
  .consequences-wrap, .consequence-summary { padding-left: 92px; }
  .consequences { display: grid; gap: 1px; border-left: 1px solid var(--border); padding-left: 8px; margin: 1px 0 3px; }
  .consequence { color: var(--text-secondary); line-height: 16px; }
  .consequence .what { font: 14px/20px var(--font-serif); }
  .consequence.jump button { width: 100%; display: block; padding: 0 4px; border-radius: var(--radius-sm); text-align: left; color: inherit; }
  .consequence.jump button:hover { background: var(--bg-field); }
  .consequence.jump button[aria-current='true'] { background: color-mix(in srgb, var(--accent) 16%, transparent); box-shadow: inset 2px 0 0 var(--accent); }
  .consequence-toggle { color: var(--text-tertiary); padding: 0 2px; text-decoration: underline dotted; }
  .consequence-toggle:hover { color: var(--text); }
</style>
