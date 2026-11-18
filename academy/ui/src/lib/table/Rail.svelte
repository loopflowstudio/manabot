<script lang="ts">
  /**
   * The rail beside the board: the stack and what happened since your last
   * move, the prompt in one plain line with its buttons, every legal choice by
   * number, and the log as a move list. The list is the guarantee that the
   * board hides nothing: it holds every offer of the frame, in order.
   */
  import type { Snippet } from 'svelte';
  import type { ExperienceFrame } from '$lib/types';
  import type { MoveTurn } from './moves';
  import type { RecentView } from './recent';
  import Thumb from './Thumb.svelte';
  import { stackName, type PlacedOffer, type StackItem, type TableView } from './view';

  interface Props {
    view: TableView;
    frame: ExperienceFrame;
    live: boolean;
    /** What to do now, in plain words. */
    line: string;
    status: string;
    whose: string;
    /** "Search 64 is choosing…" while someone else decides. */
    thinking: string | null;
    interrupted: boolean;
    canPassTurn: boolean;
    recent: RecentView;
    moves: MoveTurn[];
    opponentName: string;
    onact: (placed: PlacedOffer) => void;
    onpassturn: () => void;
    onstack: (item: StackItem, element: HTMLElement) => void;
    onpreviewname: (name: string | null, element: HTMLElement | null) => void;
    notices?: Snippet;
    result?: Snippet;
  }

  let {
    view, frame, live, line, status, whose, thinking, interrupted, canPassTurn, recent, moves, opponentName,
    onact, onpassturn, onstack, onpreviewname, notices, result,
  }: Props = $props();

  const context = $derived(frame.projection.decision_context);
  const detail = $derived(context?.instruction ?? frame.prompt?.instruction ?? '');
  const secondary = $derived(view.buttons.filter(placed => placed !== view.primary));
  const mine = (controller: number): boolean => controller === frame.projection.agent.player_index;
  const resolved = $derived(new Map(recent.resolved.map(spell => [spell.id, spell])));

  let logEnd = $state<HTMLElement | null>(null);
  $effect(() => {
    void moves.length;
    void moves.at(-1)?.moves.length;
    logEnd?.scrollIntoView({ block: 'end' });
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

<aside class="rail" aria-label="Stack, decision and legal choices">
  {#if notices}<div class="notices">{@render notices()}</div>{/if}

  <section class="block stack" aria-label="Stack">
    <h2><span class="type-rubric text-ink-2">Stack{#if view.stack.length} · {view.stack.length}{/if}</span></h2>
    {#if view.stack.length === 0}
      <p class="type-annotation text-ink-3">Empty</p>
    {:else}
      <ol>
        {#each view.stack as item, index (item.entry.id)}
          <li>
            <button type="button" class="stack-entry" class:legal={live && item.offers.length > 0} class:top={index === 0}
              data-testid="stack-entry" data-stack-card={item.card?.id}
              data-travel-id={item.card ? `card:${item.card.id}` : undefined} data-travel-side={mine(item.entry.controller) ? 'you' : 'opponent'} data-travel-name={item.card?.name}
              aria-label={`${item.entry.label}, ${mine(item.entry.controller) ? 'yours' : opponentName + "'s"}${item.offers.length ? ` — ${item.offers.map(placed => placed.offer.label).join('; ')}` : ''}`}
              onclick={event => onstack(item, event.currentTarget)}
              {...hover(item.card?.name ?? item.entry.label)}>
              <Thumb name={item.card?.name ?? item.entry.label} />
              <span class="entry-text">
                <span class="truncate">{stackName(item.entry)}</span>
                <span class="type-caption text-ink-3">{mine(item.entry.controller) ? 'You' : opponentName}{index === 0 ? ' · resolves next' : ''}</span>
              </span>
            </button>
          </li>
        {/each}
      </ol>
    {/if}
    {#if recent.facts.length}
      <div class="recent" data-testid="recent">
        <h3 class="type-rubric text-ink-3">Since your last move</h3>
        <ol>
          {#each recent.facts as fact (fact.seq)}
            {@const spell = fact.spell === null ? undefined : resolved.get(fact.spell)}
            <li class="type-caption fact" class:theirs={fact.by === 'opponent'} data-testid="recent-fact">
              {#if spell}
                <span class="resolved" data-stack-card={spell.id} tabindex="-1" {...hover(spell.name)}><Thumb name={spell.name} /></span>
              {/if}
              <span>{fact.text}</span>
            </li>
          {/each}
        </ol>
      </div>
    {/if}
  </section>

  <section class="block prompt" aria-label="Decision" data-testid="action-panel" data-action-space-kind={frame.action_space}>
    <p class="type-rubric whose" class:yours={live}>{#if thinking}<i class="thinking" aria-hidden="true"></i>{/if}{thinking ?? whose}</p>
    {#if result}
      {@render result()}
    {:else}
      {#if interrupted}
        <p class="type-annotation text-ink-2" data-testid="decision-prompt">The live session is unavailable. These choices belong to the last recorded decision and cannot be submitted.</p>
      {:else if frame.prompt}
        <p class="line type-title text-ink" data-testid="decision-prompt" data-kind={frame.action_space}>{line}</p>
        {#if detail}
          <details class="detail">
            <summary class="type-caption text-ink-3">Rules detail</summary>
            <p class="type-caption text-ink-2">{detail}</p>
          </details>
        {/if}
      {/if}
      {#if view.primary}
        <button type="button" class="btn btn-primary primary" data-testid="primary-action" data-offer-id={view.primary.offer.id}
          disabled={!live} onclick={() => onact(view.primary!)}>
          {view.primary.offer.label}<kbd>Space</kbd>
        </button>
      {/if}
      {#each secondary as placed (placed.offer.id)}
        <button type="button" class="btn btn-secondary secondary" data-testid="prompt-action" data-offer-id={placed.offer.id}
          disabled={!live} onclick={() => onact(placed)}>
          <kbd>{placed.number}</kbd>{placed.offer.label}
        </button>
      {/each}
      {#if canPassTurn}
        <button type="button" class="pass-turn type-caption" data-testid="pass-turn" onclick={onpassturn}
          title="Pass every priority window until the turn ends">Pass turn <kbd>F6</kbd></button>
      {/if}
    {/if}
    <!-- While you may act, the prompt above already says what the status would. -->
    <p class="type-caption text-ink-3 status" class:sr-only={live || thinking !== null} data-testid="current-decision" role="status" aria-live="polite" aria-atomic="true">{thinking ?? status}</p>
  </section>

  <section class="block choices" aria-label="All legal choices">
    <h2 class="heading type-rubric text-ink-2">
      <span class="type-rubric">{interrupted ? 'Recorded choices' : 'All legal choices'}</span>
      <span class="type-rubric" data-testid="legal-choice-count">{frame.offers.length}<span class="sr-only"> {interrupted ? 'recorded' : 'legal'} choices</span></span>
    </h2>
    <ol class="list" aria-label="Legal choices">
      {#each view.list as placed (placed.offer.id)}
        <li>
          <button type="button" class="row" data-testid="action-option" data-offer-id={placed.offer.id}
            data-action-type={placed.offer.action_type} data-action-description={placed.offer.label}
            aria-label={placed.offer.label} disabled={!live} onclick={() => onact(placed)}>
            <kbd class:hidden-key={placed.number > 9}>{placed.number}</kbd>
            <span class="label">{placed.offer.label}</span>
            {#if anchorText(placed)}<span class="type-caption text-ink-3">{anchorText(placed)}</span>{/if}
          </button>
        </li>
      {/each}
    </ol>
  </section>

  <section class="block log" aria-label="Moves">
    <h2><span class="type-rubric text-ink-2">Moves</span></h2>
    <ol class="turns" data-testid="game-log">
      {#each moves as turn, index (`${turn.turn}:${index}`)}
        <li class="turn" data-turn={turn.turn}>
          <p class="turn-head type-rubric"><span class="number">{turn.turn ?? '·'}</span>{#if turn.owner}<span class="owner">{turn.owner}</span>{/if}</p>
          <ol>
            {#each turn.moves as move, at (move.id)}
              <li class="move type-caption" class:auto={move.actor === 'auto'} class:you={move.actor === 'you'} data-testid="log-entry" data-actor={move.actor}>
                <!-- A score names the player once per run of their moves. -->
                <span class="who" class:repeat={turn.moves[at - 1]?.who === move.who}>{move.who}</span>
                <span class="what">{#each move.segments as segment, at (at)}{#if segment.card}<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_static_element_interactions (focus shows the card; a button would not wrap inside a line) --><span class="card-name" tabindex="0" {...hover(segment.card)}>{segment.text}</span>{:else}{segment.text}{/if}{/each}{#if move.count > 1}<span class="times">&nbsp;×{move.count}</span>{/if}</span>
              </li>
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
  .stack ol { display: grid; gap: 3px; }
  .stack-entry {
    width: 100%; display: flex; align-items: center; gap: 8px; text-align: left; padding: 3px 6px; border-radius: 5px;
    border: 1px solid var(--border); background: var(--bg-field); font-size: 13px; color: var(--text);
  }
  .stack-entry.top { border-color: var(--border-strong); }
  .stack-entry.legal { box-shadow: 0 0 0 2px var(--legal); }
  .entry-text { display: grid; min-width: 0; line-height: 16px; }
  .recent { margin-top: 8px; }
  .recent h3 { margin-bottom: 3px; }
  .recent ol { display: grid; gap: 3px; }
  .fact { display: flex; align-items: center; gap: 8px; color: var(--text-secondary); line-height: 15px; }
  .fact.theirs { color: var(--text); }
  .resolved { display: inline-flex; border-radius: 3px; outline: 1px dashed color-mix(in srgb, var(--accent) 70%, transparent); outline-offset: 1px; }
  .prompt { display: grid; grid-template-columns: minmax(0, 1fr); gap: 6px; }
  .whose { color: var(--text-tertiary); display: flex; align-items: center; gap: 6px; }
  .whose.yours { color: var(--error-text); }
  .thinking { width: 6px; height: 6px; border-radius: 3px; background: var(--warning); animation: pulse 1.2s ease-in-out infinite; }
  @keyframes pulse { 50% { opacity: 0.25; } }
  .line { font-size: 15px; line-height: 20px; }
  .detail summary { cursor: pointer; width: max-content; }
  .detail p { margin-top: 2px; max-height: 6em; overflow-y: auto; }
  .primary { height: 40px; width: 100%; font-size: 14px; justify-content: center; }
  .primary:disabled { background: var(--bg-muted); color: var(--text-secondary); opacity: 1; }
  .secondary { width: 100%; height: auto; min-height: 32px; padding-block: 5px; justify-content: flex-start; text-align: left; white-space: normal; }
  kbd {
    font-family: var(--font-mono); font-size: 10px; line-height: 14px; padding: 0 4px; border-radius: 3px;
    border: 1px solid currentColor; opacity: 0.75; min-width: 16px; text-align: center;
  }
  .pass-turn { justify-self: center; display: inline-flex; gap: 6px; align-items: center; color: var(--text-secondary); padding: 2px 6px; border-radius: 4px; }
  .pass-turn:hover { color: var(--text); background: var(--bg-field); }
  .status { min-height: 16px; }
  .choices { flex: 1 1 auto; min-height: 110px; display: flex; flex-direction: column; }
  .heading { display: flex; justify-content: space-between; }
  .list { flex: 1; min-height: 0; overflow-y: auto; display: grid; align-content: start; gap: 2px; margin: 0 -6px; padding: 2px 6px; }
  .row {
    width: 100%; display: flex; align-items: center; gap: 8px; padding: 4px 6px; border-radius: 5px; text-align: left;
    border: 1px solid transparent; font-size: 13.5px; line-height: 18px; color: var(--text); background: none;
  }
  .row:hover:not(:disabled) { border-color: var(--border-strong); background: var(--bg-field); }
  .row:disabled { color: var(--text-secondary); cursor: default; }
  .row .label { flex: 1; min-width: 0; }
  .hidden-key { visibility: hidden; }
  .log { flex: 0 1 190px; min-height: 80px; display: flex; flex-direction: column; border-bottom: 0; }
  .turns { flex: 1; min-height: 0; overflow-y: auto; mask-image: linear-gradient(transparent, #000 14px); display: grid; align-content: start; gap: 4px; }
  .turn-head { display: flex; gap: 8px; align-items: baseline; color: var(--text-secondary); border-bottom: 1px solid color-mix(in srgb, var(--border) 60%, transparent); }
  .turn-head .number { min-width: 18px; font-variant-numeric: tabular-nums; color: var(--text); }
  .move { display: flex; align-items: baseline; gap: 6px; padding-left: 26px; color: var(--text); line-height: 16px; }
  .move .what { flex: 1; min-width: 0; }
  .move .who { flex: none; width: 60px; color: var(--text-tertiary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .move .who.repeat { visibility: hidden; }
  .move.you .who { color: var(--info-text); }
  .move.auto { color: var(--text-tertiary); font-style: italic; }
  .card-name { text-decoration: underline dotted color-mix(in srgb, currentColor 45%, transparent); text-underline-offset: 2px; cursor: help; }
  .card-name:hover, .card-name:focus-visible { color: var(--accent-text); }
  .times { color: var(--text-tertiary); font-variant-numeric: tabular-nums; }
  @media (prefers-reduced-motion: reduce) {
    .thinking { animation: none; }
  }
</style>
