<script lang="ts">
  /**
   * The rail beside the board: the stack, the prompt with its buttons, every
   * legal choice by number, and the log. The list is the guarantee that the
   * board hides nothing: it holds every offer of the frame, in order.
   */
  import type { Snippet } from 'svelte';
  import type { ExperienceFrame, GameLogEntry } from '$lib/types';
  import type { PlacedOffer, StackItem, TableView } from './view';

  interface Props {
    view: TableView;
    frame: ExperienceFrame;
    live: boolean;
    status: string;
    whose: string;
    canPassTurn: boolean;
    log: GameLogEntry[];
    onact: (placed: PlacedOffer) => void;
    onpassturn: () => void;
    onstack: (item: StackItem, element: HTMLElement) => void;
    onpreviewstack: (item: StackItem | null, element: HTMLElement | null) => void;
    notices?: Snippet;
    result?: Snippet;
  }

  let { view, frame, live, status, whose, canPassTurn, log, onact, onpassturn, onstack, onpreviewstack, notices, result }: Props = $props();

  const context = $derived(frame.projection.decision_context);
  const title = $derived(frame.prompt?.title ?? '');
  const instruction = $derived(context?.instruction ?? frame.prompt?.instruction ?? '');
  const secondary = $derived(view.buttons.filter(placed => placed !== view.primary));
  let logEnd = $state<HTMLElement | null>(null);
  $effect(() => {
    void log.length;
    logEnd?.scrollIntoView({ block: 'end' });
  });

  function anchorText(placed: PlacedOffer): string {
    const anchor = placed.anchor;
    if (anchor.kind === 'player') return 'player';
    if (anchor.kind === 'tray') return 'tray';
    return '';
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
              data-testid="stack-entry"
              aria-label={`${item.entry.label}, ${item.entry.controller === frame.projection.agent.player_index ? 'yours' : "opponent's"}${item.offers.length ? ` — ${item.offers.map(placed => placed.offer.label).join('; ')}` : ''}`}
              onclick={event => onstack(item, event.currentTarget)}
              onmouseenter={event => onpreviewstack(item, event.currentTarget)}
              onmouseleave={() => onpreviewstack(null, null)}
              onfocus={event => onpreviewstack(item, event.currentTarget)}
              onblur={() => onpreviewstack(null, null)}>
              <span class="who" class:mine={item.entry.controller === frame.projection.agent.player_index}></span>
              <span class="truncate">{item.entry.label}</span>
              {#if item.entry.kind !== 'spell'}<span class="type-caption text-ink-3">{item.entry.kind}</span>{/if}
            </button>
          </li>
        {/each}
      </ol>
    {/if}
  </section>

  <section class="block prompt" aria-label="Decision" data-testid="action-panel" data-action-space-kind={frame.action_space}>
    <p class="type-rubric whose" class:yours={live}>{whose}</p>
    {#if result}
      {@render result()}
    {:else}
      {#if title}<h2 class="type-title text-ink">{title}</h2>{/if}
      {#if instruction}<!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard scrolling of a long instruction) -->
        <p tabindex="0" class="type-annotation text-ink-2 instruction" data-testid="decision-prompt" data-kind={frame.action_space}>{instruction}</p>{/if}
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
    <p class="type-caption text-ink-3 status" class:sr-only={live} data-testid="current-decision" role="status" aria-live="polite" aria-atomic="true">{status}</p>
  </section>

  <section class="block choices" aria-label="All legal choices">
    <h2 class="heading type-rubric text-ink-2">
      <span class="type-rubric">All legal choices</span>
      <span class="type-rubric" data-testid="legal-choice-count" aria-label={`${frame.offers.length} legal choices`}>{frame.offers.length}</span>
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

  <section class="block log" aria-label="Log">
    <h2><span class="type-rubric text-ink-2">Log</span></h2>
    <ol class="entries" data-testid="game-log">
      {#each log.slice(-60) as entry (entry.id)}
        <li class="type-caption" class:hero={entry.actor === 'hero'} class:system={entry.actor === 'system'}>{entry.text}</li>
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
  .stack ol { display: grid; gap: 3px; max-height: 160px; overflow-y: auto; }
  .stack-entry {
    width: 100%; display: flex; align-items: center; gap: 8px; text-align: left; padding: 3px 6px; border-radius: 5px;
    border: 1px solid var(--border); background: var(--bg-field); font-size: 13px; color: var(--text);
  }
  .stack-entry.top { border-color: var(--border-strong); }
  .stack-entry.legal { box-shadow: 0 0 0 2px var(--legal); }
  .stack-entry .who { width: 6px; height: 6px; border-radius: 3px; background: var(--neutral); flex: none; }
  .stack-entry .who.mine { background: var(--info); }
  .prompt { display: grid; gap: 6px; }
  .whose { color: var(--text-tertiary); }
  .whose.yours { color: var(--error-text); }
  .instruction { max-height: 5.2em; overflow-y: auto; }
  .primary { height: 40px; width: 100%; font-size: 14px; justify-content: center; }
  .primary:disabled { background: var(--bg-muted); color: var(--text-secondary); opacity: 1; }
  .secondary { width: 100%; justify-content: flex-start; }
  kbd {
    font-family: var(--font-mono); font-size: 10px; line-height: 14px; padding: 0 4px; border-radius: 3px;
    border: 1px solid currentColor; opacity: 0.75; min-width: 16px; text-align: center;
  }
  .pass-turn { justify-self: center; display: inline-flex; gap: 6px; align-items: center; color: var(--text-secondary); padding: 2px 6px; border-radius: 4px; }
  .pass-turn:hover { color: var(--text); background: var(--bg-field); }
  .status { min-height: 16px; }
  .choices { flex: 1 1 auto; min-height: 120px; display: flex; flex-direction: column; }
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
  .log { flex: 0 1 150px; min-height: 70px; display: flex; flex-direction: column; border-bottom: 0; }
  .entries { flex: 1; min-height: 0; overflow-y: auto; mask-image: linear-gradient(transparent, #000 18px); display: grid; align-content: start; gap: 1px; color: var(--text-secondary); }
  .entries .hero { color: var(--text); }
  .entries .system { color: var(--text-tertiary); }
</style>
