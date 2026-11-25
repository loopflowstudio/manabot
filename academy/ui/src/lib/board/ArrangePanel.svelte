<script lang="ts">
  /**
   * Arrangement panel for Scry, Surveil and OrderLibrary live groups.
   *
   * Driven solely by the authority's `ArrangeGroup` descriptor and preview
   * response; nothing commits until the user presses Confirm once.
   */
  import type { ArrangeGroup, GroupCard, GroupRefusal } from '$lib/group';
  import type { ObjectRenderId } from '$lib/types';
  import Thumb from './Thumb.svelte';
  import {
    arrangeHasChanges,
    arrangeWords,
    emptyArrangeDraft,
    orderMove,
    orderPlace,
    scryMove,
    scryPlace,
    surveilPlace,
    type ArrangeDraft,
  } from './arrange';

  interface Props {
    group: ArrangeGroup;
    draft: ArrangeDraft;
    /** Known card names by display id, for committed summaries and inspection. */
    cardNames: ReadonlyMap<number, string>;
    live: boolean;
    pending: boolean;
    readingOut: boolean;
    ready: boolean;
    refused: GroupRefusal | null;
    onchange: (next: ArrangeDraft) => void;
    onconfirm: () => void;
    onpreviewcard: (entityId: number | null, label: string | null, element: HTMLElement | null) => void;
  }

  let {
    group,
    draft,
    cardNames,
    live,
    pending,
    readingOut,
    ready,
    refused,
    onchange,
    onconfirm,
    onpreviewcard,
  }: Props = $props();

  const byId = $derived(new Map<number, GroupCard>(group.items.map(card => [card.item.entity, card])));
  const words = $derived(arrangeWords(draft, group));
  const disabled = $derived(!live || pending);
  const hasChanges = $derived(arrangeHasChanges(draft));

  const cardLabel = (id: ObjectRenderId): string =>
    byId.get(id.entity)?.label ?? cardNames.get(id.entity) ?? `Card #${id.entity}`;

  const unplaced = $derived.by((): GroupCard[] => {
    if (draft.kind === 'scry') {
      const placed = new Set([...draft.top, ...draft.bottom]);
      return group.items.filter(card => !placed.has(card.item.entity));
    }
    if (draft.kind === 'order_library') {
      const placed = new Set(draft.order);
      return group.items.filter(card => !placed.has(card.item.entity));
    }
    return [];
  });

  const surveilKept = $derived.by((): GroupCard[] => {
    if (draft.kind !== 'surveil') return [];
    const gy = new Set(draft.graveyard);
    return group.items.filter(card => !gy.has(card.item.entity));
  });

  const committedSummary = $derived.by((): { heading: string; parts: string[] } | null => {
    if (group.kind === 'scry') {
      const parts: string[] = [];
      if (group.committed_top.length) parts.push(`Top: ${group.committed_top.map(cardLabel).join(', ')}`);
      if (group.committed_bottom.length) parts.push(`Bottom: ${group.committed_bottom.map(cardLabel).join(', ')}`);
      return parts.length ? { heading: 'Already placed', parts } : null;
    }
    if (group.kind === 'surveil') {
      return group.committed_graveyard.length
        ? { heading: 'Already in graveyard', parts: [group.committed_graveyard.map(cardLabel).join(', ')] }
        : null;
    }
    return group.committed.length
      ? { heading: 'Already ordered', parts: [group.committed.map(cardLabel).join(' → ')] }
      : null;
  });

  function hoverCard(card: GroupCard) {
    return {
      onmouseenter: (event: MouseEvent) => onpreviewcard(card.item.entity, card.label, event.currentTarget as HTMLElement),
      onmouseleave: () => onpreviewcard(null, null, null),
      onfocus: (event: FocusEvent) => onpreviewcard(card.item.entity, card.label, event.currentTarget as HTMLElement),
      onblur: () => onpreviewcard(null, null, null),
    };
  }
</script>

{#snippet cardIdentity(card: GroupCard)}
  <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_static_element_interactions (focus previews the card) -->
  <span class="card-id" tabindex="0" {...hoverCard(card)}>
    <Thumb name={card.label} />
    <span class="card-name">{card.label}</span>
  </span>
{/snippet}

{#snippet orderedBucket(
  title: string,
  testId: string,
  emptyText: string,
  ids: readonly number[],
  options: {
    single?: boolean;
    onMove: (entityId: number, delta: -1 | 1) => void;
    upLabel: (label: string) => string;
    downLabel: (label: string) => string;
    swap?: { text: string; ariaLabel: (label: string) => string; onSwap: (entityId: number) => void };
    removeLabel: (label: string) => string;
    onRemove: (entityId: number) => void;
  },
)}
  <section class="bucket" class:single={options.single} aria-label={title} data-testid={testId}>
    <h3 class="type-label text-ink-2">{title}</h3>
    {#if ids.length === 0}
      <p class="type-caption text-ink-3 empty">{emptyText}</p>
    {:else}
      <ol class="card-list">
        {#each ids as entityId, index (entityId)}
          {@const card = byId.get(entityId)!}
          <li
            class="card-row"
            class:refused={refused?.item?.entity === entityId}
            data-testid="arrange-card"
            data-card-id={entityId}
          >
            <span class="rank type-caption">{index + 1}</span>
            {@render cardIdentity(card)}
            <span class="row-actions">
              <button
                type="button"
                class="step"
                disabled={disabled || index === 0}
                aria-label={options.upLabel(card.label)}
                onclick={() => options.onMove(entityId, -1)}
              >↑</button>
              <button
                type="button"
                class="step"
                disabled={disabled || index === ids.length - 1}
                aria-label={options.downLabel(card.label)}
                onclick={() => options.onMove(entityId, 1)}
              >↓</button>
              {#if options.swap}
                <button
                  type="button"
                  class="btn btn-ghost btn-xs"
                  disabled={disabled}
                  aria-label={options.swap.ariaLabel(card.label)}
                  onclick={() => options.swap!.onSwap(entityId)}
                >{options.swap.text}</button>
              {/if}
              <button
                type="button"
                class="step"
                disabled={disabled}
                aria-label={options.removeLabel(card.label)}
                onclick={() => options.onRemove(entityId)}
              >×</button>
            </span>
          </li>
        {/each}
      </ol>
    {/if}
  </section>
{/snippet}

<section
  class="arrange-panel"
  aria-label={words.title}
  data-testid="arrange-panel"
  data-group-kind={group.kind}
>
  <header class="arrange-head">
    <h2 class="type-heading text-ink">{words.title}</h2>
    <p class="type-caption text-ink-3">Nothing commits until you confirm.</p>
  </header>

  {#if committedSummary}
    <div class="committed-box" data-testid="arrange-committed">
      <p class="type-label text-ink-3">{committedSummary.heading}</p>
      {#each committedSummary.parts as line (line)}
        <p class="type-caption text-ink-2">{line}</p>
      {/each}
    </div>
  {/if}

  {#if unplaced.length > 0}
    <div class="unplaced" data-testid="arrange-unplaced">
      <p class="type-label text-ink-3">Unplaced · {unplaced.length}</p>
      <ul class="card-list">
        {#each unplaced as card (card.item.entity)}
          <li
            class="card-row"
            class:refused={refused?.item?.entity === card.item.entity}
            data-testid="arrange-card"
            data-card-id={card.item.entity}
          >
            {@render cardIdentity(card)}
            <span class="row-actions">
              {#if draft.kind === 'scry'}
                <button
                  type="button"
                  class="btn btn-secondary btn-xs"
                  disabled={disabled}
                  aria-disabled={readingOut || undefined}
                  aria-label={`Put ${card.label} on top`}
                  onclick={() => onchange(scryPlace(draft, card.item.entity, 'top'))}
                >Top</button>
                <button
                  type="button"
                  class="btn btn-secondary btn-xs"
                  disabled={disabled}
                  aria-disabled={readingOut || undefined}
                  aria-label={`Put ${card.label} on bottom`}
                  onclick={() => onchange(scryPlace(draft, card.item.entity, 'bottom'))}
                >Bottom</button>
              {:else if draft.kind === 'order_library'}
                <button
                  type="button"
                  class="btn btn-secondary btn-xs"
                  disabled={disabled}
                  aria-disabled={readingOut || undefined}
                  aria-label={`Add ${card.label} to order`}
                  onclick={() => onchange(orderPlace(draft, card.item.entity, true))}
                >Place next</button>
              {/if}
            </span>
          </li>
        {/each}
      </ul>
    </div>
  {/if}

  {#if draft.kind === 'scry'}
    <div class="buckets">
      {@render orderedBucket(
        'Top of library — drawn first at top',
        'arrange-bucket-top',
        'No cards on top',
        draft.top,
        {
          onMove: (entityId, delta) => onchange(scryMove(draft, 'top', entityId, delta)),
          upLabel: label => `Move ${label} up in top`,
          downLabel: label => `Move ${label} down in top`,
          swap: {
            text: 'Bottom',
            ariaLabel: label => `Move ${label} to bottom`,
            onSwap: entityId => onchange(scryPlace(draft, entityId, 'bottom')),
          },
          removeLabel: label => `Unplace ${label}`,
          onRemove: entityId => onchange(scryPlace(draft, entityId, 'unplaced')),
        },
      )}
      {@render orderedBucket(
        'Bottom of library — top-most at top',
        'arrange-bucket-bottom',
        'No cards on bottom',
        draft.bottom,
        {
          onMove: (entityId, delta) => onchange(scryMove(draft, 'bottom', entityId, delta)),
          upLabel: label => `Move ${label} up in bottom`,
          downLabel: label => `Move ${label} down in bottom`,
          swap: {
            text: 'Top',
            ariaLabel: label => `Move ${label} to top`,
            onSwap: entityId => onchange(scryPlace(draft, entityId, 'top')),
          },
          removeLabel: label => `Unplace ${label}`,
          onRemove: entityId => onchange(scryPlace(draft, entityId, 'unplaced')),
        },
      )}
    </div>
  {:else if draft.kind === 'surveil'}
    <div class="buckets">
      <section class="bucket" aria-label="Keep on top" data-testid="arrange-bucket-keep">
        <h3 class="type-label text-ink-2">Keep on top · {surveilKept.length}</h3>
        {#if surveilKept.length === 0}
          <p class="type-caption text-ink-3 empty">None kept on top</p>
        {:else}
          <ul class="card-list">
            {#each surveilKept as card (card.item.entity)}
              <li
                class="card-row"
                class:refused={refused?.item?.entity === card.item.entity}
                data-testid="arrange-card"
                data-card-id={card.item.entity}
              >
                {@render cardIdentity(card)}
                <span class="row-actions">
                  <button
                    type="button"
                    class="btn btn-secondary btn-xs"
                    disabled={disabled}
                    aria-label={`Put ${card.label} in graveyard`}
                    onclick={() => onchange(surveilPlace(draft, card.item.entity, 'graveyard'))}
                  >Graveyard</button>
                </span>
              </li>
            {/each}
          </ul>
        {/if}
      </section>

      <section class="bucket" aria-label="Graveyard" data-testid="arrange-bucket-graveyard">
        <h3 class="type-label text-ink-2">Graveyard · {draft.graveyard.length}</h3>
        {#if draft.graveyard.length === 0}
          <p class="type-caption text-ink-3 empty">None to graveyard</p>
        {:else}
          <ul class="card-list">
            {#each draft.graveyard as entityId (entityId)}
              {@const card = byId.get(entityId)!}
              <li
                class="card-row"
                class:refused={refused?.item?.entity === entityId}
                data-testid="arrange-card"
                data-card-id={entityId}
              >
                {@render cardIdentity(card)}
                <span class="row-actions">
                  <button
                    type="button"
                    class="btn btn-secondary btn-xs"
                    disabled={disabled}
                    aria-label={`Keep ${card.label} on top`}
                    onclick={() => onchange(surveilPlace(draft, entityId, 'keep'))}
                  >Keep on top</button>
                </span>
              </li>
            {/each}
          </ul>
        {/if}
      </section>
    </div>
  {:else if draft.kind === 'order_library'}
    {@render orderedBucket(
      words.bucketTitle ?? 'Order cards',
      'arrange-bucket-order',
      'Place each card above in the order you want',
      draft.order,
      {
        single: true,
        onMove: (entityId, delta) => onchange(orderMove(draft, entityId, delta)),
        upLabel: label => `Move ${label} up`,
        downLabel: label => `Move ${label} down`,
        removeLabel: label => `Remove ${label} from order`,
        onRemove: entityId => onchange(orderPlace(draft, entityId, false)),
      },
    )}
  {/if}

  {#if refused}
    <p class="refusal type-caption" role="alert" data-testid="arrange-refused">{refused.reason}</p>
  {/if}

  <footer class="arrange-actions">
    <button
      type="button"
      class="btn btn-primary confirm"
      data-testid="arrange-confirm"
      data-draft-ready={ready}
      disabled={disabled || !ready || refused !== null}
      aria-disabled={readingOut || undefined}
      onclick={onconfirm}
    >
      <span>{words.confirmLabel}</span>{#if ready && !disabled}<kbd>Space</kbd>{/if}
    </button>
    {#if hasChanges}
      <button
        type="button"
        class="btn btn-secondary btn-sm"
        data-testid="arrange-reset"
        disabled={disabled}
        aria-disabled={readingOut || undefined}
        onclick={() => onchange(emptyArrangeDraft(group))}
      >Reset</button>
    {/if}
  </footer>
</section>

<style>
  .arrange-panel {
    position: absolute;
    left: 50%;
    top: 46%;
    transform: translate(-50%, -50%);
    z-index: var(--layer-panel);
    width: min(640px, 92%);
    max-height: calc(100% - 32px);
    overflow-y: auto;
    display: grid;
    gap: 10px;
    padding: 16px 18px;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-md);
    background: var(--bg-surface);
    box-shadow: var(--shadow-raised);
  }
  .arrange-head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
  }
  .committed-box {
    padding: 6px 10px;
    border-left: 2px solid var(--border-strong);
    background: var(--bg-field);
    border-radius: var(--radius-sm);
    display: grid;
    gap: 2px;
  }
  .unplaced {
    display: grid;
    gap: 6px;
  }
  .buckets {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
  }
  .bucket {
    display: grid;
    align-content: start;
    gap: 6px;
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-field);
    min-height: 92px;
  }
  .empty {
    padding: 6px 0;
  }
  .card-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 6px;
  }
  .card-row {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    padding: 4px 6px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-surface);
  }
  .card-row.refused {
    border-color: var(--error);
    background: var(--error-wash);
  }
  .rank {
    flex: none;
    width: 16px;
    text-align: right;
    color: var(--text-tertiary);
    font-variant-numeric: tabular-nums;
  }
  .card-id {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: help;
  }
  .card-name {
    min-width: 0;
    font-family: var(--font-serif);
    font-weight: 600;
    font-size: 14px;
    line-height: 20px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .row-actions {
    flex: none;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .btn-xs {
    padding: 2px 7px;
    font-size: 12px;
    line-height: 16px;
    min-height: 24px;
  }
  .step {
    width: 24px;
    height: 24px;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    background: var(--bg-field);
    color: var(--text);
    font-size: 12px;
    line-height: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  .step:disabled {
    opacity: 0.4;
  }
  .refusal {
    padding: 6px 10px;
    border: 1px solid color-mix(in srgb, var(--error) 50%, transparent);
    border-radius: var(--radius-sm);
    background: var(--error-wash);
    color: var(--error-text);
  }
  .arrange-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .confirm {
    flex: 1;
    height: 38px;
    justify-content: center;
    gap: 8px;
  }
  .confirm:disabled {
    background: var(--bg-muted);
    color: var(--text-secondary);
    opacity: 1;
  }
</style>
