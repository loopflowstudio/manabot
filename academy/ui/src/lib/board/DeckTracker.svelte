<script lang="ts">
  /**
   * Untapped.gg-inspired in-game deck tracker panel for either seat.
   * Renders only viewer-safe counts (`main_counts` / `unseen_counts` / seen cards).
   */
  import type { DeckTrackerView, TrackerCardFace, TrackerRow } from './deck-tracker';

  interface Props {
    title: string;
    view: DeckTrackerView;
    faces: Readonly<Record<string, TrackerCardFace>>;
    side: 'you' | 'opponent';
    onclose: () => void;
  }

  let { title, view, faces, side, onclose }: Props = $props();
  let inspectedName = $state<string | null>(null);
  const inspectedFace = $derived(inspectedName ? faces[inspectedName] ?? null : null);

  function rowCount(row: TrackerRow): string {
    return view.mode === 'opponent-closed' ? `${row.seen}×` : String(row.remaining);
  }

  function rowRight(row: TrackerRow): string {
    if (view.mode === 'opponent-closed') return 'seen';
    return `${row.oddsPercent}%`;
  }
</script>

<aside
  class="tracker"
  class:left={side === 'you'}
  class:right={side === 'opponent'}
  aria-label={`${title} deck tracker`}
  data-testid={`deck-tracker-${side}`}
>
  <header class="head">
    <div class="head-top">
      <strong class="type-heading text-ink">{title}</strong>
      <span class="pill type-caption">
        {#if view.mode === 'self'}{view.totalRemaining} left
        {:else if view.mode === 'opponent-open'}{view.totalRemaining} unseen
        {:else}{view.totalSeen} seen{/if}
      </span>
      <button type="button" class="close type-caption" aria-label={`Close ${title} deck tracker`} onclick={onclose}>×</button>
    </div>
    <div class="odds type-caption text-ink-2">
      {#if view.mode === 'self'}
        <span>Lands {view.landsRemaining} ({view.landPercent}%)</span>
        <span>Spells {view.spellsRemaining} ({view.spellPercent}%)</span>
      {:else if view.mode === 'opponent-open'}
        <span>Hand {view.handCount} · Lib {Math.max(0, view.totalRemaining - view.handCount)}</span>
        <span>P(in hand)</span>
      {:else}
        <span>Closed list · visible cards</span>
        <span>×seen</span>
      {/if}
    </div>
  </header>

  <div class="list">
    {#if view.rows.length === 0}
      <p class="empty type-caption text-ink-3">No revealed cards yet.</p>
    {:else}
      {#each view.rows as row (row.name)}
        <button
          type="button"
          class="row"
          class:zero={view.mode !== 'opponent-closed' && row.remaining === 0}
          class:active={inspectedName === row.name}
          style:--accent-bar={row.accentColor}
          onclick={() => { inspectedName = inspectedName === row.name ? null : row.name; }}
          data-testid="tracker-row"
        >
          <span class="cnt">{rowCount(row)}</span>
          <span class="nm">{row.name}</span>
          <span class="cost">{row.cost}</span>
          <span class="pct">{rowRight(row)}</span>
        </button>
      {/each}
    {/if}

    {#if view.sideboard.length > 0}
      <p class="sec type-label text-ink-3">Sideboard / Lessons</p>
      {#each view.sideboard as row (row.name)}
        <button
          type="button"
          class="row"
          class:zero={row.remaining === 0}
          class:active={inspectedName === row.name}
          style:--accent-bar={row.accentColor}
          onclick={() => { inspectedName = inspectedName === row.name ? null : row.name; }}
          data-testid="tracker-sideboard-row"
        >
          <span class="cnt">{row.remaining}</span>
          <span class="nm">{row.name}</span>
          <span class="cost">{row.cost}</span>
          <span class="pct">{row.remaining}/{row.starting}</span>
        </button>
      {/each}
    {/if}
  </div>

  {#if inspectedName}
    <footer class="inspect" data-testid="tracker-inspect">
      <div class="inspect-top">
        <strong class="nm">{inspectedName}</strong>
        {#if inspectedFace?.cost}<span class="cost">{inspectedFace.cost.replaceAll('{', '').replaceAll('}', '')}</span>{/if}
      </div>
      {#if inspectedFace}
        <p class="type-caption text-ink-2">{inspectedFace.type_line}{#if inspectedFace.power !== null && inspectedFace.toughness !== null}{' · '}{inspectedFace.power}/{inspectedFace.toughness}{/if}</p>
        {#if inspectedFace.keywords.length > 0}
          <p class="type-caption text-ink-3">{inspectedFace.keywords.join(', ')}</p>
        {/if}
      {/if}
    </footer>
  {/if}
</aside>

<style>
  .tracker {
    position: fixed;
    top: 52px;
    bottom: 58px;
    z-index: 180;
    width: 264px;
    max-width: calc(100vw - 24px);
    display: flex;
    flex-direction: column;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-md);
    background: var(--bg-surface);
    box-shadow: var(--shadow-raised);
    overflow: hidden;
  }
  .tracker.left { left: 10px; }
  .tracker.right { right: 10px; }
  .head {
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    display: grid;
    gap: 4px;
    flex-shrink: 0;
  }
  .head-top { display: flex; align-items: center; gap: 6px; }
  .head-top strong { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .pill {
    padding: 2px 6px;
    border-radius: var(--radius-full);
    background: var(--accent-soft);
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 16px;
  }
  .close {
    border: 0;
    background: transparent;
    color: var(--text-tertiary);
    cursor: pointer;
    padding: 0 4px;
    font-size: 16px;
    line-height: 1;
    min-height: 22px;
  }
  .close:hover { color: var(--text); }
  .odds {
    display: flex;
    justify-content: space-between;
    gap: 6px;
    font-size: 12px;
    line-height: 16px;
  }
  .list {
    padding: 6px;
    display: grid;
    gap: 3px;
    align-content: start;
    overflow-y: auto;
    flex: 1;
  }
  .empty { padding: 8px 4px; margin: 0; }
  .sec { margin: 8px 4px 2px; font-size: 12px; line-height: 16px; }
  .row {
    display: grid;
    grid-template-columns: 24px 1fr auto auto;
    gap: 6px;
    align-items: center;
    padding: 4px 6px;
    min-height: 28px;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-bar, var(--border-strong));
    border-radius: var(--radius-sm);
    background: var(--bg-field);
    color: var(--text);
    cursor: pointer;
    text-align: left;
  }
  .row:hover, .row.active { border-color: var(--accent); border-left-color: var(--accent-bar, var(--accent)); }
  .row.zero { opacity: 0.44; }
  .cnt {
    font-family: var(--font-sans);
    font-weight: 600;
    font-size: 12px;
    line-height: 16px;
    text-align: center;
    background: var(--accent-soft);
    border-radius: var(--radius-sm);
    padding: 1px 0;
  }
  .nm {
    font-family: var(--font-serif);
    font-size: 12px;
    line-height: 16px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .cost {
    font-family: var(--font-sans);
    font-size: 12px;
    line-height: 16px;
    color: var(--text-tertiary);
  }
  .pct {
    font-family: var(--font-sans);
    font-weight: 600;
    font-size: 12px;
    line-height: 16px;
    color: var(--text-secondary);
    min-width: 32px;
    text-align: right;
  }
  .inspect {
    padding: 8px 10px;
    border-top: 1px solid var(--border);
    background: var(--bg);
    display: grid;
    gap: 2px;
    flex-shrink: 0;
  }
  .inspect-top { display: flex; justify-content: space-between; align-items: baseline; gap: 6px; }
  .inspect p { margin: 0; }
</style>
