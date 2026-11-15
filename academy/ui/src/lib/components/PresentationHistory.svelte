<script lang="ts">
  import { tick, untrack } from 'svelte';
  import { revealWithinScrollRegions } from '$lib/scroll-regions';
  import { visibleCardLabel } from '$lib/decision-labels';
  import { presentationBeat, presentationSubjects } from '$lib/presentation';
  import { historyContext, precedingEvent } from '$lib/presentation-history';
  import type { PresentationPlayer } from '$lib/presentation.svelte';
  import type { Observation, ObjectRenderId } from '$lib/types';
  import PresentationStage from './PresentationStage.svelte';

  let { player, observation = null, externallyDriven = false }: { player: PresentationPlayer; observation?: Observation | null; externallyDriven?: boolean } = $props();
  const PAGE_SIZE = 100;
  let region: HTMLOListElement | undefined = $state();
  let summary: HTMLElement | undefined = $state();
  let pageSeq = $state<number | null>(null);
  let following = $state(true);
  let read = $state<Set<number>>(new Set());
  let inspectionSeq: number | null = null;
  let inspection: HTMLElement | undefined = $state();
  let inspectionButton: HTMLButtonElement | null = null;
  let selected = $state<ObjectRenderId | null>(null);
  let knownScope = '';
  let readingAnchor: { seq: number; offset: number } | null = null;
  const total = $derived(player.events.length);
  const latestStart = $derived(Math.max(0, total - PAGE_SIZE));
  const start = $derived(pageSeq === null ? latestStart : Math.max(0, player.events.findIndex(event => event.seq === pageSeq)));
  const unread = $derived(player.events.filter(event => !read.has(event.seq)).length);
  const context = $derived(historyContext(player.events));
  const rows = $derived(player.events.slice(start, start + PAGE_SIZE).map((event, offset) => ({
    event,
    ...context[start + offset],
    beat: presentationBeat(event, player.labelsAt(start + offset)),
  })));
  const inspected = $derived(selected ? available(selected) : null);

  function visibleAnchor(): { seq: number; offset: number } | null {
    if (!region) return null;
    const top = region.getBoundingClientRect().top;
    const row = Array.from(region.querySelectorAll<HTMLElement>('[data-presentation-seq]'))
      .find(row => row.getBoundingClientRect().bottom > top);
    return row ? { seq: Number(row.dataset.presentationSeq), offset: row.getBoundingClientRect().top - top } : null;
  }

  function available(id: ObjectRenderId) {
    // The current legacy projection certifies incarnation zero only. Never
    // substitute by name or use a catalog to resolve unavailable subjects.
    if (!observation || id.incarnation !== 0) return null;
    for (const player of [observation.agent, observation.opponent]) {
      const card = [...player.hand, ...player.battlefield, ...player.graveyard, ...player.exile, ...player.stack, ...(player.inspected ?? [])].find(card => card.id === id.entity);
      if (card?.name) return card;
    }
    return null;
  }

  // Capture the old DOM before Svelte reconciles a recovered/selected prefix.
  // Array indices are derived; only a sequence within this scope identifies a fact.
  $effect.pre(() => {
    const scope = player.scope;
    observation; // Permission changes can invalidate an open inspection.
    const seqs = player.events.map(event => event.seq);
    untrack(() => {
      const changedScope = scope !== knownScope;
      const focused = region?.contains(document.activeElement) ? document.activeElement as HTMLElement : null;
      const anchor = visibleAnchor();
      // A parent seek can reconcile rows before this effect. Prefer the current
      // DOM for pending scroll events, unless the saved reader has disappeared.
      const saved = readingAnchor && !region?.querySelector(`[data-presentation-seq="${readingAnchor.seq}"]`) ? readingAnchor : null;
      const anchorSeq = saved?.seq ?? anchor?.seq ?? readingAnchor?.seq ?? pageSeq;
      const offset = saved?.offset ?? anchor?.offset ?? 0;
      const targetSeq = anchorSeq === null ? null : precedingEvent(seqs, anchorSeq);
      const removedAnchor = anchorSeq !== null && !seqs.includes(anchorSeq);
      const inspectionRemoved = selected !== null && (changedScope || !seqs.includes(inspectionSeq ?? -1) || !available(selected));
      const restoreInspectionFocus = inspectionRemoved && (!!inspection?.contains(document.activeElement) || document.activeElement === document.body);
      if (changedScope) {
        knownScope = scope;
        following = true;
        readingAnchor = null;
      }
      if (inspectionRemoved) {
        selected = null;
        inspectionSeq = null;
        inspectionButton = null;
      }
      if (following) {
        pageSeq = null;
        read = new Set(seqs);
      } else {
        const pageIndex = pageSeq === null ? -1 : seqs.indexOf(pageSeq);
        const anchorIndex = targetSeq === null ? -1 : seqs.indexOf(targetSeq);
        if (removedAnchor || pageIndex < 0 || anchorIndex >= pageIndex + PAGE_SIZE) {
          pageSeq = targetSeq;
        }
        readingAnchor = targetSeq === null ? null : { seq: targetSeq, offset: removedAnchor ? 0 : offset };
      }
      void tick().then(() => {
        if (!region?.isConnected) {
          if (restoreInspectionFocus || focused) summary?.focus({ preventScroll: true });
          return;
        }
        if (following) region.scrollTop = region.scrollHeight;
        else if (targetSeq !== null) {
          const row = region.querySelector<HTMLElement>(`[data-presentation-seq="${targetSeq}"]`);
          if (row) region.scrollTop += row.getBoundingClientRect().top - region.getBoundingClientRect().top - (removedAnchor ? 0 : offset);
        }
        if (restoreInspectionFocus || (focused && !focused.isConnected)) region.focus({ preventScroll: true });
      });
    });
  });

  async function inspect(id: ObjectRenderId, seq: number, button: HTMLButtonElement): Promise<void> {
    inspectionButton = button;
    inspectionSeq = seq;
    selected = id;
    await tick();
    inspection?.focus({ preventScroll: true });
    if (inspection) revealWithinScrollRegions(inspection);
  }
  async function closeInspection(): Promise<void> {
    selected = null;
    inspectionSeq = null;
    await tick();
    const target = inspectionButton?.isConnected ? inspectionButton : region;
    target?.focus({ preventScroll: true });
    if (target) revealWithinScrollRegions(target);
  }

  function onScroll(): void {
    if (!region) return;
    following = start === latestStart && region.scrollHeight - region.clientHeight - region.scrollTop <= 32;
    readingAnchor = following ? null : visibleAnchor();
    if (!following && pageSeq === null) pageSeq = player.events[start]?.seq ?? null;
    if (following) {
      pageSeq = null;
      read = new Set(player.events.map(event => event.seq));
    }
  }
  function pageBy(delta: number): void {
    following = false;
    readingAnchor = null;
    pageSeq = player.events[Math.max(0, Math.min(latestStart, start + delta * PAGE_SIZE))]?.seq ?? null;
    if (region) region.scrollTop = 0;
  }
  async function latest(): Promise<void> {
    pageSeq = null;
    readingAnchor = null;
    following = true;
    read = new Set(player.events.map(event => event.seq));
    await tick();
    if (region) region.scrollTop = region.scrollHeight;
  }
</script>

<section aria-label="Consequences" class="my-3 border-y border-line py-2" data-testid="presentation-history" data-scope={player.scope}>
  <details>
    <summary bind:this={summary} class="type-rubric cursor-pointer text-ink-2">Consequences ({total} available)</summary>
    <p class="type-caption mt-1 text-ink-2">Choices, Auto passes and observed-state summaries are in the decision log. Event attribution is unavailable.</p>
    {#if player.incomplete}<p role="status" class="type-caption text-mountain-ink">History incomplete · presentation error.</p>{/if}
    {#if player.unavailablePrefix}<p class="type-caption text-ink-2">Earlier history unavailable.</p>{/if}
    {#if total > 0}
      <nav aria-label="Consequence pages" class="my-2 flex flex-wrap items-center gap-2">
        <button class="btn btn-secondary btn-sm" disabled={start === 0} onclick={() => pageBy(-1)}>Earlier events</button>
        <span class="type-caption">{start + 1}–{Math.min(start + PAGE_SIZE, total)} of {total}</span>
        <button class="btn btn-secondary btn-sm" disabled={start >= latestStart} onclick={() => pageBy(1)}>Later events</button>
        <button class="btn btn-secondary btn-sm" onclick={() => void latest()}>Latest{unread ? ` (${unread} unread)` : ''}</button>
      </nav>
      <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard scrolling) -->
      <ol bind:this={region} onscroll={onScroll} class="mt-2 max-h-40 overflow-y-auto pl-5 text-sm text-ink" style="overflow-anchor: none" tabindex="0" aria-label="Recorded game events">
        {#each rows as row, index (row.event.seq)}
          {#if row.gap}
            <li class="type-caption border-t border-line py-2 text-ink-2" data-history-gap>Events {row.gap.first}–{row.gap.last} unavailable · turn context unknown.</li>
          {/if}
          <li data-presentation-seq={row.event.seq} class="py-1">
            {#if row.event.kind.kind === 'turn_started'}
              <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard navigation to recorded turn headings) -->
              <h3 tabindex="0" class="type-rubric border-t border-line pt-2">Turn {row.event.kind.turn_number}</h3>
            {:else if index === 0 || row.gap}
              <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard navigation to recorded turn headings) -->
              <h3 tabindex="0" class="type-rubric">{row.turn === null ? 'Turn unknown' : `Turn ${row.turn} · continued`}</h3>
            {/if}
            <p>{row.beat.detail}</p>
            {#each presentationSubjects(row.event).filter(subject => subject.kind === 'object') as subject}
              {#if subject.kind === 'object' && available(subject.id)}
                <button class="btn btn-secondary btn-sm my-1 h-auto min-h-8 max-w-full justify-start whitespace-normal py-1 text-left break-words" onclick={(event) => void inspect(subject.id, row.event.seq, event.currentTarget)}>Inspect {observation ? visibleCardLabel(observation, subject.id.entity) : ''}</button>
              {/if}
            {/each}
          </li>
        {/each}
      </ol>
      {#if inspected}
        <section bind:this={inspection} tabindex="-1" aria-label="History card inspection" class="mt-2 border-t border-line py-2">
          <h3 class="type-label">{inspected.name}</h3>
          <p class="type-caption whitespace-pre-line">{inspected.text_box ?? 'Rules text unavailable in this view.'}</p>
          <button class="btn btn-secondary btn-sm" onclick={() => void closeInspection()}>Close history inspection</button>
        </section>
      {/if}
    {:else}
      <p class="type-caption mt-1 text-ink-2">Consequences not recorded for this position.</p>
    {/if}
  </details>
  <PresentationStage {player} {externallyDriven} />
</section>
