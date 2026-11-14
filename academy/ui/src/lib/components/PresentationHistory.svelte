<script lang="ts">
  import { tick, untrack } from 'svelte';
  import { revealWithinScrollRegions } from '$lib/scroll-regions';
  import { visibleCardLabel } from '$lib/decision-labels';
  import { presentationBeat, presentationSubjects } from '$lib/presentation';
  import type { PresentationPlayer } from '$lib/presentation.svelte';
  import type { Observation, ObjectRenderId } from '$lib/types';
  import PresentationStage from './PresentationStage.svelte';

  let { player, observation = null }: { player: PresentationPlayer; observation?: Observation | null } = $props();
  const PAGE_SIZE = 100;
  let region: HTMLOListElement | undefined = $state();
  let start = $state(0);
  let following = $state(true);
  let readSeq = $state(-1);
  let inspection: HTMLElement | undefined = $state();
  let inspectionButton: HTMLButtonElement | null = null;
  let selected = $state<ObjectRenderId | null>(null);
  let knownScope = '';
  const total = $derived(player.events.length);
  const latestStart = $derived(Math.max(0, total - PAGE_SIZE));
  const unread = $derived(player.events.filter(event => event.seq > readSeq).length);
  const rows = $derived(player.events.slice(start, start + PAGE_SIZE).map((event, offset) => ({
    event,
    beat: presentationBeat(event, player.labelsAt(start + offset)),
  })));
  const precedingTurn = $derived(player.events.slice(0, start).findLast(event => event.kind.kind === 'turn_started'));
  const inspected = $derived(selected ? available(selected) : null);

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

  $effect(() => {
    const scope = player.scope;
    const length = player.events.length;
    untrack(() => {
      if (scope !== knownScope) {
        knownScope = scope;
        following = true;
        selected = null;
      }
      // A backward seek replaces the prefix instead of counting removed rows unread.
      if (start >= length || readSeq > (player.events.at(-1)?.seq ?? -1)) following = true;
      if (following) {
        start = Math.max(0, length - PAGE_SIZE);
        readSeq = player.events.at(-1)?.seq ?? -1;
        void tick().then(() => { if (region) region.scrollTop = region.scrollHeight; });
      }
    });
  });

  async function inspect(id: ObjectRenderId, button: HTMLButtonElement): Promise<void> {
    inspectionButton = button;
    selected = id;
    await tick();
    inspection?.focus({ preventScroll: true });
    if (inspection) revealWithinScrollRegions(inspection);
  }
  async function closeInspection(): Promise<void> {
    selected = null;
    await tick();
    const target = inspectionButton?.isConnected ? inspectionButton : region;
    target?.focus({ preventScroll: true });
    if (target) revealWithinScrollRegions(target);
  }

  function onScroll(): void {
    if (!region) return;
    following = start === latestStart && region.scrollHeight - region.clientHeight - region.scrollTop <= 32;
    if (following) readSeq = player.events.at(-1)?.seq ?? -1;
  }
  function pageBy(delta: number): void {
    following = false;
    start = Math.max(0, Math.min(latestStart, start + delta * PAGE_SIZE));
    if (region) region.scrollTop = 0;
  }
  async function latest(): Promise<void> {
    start = latestStart;
    following = true;
    readSeq = player.events.at(-1)?.seq ?? -1;
    await tick();
    if (region) region.scrollTop = region.scrollHeight;
  }
</script>

<section aria-label="Consequences" class="my-3 border-y border-line py-2" data-testid="presentation-history" data-scope={player.scope}>
  <details>
    <summary class="type-rubric cursor-pointer text-ink-2">Consequences ({total} available)</summary>
    <p class="type-caption mt-1 text-ink-2">Choices, Auto passes and observed-state summaries are in the decision log. Event attribution is unavailable.</p>
    {#if player.incomplete}<p role="status" class="type-caption text-mountain-ink">History incomplete · presentation error.</p>{/if}
    {#if player.unavailablePrefix}<p class="type-caption text-ink-2">Earlier history unavailable.</p>{/if}
    {#if player.hasGaps}<p class="type-caption text-ink-2">History has unavailable events between recorded sequences.</p>{/if}
    {#if total > 0}
      <nav aria-label="Consequence pages" class="my-2 flex flex-wrap items-center gap-2">
        <button class="btn btn-secondary btn-sm" disabled={start === 0} onclick={() => pageBy(-1)}>Earlier events</button>
        <span class="type-caption">{start + 1}–{Math.min(start + PAGE_SIZE, total)} of {total}</span>
        <button class="btn btn-secondary btn-sm" disabled={start >= latestStart} onclick={() => pageBy(1)}>Later events</button>
        <button class="btn btn-secondary btn-sm" onclick={() => void latest()}>Latest{unread ? ` (${unread} unread)` : ''}</button>
      </nav>
      <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard scrolling) -->
      <ol bind:this={region} onscroll={onScroll} class="mt-2 max-h-40 overflow-y-auto pl-5 text-sm text-ink" tabindex="0" aria-label="Recorded game events">
        {#each rows as row, index (row.event.seq)}
          <li data-presentation-seq={row.event.seq} class="py-1">
            {#if row.event.kind.kind === 'turn_started'}
              <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard navigation to recorded turn headings) -->
              <h3 tabindex="0" class="type-rubric border-t border-line pt-2">Turn {row.event.kind.turn_number}</h3>
            {:else if index === 0}
              <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard navigation to recorded turn headings) -->
              <h3 tabindex="0" class="type-rubric">{precedingTurn?.kind.kind === 'turn_started' ? `Turn ${precedingTurn.kind.turn_number} · continued` : 'Earlier turn unavailable'}</h3>
            {/if}
            <p>{row.beat.detail}</p>
            {#each presentationSubjects(row.event).filter(subject => subject.kind === 'object') as subject}
              {#if subject.kind === 'object' && available(subject.id)}
                <button class="btn btn-secondary btn-sm my-1 h-auto min-h-8 max-w-full justify-start whitespace-normal py-1 text-left break-words" onclick={(event) => void inspect(subject.id, event.currentTarget)}>Inspect {observation ? visibleCardLabel(observation, subject.id.entity) : ''}</button>
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
  <PresentationStage {player} />
</section>
