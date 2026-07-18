<script lang="ts">
  import { browser } from '$app/environment';

  import type { RestoredReplayDecision } from '$lib/replay-index';
  import type { TableSnapshot } from '$lib/testing-house-protocol';

  interface ScenarioSummary {
    landmark_id: string;
    label: string;
  }

  interface Props {
    table: TableSnapshot;
    scenarios: ScenarioSummary[];
    selectedScenarioId: string;
    restored: RestoredReplayDecision | null;
    branchAttemptId: string | null;
    announcement: string;
    onTransferPilot?: (viewerId: string) => void;
    onAuthorBelief?: (scenarioId: string) => void;
    onShareBelief?: (beliefId: string) => void;
    onSelectBelief?: (scenarioId: string) => void;
    onRestoreDecision?: (address: string) => void;
    onRetryDecision?: (offerId: number) => void;
    onReturnFromBranch?: () => void;
    onReturnToLive?: () => void;
  }

  let {
    table,
    scenarios,
    selectedScenarioId,
    restored,
    branchAttemptId,
    announcement,
    onTransferPilot,
    onAuthorBelief,
    onShareBelief,
    onSelectBelief,
    onRestoreDecision,
    onRetryDecision,
    onReturnFromBranch,
    onReturnToLive,
  }: Props = $props();

  let copied = $state(false);
  const viewerId = $derived(table.access.identity.viewer_id);
  const otherParticipant = $derived(
    table.participants.find((participant) => participant.viewer_id !== viewerId) ?? null,
  );
  const selectedScenario = $derived(
    scenarios.find((scenario) => scenario.landmark_id === selectedScenarioId) ?? null,
  );

  async function copyInvite(): Promise<void> {
    if (!browser || !table.watcher_invite) return;
    const link = `${window.location.origin}${window.location.pathname}${table.watcher_invite}`;
    await navigator.clipboard.writeText(link);
    copied = true;
  }
</script>

<section
  data-testid="testing-house-panel"
  data-table-id={table.table_id}
  data-table-mode={table.mode}
  aria-labelledby="testing-house-heading"
  class="border-b border-line bg-field px-5 py-4 md:px-10"
>
  <div class="flex flex-wrap items-start justify-between gap-4">
    <div>
      <p class="type-rubric text-mountain-ink">Testing house · {table.mode}</p>
      <h2 id="testing-house-heading" class="type-title mt-1 text-display">
        You are the {table.access.role}
      </h2>
      <p class="type-caption mt-1 text-ink-2">
        {table.participants.length === 2
          ? `Pilot and watcher share player 0 facts · ${otherParticipant?.connected ? 'both connected' : 'one reconnecting'}`
          : 'Waiting for the permitted watcher'}
      </p>
    </div>
    <div class="flex flex-wrap gap-2">
      {#if table.watcher_invite}
        <button
          type="button"
          data-testid="copy-watcher-invite"
          class="btn btn-secondary min-h-11"
          onclick={() => void copyInvite()}
        >
          {copied ? 'Invite copied' : 'Copy watcher link'}
        </button>
      {/if}
      {#if table.access.capabilities.includes('transfer_pilot') && otherParticipant}
        <button
          type="button"
          data-testid="transfer-pilot"
          class="btn btn-secondary min-h-11"
          onclick={() => onTransferPilot?.(otherParticipant.viewer_id)}
        >
          Make watcher pilot
        </button>
      {/if}
    </div>
  </div>

  <div class="mt-4 grid gap-5 lg:grid-cols-2">
    <div>
      <div class="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 class="type-label uppercase text-ink-3">Reads</h3>
          <p class="type-caption mt-1 text-ink-2">Personal until you explicitly share.</p>
        </div>
        <button
          type="button"
          data-testid="author-belief"
          class="btn btn-secondary min-h-11"
          disabled={!selectedScenario || !table.access.capabilities.includes('author_belief')}
          onclick={() => selectedScenario && onAuthorBelief?.(selectedScenario.landmark_id)}
        >
          Keep “{selectedScenario?.label ?? 'selected scenario'}” as my read
        </button>
      </div>
      {#if table.beliefs.length === 0}
        <p class="type-annotation mt-3 text-ink-2">No saved reads on this table.</p>
      {:else}
        <ul class="mt-3 grid gap-2" aria-label="Visible reads">
          {#each table.beliefs as belief}
            <li class="flex min-h-11 flex-wrap items-center justify-between gap-2 border-l-2 border-line-strong pl-3">
              <div>
                <button
                  type="button"
                  class="min-h-11 font-medium text-ink underline decoration-line-strong underline-offset-4 hover:text-action"
                  aria-label={`Compare advice for ${scenarios.find(({ landmark_id }) => landmark_id === belief.source.gam6_scenario_id)?.label ?? belief.source.gam6_scenario_id}`}
                  onclick={() => onSelectBelief?.(belief.source.gam6_scenario_id)}
                >
                  {scenarios.find(({ landmark_id }) => landmark_id === belief.source.gam6_scenario_id)?.label ?? belief.source.gam6_scenario_id}
                </button>
                <span class="type-caption ml-2 text-ink-2">
                  {belief.author_viewer_id === viewerId ? 'Your read' : 'Shared read'} · {belief.audience.kind}
                </span>
              </div>
              {#if belief.author_viewer_id === viewerId && belief.audience.kind === 'personal'}
                <button
                  type="button"
                  data-testid="share-belief"
                  class="btn btn-secondary min-h-11"
                  onclick={() => onShareBelief?.(belief.id)}
                >
                  Share with table
                </button>
              {/if}
            </li>
          {/each}
        </ul>
      {/if}
    </div>

    <div>
      <h3 class="type-label uppercase text-ink-3">Recorded decisions</h3>
      <p class="type-caption mt-1 text-ink-2">Explore privately; the live match keeps moving.</p>
      {#if table.decisions.length}
        <div class="mt-3 flex flex-wrap gap-2" aria-label="Committed decisions">
          {#each table.decisions as decision}
              <button
                type="button"
                data-testid="restore-table-decision"
                data-decision-address={decision.address}
                data-decision-revision={decision.revision}
                class="btn btn-secondary min-h-11"
              onclick={() => onRestoreDecision?.(decision.address)}
            >
              Decision {decision.ordinal + 1}
            </button>
          {/each}
        </div>
      {:else}
        <p class="type-annotation mt-3 text-ink-2">Decisions appear after the pilot commits them.</p>
      {/if}

      {#if restored}
        <div class="mt-3 border-l-2 border-action pl-3" data-testid="participant-study-controls">
          <p class="font-medium text-ink">Decision {restored.ordinal + 1} · isolated Study</p>
          {#if branchAttemptId}
            <p class="type-caption mt-1 text-ink-2">Branch board shown only to you. Live table truth is unchanged.</p>
            <button
              type="button"
              data-testid="return-from-participant-branch"
              class="btn btn-secondary mt-2 min-h-11"
              onclick={() => onReturnFromBranch?.()}
            >
              Return to recorded decision
            </button>
          {:else}
            <div class="mt-2 grid gap-2 sm:grid-cols-2" aria-label="Try an isolated line">
              {#each restored.frame.offers as offer}
                <button
                  type="button"
                  data-testid="retry-table-decision"
                  class="min-h-11 rounded border border-line-strong bg-panel px-3 py-2 text-left font-medium text-ink hover:border-action"
                  onclick={() => onRetryDecision?.(offer.id)}
                >
                  Try {offer.label}
                </button>
              {/each}
            </div>
          {/if}
          <button
            type="button"
            data-testid="return-to-live-table"
            class="btn btn-secondary mt-2 min-h-11"
            onclick={() => onReturnToLive?.()}
          >
            Live table
          </button>
        </div>
      {/if}
    </div>
  </div>

  <p
    class="sr-only"
    data-testid="testing-house-announcement"
    role="status"
    aria-live="polite"
    aria-atomic="true"
  >
    {announcement}
  </p>
</section>
