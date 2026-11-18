<script lang="ts">
  import { browser } from '$app/environment';

  import type { RestoredReplayDecision } from '$lib/replay-index';
  import type { TableSnapshot } from '$lib/shared-table-protocol';

  interface Props {
    table: TableSnapshot;
    restored: RestoredReplayDecision | null;
    branchAttemptId: string | null;
    announcement: string;
    onTransferPilot?: (viewerId: string) => void;
    onRestoreDecision?: (address: string) => void;
    onReturnFromBranch?: () => void;
    onReturnToLive?: () => void;
  }

  let {
    table,
    restored,
    branchAttemptId,
    announcement,
    onTransferPilot,
    onRestoreDecision,
    onReturnFromBranch,
    onReturnToLive,
  }: Props = $props();

  let copied = $state(false);
  const viewerId = $derived(table.access.identity.viewer_id);
  const otherParticipant = $derived(
    table.participants.find((participant) => participant.viewer_id !== viewerId) ?? null,
  );
  async function copyInvite(): Promise<void> {
    if (!browser || !table.watcher_invite) return;
    const link = `${window.location.origin}${window.location.pathname}${table.watcher_invite}`;
    await navigator.clipboard.writeText(link);
    copied = true;
  }
</script>

<section
  data-testid="shared-table-panel"
  data-table-id={table.table_id}
  data-table-mode={table.mode}
  aria-labelledby="shared-table-heading"
  class="border-b border-line bg-field px-5 py-4 md:px-10"
>
  <div class="flex flex-wrap items-start justify-between gap-4">
    <div>
      <p class="type-label text-mountain-ink">Shared table · {table.mode}</p>
      <h2 id="shared-table-heading" class="type-title mt-1 text-display">
        You are the {table.access.role}
      </h2>
      <p class="type-caption mt-1 text-ink-2">
        {table.participants.length === 2
          ? `Pilot and watcher share seat ${table.access.identity.rules_viewer + 1} facts · ${otherParticipant?.connected ? 'both connected' : 'one reconnecting'}`
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

  <div class="mt-4">
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
        <p class="type-body mt-3 text-ink-2">Decisions appear after the pilot commits them.</p>
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
    data-testid="shared-table-announcement"
    role="status"
    aria-live="polite"
    aria-atomic="true"
  >
    {announcement}
  </p>
</section>
