<script lang="ts">
  import { tick, untrack } from 'svelte';
  import { gameStore } from '$lib/game.svelte';
  import { recoverCurrentGame, sendDeclaration, validateDeclaration } from '$lib/socket.svelte';
  import { revealWithinScrollRegions } from '$lib/scroll-regions';
  import {
    assignmentFor, bulkAttackDestination, clearCombatDraft, declarationPlan,
    destinationKey, objectKey, selectAllAttackers, setAssignment, type CombatDraft,
  } from '$lib/combat-plan';
  import type { DeclarationProjection, ExperienceFrame, SubjectRef } from '$lib/types';
  import type { DeclarationValidationEvent } from '$lib/shared-table-protocol';

  interface Props {
    frame: ExperienceFrame;
    projection: DeclarationProjection;
    draft: CombatDraft;
    disabled?: boolean;
    readOnly?: boolean;
    inspect?: boolean;
    onChange: (draft: CombatDraft) => void;
  }
  let { frame, projection, draft, disabled = false, readOnly = false, inspect = false, onChange }: Props = $props();
  type Status = DeclarationValidationEvent['status'] | 'pending' | 'transport';
  interface Validation { key: string; signature: string; request: string; status: Status; message: string }
  let validation = $state<Validation | null>(null);
  let blocked = $state<{ signature: string; status: 'invalid' | 'capacity'; message: string } | null>(null);
  let submitted = $state<{ key: string; id: string; signature: string } | null>(null);
  let retry = $state(0);
  let assignmentsRegion: HTMLDivElement | null = $state(null);
  const signature = $derived(JSON.stringify(draft.assignments));
  const draftKey = $derived(draft.key);
  const activeKey = $derived(draft.active);
  const canEdit = $derived(!disabled && !readOnly && !inspect);
  const current = $derived(validation?.key === draftKey && validation.signature === signature ? validation : null);
  const available = $derived(projection.subjects.every(row => row.object.subject.kind === 'object'));
  const bulk = $derived(bulkAttackDestination(projection));
  const editable = $derived(projection.subjects.filter(row => !row.fixed).length);
  const fixed = $derived(projection.subjects.filter(row => row.fixed).length);
  const planned = $derived(draft.assignments.filter(answer => answer.destination !== null).length);
  const power = $derived(projection.subjects.reduce((total, row) => {
    const selected = row.fixed ? row.assigned_to != null : row.object.subject.kind === 'object' && assignmentFor(draft, row.object.subject.id) !== null;
    return total + (selected ? row.power : 0);
  }, 0));
  const canConfirm = $derived(canEdit && available && current?.status === 'valid');

  function destinationLabel(destination: SubjectRef | null | undefined): string {
    if (!destination) return 'Unassigned';
    const view = projection.destinations.find(row => destinationKey(row.subject) === destinationKey(destination));
    if (destination.kind === 'player') return `${destination.id === frame.projection.agent.player_index ? 'You' : 'Opponent'} (${view?.label ?? 'player'})`;
    if (!view) return 'Unavailable destination';
    const same = projection.destinations.filter(row => row.label === view.label);
    return same.length > 1 ? `${view.label} · creature ${same.indexOf(view) + 1}` : view.label;
  }

  function rowLabel(index: number): string {
    const row = projection.subjects[index];
    const same = projection.subjects.filter(candidate => candidate.object.label === row.object.label);
    return same.length > 1 ? `${row.object.label} · creature ${same.indexOf(row) + 1}` : row.object.label;
  }

  $effect(() => {
    draftKey;
    untrack(() => { validation = null; blocked = null; submitted = null; });
  });

  $effect(() => {
    const rejection = gameStore.lastCommandRejection;
    if (!submitted || !rejection || rejection.command_id !== submitted.id) return;
    if (submitted.key === draft.key && submitted.signature === signature) {
      blocked = { signature, status: rejection.code === 'capacity' ? 'capacity' : 'invalid', message: rejection.message };
    }
    submitted = null;
  });

  $effect(() => {
    const key = draftKey;
    const boundSignature = signature;
    retry;
    if (!canEdit || !available) return;
    if (blocked?.signature === boundSignature) {
      validation = { key, signature: boundSignature, request: '', status: blocked.status, message: blocked.message };
      return;
    }
    const request = crypto.randomUUID();
    const plan = untrack(() => declarationPlan(draft, projection));
    validation = { key, signature: boundSignature, request, status: 'pending', message: 'Checking the whole declaration…' };
    const timer = setTimeout(() => {
      if (!validateDeclaration(plan, request)) validation = { key, signature: boundSignature, request, status: 'transport', message: 'Validation is unavailable. Recover the connection or retry.' };
    }, 150);
    return () => clearTimeout(timer);
  });

  $effect(() => {
    const reply = gameStore.declarationValidation;
    if (!reply || current?.status !== 'pending' || reply.request_id !== current.request) return;
    const stale = reply.match_id !== frame.match_id || reply.revision !== frame.revision || reply.prompt_id !== frame.prompt?.id || reply.status === 'stale';
    validation = { ...current, status: stale ? 'stale' : reply.status, message: stale ? 'The game changed. Refreshing the actual declaration.' : reply.message };
    if (stale) recoverCurrentGame();
  });

  $effect(() => {
    const active = activeKey;
    if (!active) return;
    let cancelled = false;
    void tick().then(() => {
      if (cancelled) return;
      const row = assignmentsRegion?.querySelector<HTMLElement>(`[data-combat-key="${active}"]`);
      if (row) {
        row.querySelector<HTMLSelectElement>('select:not(:disabled)')?.focus({ preventScroll: true });
        revealWithinScrollRegions(row);
      }
    });
    return () => { cancelled = true; };
  });

  function retryValidation(): void { blocked = null; retry += 1; }
  function confirm(): void {
    if (!canConfirm) return;
    const id = sendDeclaration(declarationPlan(draft, projection));
    if (id) submitted = { key: draft.key, id, signature };
  }
</script>

<section class="combat-composer min-h-0 flex flex-col gap-2" aria-label={projection.stage === 'attackers' ? 'Attack declaration plan' : 'Block declaration plan'} data-testid="combat-composer" data-revision={frame.revision}>
  <h2 class="type-title text-display">Plan {projection.stage}</h2>
  <p class="type-annotation text-ink-2">Edit the whole remaining declaration, then confirm once.</p>
  <p class="type-caption" data-testid="combat-summary">{planned} planned / {editable} eligible · {fixed} already decided · {power} current power</p>
  <p class="type-caption text-ink-2">Reloading or reconnecting discards unconfirmed edits. Committed choices stay fixed.</p>
  {#if readOnly}<p class="type-caption">The pilot chooses this declaration.</p>{/if}
  {#if inspect && !readOnly}<p class="type-caption">Your draft is preserved. Choose Play to edit or confirm.</p>{/if}
  <div class="flex flex-wrap gap-2">
    {#if bulk}<button class="btn btn-secondary btn-sm" disabled={!canEdit} aria-label={`Select all eligible to attack ${destinationLabel(bulk)}`} onclick={() => onChange(selectAllAttackers(draft, projection))}>Select all eligible</button>{/if}
    <button class="btn btn-ghost btn-sm" disabled={!canEdit} onclick={() => onChange(clearCombatDraft(draft))}>Clear</button>
  </div>
  {#if projection.constraints.length > 0}
    <details open={current?.status === 'invalid'} class="type-caption text-ink-2"><summary class="cursor-pointer">Combat requirements ({projection.constraints.length})</summary><ul class="space-y-1 py-2">{#each projection.constraints as constraint}<li>{constraint}</li>{/each}</ul></details>
  {/if}
  <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard access to all assignments in a bounded margin) -->
  <div bind:this={assignmentsRegion} class="combat-assignments min-h-0 overflow-auto" role="region" aria-label="Combat assignments" tabindex="0">
    <ul>
      {#each projection.subjects as row, index (destinationKey(row.object.subject))}
        {#if row.object.subject.kind === 'object'}
          {@const subject = row.object.subject.id}
          {@const assigned = row.fixed ? row.assigned_to : assignmentFor(draft, subject)}
          <li class="border-b border-line py-2" data-combat-key={objectKey(subject)} data-fixed={row.fixed}>
            <div class="flex items-center justify-between gap-2">
              <button class="combat-subject text-left text-ink" aria-pressed={draft.active === objectKey(subject)} disabled={!canEdit || row.fixed} onclick={() => onChange({ ...draft, active: objectKey(subject) })}>{rowLabel(index)}</button>
              <span class="type-caption text-ink-2 whitespace-nowrap">Power {row.power}</span>
            </div>
            {#if row.fixed}
              <p class="type-caption text-ink-2">Committed: {assigned ? destinationLabel(assigned) : projection.stage === 'attackers' ? 'Not attacking' : 'Not blocking'}</p>
            {:else}
              <label class="type-label text-ink-2" for={`combat-${objectKey(subject)}`}>{projection.stage === 'attackers' ? 'Attack destination' : 'Block assignment'}</label>
              <select id={`combat-${objectKey(subject)}`} class="combat-destination block w-full min-w-0 rounded-sm border border-line bg-field text-ink" aria-label={`Assignment for ${rowLabel(index)}`} disabled={!canEdit} value={destinationKey(assigned)} onchange={event => onChange(setAssignment(draft, subject, row.destinations.find(destination => destinationKey(destination) === event.currentTarget.value) ?? null))}>
                <option value="">Unassigned</option>
                {#each row.destinations as destination}<option value={destinationKey(destination)}>{destinationLabel(destination)}</option>{/each}
              </select>
              {#if inspect || readOnly}<p class="type-caption text-ink-2">Destinations: {row.destinations.map(destinationLabel).join('; ') || 'none'}. Unassigned plans still require whole-declaration validation.</p>{/if}
            {/if}
          </li>
        {/if}
      {/each}
    </ul>
  </div>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard scrolling of long rules responses) -->
  <div class="combat-validation type-caption" tabindex="0" role="status" aria-live="polite" aria-atomic="true" data-testid="combat-validation" data-status={current?.status ?? 'unavailable'}>
    {#if !available}Combat view is unavailable. Recover the current declaration.
    {:else if gameStore.commandPending}Confirming the declaration…
    {:else if gameStore.connection !== 'connected'}Waiting for connection. Confirmation is unavailable.
    {:else if readOnly}Waiting for the pilot's declaration.
    {:else if inspect}Inspecting the native choices. Your unconfirmed draft is unchanged.
    {:else if current?.status === 'invalid'}
      <p>This plan is not legal. Edit the assignments before confirming.</p>
      <details><summary class="cursor-pointer">Rules response</summary><p>{current.message}</p></details>
    {:else if current?.status === 'capacity'}
      <p>The rules engine reached a capacity limit. Legality is unknown and confirmation is unavailable. Your draft is preserved.</p>
      <p>{current.message}</p>
    {:else}{current?.message ?? 'Waiting for rules validation.'}{/if}
  </div>
  <div class="flex flex-wrap items-center gap-2">
    <button class="btn btn-primary" disabled={!canConfirm} onclick={confirm}>Confirm {projection.stage}</button>
    {#if canEdit && current?.status === 'transport'}<button class="btn btn-ghost btn-sm" onclick={retryValidation}>Retry validation</button>{/if}
  </div>
</section>

<style>
  .combat-assignments { flex: 1; max-height: 26rem; scrollbar-gutter: stable; }
  .combat-validation { max-height: 5rem; overflow: auto; }
  .combat-validation[data-status='invalid'], .combat-validation[data-status='capacity'], .combat-validation[data-status='transport'] { color: var(--error-text); background: var(--error-wash); padding: 8px; }
  .combat-subject { min-height: 32px; border-bottom: 1px solid transparent; }
  .combat-subject:hover:not(:disabled), .combat-subject[aria-pressed='true'] { border-color: var(--accent); }
  .combat-subject:active:not(:disabled) { background: var(--bg-muted); }
  .combat-subject:disabled { color: var(--text-secondary); }
  .combat-destination { min-height: 32px; padding: 4px 8px; }
  .combat-destination:hover:not(:disabled) { border-color: var(--accent); }
  .combat-destination:disabled { color: var(--text-secondary); }
  @media (min-width: 1024px) and (min-height: 650px) { .combat-composer { flex: 1; } .combat-assignments { max-height: none; } }
  @media (max-width: 640px) { .combat-subject, .combat-destination { min-height: 44px; } }
</style>
