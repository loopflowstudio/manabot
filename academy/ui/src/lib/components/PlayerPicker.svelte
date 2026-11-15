<script lang="ts">
  import { onMount } from 'svelte';
  import { runnerReplacement, rememberPolicy, rememberedPolicy } from '$lib/policy-selection';
  import type { PolicyRuntimeIdentity } from '$lib/types';
  import { policyChoice, type PlayerChoice } from '$lib/game.svelte';
  import type { Decklists, TrainedOpponent } from '$lib/types';
  import type { MirrorSearchSpec } from '$lib/watch';
  import SearchControl from './SearchControl.svelte';

  interface Props {
    value?: PlayerChoice;
    trained?: TrainedOpponent[];
    extension?: string | null;
    catalogError?: string | null;
    decklists?: Decklists;
    disabled?: boolean;
    label?: string;
    referencePlayers?: string[];
    onChange?: (value: PlayerChoice) => void;
    search?: MirrorSearchSpec | null;
    onSearchChange?: (value: MirrorSearchSpec | null) => void;
  }
  let { value = 'search-64', trained = [], extension = undefined, catalogError = null,
    decklists = 'open', disabled = false, label = 'Opponent', referencePlayers,
    onChange, search = null, onSearchChange }: Props = $props();
  const uid = $props.id();
  const selected = $derived(trained.find(entry => policyChoice(entry.sha256) === value));
  const custom = $derived(value.startsWith('policy:'));
  let chosenGroup = $state('');
  let prior = $state<PolicyRuntimeIdentity | null>(null);
  let mounted = $state(false);
  onMount(() => { mounted = true; });
  $effect(() => {
    if (!mounted) return;
    prior = custom && !selected ? rememberedPolicy(value.slice(7)) : null;
    if (selected?.policy) rememberPolicy(selected.policy);
  });
  const replacement = $derived(prior ? runnerReplacement(prior, trained) : null);
  function groupKey(entry: TrainedOpponent): string {
    return entry.group && entry.group !== entry.producer
      ? `${entry.producer} · ${entry.group}` : entry.producer;
  }
  const groups = $derived([...new Set(trained.map(groupKey))].sort());
  const group = $derived(chosenGroup || (selected ? groupKey(selected) : ''));
  const entries = $derived(trained.filter(entry => groupKey(entry) === group));
  const native = $derived(referencePlayers
    ? ['random', 'search-16', 'search-64', 'search-256', ...referencePlayers.filter(p => !['random', 'search'].includes(p))]
    : ['search-16', 'search-64', 'search-256', 'random', 'passive']);
  function title(entry: TrainedOpponent): string {
    return `${entry.name} · ${entry.sha256.slice(0, 10)}${entry.available ? '' : ` — ${entry.reason ?? 'unavailable'}`}`;
  }
</script>

<div class="flex min-w-0 max-w-full flex-wrap items-end gap-3">
  <label class="flex min-w-0 max-w-full flex-col gap-1.5">
    <span class="type-label text-ink-2">{label}</span>
    <select data-testid="opponent-select" class="w-full min-w-0 max-w-[28rem] rounded border border-line bg-field px-3 py-2"
      value={custom ? 'policy' : value} {disabled} aria-describedby={`${uid}-details`}
      onchange={event => { chosenGroup = ''; onChange?.(event.currentTarget.value === 'policy' ? 'policy:' : event.currentTarget.value as PlayerChoice); }}>
      {#each native as kind}
        <option value={kind} disabled={kind.startsWith('search-') && decklists === 'closed'}>
          {kind.startsWith('search-') ? `Search ${kind.slice(7)}` : kind.charAt(0).toUpperCase() + kind.slice(1)}{kind.startsWith('search-') && decklists === 'closed' ? ' — needs open decklists' : ''}
        </option>
      {/each}
      <option value="policy">Registered policy</option>
    </select>
  </label>
  {#if custom}
    <label class="flex min-w-0 max-w-full flex-col gap-1.5">
      <span class="type-label text-ink-2">Provider / group</span>
      <select class="min-w-0 max-w-[24rem] rounded border border-line bg-field px-3 py-2" value={group} {disabled}
        onchange={event => { chosenGroup = event.currentTarget.value; onChange?.('policy:'); }}>
        <option value="" disabled>Choose a provider or group</option>
        {#each groups as key}<option value={key}>{key}</option>{/each}
      </select>
    </label>
    <label class="flex min-w-0 max-w-full flex-col gap-1.5">
      <span class="type-label text-ink-2">Policy</span>
      <select data-testid="checkpoint-select" class="min-w-0 max-w-[36rem] rounded border border-line bg-field px-3 py-2" {value} {disabled}
        onchange={event => onChange?.(event.currentTarget.value as PlayerChoice)}>
        <option value="policy:" disabled>Choose a policy</option>
        {#each entries as entry (entry.sha256)}
          <option data-testid="trained-opponent" value={policyChoice(entry.sha256)} disabled={!entry.available}>{title(entry)}</option>
        {/each}
        {#if !selected && value !== 'policy:'}<option {value}>Unavailable · {value.slice(7, 17)}</option>{/if}
      </select>
    </label>
  {/if}
  <p id={`${uid}-details`} data-testid="opponent-details" class="type-caption min-w-0 w-full break-words text-ink-2">
    {#if selected}
      Policy ID: {selected.sha256} · {selected.producer}
      {#each selected.details as detail} · {detail.label}: {detail.value}{/each}
      {#if !selected.available} · {selected.reason}{/if}
      {#if !selected.supports_search} · Mirror search unavailable: {selected.search_reason ?? 'this policy does not advertise compatible search evaluation.'}{/if}

    {:else if replacement}The runner changed since this was selected. Reselect the policy to use the current runner.
    {:else if custom && value !== 'policy:'}The selected policy is no longer available.
    {:else if catalogError}{catalogError}
    {:else if !trained.length && extension !== undefined}No policies registered on this server. Supply a Python policy factory or a saved specification.{/if}
  </p>
  {#if replacement}
    <button type="button" {disabled} class="rounded border border-line px-3 py-2"
      onclick={() => { chosenGroup = groupKey(replacement); onChange?.(policyChoice(replacement.sha256)); }}>Reselect {replacement.name}</button>
  {/if}
  {#if selected?.policy}
    <details class="type-caption min-w-0 w-full break-words text-ink-2">
      <summary>Policy provenance</summary>
      <p>Runner: {selected.policy.runner}</p>
      {#each Object.entries(selected.policy.artifacts) as [role, digest]}<p>{role}: {digest}</p>{/each}
    </details>
  {/if}
</div>
{#if onSearchChange && (custom || search !== null)}
  <SearchControl value={search} supported={selected?.supports_search === true && decklists === 'open'}
    reason={decklists === 'closed' ? 'Mirror search requires open decklists.' : selected?.search_reason}
    label={`${label} search`} {disabled} onChange={value => onSearchChange?.(value)} />
{/if}
