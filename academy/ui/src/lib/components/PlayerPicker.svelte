<script lang="ts">
  import { onMount } from 'svelte';
  import { runnerReplacement, rememberPolicy, rememberedPolicy } from '$lib/policy-selection';
  import type { Decklists, PolicyRuntimeIdentity, SettingsFitReport, TrainedOpponent } from '$lib/types';
  import { policyChoice, type PlayerChoice } from '$lib/game.svelte';
  import type { SearchConfig } from '$lib/search';
  import SearchControl from './SearchControl.svelte';
  import { builtinOpponentName, catalogOpponentName } from '$lib/opponent-name';

  interface Props {
    value?: PlayerChoice;
    trained?: TrainedOpponent[];
    extension?: string | null;
    catalogError?: string | null;
    decklists?: Decklists;
    manaPayment?: 'manual' | 'auto';
    manaFloats?: 'off' | 'usable' | 'all';
    botRecords?: Record<string, { wins: number; losses: number }>;
    setup?: Record<string, unknown>;
    disabled?: boolean;
    label?: string;
    onChange?: (value: PlayerChoice) => void;
    search?: SearchConfig | null;
    onSearchChange?: (value: SearchConfig | null) => void;
  }
  let {
    value = 'search-64',
    trained = [],
    extension = undefined,
    catalogError = null,
    setup,
    decklists = 'open',
    manaPayment = 'manual',
    manaFloats = 'off',
    botRecords = {},
    disabled = false,
    label = 'Opponent',
    onChange,
    search = null,
    onSearchChange,
  }: Props = $props();
  const uid = $props.id();
  const selected = $derived(trained.find(entry => policyChoice(entry.sha256) === value));
  const custom = $derived(value.startsWith('policy:'));
  let prior = $state<PolicyRuntimeIdentity | null>(null);
  let mounted = $state(false);
  onMount(() => { mounted = true; });
  $effect(() => {
    if (!mounted) return;
    prior = custom && !selected ? rememberedPolicy(value.slice(7)) : null;
    if (selected?.policy) rememberPolicy(selected.policy);
  });

  // Built-in rollout Search requires open decklists; policy-backed search uses
  // physical-deal belief and works on closed lists.
  $effect(() => {
    if (decklists === 'closed' && value.startsWith('search-') && onChange) {
      const preferred = trained.find(entry => entry.recommended && entry.available)
        ?? trained.find(entry => entry.available);
      onChange(preferred ? policyChoice(preferred.sha256) : 'random');
    }
  });

  let compatibility = $state<string | null>(null);
  let fitReport = $state<SettingsFitReport | null>(null);
  let checking = $state(false);
  $effect(() => {
    const config = setup;
    const policy = selected;
    const disclosure = decklists;
    const payment = manaPayment;
    const floats = manaFloats;
    compatibility = null;
    fitReport = null;
    checking = false;
    if (!mounted || !config || !policy?.available) return;
    const controller = new AbortController();
    checking = true;
    const timer = setTimeout(() => {
      void fetch('/api/policies/check', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          ...config,
          decklists: disclosure,
          mana_payment: payment,
          mana_floats: floats,
          villain_type: 'policy',
          opponent_sha256: policy.sha256,
        }),
      }).then(async response => {
        if (!response.ok) throw new Error('Setup compatibility could not be checked. Start will check again.');
        return response.json() as Promise<{ compatible: boolean; reason: string | null; fit?: SettingsFitReport | null }>;
      }).then(result => {
        if (!controller.signal.aborted) {
          compatibility = result.compatible ? null : result.reason;
          fitReport = result.fit ?? null;
        }
      }).catch(error => {
        if (!controller.signal.aborted) compatibility = error instanceof Error ? error.message : 'Setup compatibility could not be checked.';
      }).finally(() => { if (!controller.signal.aborted) checking = false; });
    }, 150);
    return () => { clearTimeout(timer); controller.abort(); };
  });

  const replacement = $derived(prior ? runnerReplacement(prior, trained) : null);
  const native = ['search-16', 'search-64', 'search-256', 'random', 'passive'];
  const groupedTrained = $derived.by(() => {
    const groups = new Map<string, TrainedOpponent[]>();
    for (const entry of trained) {
      const key = entry.group || entry.producer || 'Policies';
      const bucket = groups.get(key) ?? [];
      bucket.push(entry);
      groups.set(key, bucket);
    }
    return [...groups.entries()];
  });

  function title(entry: TrainedOpponent): string {
    const rec = botRecords[entry.sha256];
    const score = rec && (rec.wins + rec.losses) > 0
      ? ` · Bot ${rec.wins}W–${rec.losses}L`
      : '';
    return `${catalogOpponentName(entry)}${score} · ${entry.sha256.slice(0, 10)}${entry.available ? '' : ` — ${entry.reason ?? 'unavailable'}`}`;
  }

  const selectedRecord = $derived.by(() => {
    const key = selected ? selected.sha256 : value;
    const rec = botRecords[key];
    if (!rec || rec.wins + rec.losses === 0) return null;
    const total = rec.wins + rec.losses;
    return `Bot record: ${rec.wins}W–${rec.losses}L (${Math.round((rec.wins / total) * 100)}%)`;
  });
</script>

<div class="flex min-w-0 max-w-full flex-wrap items-end gap-3">
  <label class="flex min-w-0 max-w-full flex-col gap-1.5">
    <span class="type-label text-ink-2">{label}</span>
    <select data-testid="opponent-select" class="w-full min-w-0 max-w-[28rem] rounded border border-line bg-field px-3 py-2"
      value={custom ? 'policy' : value} {disabled} aria-describedby={`${uid}-details`}
      onchange={event => {
        const next = event.currentTarget.value;
        if (next === 'policy') {
          const preferred = trained.find(entry => entry.recommended && entry.available)
            ?? trained.find(entry => entry.available);
          onChange?.(preferred ? policyChoice(preferred.sha256) : 'policy:');
        } else {
          onChange?.(next as PlayerChoice);
        }
      }}>
      {#each native as kind}
        <option value={kind} disabled={kind.startsWith('search-') && decklists === 'closed'}>
          {builtinOpponentName(kind as PlayerChoice)}{kind.startsWith('search-') && decklists === 'closed' ? ' — needs open decklists' : ''}
        </option>
      {/each}
      <option value="policy">Registered policy</option>
    </select>
  </label>
  {#if custom}
    <label class="flex min-w-0 max-w-full flex-col gap-1.5">
      <span class="type-label text-ink-2">Policy</span>
      <select data-testid="checkpoint-select" class="min-w-0 max-w-[36rem] rounded border border-line bg-field px-3 py-2" {value} {disabled}
        onchange={event => onChange?.(event.currentTarget.value as PlayerChoice)}>
        <option value="policy:" disabled>Choose a policy</option>
        {#if groupedTrained.length > 1}
          {#each groupedTrained as [groupLabel, entries] (groupLabel)}
            <optgroup label={groupLabel}>
              {#each entries as entry (entry.sha256)}
                <option data-testid="trained-opponent" value={policyChoice(entry.sha256)} disabled={!entry.available}>{title(entry)}</option>
              {/each}
            </optgroup>
          {/each}
        {:else}
          {#each trained as entry (entry.sha256)}
            <option data-testid="trained-opponent" value={policyChoice(entry.sha256)} disabled={!entry.available}>{title(entry)}</option>
          {/each}
        {/if}
        {#if !selected && value !== 'policy:'}<option {value}>Unavailable · {value.slice(7, 17)}</option>{/if}
      </select>
    </label>
  {/if}
  <p id={`${uid}-details`} data-testid="opponent-details" class="type-caption min-w-0 w-full break-words text-ink-2">
    {#if selected}
      {#if selected.summary}{selected.summary} · {/if}
      {#if selectedRecord}{selectedRecord} · {/if}
      <span class="text-ink-3">Policy ID: {selected.sha256.length > 16 ? `${selected.sha256.slice(0, 12)}…` : selected.sha256}</span>
      {#if !selected.available}{' · '}{selected.reason}{/if}
      {#if !selected.supports_search}{' · '}Search unavailable: {selected.search_reason ?? 'this policy does not advertise compatible search evaluation.'}{/if}
    {:else if replacement}The runner changed since this was selected. Reselect the policy to use the current runner.
    {:else if custom && value !== 'policy:'}The selected policy is no longer available.
    {:else if value.startsWith('search-')}Built-in: {value.slice(7)} random playouts per legal action, no policy{#if selectedRecord}{' · '}{selectedRecord}{/if}
    {:else if value === 'random'}Built-in: uniform random legal action{#if selectedRecord}{' · '}{selectedRecord}{/if}
    {:else if value === 'passive'}Built-in: passes when possible, otherwise takes the first legal action{#if selectedRecord}{' · '}{selectedRecord}{/if}
    {/if}
    {#if !selected && catalogError}{' '}{catalogError}
    {:else if !selected && !trained.length && extension !== undefined}{' '}No policies registered on this server. Supply a Python policy or install manabot for hosted policies.{/if}
  </p>
  {#if checking || compatibility}
    <p role="status" data-testid="policy-compatibility" class="type-caption w-full text-mountain-ink">{checking ? 'Checking setup compatibility…' : compatibility}</p>
  {/if}
  {#if !compatibility && fitReport && fitReport.status === 'mismatch' && fitReport.summary}
    <p role="status" data-testid="policy-fit" class="type-caption w-full rounded bg-plains-wash px-2.5 py-1.5 text-plains-ink">
      Trained outside these settings: {fitReport.summary}. Informs only; you can still play.
    </p>
  {:else if !compatibility && fitReport && fitReport.status === 'unknown'}
    <p data-testid="policy-fit" class="type-caption w-full text-ink-3">Training settings unrecorded for this policy.</p>
  {/if}
  {#if replacement}
    <button type="button" {disabled} class="rounded border border-line px-3 py-2"
      onclick={() => { onChange?.(policyChoice(replacement.sha256)); }}>Reselect {replacement.name}</button>
  {/if}
  {#if selected}
    <details class="type-caption min-w-0 w-full break-words text-ink-2">
      <summary>Policy provenance</summary>
      <p>Policy ID: {selected.sha256} · {selected.producer}{#if selected.group}{' · '}{selected.group}{/if}</p>
      {#if selected.policy}<p>Runner: {selected.policy.runner}</p>{/if}
      {#each selected.details as detail}<p>{detail.label}: {detail.value}</p>{/each}
      {#if selected.policy}{#each Object.entries(selected.policy.artifacts) as [role, digest]}<p>{role}: {digest}</p>{/each}{/if}
    </details>
  {/if}
</div>
{#if onSearchChange && (custom || search !== null)}
  <SearchControl value={search} supported={selected?.supports_search === true}
    reason={selected?.search_reason}
    label={`${label} search`} {disabled} onChange={value => onSearchChange?.(value)} />
{/if}
