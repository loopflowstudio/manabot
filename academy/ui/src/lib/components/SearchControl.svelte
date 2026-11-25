<script lang="ts">
  import type { SearchConfig } from '$lib/search';

  interface Props {
    value: SearchConfig | null;
    supported: boolean;
    reason?: string | null;
    label?: string;
    disabled?: boolean;
    onChange: (value: SearchConfig | null) => void;
  }
  let { value, supported, reason = null, label = 'Search', disabled = false, onChange }: Props = $props();

  interface SearchPreset {
    id: string;
    label: string;
    worlds: number;
    depth: number;
    alpha: number;
    beta: number;
    decision_seconds: number;
  }

  // Mirror-search v2 uniform-hand settings (experiments/reports/search-mirror.md):
  // depth 20, alpha 0.02, beta 0.01 across 4 / 8 / 16 worlds.
  const PRESETS: readonly SearchPreset[] = [
    { id: 'quick', label: 'Quick (4 worlds · depth 20)', worlds: 4, depth: 20, alpha: 0.02, beta: 0.01, decision_seconds: 30 },
    { id: 'standard', label: 'Standard (8 worlds · depth 20)', worlds: 8, depth: 20, alpha: 0.02, beta: 0.01, decision_seconds: 30 },
    { id: 'strong', label: 'Strong (16 worlds · depth 20)', worlds: 16, depth: 20, alpha: 0.02, beta: 0.01, decision_seconds: 30 },
  ];

  let customMode = $state(false);

  const matchedPreset = $derived(
    value
      ? PRESETS.find(
          preset =>
            preset.worlds === value.worlds
            && preset.depth === value.depth
            && preset.alpha === value.alpha
            && preset.beta === value.beta,
        ) ?? null
      : null,
  );

  const selectedPreset = $derived.by(() => {
    if (!value) return 'none';
    if (customMode || !matchedPreset) return 'custom';
    return matchedPreset.id;
  });

  function selectLevel(level: string): void {
    if (level === 'none') {
      customMode = false;
      onChange(null);
      return;
    }
    if (level === 'custom') {
      customMode = true;
      if (!value) {
        const fallback = PRESETS[0];
        onChange({
          algorithm: 'local_update',
          belief: 'physical_deal',
          worlds: fallback.worlds,
          depth: fallback.depth,
          decision_seconds: fallback.decision_seconds,
          alpha: fallback.alpha,
          beta: fallback.beta,
        });
      }
      return;
    }
    customMode = false;
    const preset = PRESETS.find(item => item.id === level) ?? PRESETS[0];
    onChange({
      algorithm: 'local_update',
      belief: 'physical_deal',
      worlds: preset.worlds,
      depth: preset.depth,
      decision_seconds: preset.decision_seconds,
      alpha: preset.alpha,
      beta: preset.beta,
    });
  }

  function setBudget(field: 'worlds' | 'depth', event: Event): void {
    const input = event.target as HTMLInputElement;
    if (value) onChange({ ...value, [field]: Number(input.value) });
  }
</script>

<fieldset class="min-w-0 space-y-2 border-0 p-0">
  <legend class="type-label mb-1 text-ink-2">{label}</legend>
  <select
    aria-label={label}
    class="w-full rounded border border-line bg-panel px-3 py-2 text-ink"
    value={selectedPreset}
    {disabled}
    onchange={(event) => selectLevel(event.currentTarget.value)}
  >
    <option value="none">None (raw policy)</option>
    {#each PRESETS as preset (preset.id)}
      <option value={preset.id} disabled={!supported}>{preset.label}</option>
    {/each}
    <option value="custom" disabled={!supported}>Custom…</option>
  </select>
  {#if !supported}
    <p class="type-caption text-ink-2">{reason ?? 'This policy does not support Search.'}</p>
  {/if}
  {#if value}
    {#if selectedPreset === 'custom'}
      <div class="flex flex-wrap gap-3">
        <label class="type-caption">Worlds
          <input class="ml-1 w-20 rounded border border-line bg-panel px-2 py-1" aria-label={`${label} worlds`} type="number" min="1" max="64" step="1" required {disabled} value={value.worlds} onchange={(event) => setBudget('worlds', event)} />
        </label>
        <label class="type-caption">Depth
          <input class="ml-1 w-20 rounded border border-line bg-panel px-2 py-1" aria-label={`${label} depth`} type="number" min="1" max="40" step="1" required {disabled} value={value.depth} onchange={(event) => setBudget('depth', event)} />
        </label>
      </div>
    {/if}
    <p class="type-caption text-ink-2">Uniform belief (physical deal) · α = {value.alpha} · β = {value.beta} · {value.decision_seconds} s per decision. A timeout keeps the decision for explicit retry.</p>
  {/if}
</fieldset>
