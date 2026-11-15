<script lang="ts">
  import type { MirrorSearch } from '$lib/mirror';

  interface Props {
    value: MirrorSearch | null;
    supported: boolean;
    reason?: string | null;
    label?: string;
    disabled?: boolean;
    onChange: (value: MirrorSearch | null) => void;
  }
  let { value, supported, reason = null, label = 'Search', disabled = false, onChange }: Props = $props();

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
    value={value ? 'mirror' : 'none'}
    {disabled}
    onchange={(event) => onChange(event.currentTarget.value === 'mirror' ? {
      algorithm: 'local_update', belief: 'physical_deal', worlds: 1, depth: 1,
      decision_seconds: 30, alpha: 0.02, beta: 1,
    } : null)}
  >
    <option value="none">None (raw policy)</option>
    <option value="mirror" disabled={!supported}>Mirror search</option>
  </select>
  {#if !supported}
    <p class="type-caption text-ink-2">{reason ?? 'This policy does not support Mirror search.'}</p>
  {/if}
  {#if value}
    <div class="flex flex-wrap gap-3">
      <label class="type-caption">Worlds
        <input class="ml-1 w-20 rounded border border-line bg-panel px-2 py-1" aria-label={`${label} worlds`} type="number" min="1" max="64" step="1" required {disabled} value={value.worlds} onchange={(event) => setBudget('worlds', event)} />
      </label>
      <label class="type-caption">Depth
        <input class="ml-1 w-20 rounded border border-line bg-panel px-2 py-1" aria-label={`${label} depth`} type="number" min="1" max="40" step="1" required {disabled} value={value.depth} onchange={(event) => setBudget('depth', event)} />
      </label>
    </div>
    <p class="type-caption text-ink-2">Uniform belief (physical deal) · {value.decision_seconds} s per decision. A timeout keeps the decision for explicit retry.</p>
  {/if}
</fieldset>
