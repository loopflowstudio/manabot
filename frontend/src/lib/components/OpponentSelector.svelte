<script lang="ts">
  export let value: 'passive' | 'random' = 'passive';
  export let disabled = false;
  export let onChange: ((value: 'passive' | 'random') => void) | undefined = undefined;
</script>

<label class="flex items-center gap-2 text-sm text-slate-300">
  Opponent
  <select
    class="rounded border border-slate-600 bg-slate-900 px-3 py-2 text-sm"
    value={value}
    disabled={disabled}
    on:change={(event) => onChange?.((event.currentTarget as HTMLSelectElement).value as 'passive' | 'random')}
  >
    <option value="passive">Passive</option>
    <option value="random">Random</option>
  </select>
</label>
