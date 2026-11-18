<script lang="ts">
  /** A printed mana cost as symbols; the accessible text is the cost itself. */
  import { costSymbols } from '$lib/card-faces.svelte';

  let { cost, size = 12 }: { cost: string; size?: number } = $props();
  const symbols = $derived(costSymbols(cost));
  const COLORS: Record<string, string> = { W: 'w', U: 'u', B: 'b', R: 'r', G: 'g', C: 'c' };
  function kind(symbol: string): string {
    if (symbol.includes('/')) return `hybrid ${symbol.split('/').map(part => COLORS[part] ?? '').join(' ')}`;
    return COLORS[symbol] ?? 'generic';
  }
  function style(symbol: string): string | undefined {
    if (!symbol.includes('/')) return undefined;
    const [left, right] = symbol.split('/').map(part => `var(--pip-${COLORS[part] ?? 'c'})`);
    return `background: linear-gradient(135deg, ${left} 50%, ${right} 50%)`;
  }
</script>

{#if symbols.length}
  <span class="pips" style:--pip={`${size}px`} title={cost}>
    <span class="sr-only">{cost}</span>
    {#each symbols as symbol, index (index)}
      <i class={`pip ${kind(symbol)}`} style={style(symbol)} aria-hidden="true">{COLORS[symbol] || symbol.includes('/') ? '' : symbol}</i>
    {/each}
  </span>
{/if}

<style>
  .pips { display: inline-flex; gap: max(1px, calc(var(--pip) * 0.12)); align-items: center; flex: none; }
  .pip {
    width: var(--pip); height: var(--pip); border-radius: 50%; display: inline-grid; place-items: center;
    font: 700 calc(var(--pip) * 0.72)/1 var(--font-sans); font-style: normal; color: #2a2116;
    box-shadow: 0 0 0 1px rgb(20 14 8 / 0.45);
    background: var(--pip-generic);
  }
  .w { background: var(--pip-w); } .u { background: var(--pip-u); } .b { background: var(--pip-b); }
  .r { background: var(--pip-r); } .g { background: var(--pip-g); } .c { background: var(--pip-c); }
</style>
