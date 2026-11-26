<script lang="ts">
  /**
   * Mana and action symbols ({W}{U}{B}{R}{G}{C}{T}{Q}{2}…); the accessible
   * text stays the exact braced string so screen readers and text matchers read
   * it verbatim. Coloured mana uses the bundled Scryfall SVGs under `/mana/`;
   * tap/untap use crisp inline SVGs; generic and hybrid use circular pips.
   */
  import { costSymbols } from '$lib/card-faces.svelte';

  let { cost, size = 12 }: { cost: string; size?: number } = $props();
  const symbols = $derived(costSymbols(cost));
  const SVG_MANA = new Set(['W', 'U', 'B', 'R', 'G']);
  const COLORS: Record<string, string> = { W: 'w', U: 'u', B: 'b', R: 'r', G: 'g', C: 'c' };
  function kind(symbol: string): string {
    if (symbol.includes('/')) return `hybrid ${symbol.split('/').map(part => COLORS[part] ?? '').join(' ')}`;
    return COLORS[symbol] ?? 'generic';
  }
  const GLYPHS: Record<string, string> = { W: 'W', U: 'U', B: 'B', R: 'R', G: 'G', C: '◇' };
  function glyph(symbol: string): string {
    if (symbol.includes('/')) return '';
    return GLYPHS[symbol] ?? symbol;
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
      {#if SVG_MANA.has(symbol)}
        <img class="pip-img" src={`/mana/${symbol}.svg`} alt="" aria-hidden="true" />
      {:else if symbol === 'T'}
        <i class="pip tap" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
            <path d="M5.5 13a6.5 6.5 0 0 1 11.4-4.2L19 11" />
            <path d="M19 6.5V11h-4.5" />
          </svg>
        </i>
      {:else if symbol === 'Q'}
        <i class="pip untap" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
            <path d="M18.5 13a6.5 6.5 0 0 0-11.4-4.2L5 11" />
            <path d="M5 6.5V11h4.5" />
          </svg>
        </i>
      {:else}
        <i class={`pip ${kind(symbol)}`} style={style(symbol)} data-glyph={glyph(symbol)} aria-hidden="true"></i>
      {/if}
    {/each}
  </span>
{/if}

<style>
  .pips { display: inline-flex; gap: max(1px, calc(var(--pip) * 0.12)); align-items: center; flex: none; }
  .pip-img {
    width: var(--pip); height: var(--pip); border-radius: 50%; display: inline-block; flex: none;
    box-shadow: 0 0 0 1px var(--pip-edge);
  }
  .pip {
    width: var(--pip); height: var(--pip); border-radius: 50%; display: inline-grid; place-items: center; flex: none;
    font: 600 calc(var(--pip) * 0.7)/1 var(--font-sans); color: var(--pip-ink);
    box-shadow: 0 0 0 1px var(--pip-edge);
    background: var(--pip-generic);
  }
  .pip::before { content: attr(data-glyph); }
  .pip svg { width: 82%; height: 82%; display: block; }
  .tap::before, .untap::before { content: none; }
  .untap { background: var(--pip-ink); color: var(--pip-generic); }
  .w { background: var(--pip-w); } .u { background: var(--pip-u); } .b { background: var(--pip-b); }
  .r { background: var(--pip-r); } .g { background: var(--pip-g); } .c { background: var(--pip-c); }
  .w, .u, .b, .r, .g { font-size: calc(var(--pip) * 0.62); }
  .c { font-size: calc(var(--pip) * 0.74); font-weight: 400; }
</style>
