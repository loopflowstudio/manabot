<script lang="ts">
  /**
   * Everything the table knows about one card, typeset: art, name and cost,
   * type line, keywords (granted ones with their source), rules text, where
   * its power and toughness come from, and its state on the battlefield.
   * The floating preview and the docked pane both render this.
   */
  import { artUrl, loadArt } from '$lib/card-art';
  import { cardFace } from '$lib/card-faces.svelte';
  import type { CardState, PermanentState } from '$lib/types';
  import { keywordsOf, ptProvenance, signed } from './inspect';
  import Pips from './Pips.svelte';

  interface Props {
    name: string;
    text: string | null;
    permanent?: PermanentState | null;
    card?: CardState | null;
    /** The name of the attacker this permanent blocks. */
    blocks?: string | null;
    /** Art height in px; the docked pane uses a shorter strip. */
    art?: number;
  }

  let { name, text, permanent = null, card = null, blocks = null, art = 128 }: Props = $props();

  const face = $derived(cardFace(permanent?.copy_of?.name ?? name));
  const keywords = $derived(permanent ? keywordsOf(permanent, face) : (face?.keywords ?? []).map(keyword => ({ keyword, source: null })));
  const pt = $derived(permanent ? ptProvenance(permanent, face) : null);
  const plain = $derived(!permanent && card?.types.is_creature ? `${card.power}/${card.toughness}` : face?.power != null && !permanent ? `${face.power}/${face.toughness}` : null);
  const facts = $derived(permanent ? [
    permanent.tapped ? 'Tapped' : 'Untapped',
    permanent.summoning_sick && permanent.types?.is_creature ? 'summoning sick' : '',
    permanent.attacking ? 'attacking' : '',
    permanent.blocking !== undefined ? `blocking ${blocks ?? 'an attacker'}` : '',
    permanent.damage ? `${permanent.damage} damage marked` : '',
    ...Object.entries(permanent.counters ?? {}).filter(([, n]) => n > 0).map(([kind, n]) => `${n} ${kind.replace(/_/g, ' ')} counter${n === 1 ? '' : 's'}`),
    permanent.abilities_lost ? 'has lost all abilities' : '',
    permanent.is_token ? 'token' : '',
  ].filter(Boolean).join(' · ') : null);

  const url = $derived(artUrl(name));
  let loaded = $state<string | null>(null);
  $effect(() => {
    const requested = url;
    let active = true;
    void loadArt(requested, name).then(ok => { if (active && ok) loaded = requested; });
    return () => { active = false; };
  });
</script>

<div class="inspect" data-testid="inspect-card">
  <div class="art" style:height={`${art}px`} style:background-image={loaded === url ? `url('${url}')` : undefined}></div>
  <p class="name"><span>{name}</span>{#if face?.cost}<Pips cost={face.cost} size={14} />{/if}</p>
  {#if face}<p class="type">{face.type_line}</p>{/if}
  {#if keywords.length}
    <p class="keywords">
      {#each keywords as fact (fact.keyword)}<span class:granted={fact.source !== null} title={fact.source ?? 'Printed'}>{fact.keyword.replace(/_/g, ' ')}{#if fact.source}<small> · {fact.source}</small>{/if}</span>{/each}
    </p>
  {/if}
  {#if text}<p class="text" data-testid="card-rules-text">{text}</p>{/if}
  {#if pt}
    <div class="pt" data-testid="pt-provenance">
      <p class="total"><b>{pt.power}/{pt.toughness}</b>{#if permanent?.damage}<span> · {permanent.toughness! - permanent.damage} toughness left after damage</span>{/if}</p>
      {#if pt.parts.length > 1 || (pt.parts.length === 1 && !pt.parts[0].base)}
        <ol>
          {#each pt.parts as part, index (index)}
            <li><span class="amount">{part.base ? `${part.power}/${part.toughness}` : signed(part.power, part.toughness)}</span><span>{part.label}</span></li>
          {/each}
        </ol>
      {/if}
    </div>
  {:else if plain}
    <p class="total"><b>{plain}</b></p>
  {/if}
  {#if facts}<p class="state">{facts}</p>{/if}
</div>

<style>
  .inspect { display: grid; gap: 4px; min-width: 0; }
  .art { border-radius: 5px; background: var(--bg-muted) center 28% / cover no-repeat; border: 1px solid var(--border); margin-bottom: 2px; }
  .name { display: flex; justify-content: space-between; align-items: center; gap: 8px; font: 600 19px/1.15 var(--font-serif); color: var(--text); }
  .type { font: 12px/1.3 var(--font-sans); color: var(--text-tertiary); }
  .keywords { display: flex; flex-wrap: wrap; gap: 3px; }
  .keywords span {
    font: 600 11px/16px var(--font-sans); text-transform: capitalize; padding: 0 6px; border-radius: 3px;
    background: var(--bg-muted); color: var(--text);
  }
  .keywords span.granted { background: var(--success-wash); box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--success) 50%, transparent); }
  .keywords small { text-transform: none; font-weight: 400; color: var(--text-secondary); }
  .text { font: 12.5px/1.42 var(--font-sans); color: var(--text); white-space: pre-line; }
  .pt { border-top: 1px solid var(--border); padding-top: 4px; display: grid; gap: 2px; }
  .total { font: 12px/1.4 var(--font-sans); color: var(--text-secondary); }
  .total b { font: 700 13px/1.4 var(--font-mono); color: var(--text); }
  .pt ol { display: grid; gap: 1px; }
  .pt li { display: flex; gap: 8px; font: 11.5px/1.35 var(--font-sans); color: var(--text-secondary); }
  .pt .amount { flex: none; min-width: 44px; font-family: var(--font-mono); color: var(--text); }
  .state { font: 11.5px/1.35 var(--font-mono); color: var(--text-secondary); border-top: 1px solid var(--border); padding-top: 4px; }
</style>
