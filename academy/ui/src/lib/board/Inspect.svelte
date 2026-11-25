<script lang="ts">
  /**
   * Everything the board knows about one card, typeset: art, name and cost,
   * type line, keywords (granted ones with their source), rules text, where
   * its power and toughness come from, and its state on the battlefield.
   * The floating preview and the docked pane both render this.
   */
  import { artUrl, loadArt } from '$lib/card-art';
  import { cardFace } from '$lib/card-faces.svelte';
  import type { CardState, PermanentState } from '$lib/types';
  import { counterTerms, keywordsOf, labelText, ptProvenance, signed, statusesOf } from './inspect';
  import Pips from './Pips.svelte';

  interface Props {
    name: string;
    text: string | null;
    permanent?: PermanentState | null;
    card?: CardState | null;
    castTerms?: string[];
    /** The name of the attacker this permanent blocks. */
    blocks?: string | null;
    /** Possessive name of the player this permanent entered under when control changed (e.g. "your", "Opponent's"). */
    enteredUnder?: string | null;
    /** Art height in px; the docked pane uses a shorter strip. */
    art?: number;
  }

  let { name, text, permanent = null, card = null, castTerms = [], blocks = null, enteredUnder = null, art = 128 }: Props = $props();

  const SOURCE_CARD_TYPES = new Set([
    'artifact', 'battle', 'creature', 'enchantment',
    'instant', 'kindred', 'land', 'planeswalker', 'sorcery',
  ]);
  function sourceQuality(quality: string): string {
    return SOURCE_CARD_TYPES.has(quality) ? `${quality} sources` : quality;
  }

  const face = $derived(cardFace(permanent?.copy_of?.name ?? name));
  const keywords = $derived(permanent ? keywordsOf(permanent, face) : (face?.keywords ?? []).map(keyword => ({ keyword, source: null })));
  const statuses = $derived(permanent ? statusesOf(permanent) : []);
  const pt = $derived(permanent ? ptProvenance(permanent, face) : null);
  const plain = $derived(!permanent && card?.types.is_creature ? `${card.power}/${card.toughness}` : face?.power != null && !permanent ? `${face.power}/${face.toughness}` : null);
  const facts = $derived(permanent ? [
    permanent.tapped ? 'Tapped' : 'Untapped',
    permanent.summoning_sick && permanent.types?.is_creature ? 'summoning sick' : '',
    ...(permanent.hexproof_from ?? []).map(quality => `hexproof from ${sourceQuality(quality)}`),
    ...(permanent.protection_from ?? []).map(quality => `protection from ${sourceQuality(quality)}`),
    permanent.blocked ? 'blocked' : '',
    permanent.entry_controller_id !== undefined && permanent.entry_controller_id !== permanent.controller_id
      ? `entered under ${enteredUnder ?? `player #${permanent.entry_controller_id}'s`} control`
      : '',
    permanent.first_strike_participant ? 'first-strike participant' : '',
    permanent.attacking ? 'attacking' : '',
    permanent.blocking !== undefined ? `blocking ${blocks ?? 'an attacker'}` : '',
    ...(permanent.combat_damage_to_players ?? []).map(seat => `dealt combat damage to seat ${seat} this turn`),
    permanent.damage ? `${permanent.damage} damage marked` : '',
    ...counterTerms(permanent.counters),
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
  {#if keywords.length || statuses.length}
    <p class="keywords">
      {#each statuses as status (status)}<span class="granted">{status}</span>{/each}
      {#each keywords as fact (fact.keyword)}<span class:granted={fact.source !== null} title={fact.source ?? 'Printed'}>{labelText(fact.keyword)}{#if fact.source}<small> · {fact.source}</small>{/if}</span>{/each}
    </p>
  {/if}
  {#if text}<p class="text" data-testid="card-rules-text">{text}</p>{/if}
  {#if !permanent}{#each counterTerms(card?.counters) as term (term)}<p class="state">{term}</p>{/each}{/if}
  {#each castTerms as term}<p class="state">{term}</p>{/each}
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
  .art { border-radius: var(--radius-sm); background: var(--bg-muted) center 28% / cover no-repeat; border: 1px solid var(--border); margin-bottom: 2px; }
  .name { display: flex; justify-content: space-between; align-items: center; gap: 8px; font: 600 20px/28px var(--font-serif); color: var(--text); }
  .type { font: 12px/16px var(--font-sans); color: var(--text-tertiary); }
  .keywords { display: flex; flex-wrap: wrap; gap: 3px; }
  .keywords span {
    font: 600 12px/16px var(--font-sans); padding: 0 6px; border-radius: var(--radius-sm);
    background: var(--bg-muted); color: var(--text);
  }
  .keywords span.granted { background: var(--success-wash); box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--success) 50%, transparent); }
  .keywords small { font-weight: 400; color: var(--text-secondary); }
  .text { font: 14px/20px var(--font-sans); color: var(--text); white-space: pre-line; }
  .pt { border-top: 1px solid var(--border); padding-top: 4px; display: grid; gap: 2px; }
  .total { font: 12px/16px var(--font-sans); color: var(--text-secondary); }
  .total b { font: 600 14px/20px var(--font-sans); color: var(--text); }
  .pt ol { display: grid; gap: 1px; }
  .pt li { display: flex; gap: 8px; font: 12px/16px var(--font-sans); color: var(--text-secondary); }
  .pt .amount { flex: none; min-width: 44px; font-weight: 600; color: var(--text); }
  .state { font: 12px/16px var(--font-sans); color: var(--text-secondary); border-top: 1px solid var(--border); padding-top: 4px; }
</style>
