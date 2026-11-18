<script lang="ts">
  /**
   * One card on the table: a permanent, a card in hand, a spell on the stack
   * or a tray candidate. It is a button whenever the viewer may inspect it;
   * its outline says a legal offer acts on it, and its accessible name says
   * which.
   *
   * Two frames. `portrait` (hand, tray) is a printed card's shape: art, then a
   * bar with the name and type line, cost pips on the art. `landscape`
   * (battlefield) puts the name bar on top so names stay readable when a
   * crowded row shrinks its cards, with the art below. Both keep power and
   * toughness on an upright plate. A tapped card keeps its frame and place:
   * its art is desaturated and labelled, so rows never reflow on tap.
   * Rules text lives in the inspect view.
   */
  import { artUrl, loadArt } from '$lib/card-art';
  import { cardFace } from '$lib/card-faces.svelte';
  import type { CardState, PermanentState } from '$lib/types';
  import { getContext } from 'svelte';
  import { keywordsOf } from './inspect';
  import { cardEvidence, EVIDENCE, formatProbability, type OfferEvidence } from './replay';
  import type { PlannedSource } from './payment';
  import Pips from './Pips.svelte';
  import { CARD_RATIO, FRAME_RATIO, type PlacedOffer } from './view';

  interface Props {
    name: string;
    width: number;
    shape?: 'portrait' | 'landscape';
    permanent?: PermanentState | null;
    card?: CardState | null;
    offers?: PlacedOffer[];
    /** Offers may be committed now. */
    live?: boolean;
    asked?: boolean;
    /** The side of the centre line this card sits on; attacks lean toward it. */
    toward?: 'up' | 'down' | null;
    /** The name of the attacker this permanent blocks; the line on the table draws it. */
    blocks?: string | null;
    /** Whose plate a card new to the table travels from. */
    side?: 'you' | 'opponent';
    /** It attacked, blocked, was dealt damage or was targeted since your last move. */
    touched?: boolean;
    /** The opponent's move being read out is about this card. */
    beat?: boolean;
    /** Damage it was dealt since your last move, shown as it lands. */
    dealt?: number;
    /** Its place in the automatic payer's plan: "would tap, for {U}". */
    planned?: PlannedSource | null;
    /** Copies in this card's pile, shown on the top copy. */
    count?: number | null;
    onactivate?: (element: HTMLElement) => void;
    onpreview?: (element: HTMLElement | null) => void;
  }

  let {
    name, width, shape = 'portrait', permanent = null, card = null, offers = [], live = false, asked = false,
    toward = null, blocks = null, side = 'you', touched = false, beat = false, dealt = 0, planned = null, count = null, onactivate, onpreview,
  }: Props = $props();

  // At a recorded choice: the policy's mass on this card's offers, and whether one was played.
  const recordedPolicy = getContext<(() => OfferEvidence | null) | undefined>(EVIDENCE);
  const recorded = $derived(cardEvidence(recordedPolicy?.() ?? null, offers.map(placed => placed.offer.id)));

  const landscape = $derived(shape === 'landscape');
  const height = $derived(Math.round(width / (landscape ? FRAME_RATIO : CARD_RATIO)));
  const tapped = $derived(permanent?.tapped ?? false);
  const creature = $derived(permanent ? permanent.types?.is_creature ?? permanent.power !== null : card?.types.is_creature ?? false);
  const power = $derived(permanent?.power ?? card?.power ?? null);
  const toughness = $derived(permanent?.toughness ?? card?.toughness ?? null);
  const face = $derived(cardFace(permanent?.copy_of?.name ?? name));
  const keywords = $derived(permanent ? keywordsOf(permanent, face) : (face?.keywords ?? []).map(keyword => ({ keyword, source: null })));
  const counters = $derived(Object.entries({
    ...(permanent?.plus1_counters ? { '+1/+1': permanent.plus1_counters } : {}),
    ...(permanent?.counters ?? {}),
  }).filter(([, amount]) => amount > 0));

  // Type sizes follow the frame, within a readable floor and a quiet ceiling.
  const clamp = (value: number, low: number, high: number): number => Math.max(low, Math.min(high, value));
  const bar = $derived(landscape ? clamp(Math.round(height * 0.2), 14, 22) : Math.round(height * 0.26));
  const nameSize = $derived(landscape ? clamp(bar - 4.5, 9.5, 14) : clamp(height * 0.078, 9, 15));
  const typeSize = $derived(clamp(height * 0.05, 8.5, 11));
  const showType = $derived(!landscape && height >= 110);
  const pip = $derived(landscape ? clamp(bar - 5, 10, 14) : clamp(height * 0.07, 10, 14));
  const showCost = $derived(landscape ? width >= 96 : width >= 58);
  const plate = $derived(clamp(height * (landscape ? 0.17 : 0.085), 10, 14));
  const showKeywords = $derived(landscape && height >= 62 && keywords.length > 0);

  const url = $derived(artUrl(name));
  let loaded = $state<string | null>(null);
  $effect(() => {
    const requested = url;
    let active = true;
    void loadArt(requested, name).then(ok => { if (active && ok) loaded = requested; });
    return () => { active = false; };
  });
  const art = $derived(loaded === url ? url : null);

  const facts = $derived([
    tapped ? 'tapped' : '',
    permanent?.summoning_sick && creature ? 'summoning sick' : '',
    permanent?.attacking ? 'attacking' : '',
    permanent?.blocking !== undefined ? `blocking ${blocks ?? 'an attacker'}` : '',
    permanent?.damage ? `${permanent.damage} damage` : '',
    ...counters.map(([kind, amount]) => `${amount} ${kind} counter${amount === 1 ? '' : 's'}`),
    permanent?.is_token ? 'token' : '',
    dealt ? `dealt ${dealt} since your last move` : '',
    planned ? `would tap for ${planned.output || 'mana'} (payment plan, source ${planned.order})` : '',
    recorded?.p != null ? `recorded policy ${formatProbability(recorded.p)}` : '',
    recorded?.chosen ? 'chosen' : '',
  ].filter(Boolean).join(', '));
  const label = $derived([
    `${name}${creature && power !== null ? ` ${power}/${toughness}` : ''}`,
    facts,
    offers.length === 1 ? `${offers[0].offer.label} (${offers[0].number})`
      : offers.length > 1 ? `${offers.length} choices: ${offers.map(placed => placed.offer.label).join('; ')}` : '',
  ].filter(Boolean).join(' — '));
</script>

<div
  class="slot"
  class:lean-up={permanent?.attacking && toward === 'up'}
  class:lean-down={permanent?.attacking && toward === 'down'}
  style:width={`${width}px`}
  style:height={`${height}px`}
>
  <button
    type="button"
    class="tcard"
    class:landscape
    class:tapped
    class:legal={live && offers.length > 0}
    class:offered={!live && recorded !== null}
    class:chosen={recorded?.chosen}
    class:asked
    class:attacking={permanent?.attacking}
    class:touched
    class:beat
    class:token={permanent?.is_token}
    class:planned={planned !== null}
    data-testid="table-card"
    data-travel-id={permanent ? `perm:${permanent.id}` : card ? `card:${card.id}` : undefined}
    data-travel-name={name}
    data-travel-side={side}
    data-object-id={permanent?.id ?? card?.id}
    data-offer-count={offers.length}
    data-payment-source={planned?.order}
    data-card-name={name}
    aria-label={label}
    style:width={`${width}px`}
    style:height={`${height}px`}
    style:--bar={`${bar}px`}
    onclick={event => onactivate?.(event.currentTarget)}
    onmouseenter={event => onpreview?.(event.currentTarget)}
    onmouseleave={() => onpreview?.(null)}
    onfocus={event => onpreview?.(event.currentTarget)}
    onblur={() => onpreview?.(null)}
  >
    <span class="art" aria-hidden="true" style:background-image={art ? `url('${art}')` : undefined}>
      {#if !art}<span class="no-art">{face?.type_line ?? ''}</span>{/if}
    </span>
    <span class="bar" aria-hidden="true">
      <span class="name" style:font-size={`${nameSize}px`}>{name}</span>
      {#if count}<span class="count" style:font-size={`${Math.max(9, nameSize - 1)}px`}>×{count}</span>{/if}
      {#if landscape && showCost && face?.cost}<Pips cost={face.cost} size={pip} />{/if}
      {#if showType && face}<span class="type" style:font-size={`${typeSize}px`}>{face.type_line}</span>{/if}
    </span>
    {#if !landscape && showCost && face?.cost}<span class="cost" aria-hidden="true"><Pips cost={face.cost} size={pip} /></span>{/if}
    {#if tapped}<span class="tapped-mark" aria-hidden="true">Tapped</span>{/if}
    {#if counters.length && width >= 56}
      <span class="counters" aria-hidden="true">{counters.map(([kind, amount]) => `${kind === '+1/+1' ? '+1' : kind.replace(/_/g, ' ')} ×${amount}`).join(' · ')}</span>
    {/if}
    {#if showKeywords}
      <span class="keywords" aria-hidden="true">
        {#each keywords.slice(0, width >= 120 ? 3 : 1) as fact (fact.keyword)}<i class:granted={fact.source !== null}>{fact.keyword.replace(/_/g, ' ')}</i>{/each}
        {#if keywords.length > (width >= 120 ? 3 : 1)}<i>+{keywords.length - (width >= 120 ? 3 : 1)}</i>{/if}
      </span>
    {/if}
    {#if creature && power !== null}
      <span class="pt" class:hurt={(permanent?.damage ?? 0) > 0}
        class:up={permanent && ((permanent.power ?? 0) > (face?.power ?? Infinity) || (permanent.toughness ?? 0) > (face?.toughness ?? Infinity))}
        style:font-size={`${plate}px`} aria-hidden="true">{power}/{toughness}{#if permanent?.damage}<small>−{permanent.damage}</small>{/if}</span>
    {/if}
    {#if offers.length > 1 && live}<span class="many" aria-hidden="true">{offers.length}</span>{/if}
    {#if planned}
      <span class="plan" aria-hidden="true"><b>{planned.order}</b>{#if planned.output}<Pips cost={planned.output} size={Math.max(10, pip)} />{/if}</span>
    {/if}
    {#if recorded && (recorded.p !== null || recorded.chosen)}
      <span class="policy" aria-hidden="true" data-testid="card-policy" data-chosen={recorded.chosen}>{#if recorded.chosen}✓{/if}{recorded.p !== null ? formatProbability(recorded.p) : ''}</span>
    {/if}
    {#if dealt}{#key dealt}<span class="dealt" aria-hidden="true" data-testid="damage-dealt">−{dealt}</span>{/key}{/if}
  </button>
</div>

<style>
  .slot { position: relative; flex: none; transition: transform var(--motion-move) var(--motion-move-ease); }
  /* An attacker steps toward the player it attacks by `--lean`, which its row sizes (Field). */
  .slot.lean-up { transform: translateY(calc(-1 * var(--lean, 20%))); }
  .slot.lean-down { transform: translateY(var(--lean, 20%)); }

  .tcard {
    position: relative; display: block; overflow: hidden; padding: 0; cursor: pointer; text-align: left;
    border-radius: max(4px, 5%); border: 1px solid rgb(20 14 8 / 0.55);
    background: var(--bg-surface); color: var(--text);
    box-shadow: 0 1px 2px rgb(20 14 8 / 0.25);
    transition: transform var(--motion-move) var(--motion-move-ease), box-shadow var(--motion-fast) ease-out;
  }
  .tcard:focus-visible { outline: 2px solid var(--text); outline-offset: 3px; }

  /* Portrait: art over a bar of name and type line. */
  .art { position: absolute; left: 0; right: 0; top: 0; bottom: var(--bar); background: var(--bg-muted) center 30% / cover no-repeat; transition: filter var(--motion-fast); }
  .no-art { position: absolute; inset: 4px; display: grid; place-items: center; text-align: center; font: italic 10px/1.2 var(--font-serif); color: var(--text-tertiary); overflow: hidden; }
  .bar {
    position: absolute; left: 0; right: 0; bottom: 0; height: var(--bar); padding: 3px 5px 0; min-width: 0;
    display: flex; flex-direction: column; gap: 1px; background: var(--bg-surface); border-top: 1px solid var(--border);
  }
  .name { font-family: var(--font-sans); font-weight: 700; line-height: 1.15; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .type { font-family: var(--font-sans); line-height: 1.2; color: var(--text-tertiary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .cost { position: absolute; top: 4px; right: 4px; padding: 1px 2px; border-radius: 8px; background: rgb(247 240 224 / 0.82); display: flex; }

  /* Landscape: a name bar on top, art below; cost at the bar's end. */
  .landscape .art { top: var(--bar); bottom: 0; background-position: center 28%; }
  .landscape .bar {
    top: 0; bottom: auto; flex-direction: row; align-items: center; gap: 4px; padding: 0 4px 0 5px;
    border-top: 0; border-bottom: 1px solid var(--border);
  }
  .landscape .name { flex: 1 1 auto; min-width: 0; }

  .token .bar { background: color-mix(in srgb, var(--bg-muted) 60%, var(--bg-surface)); }

  /* Tapped: same frame and place, spent colour, labelled. */
  .tapped .art { filter: saturate(0.25) brightness(0.78); }
  .tapped .bar { background: var(--bg-muted); }
  .tapped .name { color: var(--text-secondary); }
  .tapped-mark {
    position: absolute; left: 0; right: 0; top: calc(var(--bar) + (100% - var(--bar)) / 2 - 7px); text-align: center;
    font: 600 9px/14px var(--font-mono); letter-spacing: 0.14em; text-transform: uppercase;
    color: #f8f1e0; background: rgb(20 14 8 / 0.55);
  }
  .tcard:not(.landscape) .tapped-mark { top: calc((100% - var(--bar)) / 2 - 7px); }

  .tcard.attacking { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent), 0 6px 14px rgb(20 14 8 / 0.3); }
  .tcard.touched { box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 55%, transparent), 0 1px 2px rgb(20 14 8 / 0.25); }
  .tcard.legal { box-shadow: 0 0 0 2px var(--legal), 0 0 12px color-mix(in srgb, var(--legal) 45%, transparent); }
  .tcard.legal:hover { box-shadow: 0 0 0 3px var(--legal), 0 0 16px color-mix(in srgb, var(--legal) 60%, transparent); transform: translateY(-2px); }
  /* A recorded choice's legal space, read-only: offered cards are outlined thinly, the played one solidly. */
  .tcard.offered { box-shadow: 0 0 0 1px color-mix(in srgb, var(--legal) 70%, transparent); }
  .tcard.chosen { box-shadow: 0 0 0 2px var(--text), 0 4px 10px rgb(20 14 8 / 0.3); }
  .policy {
    position: absolute; left: 50%; bottom: 3px; transform: translateX(-50%); padding: 0 5px; border-radius: 8px;
    font: 700 10.5px/15px var(--font-sans); font-variant-numeric: tabular-nums; white-space: nowrap;
    color: var(--text); background: var(--bg-surface); border: 1px solid var(--border-strong);
  }
  .policy[data-chosen='true'] { color: var(--bg-surface); background: var(--text); border-color: var(--text); }
  .tcard:not(.landscape) .policy { bottom: calc(var(--bar) + 3px); }
  .tcard.beat { box-shadow: 0 0 0 3px var(--accent), 0 0 18px color-mix(in srgb, var(--accent) 55%, transparent); transition: box-shadow 160ms ease-out; }
  .tcard.asked { box-shadow: 0 0 0 2px var(--accent), 0 8px 18px rgb(20 14 8 / 0.35); transform: translateY(-8px); }

  /* Planned payment: dashed, and what tapping it adds. Distinct from the solid "legal" outline. */
  .tcard.planned { outline: 2px dashed var(--planned); outline-offset: 2px; }
  .plan {
    position: absolute; left: 50%; top: calc(var(--bar) + 3px); transform: translateX(-50%);
    display: inline-flex; align-items: center; gap: 3px; padding: 1px 4px 1px 3px; border-radius: 9px;
    background: var(--bg-surface); border: 1px dashed var(--planned); box-shadow: 0 1px 3px rgb(20 14 8 / 0.3);
  }
  .tcard:not(.landscape) .plan { top: 3px; }
  .plan b { font: 700 10px/14px var(--font-sans); color: var(--planned-text); min-width: 9px; text-align: center; }
  .dealt {
    position: absolute; left: 50%; top: 50%; transform: translate(-50%, -30%);
    font: 800 clamp(13px, 30%, 22px)/1 var(--font-sans); font-size: max(13px, calc(var(--bar) * 1.05)); color: #fff;
    padding: 2px 6px; border-radius: 5px; background: #9b2f22; box-shadow: 0 2px 6px rgb(20 14 8 / 0.4);
    animation: land calc(var(--motion-move) * 1.6) var(--motion-move-ease);
  }
  @keyframes land { from { transform: translate(-50%, -90%) scale(1.25); opacity: 0; } }
  .pt {
    position: absolute; right: 3px; bottom: 3px; padding: 0 0.35em; border-radius: 4px;
    font-family: var(--font-sans); font-weight: 700; line-height: 1.35; font-variant-numeric: tabular-nums;
    color: #2a2116; background: #f3ead6; border: 1px solid rgb(20 14 8 / 0.5);
  }
  .tcard:not(.landscape) .pt { bottom: calc(var(--bar) + 3px); }
  .pt.up { color: #2f5f26; }
  .pt.hurt { color: #7e2a20; }
  .pt small { font-size: 0.75em; margin-left: 2px; }
  .keywords { position: absolute; left: 3px; bottom: 3px; display: flex; gap: 2px; max-width: 62%; overflow: hidden; }
  .keywords i {
    font: 600 9px/13px var(--font-sans); font-style: normal; text-transform: capitalize; white-space: nowrap;
    padding: 0 4px; border-radius: 3px; color: #f8f1e0; background: rgb(12 9 5 / 0.72);
  }
  .keywords i.granted { box-shadow: inset 0 0 0 1px #9ecb8b; }
  .many {
    position: absolute; top: calc(var(--bar) + 3px); right: 3px; min-width: 17px; height: 17px; border-radius: 9px; padding: 0 4px;
    font-family: var(--font-sans); font-size: 10.5px; font-weight: 700; line-height: 15px; text-align: center;
    color: #f8f1e0; background: var(--legal); border: 1px solid rgb(20 14 8 / 0.5);
  }
  .tcard:not(.landscape) .many { top: 3px; left: 3px; right: auto; }
  .count {
    flex: none; padding: 0 4px; border-radius: 3px; font-family: var(--font-sans); font-weight: 700; line-height: 1.25;
    font-variant-numeric: tabular-nums; color: var(--bg-surface); background: var(--text);
  }
  .tcard:not(.landscape) .count { position: absolute; top: 3px; right: 3px; }
  .counters {
    position: absolute; top: 3px; left: 3px; max-width: calc(100% - 6px); padding: 0 4px; border-radius: 4px; text-align: left;
    font-family: var(--font-sans); font-size: 9.5px; font-weight: 700; line-height: 14px;
    color: #f8f1e0; background: rgb(12 9 5 / 0.72); white-space: nowrap; overflow: hidden;
  }
  .landscape .counters { top: calc(var(--bar) + 3px); }
  @media (prefers-reduced-motion: reduce) {
    .slot, .tcard, .art { transition: none; }
    .dealt { animation: none; }
  }
</style>
